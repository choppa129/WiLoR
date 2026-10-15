"""Fit weak cameras on a reproducible training subset, without network predictions."""
import json
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wilor.configs import dataset_config
from wilor.datasets.finite_freihand import FiniteFreiHAND
from wilor.models import MANO
from wilor.utils.geometry import aa_to_rotmat, weak_perspective_projection

torch.set_num_threads(8)
with initialize_config_dir(version_base='1.2',config_dir=str(ROOT/'configs_hydra')):
    cfg=compose(config_name='train',overrides=['experiment=wilor_paper'])
data=FiniteFreiHAND(cfg,dataset_config(cfg.dataset_config)['FREIHAND-TRAIN'].URLS,train=True)
ids=np.random.default_rng(2026).choice(len(data),512,replace=False)
records=[]
for i in ids:
    p,_,_,_,offset,size,index=data.entries[i]
    with open(p,'rb') as f:
        f.seek(offset)
        records.append(pickle.loads(f.read(size))[index])
mano=MANO(**{k.lower():v for k,v in cfg.MANO.items()})
with torch.no_grad():
    rotations=aa_to_rotmat(torch.tensor(np.stack([r['hand_pose'] for r in records])).float().reshape(-1,3)).reshape(-1,16,3,3)
    shape=torch.tensor(np.stack([r['betas'] for r in records])).float()
    xyz=mano(global_orient=rotations[:,:1],hand_pose=rotations[:,1:],betas=shape,pose2rot=False).joints
    y=torch.tensor(np.stack([(r['keypoints_2d'][:,:2]-r['center'])/np.max(r['scale']*200) for r in records])).float()
    w=torch.tensor(np.stack([r['keypoints_2d'][:,2] for r in records])).float()[...,None]
    x=xyz[...,:2]
    mx=(x*w).sum(1)/w.sum(1).clamp_min(1e-8)
    my=(y*w).sum(1)/w.sum(1).clamp_min(1e-8)
    dx,dy=x-mx[:,None],y-my[:,None]
    a=(w*dx*dy).sum((1,2))/(w*dx.square()).sum((1,2)).clamp_min(1e-12)
    fit=torch.cat([2*a[:,None],(my-a[:,None]*mx)/a[:,None]],-1)
    cameras={'official':torch.tensor([4.,0,0]).repeat(len(ids),1),
             'fitted_mean':torch.tensor(np.load(cfg.MANO.MEAN_PARAMS)['cam']).repeat(len(ids),1),
             'per_sample_fit':fit}
    report={'samples':len(ids),'units':'reprojection error in 256px full crop; GT MANO geometry, unaugmented',
            'confidence_min_max':[float(w.min()),float(w.max())],
            'fitted_camera_mean':fit.mean(0).tolist(),
            'fitted_camera_quantiles':{str(q):torch.quantile(fit,q,dim=0).tolist() for q in [.05,.5,.95]},
            'negative_fitted_scales':int((a<=0).sum()),'projection_errors':{}}
    for name,cam in cameras.items():
        error=torch.linalg.vector_norm(weak_perspective_projection(xyz,cam)-y,dim=-1)*256
        per_sample=(error*w[...,0]).sum(1)/w[...,0].sum(1).clamp_min(1)
        report['projection_errors'][name]={'mean_px':float(per_sample.mean()),'median_px':float(per_sample.median())}
report['camera_only_optimization']={'description':'Diagnostic only: fixed GT MANO joints, learn a separate (s,tx,ty) for each sample; L1 projection, Adam lr=0.01, 1000 steps. Not WiLoR network training.', 'runs':{}}
for name in ['official','fitted_mean']:
    cam=torch.nn.Parameter(cameras[name].clone())
    optimizer=torch.optim.Adam([cam],lr=.01)
    history=[]
    for step in range(1001):
        projected=weak_perspective_projection(xyz,cam)
        loss=((projected-y).abs()*w).mean()
        if step in [0,50,100,500,1000]:
            with torch.no_grad():
                e=torch.linalg.vector_norm(projected-y,dim=-1)*256
                history.append({'step':step,'reprojection_px':float(e.mean()),'mean_camera':cam.mean(0).tolist()})
        if step==1000:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    report['camera_only_optimization']['runs'][name]=history
out=ROOT/'evaluation/camera_initialization_20261014/geometry_audit.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
