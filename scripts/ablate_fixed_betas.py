"""Hold trained pose/camera fixed and intervene only on MANO shape."""
import sys
import csv
import json
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wilor.models.mano_wrapper import MANO
from swap_scale_cameras import project, extent

BASE = ROOT / 'evaluation/scale_diagnostic'


def main():
    torch.set_num_threads(4)
    mano = MANO(model_path=str(ROOT/'mano_data'), gender='neutral',
                num_hand_joints=15, create_body_pose=False).eval()
    out = BASE/'betas_ablation'
    out.mkdir(exist_ok=True)
    records = json.loads((BASE/'fixed_inputs/manifest.json').read_text())['samples']
    rows, errors = [], []
    for rec in records:
        sid = rec['id']
        def load(model, stage):
            with np.load(BASE/'predictions'/model/f'{sid}_{stage}.npz') as z:
                return dict(z)
        trained = load('trained','after')
        official = load('official','after')
        hamer = load('hamer','final')
        shapes = {'trained':trained['betas'], 'official':official['betas'],
                  'hamer':hamer['betas'], 'zero':np.zeros(10, np.float32)}
        panels=[]
        for source, betas in shapes.items():
            with torch.inference_mode():
                result = mano(global_orient=torch.from_numpy(trained['global_orient']).unsqueeze(0),
                    hand_pose=torch.from_numpy(trained['hand_pose']).unsqueeze(0),
                    betas=torch.from_numpy(betas).unsqueeze(0), pose2rot=False)
            verts = result.vertices[0].numpy()
            joints = result.joints[0].numpy()
            assert np.isfinite(verts).all() and np.isfinite(joints).all()
            if source == 'trained':
                err = max(float(np.max(np.abs(verts-trained['vertices']))),
                          float(np.max(np.abs(joints-trained['joints_3d']))))
                errors.append(err)
                assert err < 1e-5, f'{sid}: MANO reproduction failed: {err}'
            mesh = dict(trained, joints_3d=joints)
            uv = project(mesh,trained,'joints_3d')
            mirrored=verts.copy(); mirrored[:,0] *= 2*float(trained['right'])-1
            vv = project(dict(trained,vertices_full=mirrored),trained,'vertices_full')
            row = dict(sample=sid, betas_source=source, beta_norm=float(np.linalg.norm(betas)),
                joint_extent_px=extent(uv), vertex_extent_px=extent(vv),
                joint_ratio_to_official=extent(uv)/extent(official['joints_2d_full']),
                mesh_diag_m=extent(verts), palm_width_m=float(np.linalg.norm(joints[5]-joints[17])))
            for finger, start in zip(['thumb','index','middle','ring','pinky'], [1,5,9,13,17]):
                row[f'{finger}_chain_m']=float(np.linalg.norm(np.diff(joints[start:start+4],axis=0),axis=1).sum())
            rows.append(row)
            np.savez_compressed(out/f'{sid}_{source}.npz',betas=betas,vertices=verts,
                joints_3d=joints,joints_2d_full=uv,vertices_2d_full=vv,
                global_orient=trained['global_orient'],hand_pose=trained['hand_pose'],
                cam_t_full=trained['cam_t_full'],right=trained['right'])
            image=cv2.imread(str(ROOT/rec['image']))
            center=np.array(rec['box_center']); scale=360/rec['box_size']
            trans=np.array([[scale,0,180-scale*center[0]],[0,scale,180-scale*center[1]]])
            panel=cv2.warpAffine(image,trans,(360,360))
            for points,color in [(trained['joints_2d_full'],(160,160,160)),(uv,(0,220,0))]:
                pts=points*scale+trans[:,2]
                for x,y in pts:
                    if 0<=x<360 and 0<=y<360: cv2.circle(panel,(round(x),round(y)),3,color,-1)
            panel=cv2.copyMakeBorder(panel,35,0,0,0,cv2.BORDER_CONSTANT)
            cv2.putText(panel,f'betas={source} extent={extent(uv):.1f}px',(5,23),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            panels.append(panel)
        cv2.imwrite(str(out/f'{sid}.jpg'),np.concatenate(panels,axis=1))
    with (out/'per_sample.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    baseline={r['sample']:r for r in rows if r['betas_source']=='trained'}
    lines=['# Betas ablation', '', '45 fixed crops; trained final pose and camera held constant; CPU MANO.',
        f'Maximum baseline MANO reproduction error: {max(errors):.9g} m.', '',
        '| Betas | Median joint extent / trained | Median joint extent / official | Median palm width / trained | Median vertex extent / trained |',
        '|---|---:|---:|---:|---:|']
    for source in shapes:
        subset=[r for r in rows if r['betas_source']==source]
        def ratio(key): return np.median([r[key]/baseline[r['sample']][key] for r in subset])
        lines.append(f'| {source} | {ratio("joint_extent_px"):.4f} | '+
            f'{np.median([r["joint_ratio_to_official"] for r in subset]):.4f} | '+
            f'{ratio("palm_width_m"):.4f} | {ratio("vertex_extent_px"):.4f} |')
    lines += ['', 'Palm width is joint 5 to 17 (index MCP to little-finger MCP).',
        'Finger lengths in CSV are sums of three segments from MCP to fingertip, not skin-surface lengths.',
        'Gray dots: trained baseline; green: beta intervention. All panels use the same crop window.',
        'Official/HaMeR betas are reference predictions, not ground truth. Zero betas means MANO mean shape.',
        'Changes describe sensitivity at fixed pose/camera; they do not prove better accuracy or identify a training defect.',
        'Results across 45 detector proposals from 8 images are descriptive and may include false positives.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__': main()
