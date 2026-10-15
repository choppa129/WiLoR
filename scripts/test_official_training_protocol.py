"""Check two-GPU recipe and numerical agreement with official FreiHAND metrics."""
import sys
from pathlib import Path
import numpy as np
import torch
from hydra import compose, initialize_config_dir
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wilor.utils.official_freihand_eval import evaluate_predictions
from wilor.utils.freihand_metrics import align_freihand, mesh_metrics

with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs_hydra'),version_base='1.2'):
    cfg=compose(config_name='train',overrides=['experiment=wilor_b200_2gpu'])
assert cfg.trainer.devices*cfg.TRAIN.BATCH_SIZE==1024
assert cfg.trainer.max_epochs==1000 and cfg.trainer.max_steps==-1
assert cfg.TRAIN.OPTIMIZER=='adamw' and cfg.TRAIN.OFFICIAL_LOSSES
assert cfg.MODEL.NUM_POSE_TOKENS==16 and cfg.MODEL.PROJECTION=='perspective'
assert cfg.MODEL.BACKBONE.USE_CHECKPOINT
assert cfg.DATASETS.CONFIG.TRANS_FACTOR==.02 and cfg.DATASETS.CONFIG.TRANS_AUG_RATE==.5
np.testing.assert_array_equal(np.load(cfg.MANO.MEAN_PARAMS)['cam'],[4,0,0])
g=torch.arange(1024.,dtype=torch.float64)
a=sum(t.sum() for t in g.split(128))/8
b=sum(t.sum()*(cfg.TRAIN.REFERENCE_PER_RANK_BATCH_SIZE/512) for t in g.split(512))/2
torch.testing.assert_close(a,b,rtol=0,atol=0)
rng=np.random.default_rng(42)
j=rng.normal(size=(3,21,3))*.03;v=rng.normal(size=(3,778,3))*.03
pj=j*1.4+.2+rng.normal(size=j.shape)*.003;pv=v*.8-.1+rng.normal(size=v.shape)*.003
result=evaluate_predictions(pj,pv,j,v)
metrics=mesh_metrics(torch.tensor(pv),torch.tensor(v))
pa=(align_freihand(torch.tensor(pj),torch.tensor(j))-torch.tensor(j)).norm(dim=-1).mean()*1000
np.testing.assert_allclose(result['pa_mpjpe_mm'],pa,atol=1e-9)
for k,x in metrics.items():np.testing.assert_allclose(result[k],x,atol=1e-7)
identity=evaluate_predictions(j,v,j,v)
assert identity['pa_mpjpe_mm']<1e-5 and identity['pa_mpvpe_mm']<1e-5 and identity['f_at_5mm']==1
print('PASS: composed 2x512 recipe, reference DDP loss scale, official PA/F-score agreement')
