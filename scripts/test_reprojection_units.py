"""Exercise the actual validation method with known pixel displacements."""
import sys
from pathlib import Path
from types import SimpleNamespace
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilor.models.wilor import WiLoR

torch.manual_seed(42)
for size in [128, 256]:
    from omegaconf import OmegaConf
    owner = SimpleNamespace(cfg=OmegaConf.create({'MODEL': {'IMAGE_SIZE': size}}))
    xyz = torch.randn(1, 21, 3)
    gt = torch.zeros(1,21,3)
    gt[...,:2] = -.25
    gt[0,:2,2] = torch.tensor([1.,.5])
    pred = gt[...,:2].clone()
    pred[0,0] += torch.tensor([3.,4.])/size
    pred[0,1] += torch.tensor([6.,8.])/size
    pred[0,2:] += 100  # invisible joints must not affect the metric
    batch = {'keypoints_2d':gt, 'keypoints_3d':torch.cat([xyz,torch.ones(1,21,1)],-1),
             'has_mano_params':{k:torch.zeros(1,dtype=torch.bool) for k in ['betas','hand_pose','global_orient']}}
    result=WiLoR.compute_validation_metrics(owner,batch,{'pred_keypoints_3d':xyz,'pred_keypoints_2d':pred})
    torch.testing.assert_close(result['val/reprojection_px'],torch.tensor(10/1.5))
    batch['keypoints_2d'][...,2]=0
    result=WiLoR.compute_validation_metrics(owner,batch,{'pred_keypoints_3d':xyz,'pred_keypoints_2d':pred})
    assert result['val/reprojection_px']==0
print('PASS: 3-4-5 and 6-8-10 pixel errors, confidence weights, masked joints, 128/256 crops.')
