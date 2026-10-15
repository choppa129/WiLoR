"""Numerical checks of the paper-specific representation and FreiHAND protocol."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilor.utils.geometry import rotmat_to_axis_angle
from wilor.utils.freihand_metrics import align_freihand, mesh_metrics
from wilor.models.backbones.vit import ViT
from scripts.prepare_paper_mean_params import encode_rotation_mean

torch.set_num_threads(2)
torch.manual_seed(42)
rotations = Rotation.random(100, random_state=42)
matrix = torch.tensor(rotations.as_matrix(), dtype=torch.double, requires_grad=True)
np.testing.assert_allclose(rotmat_to_axis_angle(matrix).detach(), rotations.as_rotvec(), atol=1e-12)
assert torch.autograd.gradcheck(rotmat_to_axis_angle, (matrix[:3],))
from wilor.utils.geometry import rot6d_to_rotmat
torch.testing.assert_close(rot6d_to_rotmat(encode_rotation_mean(matrix.detach())), matrix.detach())
for angle in [0., 1e-10, np.pi-1e-7, np.pi]:
    r = torch.tensor(Rotation.from_rotvec([angle,0,0]).as_matrix(), requires_grad=True)
    value = rotmat_to_axis_angle(r)
    value.square().sum().backward()
    assert torch.isfinite(r.grad).all()
    np.testing.assert_allclose(value.detach(), [angle,0,0], atol=1e-12)

pred = torch.randn(3, 778, 3, dtype=torch.double) * .03
target = torch.randn_like(pred) * .03
aligned = align_freihand(pred, target).numpy()
for i in range(len(pred)):
    a, b = target[i].numpy(), pred[i].numpy()
    ma, mb = a.mean(0), b.mean(0)
    na, nb = np.linalg.norm(a-ma)+1e-8, np.linalg.norm(b-mb)+1e-8
    r, scale = orthogonal_procrustes((a-ma)/na, (b-mb)/nb)
    expected = (b-mb)/nb @ r.T * scale * na + ma
    np.testing.assert_allclose(aligned[i], expected, atol=1e-12)
reflected = pred * torch.tensor([-2.,2.,2.]) + .4
metrics = mesh_metrics(reflected, pred)
assert metrics['pa_mpvpe_mm'] < 1e-5
assert metrics['f_at_5mm'] == 1 and metrics['f_at_15mm'] == 1

with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp)/'means.npz')
    np.savez(path, pose=np.tile(np.eye(3)[:,:2].T.reshape(-1),16), shape=np.zeros(10), cam=[1.,0,0])
    cfg = OmegaConf.create({'MODEL':{'PAPER_LITERAL':True,'MANO_HEAD':{}},
                            'MANO':{'NUM_HAND_JOINTS':15,'MEAN_PARAMS':path}})
    model = ViT(img_size=(256,192), embed_dim=32, depth=1, num_heads=4, cfg=cfg)
    observed = []
    hook = model.blocks[0].register_forward_pre_hook(lambda m,args: observed.append(args[0].shape))
    params, cam, feats, image = model(torch.randn(2,3,256,192))
    assert observed == [torch.Size([2,195,32])]
    assert feats['hand_pose'].shape == (2,96) and cam.shape == (2,3)
    assert image.shape == (2,32,16,12)
    (sum(t.square().sum() for t in params.values()) + cam.square().sum()).backward()
    assert model.pose_emb.weight.grad is not None
    assert torch.isfinite(model.pose_emb.weight.grad).all()
    hook.remove()
print('PASS: 3 tokens, 96D output, axis-angle values/gradients, official Procrustes and mesh F-scores')
