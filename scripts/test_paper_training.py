"""CPU checks for the paper training path, without the large ViT allocation."""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilor.models.wilor import WiLoR
from wilor.models.losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss


class TinyMANO:
    def __call__(self, **params):
        assert torch.isfinite(params['betas']).all()
        count = len(params['betas'])
        return SimpleNamespace(vertices=torch.zeros(count, 4, 3))


cfg = OmegaConf.create({
    'MODEL': {'PAPER_LITERAL': True},
    'TRAIN': {'PAPER_MODE': True, 'LR': 1e-5, 'WEIGHT_DECAY': 1e-4},
    'LOSS_WEIGHTS': {'VERTICES_3D': .05, 'KEYPOINTS_3D': 0.,
                     'KEYPOINTS_2D': .01, 'GLOBAL_ORIENT': .001,
                     'HAND_POSE': .001, 'BETAS': .0005, 'ADVERSARIAL': .0005},
})
owner = SimpleNamespace(cfg=cfg, mano=TinyMANO(),
                        keypoint_2d_loss=Keypoint2DLoss(),
                        keypoint_3d_loss=Keypoint3DLoss(),
                        mano_parameter_loss=ParameterLoss())
owner.compute_vertex_loss = lambda b, o: WiLoR.compute_vertex_loss(owner, b, o)
batch = {
    'mano_params': {'global_orient': torch.zeros(2, 3),
                    'hand_pose': torch.zeros(2, 45), 'betas': torch.zeros(2, 10)},
    'has_mano_params': {k: torch.tensor([1., 0.])
                        for k in ('global_orient', 'hand_pose', 'betas')},
    'mano_params_is_axis_angle': {k: torch.ones(2, dtype=torch.bool)
                                 for k in ('global_orient', 'hand_pose', 'betas')},
    'keypoints_2d': torch.zeros(2, 21, 3),
    'keypoints_3d': torch.zeros(2, 21, 4),
}
pred = torch.ones(2, 4, 3, requires_grad=True)
batch['mano_params_is_axis_angle']['betas'].zero_()
output = {'pred_vertices': pred, 'pred_keypoints_2d': torch.zeros(2, 21, 2),
          'pred_keypoints_3d': torch.zeros(2, 21, 3),
          'pred_mano_params': {
              'global_orient': torch.eye(3).repeat(2, 1, 1, 1),
              'hand_pose': torch.eye(3).repeat(2, 15, 1, 1),
              'betas': torch.zeros(2, 10)}}
loss = WiLoR.compute_loss(owner, batch, output)
torch.testing.assert_close(loss, torch.tensor(.3))  # .05 * 12 / batch size 2
loss.backward()
torch.testing.assert_close(pred.grad[0], torch.full((4, 3), .025))
assert pred.grad[1].count_nonzero() == 0
# Check the actual loss is squared distance in axis-angle space, not matrices.
from wilor.utils.geometry import aa_to_rotmat
angle = torch.tensor([[0., 0., .7], [0., 0., 0.]], requires_grad=True)
output['pred_mano_params']['global_orient'] = aa_to_rotmat(angle).reshape(2,1,3,3)
axis_loss = WiLoR.compute_loss(owner, batch, output)
torch.testing.assert_close(axis_loss, torch.tensor(.3 + .001 * .7**2 / 2))
axis_loss.backward()
torch.testing.assert_close(angle.grad[0,2], torch.tensor(.0007))
# Missing annotations must never be decoded, even if placeholders contain NaNs.
batch['mano_params']['betas'][1] = float('nan')
assert owner.compute_vertex_loss(batch, output).item() == 12
for mask in batch['has_mano_params'].values():
    mask.zero_()
assert owner.compute_vertex_loss(batch, output).item() == 0

backbone, refine, disc = [torch.nn.Linear(2, 2) for _ in range(3)]
owner.get_parameters = lambda: list(backbone.parameters()) + list(refine.parameters())
owner.discriminator = disc
optim, optim_disc = WiLoR.configure_optimizers(owner)
assert type(optim) is torch.optim.Adam and type(optim_disc) is torch.optim.Adam
assert {id(p) for g in optim.param_groups for p in g['params']} == {
    id(p) for p in owner.get_parameters()}
cfg.TRAIN.PAPER_MODE = False
assert type(WiLoR.configure_optimizers(owner)[0]) is torch.optim.AdamW
cfg.LOSS_WEIGHTS.ADVERSARIAL = 0
assert isinstance(WiLoR.configure_optimizers(owner), torch.optim.Optimizer)
print('PASS: paper loss, gradient masking, absent labels, optimizer coverage and legacy mode')
