"""Analytic weak-camera and multi-scale crop-coordinate checks."""
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wilor.utils.geometry import weak_perspective_projection, weak_perspective_feature_grid

points = torch.tensor([[[.1, -.2, 0.], [.1, -.2, 100.]]], dtype=torch.float64,
                      requires_grad=True)
camera = torch.tensor([[2., .3, .4]], dtype=torch.float64, requires_grad=True)
xy = weak_perspective_projection(points, camera)
torch.testing.assert_close(xy, torch.tensor([[[.4, .2], [.4, .2]]], dtype=torch.float64))
assert torch.autograd.gradcheck(weak_perspective_projection, (points, camera))
assert torch.autograd.gradcheck(weak_perspective_feature_grid, (points, camera))
xy.sum().backward()
assert points.grad[..., 2].count_nonzero() == 0

# Construct coordinate-ramp feature maps. A known full-image pixel must sample
# the same physical position on every pyramid scale after the horizontal crop.
pixels = torch.tensor([[[80., 90.], [170., 180.]]], dtype=torch.float64)
xyz = torch.cat((pixels / 256 - .5, torch.zeros(1, 2, 1, dtype=torch.float64)), -1)
cam = torch.tensor([[2., 0., 0.]], dtype=torch.float64)
grid = weak_perspective_feature_grid(xyz, cam).unsqueeze(2)
for h, w in [(16, 12), (32, 24), (64, 48)]:
    # align_corners=False samples centers; express features in full-crop pixels.
    xs = (torch.arange(w, dtype=torch.float64) + .5) * 192 / w + 32 - .5
    ys = (torch.arange(h, dtype=torch.float64) + .5) * 256 / h - .5
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    features = torch.stack((xx, yy)).unsqueeze(0)
    sampled = F.grid_sample(features, grid, align_corners=False).squeeze(-1).transpose(1, 2)
    torch.testing.assert_close(sampled, pixels)
print('PASS: weak-camera analytic values, depth invariance, gradients and three-scale crop sampling')
