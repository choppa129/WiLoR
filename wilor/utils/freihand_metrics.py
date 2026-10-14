"""FreiHAND protocol geometry (meters), matching the official eval.py.

Reference: https://github.com/lmb-freiburg/freihand/blob/master/eval.py
Orthogonal Procrustes permits reflections, as scipy.orthogonal_procrustes does
in the official evaluator. Joints and vertices are aligned independently.
"""
import torch


def align_freihand(pred, target):
    pred_mean, target_mean = pred.mean(1, keepdim=True), target.mean(1, keepdim=True)
    x, y = pred - pred_mean, target - target_mean
    nx = torch.linalg.vector_norm(x, dim=(1, 2), keepdim=True) + 1e-8
    ny = torch.linalg.vector_norm(y, dim=(1, 2), keepdim=True) + 1e-8
    x, y = x / nx, y / ny
    u, singular, vh = torch.linalg.svd(x.transpose(1, 2) @ y)
    return (x @ (u @ vh)) * singular.sum(-1)[:, None, None] * ny + target_mean


def mesh_metrics(pred, target):
    aligned = align_freihand(pred, target)
    result = {'pa_mpvpe_mm': torch.linalg.vector_norm(aligned - target, dim=-1).mean() * 1000}
    scores = {5: [], 15: []}
    for a, b in zip(aligned.split(16), target.split(16)):
        distances = torch.cdist(a, b)
        near_pred, near_gt = distances.amin(-1), distances.amin(-2)
        for mm in scores:
            precision = (near_pred < mm / 1000).float().mean(-1)
            recall = (near_gt < mm / 1000).float().mean(-1)
            scores[mm].append(2 * precision * recall / (precision + recall).clamp_min(1e-12))
    result.update({f'f_at_{mm}mm': torch.cat(values).mean() for mm, values in scores.items()})
    return result
