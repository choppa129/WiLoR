"""Compute training-only FreiHAND pose/shape/camera means for paper mode.

The paper does not publish its averaging or camera fitting procedure. Here we
average annotated axis-angle/shape vectors and fit a shared XY scale + translation
to unaugmented crop keypoints for each annotated MANO mesh. These are recorded
implementation choices, not a claim about the authors' unpublished preprocessing.
"""
import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import initialize_config_dir, compose

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wilor.configs import dataset_config
from wilor.datasets.finite_freihand import FiniteFreiHAND
from wilor.models import MANO
from wilor.utils.geometry import aa_to_rotmat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    with initialize_config_dir(version_base='1.2', config_dir=str(ROOT / 'configs_hydra')):
        cfg = compose(config_name='train', overrides=['experiment=wilor_paper'])
    registry = dataset_config(cfg.dataset_config)
    dataset = FiniteFreiHAND(cfg, registry['FREIHAND-TRAIN'].URLS, train=True)
    mano = MANO(**{k.lower():v for k,v in cfg.MANO.items()}).to(args.device)
    poses, shapes, cameras = [], [], []
    rejected = 0
    with torch.no_grad():
        for start in range(0, len(dataset), 512):
            records = []
            for entry in dataset.entries[start:start+512]:
                path, _, _, _, offset, size, index = entry
                with open(path, 'rb') as stream:
                    stream.seek(offset)
                    record = pickle.loads(stream.read(size))[index]
                if record['has_hand_pose'] and record['has_betas']:
                    records.append(record)
            if not records:
                continue
            pose = np.stack([r['hand_pose'] for r in records])
            shape = np.stack([r['betas'] for r in records])
            rotations = aa_to_rotmat(torch.tensor(pose, device=args.device).float().reshape(-1,3)).reshape(-1,16,3,3)
            xyz = mano(global_orient=rotations[:, :1], hand_pose=rotations[:, 1:],
                       betas=torch.tensor(shape,device=args.device).float(), pose2rot=False).joints
            targets, weights = [], []
            for r in records:
                box = np.max(r['scale'] * 200)
                targets.append((r['keypoints_2d'][:, :2] - r['center']) / box)
                weights.append(r['keypoints_2d'][:, 2])
            y = torch.tensor(np.stack(targets), device=args.device).float()
            w = torch.tensor(np.stack(weights), device=args.device).float()[...,None]
            x = xyz[..., :2]
            mx = (x*w).sum(1) / w.sum(1).clamp_min(1e-8)
            my = (y*w).sum(1) / w.sum(1).clamp_min(1e-8)
            dx, dy = x-mx[:,None], y-my[:,None]
            denom = (w*dx.square()).sum((1,2))
            a = (w*dx*dy).sum((1,2)) / denom.clamp_min(1e-12)
            valid = (a>1e-8) & (denom>1e-12) & torch.isfinite(a)
            cam = torch.cat((2*a[:,None], (my-a[:,None]*mx)/a[:,None].clamp_min(1e-8)), -1)
            cameras.append(cam[valid].cpu().numpy())
            rejected += int((~valid).sum())
            poses.append(pose); shapes.append(shape)
    if not poses or not cameras or not any(len(c) for c in cameras):
        raise RuntimeError('No valid training annotations / camera fits for initialization')
    camera_mean = np.concatenate(cameras).astype(np.float64).mean(0).astype(np.float32)
    if not np.isfinite(camera_mean).all():
        raise RuntimeError('Non-finite training camera mean')
    pose_mean = np.concatenate(poses).astype(np.float64).mean(0)
    mean_rot = aa_to_rotmat(torch.tensor(pose_mean).float().reshape(-1,3))
    destination = Path(cfg.MANO.MEAN_PARAMS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, pose=mean_rot[:, :, :2].reshape(96).numpy(),
             shape=np.concatenate(shapes).mean(0).astype(np.float32),
             cam=camera_mean)
    provenance = {'training_examples':len(dataset), 'rejected_camera_fits':rejected,
                  'seed':cfg.seed, 'holdout':cfg.DATASETS.FREIHAND_HOLDOUT,
                  'training_ids_sha256':hashlib.sha256('\n'.join(e[1] for e in dataset.entries).encode()).hexdigest(),
                  'method':__doc__, 'mean_camera':np.concatenate(cameras).mean(0).tolist()}
    destination.with_suffix('.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__ == '__main__':
    main()
