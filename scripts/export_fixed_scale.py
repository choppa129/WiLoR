"""Export comparable FP32 reconstructions from immutable diagnostic crops."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'hamer_official/code')]
FIXED = ROOT / 'evaluation/scale_diagnostic/fixed_inputs'
OUT = ROOT / 'evaluation/scale_diagnostic/predictions'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['official', 'trained', 'hamer'], required=True)
    args = parser.parse_args()
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.chdir(ROOT)
    if args.model == 'hamer':
        os.chdir(ROOT / 'hamer_official/code')
        from hamer.models import load_hamer
        ckpt = Path.cwd() / '_DATA/hamer_ckpts/checkpoints/hamer.ckpt'
        model, cfg = load_hamer(str(ckpt))
    else:
        from wilor.models import load_wilor
        base = ROOT / ('pretrained_models' if args.model == 'official' else
                       'logs/train/runs/wilor_freihand_full_3gpu')
        ckpt = base / ('wilor_final.ckpt' if args.model == 'official' else 'checkpoints/last.ckpt')
        model, cfg = load_wilor(str(ckpt), str(base / 'model_config.yaml'))
    model = model.float().cuda().eval()
    if hasattr(model.backbone, 'skip_blocks'):
        model.backbone.skip_blocks = False
    assert cfg.MODEL.IMAGE_SIZE == 256
    np.testing.assert_allclose(cfg.MODEL.IMAGE_MEAN, [.485,.456,.406])
    np.testing.assert_allclose(cfg.MODEL.IMAGE_STD, [.229,.224,.225])
    manifest = json.loads((FIXED / 'manifest.json').read_text())
    dest = OUT / args.model
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for rec in manifest['samples']:
        path = FIXED / rec['tensor_file']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == rec['sha256']
        with np.load(path, allow_pickle=False) as data:
            batch = {k: torch.as_tensor(data[k], device='cuda', dtype=torch.float32).unsqueeze(0)
                     for k in ['img', 'box_center', 'box_size', 'img_size', 'right']}
        with torch.inference_mode():
            pred = (model.forward_step(batch, train=False) if args.model == 'hamer' else
                    model.forward_step(batch, train=False, return_initial=True))
        for stage, prefix in ([('final', '')] if args.model == 'hamer' else
                              [('before', 'initial_'), ('after', '')]):
            def arr(key):
                return pred[prefix + key][0].detach().float().cpu().numpy().copy()
            cam = arr('pred_cam')
            vertices, joints = arr('pred_vertices'), arr('pred_keypoints_3d')
            params = {k: v[0].detach().float().cpu().numpy().copy()
                      for k,v in pred[prefix+'pred_mano_params'].items()}
            right = rec['right']
            width, height = batch['img_size'][0].cpu().numpy()
            focal = float(cfg.EXTRA.FOCAL_LENGTH) / 256 * max(width, height)
            size = rec['box_size']
            center = np.array(rec['box_center'])
            # Convert a cloned camera; preserve canonical network coordinates.
            full_cam = cam.copy()
            full_cam[1] *= 2*right-1
            bs = size * full_cam[0] + 1e-9
            translation = np.array([full_cam[1]+2*(center[0]-width/2)/bs,
                                    full_cam[2]+2*(center[1]-height/2)/bs, 2*focal/bs])
            world_joints, world_vertices = joints.copy(), vertices.copy()
            world_joints[:,0] *= 2*right-1
            world_vertices[:,0] *= 2*right-1
            def project(points):
                q = points + translation
                assert (q[:,2] > 0).all(), rec['id']
                return q[:,:2]/q[:,2:] * focal + [width/2,height/2]
            uv = project(world_joints)
            values = dict(pred_cam=cam, cam_t_full=translation, vertices=vertices,
                joints_3d=joints, vertices_full=world_vertices, joints_2d_full=uv,
                vertices_2d_full=project(world_vertices), betas=params['betas'],
                hand_pose=params['hand_pose'], global_orient=params['global_orient'],
                right=right, focal_length=focal, box_size=size, box_center=center,
                img_size=np.array([width,height]))
            assert all(np.isfinite(v).all() for v in values.values())
            np.savez_compressed(dest / f"{rec['id']}_{stage}.npz", **values)
            rows.append(dict(sample=rec['id'], model=args.model, stage=stage,
                camera_scale=float(cam[0]), camera_depth=float(translation[2]),
                beta_norm=float(np.linalg.norm(params['betas'])),
                mesh_diag=float(np.linalg.norm(np.ptp(vertices, axis=0))),
                joint_extent_px=float(np.linalg.norm(np.ptp(uv, axis=0)))))
    with (dest/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (dest/'provenance.json').write_text(json.dumps(dict(checkpoint=str(ckpt),
        precision='float32; TF32 disabled; no layer skipping', samples=len(manifest['samples']),
        input_manifest_sha256=hashlib.sha256((FIXED/'manifest.json').read_bytes()).hexdigest(),
        coordinates='vertices/joints_3d/pred_cam canonical right-hand; *_full mirrored for left hands',
        interpretation='fixed detector crops, not official end-to-end detector comparison'), indent=2))
    print(f'{args.model}: exported {len(rows)} predictions', flush=True)


if __name__ == '__main__':
    main()
