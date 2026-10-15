"""Paired short training experiment: only backbone.init_cam differs between arms.

Uses the current paper-mode architecture, not a reconstruction of unpublished
author training. Cached augmentations, samples and priors are identical for all
arms; model/random seeds are reset within each pair. No production config edits.
"""
import argparse
import gc
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule
from wilor.models import WiLoR
from wilor.utils.geometry import aa_to_rotmat, weak_perspective_feature_grid


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move(value):
    if torch.is_tensor(value):
        return value.cuda()
    if isinstance(value, dict):
        return {k: move(v) for k, v in value.items()}
    return value


@torch.no_grad()
def evaluate(model, batches):
    model.eval()
    totals = defaultdict(float)
    count = 0
    for cpu in batches:
        batch = move(cpu)
        out = model.forward_step(batch, return_initial=True)
        loss = model.compute_loss(batch, out)
        n = len(batch['img'])
        totals['supervised_loss'] += float(loss) * n
        for k, v in model.compute_validation_metrics(batch, out).items():
            totals[k] += float(v) * n
        confidence = batch['keypoints_2d'][..., 2]
        error = torch.linalg.vector_norm(out['initial_pred_keypoints_2d'] - batch['keypoints_2d'][..., :2], dim=-1)
        totals['initial_reprojection_px'] += float((error*confidence).sum()/confidence.sum().clamp_min(1)*256) * n
        params = out['initial_pred_mano_params']
        vertices = model.mano(**params, pose2rot=False).vertices
        grid = weak_perspective_feature_grid(vertices, out['initial_pred_cam'])
        totals['initial_vertices_in_grid_fraction'] += float((grid.abs() <= 1).all(-1).float().mean()) * n
        for j, key in enumerate(['scale', 'tx', 'ty']):
            totals['pred_camera_'+key] += float(out['pred_cam'][:, j].mean())*n
        count += n
    model.train()
    return {k: v/count for k,v in totals.items()}


def run(cfg, train, val, camera, seed, steps, directory, eval_every):
    seed_all(seed)
    model = WiLoR(cfg, init_renderer=False).cuda()
    with torch.no_grad():
        model.backbone.init_cam.copy_(torch.tensor(camera, device='cuda')[None])
    opt, disc_opt = model.configure_optimizers()
    probes = ['backbone.patch_embed.proj.weight', 'backbone.cam_emb.weight',
              'backbone.deccam.weight', 'refine_net.dec_cam.weight']
    before = {k: dict(model.named_parameters())[k].detach().clone() for k in probes}
    result = {'seed':seed, 'camera':camera, 'steps':steps, 'evaluations':[], 'training':[]}
    directory.mkdir(parents=True, exist_ok=True)
    def save():
        (directory/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    result['evaluations'].append({'step':0, **evaluate(model,val)})
    save()
    started = time.monotonic()
    for step in range(1,steps+1):
        joint = move(train[(step-1)%len(train)])
        batch, prior = joint['img'], joint['mocap']
        n = len(batch['img'])
        opt.zero_grad(set_to_none=True)
        out = model.forward_step(batch, train=True)
        loss = model.compute_loss(batch,out,train=True)
        params = out['pred_mano_params']
        disc_scores = model.discriminator(params['hand_pose'].reshape(n,-1), params['betas'])
        adv = torch.linalg.vector_norm(disc_scores.float()-1,dim=-1).mean()
        total = loss + cfg.LOSS_WEIGHTS.ADVERSARIAL*adv
        if not torch.isfinite(total):
            raise RuntimeError(f'Nonfinite generator loss at step {step}')
        total.backward()
        grad = float(model.backbone.deccam.weight.grad.norm())
        if not np.isfinite(grad):
            raise RuntimeError(f'Nonfinite camera gradient at step {step}')
        opt.step()
        disc_opt.zero_grad(set_to_none=True)
        gt_rot = aa_to_rotmat(prior['hand_pose'].reshape(-1,3)).reshape(n,-1,3,3)
        fake = model.discriminator(params['hand_pose'].detach().reshape(n,-1), params['betas'].detach())
        real = model.discriminator(gt_rot, prior['betas'])
        disc_loss = (fake.square().sum() + (real-1).square().sum())/n
        if not torch.isfinite(disc_loss):
            raise RuntimeError(f'Nonfinite discriminator loss at step {step}')
        (cfg.LOSS_WEIGHTS.ADVERSARIAL*disc_loss).backward()
        disc_opt.step()
        result['training'].append({'step':step, 'loss':float(total.detach()),
                                   'supervised_loss':float(loss.detach()), 'generator_adv':float(adv.detach()),
                                   'discriminator_loss':float(disc_loss.detach()), 'camera_head_grad_norm':grad})
        if step % eval_every == 0 or step == steps:
            metrics = {'step':step, **evaluate(model,val)}
            result['evaluations'].append(metrics)
            result['seconds'] = time.monotonic()-started
            save()
            print(json.dumps({'seed':seed,'camera':camera,'seconds':result['seconds'],**metrics}),flush=True)
    result['parameter_max_updates'] = {k:float((dict(model.named_parameters())[k].detach()-v).abs().max()) for k,v in before.items()}
    result['all_model_parameters_finite'] = all(bool(torch.isfinite(p).all()) for p in model.parameters())
    result['optimizer_steps'] = [sorted({int(v['step']) for v in o.state.values()}) for o in [opt,disc_opt]]
    result['final_training_subset_metrics'] = evaluate(model,[b['img'] for b in train[:max(1,64//cfg.TRAIN.BATCH_SIZE)]])
    save()
    del model,opt,disc_opt,before,out,params,total,loss,disc_scores,adv,fake,real,disc_loss
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42,123])
    parser.add_argument('--eval-every',type=int,default=20)
    parser.add_argument('--output',default='evaluation/camera_initialization_20261014')
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    seed_all(2026)
    with initialize_config_dir(version_base='1.2',config_dir=str(ROOT/'configs_hydra')):
        cfg = compose(config_name='train',overrides=['experiment=wilor_paper',f'TRAIN.BATCH_SIZE={args.batch_size}','GENERAL.NUM_WORKERS=0'])
    data = WiLoRDataModule(cfg,dataset_config(cfg.dataset_config))
    data.setup()
    rng = np.random.default_rng(2026)
    ti = rng.choice(len(data.train_dataset),256,replace=False).tolist()
    vi = rng.choice(len(data.val_dataset),64,replace=False).tolist()
    train = list(DataLoader(Subset(data.train_dataset,ti),batch_size=args.batch_size,num_workers=0))
    val = list(DataLoader(Subset(data.val_dataset,vi),batch_size=4,num_workers=0))
    cameras = {'official':np.load('mano_data/mano_mean_params.npz')['cam'].tolist(),
               'fitted':np.load(cfg.MANO.MEAN_PARAMS)['cam'].tolist()}
    directory = Path(args.output)
    directory.mkdir(parents=True,exist_ok=True)
    manifest = {'train_ids':[data.train_dataset.entries[i][1] for i in ti],
                'val_ids':[data.val_dataset.entries[i][1] for i in vi],
                'batch_size':args.batch_size,'steps_per_run':args.steps,'seeds':args.seeds,'cameras':cameras,
                'mean_file_sha256':hashlib.sha256(Path(cfg.MANO.MEAN_PARAMS).read_bytes()).hexdigest(),
                'pose_mean_metadata':json.loads(Path(cfg.MANO.MEAN_PARAMS).with_suffix('.json').read_text()),
                'precision':'float32','optimizer':'Adam','learning_rate':1e-5,'weight_decay':1e-4,
                'architecture':'current paper mode: 3 parameter tokens, weak perspective',
                'controls':'Same fitted pose/shape means, pretrained weights, cached images/augmentations/prior batches, order and paired RNG seeds; only init_cam changed.',
                'limitation':'Small cached subset and small batch, not convergence or author-provenance evidence. CUDA grid_sample backward is not bitwise deterministic.'}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    results={}
    for seed in args.seeds:
        for arm,cam in cameras.items():
            key=f'{arm}_seed{seed}'
            print('START '+key,flush=True)
            results[key]=run(cfg,train,val,cam,seed,args.steps,directory/key,args.eval_every)
            (directory/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ALL RUNS COMPLETE',flush=True)


if __name__ == '__main__':
    main()
