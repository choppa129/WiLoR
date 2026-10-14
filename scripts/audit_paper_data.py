"""Check the real finite FreiHAND split, workers and distributed epoch sampling."""
import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DistributedSampler, DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule

with initialize_config_dir(version_base='1.2', config_dir=str(ROOT / 'configs_hydra')):
    cfg = compose(config_name='train', overrides=['experiment=wilor_paper'])
data = WiLoRDataModule(cfg, dataset_config(cfg.dataset_config))
data.setup()
train, val = data.train_dataset, data.val_dataset
ids = lambda d: [int(Path(e[1]).name) for e in d.entries]
ti, vi = ids(train), ids(val)
assert len(set(ti)) == len(ti) and len(set(vi)) == len(vi)
assert set(ti).isdisjoint(vi)
assert {i % 32560 for i in ti}.isdisjoint({i % 32560 for i in vi})
assert len(ti) == 117216 and len(vi) == 13024
for epoch in (0, 1):
    partitions = []
    for rank in range(8):
        sampler = DistributedSampler(train, num_replicas=8, rank=rank, seed=42)
        sampler.set_epoch(epoch)
        partitions.append(list(sampler))
    flat = sum(partitions, [])
    assert len(flat) == len(set(flat)) == len(train)
    assert len({len(p) for p in partitions}) == 1
    retained = [p[:len(p)//cfg.TRAIN.BATCH_SIZE*cfg.TRAIN.BATCH_SIZE] for p in partitions]
    assert len(retained[0]) // cfg.TRAIN.BATCH_SIZE == 114
    assert len(set(sum(retained, []))) == 116736
    if epoch == 0:
        first = partitions[0][:10]
    else:
        assert first != partitions[0][:10]
for workers in (0, 2):
    loader = DataLoader(Subset(train, range(9)), batch_size=4, num_workers=workers)
    batches = list(loader)
    assert [len(b['img']['img']) for b in batches] == [4, 4, 1]
    assert len({key for b in batches for key in b['img']['imgname']}) == 9
    for b in batches:
        assert torch.isfinite(b['img']['img']).all()
        assert len(b['mocap']['hand_pose']) == len(b['img']['img'])
print(json.dumps({'train_images':len(train), 'validation_images':len(val),
                  'train_capture_ids':len(set(i % 32560 for i in ti)),
                  'validation_capture_ids':len(set(i % 32560 for i in vi)),
                  'train_prior_labels':len(train.prior_poses),
                  'eight_rank_samples_per_rank':len(partitions[0]),
                  'full_batches_per_rank':114,
                  'retained_images_per_epoch':116736,
                  'dropped_images_per_epoch':480,
                  'status':'PASS: disjoint groups, full epoch, no rank overlap, worker loading'}))
