"""Finite, indexed access to local FreiHAND tar shards (no resampling).

An epoch visits the selected examples once; Lightning's distributed sampler
may pad the final rank partition. Pickled labels are the trusted local training
assets already consumed by ImageDataset. Indexing reads headers and labels,
not decoded images. Each worker opens files independently for random access.
"""
import io
import pickle
import tarfile
from pathlib import PurePosixPath

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .image_dataset import ImageDataset, expand_urls


class FiniteFreiHAND(Dataset):
    def __init__(self, cfg, urls, train=False):
        self.cfg, self.train = cfg, train
        self.entries = []
        poses, betas = [], []
        holdout = cfg.DATASETS.get('FREIHAND_HOLDOUT', 0.0)
        if not 0 <= holdout < 1:
            raise ValueError('FREIHAND_HOLDOUT must be in [0, 1)')
        # FreiHAND has four appearance variants of each of 32,560 captures.
        # Group all variants together; seed and held-out fraction are local
        # validation choices, not the paper's official evaluation protocol.
        val_ids = set(np.random.default_rng(cfg.get('seed', 42)).permutation(32560)
                      [:int(32560 * holdout)].tolist())
        for path in expand_urls(urls):
            with tarfile.open(path, 'r:') as archive:
                members = {m.name: m for m in archive if m.isfile()}
                for name, label in members.items():
                    if not name.endswith('.data.pyd'):
                        continue
                    key = name[:-len('.data.pyd')]
                    if holdout:
                        sample_id = int(PurePosixPath(key).name)
                        if not 0 <= sample_id < 130240:
                            raise ValueError(f'Unexpected FreiHAND sample ID: {key}')
                        if ((sample_id % 32560) in val_ids) == train:
                            continue
                    image = next((members.get(key + ext) for ext in ('.jpg', '.jpeg', '.png')
                                  if key + ext in members), None)
                    if image is None:
                        raise ValueError(f'Missing image for {path}:{key}')
                    records = pickle.loads(archive.extractfile(label).read())
                    for record_index in range(len(records)):
                        self.entries.append((path, key, image.offset_data, image.size,
                                             label.offset_data, label.size, record_index))
                        record = records[record_index]
                        if train and record['has_hand_pose'] and record['has_betas']:
                            poses.append(record['hand_pose'][3:])
                            betas.append(record['betas'])
        if not self.entries:
            raise ValueError('No FreiHAND samples found')
        if train:
            if not poses:
                raise ValueError('No annotated training poses for the discriminator')
            self.prior_poses = np.asarray(poses, dtype=np.float32)
            self.prior_betas = np.asarray(betas, dtype=np.float32)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, key, image_offset, image_size, label_offset, label_size, record_index = self.entries[index]
        with open(path, 'rb') as stream:
            stream.seek(image_offset)
            image = np.array(Image.open(io.BytesIO(stream.read(image_size))).convert('RGB'))
            stream.seek(label_offset)
            label = pickle.loads(stream.read(label_size))[record_index]
        item = {'__key__': key, 'jpg': image, 'data.pyd': label,
                'mask': np.ones(image.shape[:2], dtype=bool)}
        cfg = self.cfg
        item = ImageDataset.process_webdataset_tar_item(
            item, self.train, augm_config=cfg.DATASETS.CONFIG,
            MEAN=255 * np.array(cfg.MODEL.IMAGE_MEAN),
            STD=255 * np.array(cfg.MODEL.IMAGE_STD), IMG_SIZE=cfg.MODEL.IMAGE_SIZE,
            BBOX_SHAPE=cfg.MODEL.get('BBOX_SHAPE', None))
        if not self.train:
            return item
        # Independent prior example; image dataset alone determines epoch length.
        prior_index = np.random.randint(len(self.prior_poses))
        return {'img': item, 'mocap': {'hand_pose': self.prior_poses[prior_index].copy(),
                                      'betas': self.prior_betas[prior_index].copy()}}
