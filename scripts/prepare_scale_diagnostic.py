"""Freeze demo detections and audit reconstruction inputs on CPU."""
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from yacs.config import CfgNode
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'hamer_official/code'))
from wilor.datasets.vitdet_dataset import ViTDetDataset as WDataset
from hamer.datasets.vitdet_dataset import ViTDetDataset as HDataset
from wilor.datasets.utils import generate_image_patch_cv2


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT / 'evaluation/scale_diagnostic/fixed_inputs'
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Fixed manifest already exists; preserve it for reproducibility.')
    cfg = CfgNode({'MODEL': {'IMAGE_SIZE': 256, 'BBOX_SHAPE': [192, 256],
                   'IMAGE_MEAN': [0.485, 0.456, 0.406], 'IMAGE_STD': [0.229, 0.224, 0.225]}})
    detector_path = ROOT / 'pretrained_models/detector.pt'
    detector = YOLO(str(detector_path))
    records, images = [], []
    for path in sorted((ROOT / 'demo_img').iterdir()):
        if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        im = cv2.imread(str(path))
        assert im is not None, path
        result = detector(im, conf=0.3, device='cpu', verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        rights = result.boxes.cls.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        assert np.isin(rights, [0, 1]).all()
        images.append({'path': str(path.relative_to(ROOT)), 'sha256': digest(path),
                       'width': im.shape[1], 'height': im.shape[0], 'detections': len(boxes)})
        wd = WDataset(cfg, im, boxes, rights, rescale_factor=2.0)
        hd = HDataset(cfg, im, boxes, rights, rescale_factor=2.0)
        overlay = im.copy()
        for i in range(len(boxes)):
            w = wd[i]
            with contextlib.redirect_stdout(io.StringIO()):
                h = hd[i]
            for key in ['img', 'box_center', 'box_size', 'img_size', 'right']:
                np.testing.assert_array_equal(w[key], h[key], err_msg=f'{path.name}/{i}/{key}')
            assert np.isfinite(w['img']).all()
            size = float(w['box_size'])
            center = w['box_center']
            _, trans = generate_image_patch_cv2(im, *center, size, size, 256, 256,
                rights[i] == 0, 1.0, 0, border_mode=cv2.BORDER_CONSTANT)
            affine = np.eye(3)
            affine[:2] = trans
            if rights[i] == 0:
                affine = affine @ np.array([[-1, 0, im.shape[1]-1], [0, 1, 0], [0, 0, 1]])
            sid = f'{path.stem}_{i:03d}'
            target = out / f'{sid}.npz'
            np.savez_compressed(target, img=w['img'], box_center=center, box_size=size,
                img_size=w['img_size'], right=rights[i], bbox=boxes[i],
                original_to_crop=affine, crop_to_original=np.linalg.inv(affine))
            crop = w['img'].transpose(1, 2, 0) * np.array(cfg.MODEL.IMAGE_STD) + np.array(cfg.MODEL.IMAGE_MEAN)
            cv2.imwrite(str(out / f'{sid}.jpg'), np.clip(crop[:, :, ::-1]*255, 0, 255).astype(np.uint8))
            x1, y1, x2, y2 = boxes[i].astype(int)
            cv2.rectangle(overlay, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(overlay, f'{sid} {"R" if rights[i] else "L"}', (x1,max(15,y1)),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 1)
            records.append({'id': sid, 'image': str(path.relative_to(ROOT)), 'bbox': boxes[i].tolist(),
                'right': int(rights[i]), 'confidence': float(scores[i]), 'box_center': center.tolist(),
                'box_size': size, 'tensor_file': target.name, 'sha256': digest(target),
                'preprocessing_max_abs_diff': 0.0})
        cv2.imwrite(str(out / f'{path.stem}_boxes.jpg'), overlay)
        print(f'{path.name}: {len(boxes)} fixed detections', flush=True)
    manifest = {'scope': 'demo images; detector proposals, not ground truth',
        'detector_sha256': digest(detector_path), 'detector_classes': detector.names,
        'rescale_factor': 2.0, 'image_size': 256, 'bbox_shape': [192,256],
        'precision': 'float32', 'images': images, 'samples': records,
        'note': 'All reconstruction models must reuse saved tensors. IDs are new and do not imply correspondence with old output IDs.'}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'PASS: {len(images)} images, {len(records)} crops; WiLoR/HaMeR tensors exactly equal.')


if __name__ == '__main__':
    main()
