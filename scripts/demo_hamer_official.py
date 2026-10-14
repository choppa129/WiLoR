"""Run the official HaMeR checkpoint with WiLoR's YOLO hand detections."""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
HAMER_ROOT = ROOT / "hamer_official" / "code"
sys.path.insert(0, str(HAMER_ROOT))
os.chdir(HAMER_ROOT)

from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import load_hamer
from hamer.utils import recursive_to
from hamer.utils.renderer import Renderer, cam_crop_to_full

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def project_path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def project(points, camera_translation, focal_length, image_size):
    points = points + camera_translation
    points = points / points[..., -1:]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = intrinsic[1, 1] = focal_length
    intrinsic[0, 2] = image_size[0] / 2.0
    intrinsic[1, 2] = image_size[1] / 2.0
    return (intrinsic @ points.T).T[..., :2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-folder", default=str(ROOT / "demo_img"))
    parser.add_argument("--out-folder", default=str(ROOT / "demo_out_hamer"))
    parser.add_argument("--checkpoint", default=str(HAMER_ROOT / "_DATA/hamer_ckpts/checkpoints/hamer.ckpt"))
    parser.add_argument("--detector", default=str(ROOT / "pretrained_models/detector.pt"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    args = parser.parse_args()

    image_dir = project_path(args.img_folder)
    output_dir = project_path(args.out_folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_hamer(str(project_path(args.checkpoint)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    detector = YOLO(str(project_path(args.detector))).to(device)
    renderer = Renderer(cfg, faces=model.mano.faces)

    image_paths = sorted([*image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg"), *image_dir.glob("*.png")])
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        detections = detector(image_bgr, conf=0.3, verbose=False)[0]
        boxes, handedness = [], []
        for detection in detections:
            values = detection.boxes.data.cpu().numpy().reshape(-1)
            boxes.append(values[:4])
            handedness.append(float(detection.boxes.cls.cpu().item()))
        if not boxes:
            print(f"No hands detected: {image_path.name}")
            continue

        dataset = ViTDetDataset(cfg, image_bgr, np.stack(boxes), np.asarray(handedness),
                                rescale_factor=args.rescale_factor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
        all_vertices, all_cameras, all_right = [], [], []
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.inference_mode():
                output = model(batch)
            multiplier = 2 * batch["right"] - 1
            pred_camera = output["pred_cam"].clone()
            pred_camera[:, 1] = multiplier * pred_camera[:, 1]
            image_size = batch["img_size"].float()
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * image_size.max()
            camera_full = cam_crop_to_full(pred_camera, batch["box_center"].float(),
                                           batch["box_size"].float(), image_size, focal).cpu().numpy()
            for index in range(batch["img"].shape[0]):
                person_id = int(batch["personid"][index])
                stem = image_path.stem
                vertices = output["pred_vertices"][index].float().cpu().numpy()
                joints = output["pred_keypoints_3d"][index].float().cpu().numpy()
                is_right = float(batch["right"][index].cpu())
                vertices[:, 0] *= 2 * is_right - 1
                joints[:, 0] *= 2 * is_right - 1
                camera = camera_full[index]
                joints_2d = project(joints, camera, float(focal), image_size[index].cpu().numpy())
                projection = image_bgr.copy()
                for joint_id, (x, y) in enumerate(joints_2d):
                    if 0 <= x < projection.shape[1] and 0 <= y < projection.shape[0]:
                        cv2.circle(projection, (int(x), int(y)), 5, (0, 255, 0), -1)
                        cv2.putText(projection, str(joint_id), (int(x)+3, int(y)-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                cv2.imwrite(str(output_dir / f"{stem}_{person_id}_joints_projection.jpg"), projection)
                np.savez(output_dir / f"{stem}_{person_id}_params.npz",
                         pred_cam=output["pred_cam"][index].float().cpu().numpy(), cam_t=camera,
                         joints_3d=joints, joints_2d=joints_2d, vertices=vertices,
                         mano_pose=output["pred_mano_params"]["hand_pose"][index].float().cpu().numpy(),
                         mano_shape=output["pred_mano_params"]["betas"][index].float().cpu().numpy(),
                         global_orient=output["pred_mano_params"]["global_orient"][index].float().cpu().numpy(),
                         is_right=np.asarray(is_right))
                mesh = renderer.vertices_to_trimesh(vertices, camera.copy(), LIGHT_BLUE, is_right=is_right)
                mesh.export(output_dir / f"{stem}_{person_id}.obj")
                all_vertices.append(vertices); all_cameras.append(camera); all_right.append(is_right)
        if all_vertices:
            rgba = renderer.render_rgba_multiple(all_vertices, cam_t=all_cameras,
                render_res=image_size[-1], is_right=all_right, mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1), focal_length=focal)
            rgb = image_bgr[:, :, ::-1].astype(np.float32) / 255.0
            overlay = rgb * (1-rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
            cv2.imwrite(str(output_dir / f"{image_path.stem}.jpg"), 255 * overlay[:, :, ::-1])
        print(f"Processed {image_path.name}: {len(all_vertices)} hands")


if __name__ == "__main__":
    main()
