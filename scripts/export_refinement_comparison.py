"""Export matched WiLoR predictions before and after RefineNet."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import Renderer, cam_crop_to_full

COLOR = (0.25098039, 0.274117647, 0.65882353)


def project(points, camera, focal, image_size):
    points = points + camera
    points = points / points[..., -1:]
    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = intrinsic[1, 1] = focal
    intrinsic[0, 2] = image_size[0] / 2
    intrinsic[1, 2] = image_size[1] / 2
    return (intrinsic @ points.T).T[..., :2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-folder", default="demo_img")
    parser.add_argument("--out-folder", default="demo_out_refinement")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    image_dir = (ROOT / args.img_folder).resolve()
    output_dir = (ROOT / args.out_folder).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_wilor(
        str(ROOT / "logs/train/runs/wilor_freihand_full_3gpu/checkpoints/last.ckpt"),
        str(ROOT / "logs/train/runs/wilor_freihand_full_3gpu/model_config.yaml"),
    )
    device = torch.device(args.device)
    model = model.to(device).eval()
    detector = YOLO(str(ROOT / "pretrained_models/detector.pt")).to(device)
    renderer = Renderer(cfg, faces=model.mano.faces)
    image_paths = sorted([*image_dir.glob("*.jpg"), *image_dir.glob("*.jpeg"), *image_dir.glob("*.png")])

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        detections = detector(image, conf=0.3, verbose=False)[0]
        boxes, rights = [], []
        for detection in detections:
            row = detection.boxes.data.cpu().numpy().reshape(-1)
            boxes.append(row[:4]); rights.append(float(detection.boxes.cls.cpu().item()))
        if not boxes:
            continue
        dataset = ViTDetDataset(cfg, image, np.stack(boxes), np.asarray(rights), rescale_factor=2.0)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
        scenes = {"before_refine": ([], [], []), "after_refine": ([], [], [])}
        for batch in loader:
            batch = recursive_to(batch, device)
            with torch.inference_mode():
                prediction = model.forward_step(batch, train=False, return_initial=True)
            image_size = batch["img_size"].float()
            focal = cfg.EXTRA.FOCAL_LENGTH / cfg.MODEL.IMAGE_SIZE * image_size.max()
            stage_keys = {
                "before_refine": ("initial_pred_cam", "initial_pred_vertices", "initial_pred_keypoints_3d", "initial_pred_mano_params"),
                "after_refine": ("pred_cam", "pred_vertices", "pred_keypoints_3d", "pred_mano_params"),
            }
            for stage, (cam_key, vertex_key, joint_key, mano_key) in stage_keys.items():
                camera_crop = prediction[cam_key].clone()
                camera_crop[:, 1] *= 2 * batch["right"] - 1
                camera_full = cam_crop_to_full(camera_crop, batch["box_center"].float(),
                    batch["box_size"].float(), image_size, focal).cpu().numpy()
                for index in range(batch["img"].shape[0]):
                    person_id = int(batch["personid"][index])
                    right = float(batch["right"][index].cpu())
                    vertices = prediction[vertex_key][index].float().cpu().numpy().copy()
                    joints = prediction[joint_key][index].float().cpu().numpy().copy()
                    vertices[:, 0] *= 2 * right - 1; joints[:, 0] *= 2 * right - 1
                    camera = camera_full[index]
                    joints_2d = project(joints, camera, float(focal), image_size[index].cpu().numpy())
                    stem = f"{image_path.stem}_{person_id}_{stage}"
                    overlay = image.copy()
                    for joint_id, (x, y) in enumerate(joints_2d):
                        if 0 <= x < overlay.shape[1] and 0 <= y < overlay.shape[0]:
                            cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)
                            cv2.putText(overlay, str(joint_id), (int(x)+3, int(y)-3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
                    cv2.imwrite(str(output_dir / f"{stem}_joints.jpg"), overlay)
                    mano = prediction[mano_key]
                    np.savez(output_dir / f"{stem}_params.npz", pred_cam=prediction[cam_key][index].float().cpu().numpy(),
                        cam_t=camera, vertices=vertices, joints_3d=joints, joints_2d=joints_2d,
                        mano_pose=mano["hand_pose"][index].float().cpu().numpy(),
                        mano_shape=mano["betas"][index].float().cpu().numpy(),
                        global_orient=mano["global_orient"][index].float().cpu().numpy(), is_right=np.asarray(right))
                    mesh = renderer.vertices_to_trimesh(vertices, camera.copy(), COLOR, is_right=right)
                    mesh.export(output_dir / f"{stem}.obj")
                    scenes[stage][0].append(vertices); scenes[stage][1].append(camera); scenes[stage][2].append(right)
        for stage, (vertices, cameras, rights) in scenes.items():
            rgba = renderer.render_rgba_multiple(vertices, cam_t=cameras, render_res=image_size[-1],
                is_right=rights, mesh_base_color=COLOR, scene_bg_color=(1, 1, 1), focal_length=focal)
            rgb = image[:, :, ::-1].astype(np.float32) / 255
            composed = rgb * (1-rgba[:, :, 3:]) + rgba[:, :, :3] * rgba[:, :, 3:]
            cv2.imwrite(str(output_dir / f"{image_path.stem}_{stage}.jpg"), 255*composed[:, :, ::-1])
        print(f"Processed {image_path.name}: {len(boxes)} hands")


if __name__ == "__main__":
    main()
