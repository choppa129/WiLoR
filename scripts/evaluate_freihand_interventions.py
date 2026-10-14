"""Evaluate camera and pose interventions against FreiHAND annotations."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_freihand import build_cfg, make_loader, move
from wilor.models import WiLoR
from wilor.utils.geometry import aa_to_rotmat, perspective_projection
from wilor.utils.pose_utils import reconstruction_error


def load_model(cfg, checkpoint, device):
    model = WiLoR(cfg, init_renderer=False)
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(checkpoint_data.get("state_dict", checkpoint_data), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return model.to(device).eval()


def make_output(mano, orient, pose, betas, camera, focal, image_size):
    result = mano(global_orient=orient, hand_pose=pose, betas=betas, pose2rot=False)
    joints, vertices = result.joints, result.vertices
    joints_2d = perspective_projection(joints, translation=camera,
        focal_length=focal / image_size)
    return dict(pred_keypoints_3d=joints, pred_vertices=vertices,
                pred_keypoints_2d=joints_2d, pred_cam_t=camera)


def sample_metrics(batch, output, gt_vertices):
    pred_2d = output["pred_keypoints_2d"].float()
    gt_2d = batch["keypoints_2d"][..., :2].float()
    conf_2d = batch["keypoints_2d"][..., 2].float()
    pred_3d = output["pred_keypoints_3d"].float()
    gt_3d = batch["keypoints_3d"][..., :3].float()
    conf_3d = batch["keypoints_3d"][..., 3].float()
    image_size = float(pred_2d.new_tensor(256.0))
    reprojection = torch.linalg.vector_norm(pred_2d-gt_2d, dim=-1) * image_size
    reprojection = (reprojection*conf_2d).sum(-1)/conf_2d.sum(-1).clamp_min(1)
    pred_root = pred_3d-pred_3d[:, [0]]
    gt_root = gt_3d-gt_3d[:, [0]]
    mpjpe = 1000*(torch.linalg.vector_norm(pred_root-gt_root,dim=-1)*conf_3d).sum(-1)/conf_3d.sum(-1).clamp_min(1)
    pa = 1000*reconstruction_error(pred_root.float(), gt_root.float())
    pred_vertices = output["pred_vertices"].float()-pred_3d[:, [0]]
    gt_vertices = gt_vertices.float()-batch["gt_mano_joints"].float()[:, [0]]
    pve = 1000*torch.linalg.vector_norm(pred_vertices-gt_vertices,dim=-1).mean(-1)
    pred_extent, gt_extent = [], []
    for pred, gt, conf in zip(pred_2d, gt_2d, conf_2d):
        valid = conf > 0
        if valid.sum() < 2:
            pred_extent.append(torch.tensor(float("nan"),device=pred.device))
            gt_extent.append(torch.tensor(float("nan"),device=pred.device))
        else:
            pred_range = pred[valid].max(dim=0).values-pred[valid].min(dim=0).values
            gt_range = gt[valid].max(dim=0).values-gt[valid].min(dim=0).values
            pred_extent.append(torch.linalg.vector_norm(pred_range)*image_size)
            gt_extent.append(torch.linalg.vector_norm(gt_range)*image_size)
    pred_extent, gt_extent = torch.stack(pred_extent), torch.stack(gt_extent)
    return dict(reprojection_px=reprojection, mpjpe_mm=mpjpe, pa_mpjpe_mm=pa,
        pve_mm=pve, pred_extent_px=pred_extent, gt_extent_px=gt_extent,
        extent_ratio=pred_extent/gt_extent,
        extent_abs_error_px=(pred_extent-gt_extent).abs())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="evaluation/freihand_annotated_interventions_512")
    args = parser.parse_args()
    assert args.samples % args.batch_size == 0
    torch.manual_seed(42); np.random.seed(42)
    device = torch.device(args.device)
    cfg = build_cfg(args.batch_size, args.workers)
    trained = load_model(cfg, ROOT/"logs/train/runs/wilor_freihand_full_3gpu/checkpoints/last.ckpt", device)
    official = load_model(cfg, ROOT/"pretrained_models/wilor_final.ckpt", device)
    loader = make_loader(cfg)
    rows = []
    processed = 0
    for batch in loader:
        if processed >= args.samples:
            break
        batch = move(batch, str(device))
        with torch.inference_mode():
            trained_out = trained.forward_step(batch, train=False)
            official_out = official.forward_step(batch, train=False)
            batch_size = batch["img"].shape[0]
            gt = {}
            for key, value in batch["mano_params"].items():
                if key == "betas": gt[key] = value.float()
                else: gt[key] = aa_to_rotmat(value.float().reshape(-1,3)).reshape(
                    batch_size, -1, 3, 3)
            gt_mano = trained.mano(**gt, pose2rot=False)
            batch["gt_mano_joints"] = gt_mano.joints
            tparams, oparams = trained_out["pred_mano_params"], official_out["pred_mano_params"]
            focal = trained_out["focal_length"].float()
            variants = {
                "trained": trained_out,
                "official": official_out,
                "trained_geometry_official_camera": dict(trained_out,
                    pred_keypoints_2d=perspective_projection(trained_out["pred_keypoints_3d"],
                        translation=official_out["pred_cam_t"], focal_length=focal/cfg.MODEL.IMAGE_SIZE),
                    pred_cam_t=official_out["pred_cam_t"]),
                "official_geometry_trained_camera": dict(official_out,
                    pred_keypoints_2d=perspective_projection(official_out["pred_keypoints_3d"],
                        translation=trained_out["pred_cam_t"], focal_length=focal/cfg.MODEL.IMAGE_SIZE),
                    pred_cam_t=trained_out["pred_cam_t"]),
                "trained_with_gt_orientation": make_output(trained.mano, gt["global_orient"],
                    tparams["hand_pose"], tparams["betas"], trained_out["pred_cam_t"], focal, cfg.MODEL.IMAGE_SIZE),
                "trained_with_gt_fingers": make_output(trained.mano, tparams["global_orient"],
                    gt["hand_pose"], tparams["betas"], trained_out["pred_cam_t"], focal, cfg.MODEL.IMAGE_SIZE),
                "trained_with_gt_pose": make_output(trained.mano, gt["global_orient"],
                    gt["hand_pose"], tparams["betas"], trained_out["pred_cam_t"], focal, cfg.MODEL.IMAGE_SIZE),
            }
            for name, output in variants.items():
                result = sample_metrics(batch, output, gt_mano.vertices)
                for index in range(batch_size):
                    row = dict(sample=processed+index, key=batch["__key__"][index], variant=name)
                    row.update({key:float(value[index].detach().cpu()) for key,value in result.items()})
                    rows.append(row)
        processed += batch_size
        print(f"processed {processed}/{args.samples}", flush=True)
    assert processed == args.samples
    out = ROOT/args.output_dir; out.mkdir(parents=True,exist_ok=True)
    with (out/"per_sample.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    variants=list(dict.fromkeys(row["variant"] for row in rows))
    summary=[]
    for name in variants:
        subset=[row for row in rows if row["variant"]==name]
        summary.append(dict(variant=name,samples=len(subset),
            reprojection_px_mean=np.mean([r["reprojection_px"] for r in subset]),
            mpjpe_mm_mean=np.mean([r["mpjpe_mm"] for r in subset]),
            pa_mpjpe_mm_mean=np.mean([r["pa_mpjpe_mm"] for r in subset]),
            pve_mm_mean=np.mean([r["pve_mm"] for r in subset]),
            extent_ratio_median=np.nanmedian([r["extent_ratio"] for r in subset]),
            extent_abs_error_px_median=np.nanmedian([r["extent_abs_error_px"] for r in subset])))
    with (out/"summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    by_variant = {
        name: {row["key"]: row for row in rows if row["variant"] == name}
        for name in variants
    }
    paired_specs = [
        ("official", "trained"),
        ("trained_geometry_official_camera", "trained"),
        ("official_geometry_trained_camera", "official"),
        ("trained_with_gt_orientation", "trained"),
        ("trained_with_gt_fingers", "trained"),
        ("trained_with_gt_pose", "trained"),
    ]
    paired = []
    paired_metrics = ["reprojection_px", "mpjpe_mm", "pa_mpjpe_mm", "pve_mm",
                      "extent_abs_error_px"]
    for candidate, baseline in paired_specs:
        assert set(by_variant[candidate]) == set(by_variant[baseline])
        for metric in paired_metrics:
            candidate_values = np.asarray([
                by_variant[candidate][key][metric] for key in by_variant[baseline]
            ])
            baseline_values = np.asarray([
                by_variant[baseline][key][metric] for key in by_variant[baseline]
            ])
            valid = np.isfinite(candidate_values) & np.isfinite(baseline_values)
            relative_change = (candidate_values[valid] - baseline_values[valid]) / np.maximum(
                np.abs(baseline_values[valid]), 1e-12)
            paired.append(dict(candidate=candidate, baseline=baseline, metric=metric,
                samples=int(valid.sum()), improved=int((candidate_values[valid] < baseline_values[valid]).sum()),
                median_relative_change_pct=float(100*np.median(relative_change))))
    with (out/"paired_summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(paired[0]));writer.writeheader();writer.writerows(paired)
    lines=["# Annotated FreiHAND intervention pilot", "",
        f"First {args.samples} deterministic samples from the configured validation WebDataset; batch={args.batch_size}.",
        "All samples report valid 2D, 3D and MANO annotations. This is a local shard split, not the official benchmark.",
        "Camera swaps change only 2D projection; their 3D metrics duplicate the geometry source by construction.",
        "GT pose interventions keep trained betas and trained camera fixed.", "",
        "| Variant | Reproj px mean | MPJPE mm mean | PA-MPJPE mm mean | PVE mm mean | Median extent / GT | Median abs extent error px |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"| {r['variant']} | {r['reprojection_px_mean']:.4f} | {r['mpjpe_mm_mean']:.4f} | "+
            f"{r['pa_mpjpe_mm_mean']:.4f} | {r['pve_mm_mean']:.4f} | {r['extent_ratio_median']:.4f} | "+
            f"{r['extent_abs_error_px_median']:.4f} |")
    lines += ["", "Means are per-sample means; extent statistics are paired medians over visible annotated joints.",
        "GT interventions are diagnostic counterfactuals, not deployable predictions. Official checkpoint overlap with this split is unknown.",
        "", "## Paired comparisons", "",
        "Lower is better for every metric below. Relative change is the paired median of (candidate - baseline) / baseline.", "",
        "| Candidate vs baseline | Metric | Improved samples | Median relative change |",
        "|---|---|---:|---:|"]
    for r in paired:
        lines.append(f"| {r['candidate']} vs {r['baseline']} | {r['metric']} | "
                     f"{r['improved']}/{r['samples']} | {r['median_relative_change_pct']:+.2f}% |")
    (out/"report.md").write_text("\n".join(lines)+"\n")
    (out/"provenance.json").write_text(json.dumps(dict(samples=args.samples,seed=42,
        trained_checkpoint=str(ROOT/"logs/train/runs/wilor_freihand_full_3gpu/checkpoints/last.ckpt"),
        official_checkpoint=str(ROOT/"pretrained_models/wilor_final.ckpt"),
        metric_convention="2D normalized xy = pixel/IMAGE_SIZE - 0.5; errors multiplied by IMAGE_SIZE"),indent=2)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
