"""Audit weighted loss contributions and gradient norms on fixed FreiHAND batches."""
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_freihand import build_cfg, make_loader, move
from wilor.models import WiLoR
from wilor.utils.geometry import aa_to_rotmat


def load_model(cfg, checkpoint, device):
    model = WiLoR(cfg, init_renderer=False)
    data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    incompatible = model.load_state_dict(data.get("state_dict", data), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Checkpoint mismatch: {incompatible}")
    return model.to(device)


def differentiable_losses(model, batch, output):
    pred_params = output["pred_mano_params"]
    batch_size = batch["img"].shape[0]
    losses = {
        "keypoints_2d": model.keypoint_2d_loss(
            output["pred_keypoints_2d"], batch["keypoints_2d"]),
        "keypoints_3d": model.keypoint_3d_loss(
            output["pred_keypoints_3d"], batch["keypoints_3d"], pelvis_id=0),
    }
    for key, pred in pred_params.items():
        target = batch["mano_params"][key].view(batch_size, -1)
        if batch["mano_params_is_axis_angle"][key].all():
            target = aa_to_rotmat(target.reshape(-1, 3)).view(batch_size, -1, 3, 3)
        losses[key] = model.mano_parameter_loss(
            pred.reshape(batch_size, -1), target.reshape(batch_size, -1),
            batch["has_mano_params"][key])
    disc_out = model.discriminator(
        pred_params["hand_pose"].reshape(batch_size, -1),
        pred_params["betas"].reshape(batch_size, -1),
    )
    losses["adversarial"] = ((disc_out - 1.0) ** 2).sum() / batch_size
    return losses


def parameter_groups(model):
    named = [(name, param) for name, param in model.named_parameters()
             if param.requires_grad and (name.startswith("backbone.") or name.startswith("refine_net."))]
    names = [name for name, _ in named]
    params = [param for _, param in named]
    prefixes = {
        "all_generator": ("backbone.", "refine_net."),
        "backbone": ("backbone.",),
        "refinement": ("refine_net.",),
        "refine_shared": ("refine_net.deconv.",),
        "refine_pose_head": ("refine_net.dec_pose.",),
        "refine_shape_head": ("refine_net.dec_shape.",),
        "refine_camera_head": ("refine_net.dec_cam.",),
        "initial_pose_head": ("backbone.decpose.",),
        "initial_shape_head": ("backbone.decshape.",),
        "initial_camera_head": ("backbone.deccam.",),
    }
    indices = {group: [i for i, name in enumerate(names) if name.startswith(group_prefixes)]
               for group, group_prefixes in prefixes.items()}
    if any(not values for values in indices.values()):
        missing = [key for key, values in indices.items() if not values]
        raise RuntimeError(f"Empty parameter groups: {missing}")
    return names, params, indices


def gradient_norms(loss, params, group_indices, retain_graph):
    grads = torch.autograd.grad(loss, params, retain_graph=retain_graph,
                                allow_unused=True, materialize_grads=False)
    result = {}
    for group, indices in group_indices.items():
        squared = [grads[index].detach().float().square().sum()
                   for index in indices if grads[index] is not None]
        result[group] = math.sqrt(float(torch.stack(squared).sum().cpu())) if squared else 0.0
    del grads
    return result


def summarize(rows, loss_order, group_names):
    summary = []
    for name in loss_order:
        subset = [row for row in rows if row["loss"] == name]
        item = {
            "loss": name,
            "batches": len(subset),
            "raw_per_sample_mean": float(np.mean([row["raw_per_sample"] for row in subset])),
            "weight": subset[0]["weight"],
            "weighted_mean": float(np.mean([row["weighted"] for row in subset])),
            "weighted_fraction_mean": float(np.mean([row["weighted_fraction"] for row in subset])),
        }
        for group in group_names:
            item[f"grad_{group}_median"] = float(np.median([row[f"grad_{group}"] for row in subset]))
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="logs/train/runs/wilor_freihand_full_3gpu/checkpoints/last.ckpt")
    parser.add_argument("--output-dir", default="evaluation/freihand_loss_gradient_audit")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    cfg = build_cfg(args.batch_size, args.workers)
    device = torch.device(args.device)
    model = load_model(cfg, ROOT / args.checkpoint, device)
    model.train()
    loader = make_loader(cfg)
    _, params, group_indices = parameter_groups(model)
    group_names = list(group_indices)
    weights = {
        "keypoints_2d": float(cfg.LOSS_WEIGHTS.KEYPOINTS_2D),
        "keypoints_3d": float(cfg.LOSS_WEIGHTS.KEYPOINTS_3D),
        "global_orient": float(cfg.LOSS_WEIGHTS.GLOBAL_ORIENT),
        "hand_pose": float(cfg.LOSS_WEIGHTS.HAND_POSE),
        "betas": float(cfg.LOSS_WEIGHTS.BETAS),
        "adversarial": float(cfg.LOSS_WEIGHTS.ADVERSARIAL),
    }
    loss_order = list(weights)
    rows = []
    sample_keys = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= args.batches:
            break
        batch = move(batch, str(device))
        sample_keys.extend(batch["__key__"])
        output = model.forward_step(batch, train=True)
        raw_losses = differentiable_losses(model, batch, output)
        weighted_losses = {name: raw_losses[name] * weights[name] for name in loss_order}
        total = sum(weighted_losses.values())
        total_value = float(total.detach().cpu())
        for name in loss_order:
            norms = gradient_norms(weighted_losses[name], params, group_indices, retain_graph=True)
            raw_normalizer = 1 if name == "adversarial" else batch["img"].shape[0]
            row = {
                "batch": batch_index,
                "loss": name,
                "batch_size": batch["img"].shape[0],
                "raw": float(raw_losses[name].detach().cpu()),
                "raw_per_sample": float(raw_losses[name].detach().cpu()) / raw_normalizer,
                "weight": weights[name],
                "weighted": float(weighted_losses[name].detach().cpu()),
                "weighted_fraction": float(weighted_losses[name].detach().cpu()) / total_value,
            }
            row.update({f"grad_{group}": value for group, value in norms.items()})
            rows.append(row)
        total_norms = gradient_norms(total, params, group_indices, retain_graph=False)
        row = {
            "batch": batch_index,
            "loss": "total",
            "batch_size": batch["img"].shape[0],
            "raw": total_value,
            "raw_per_sample": total_value / batch["img"].shape[0],
            "weight": 1.0,
            "weighted": total_value,
            "weighted_fraction": 1.0,
        }
        row.update({f"grad_{group}": value for group, value in total_norms.items()})
        rows.append(row)
        print(f"audited batch {batch_index + 1}/{args.batches}", flush=True)
        del output, raw_losses, weighted_losses, total, batch

    if len(sample_keys) != args.batch_size * args.batches:
        raise RuntimeError(f"Expected {args.batch_size * args.batches} samples, got {len(sample_keys)}")
    if len(set(sample_keys)) != len(sample_keys):
        raise RuntimeError("Fixed audit sample keys are not unique")

    loss_order_with_total = loss_order + ["total"]
    summary = summarize(rows, loss_order_with_total, group_names)
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "per_batch.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    lines = [
        "# FreiHAND loss and gradient audit", "",
        f"Checkpoint: `{(ROOT / args.checkpoint).resolve()}`",
        f"Fixed deterministic validation subset: {len(sample_keys)} unique samples in {args.batches} batches of {args.batch_size}.",
        "Model is in training mode; gradients are FP32 diagnostics without optimizer updates or gradient scaling.",
        "Supervised losses are sums over the local batch in the training code; adversarial loss is divided by batch size.", "",
        "| Loss | Config weight | Raw / sample mean | Weighted contribution mean | Mean fraction of total | Median all-generator grad norm | Median refinement grad norm |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['loss']} | {row['weight']:.6g} | {row['raw_per_sample_mean']:.6f} | "
            f"{row['weighted_mean']:.6f} | {100*row['weighted_fraction_mean']:.2f}% | "
            f"{row['grad_all_generator_median']:.6g} | {row['grad_refinement_median']:.6g} |"
        )
    lines += ["", "## Direct head gradient norms (paired medians across batches)", "",
              "| Loss | Refine pose | Refine shape | Refine camera | Initial pose | Initial shape | Initial camera |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(
            f"| {row['loss']} | {row['grad_refine_pose_head_median']:.6g} | "
            f"{row['grad_refine_shape_head_median']:.6g} | {row['grad_refine_camera_head_median']:.6g} | "
            f"{row['grad_initial_pose_head_median']:.6g} | {row['grad_initial_shape_head_median']:.6g} | "
            f"{row['grad_initial_camera_head_median']:.6g} |"
        )
    lines += ["", "Gradient magnitudes are diagnostic at this checkpoint and sample subset; they do not by themselves determine an optimal loss weight."]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "provenance.json").write_text(json.dumps({
        "checkpoint": str((ROOT / args.checkpoint).resolve()),
        "sample_keys": sample_keys,
        "seed": 42,
        "batch_size": args.batch_size,
        "batches": args.batches,
        "mode": "train",
        "precision": "fp32",
        "optimizer_updates": 0,
        "loss_weights": weights,
    }, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
