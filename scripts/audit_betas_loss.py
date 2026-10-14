"""Audit beta supervision, weighted contribution, and gradient connectivity."""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule
from wilor.models import load_wilor


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    return value


def main():
    checkpoint = ROOT / "logs/train/runs/wilor_freihand_full_3gpu/checkpoints/last.ckpt"
    config = ROOT / "logs/train/runs/wilor_freihand_full_3gpu/model_config.yaml"
    model, cfg = load_wilor(str(checkpoint), str(config))
    cfg.defrost(); cfg.TRAIN.BATCH_SIZE = 2; cfg.GENERAL.NUM_WORKERS = 0; cfg.freeze()
    data = WiLoRDataModule(cfg, dataset_config(cfg.dataset_config)); data.setup()
    batch = move(next(iter(data.val_dataloader())), "cuda")
    model = model.cuda().eval()
    output = model.forward_step(batch, train=False, return_initial=True)
    total = model.compute_loss(batch, output, train=False)
    losses = output["losses"]
    weights = {"loss_keypoints_3d": cfg.LOSS_WEIGHTS.KEYPOINTS_3D,
               "loss_keypoints_2d": cfg.LOSS_WEIGHTS.KEYPOINTS_2D,
               "loss_global_orient": cfg.LOSS_WEIGHTS.GLOBAL_ORIENT,
               "loss_hand_pose": cfg.LOSS_WEIGHTS.HAND_POSE,
               "loss_betas": cfg.LOSS_WEIGHTS.BETAS}
    contributions = {name: float(losses[name] * weight) for name, weight in weights.items()}

    predicted = output["pred_mano_params"]["betas"]
    target = batch["mano_params"]["betas"].view_as(predicted)
    mask = batch["has_mano_params"]["betas"]
    raw_beta = model.mano_parameter_loss(predicted, target, mask)
    initial_raw_beta = model.mano_parameter_loss(
        output["initial_pred_mano_params"]["betas"], target, mask
    )
    weighted_beta = raw_beta * cfg.LOSS_WEIGHTS.BETAS
    model.zero_grad(set_to_none=True)
    weighted_beta.backward()
    refine_grad = model.refine_net.dec_shape.weight.grad
    backbone_grad = model.backbone.decshape.weight.grad

    print(f"batch_size={predicted.shape[0]}")
    print(f"has_betas_fraction={mask.float().mean().item():.6f}")
    print(f"betas_weight={cfg.LOSS_WEIGHTS.BETAS:.8f}")
    print(f"raw_beta_loss={raw_beta.item():.8f}")
    print(f"initial_raw_beta_loss={initial_raw_beta.item():.8f}")
    print(f"refinement_beta_loss_ratio={raw_beta.item()/initial_raw_beta.item():.8f}")
    print(f"weighted_beta_loss={weighted_beta.item():.8f}")
    print(f"reported_total_loss={total.item():.8f}")
    print("weighted_contributions=" + str(contributions))
    print(f"weighted_beta_fraction={contributions['loss_betas']/sum(contributions.values()):.8f}")
    print(f"refine_dec_shape_grad_norm={refine_grad.norm().item():.8e}")
    print(f"backbone_decshape_grad_norm={backbone_grad.norm().item():.8e}")
    print(f"gt_beta_norm_mean={target.norm(dim=1).mean().item():.8f}")
    print(f"initial_beta_norm_mean={output['initial_pred_mano_params']['betas'].norm(dim=1).mean().item():.8f}")
    print(f"refined_beta_norm_mean={predicted.norm(dim=1).mean().item():.8f}")


if __name__ == "__main__":
    main()
