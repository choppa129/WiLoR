"""Overfit one cached FreiHAND batch to validate the reconstruction training path."""

import argparse
from pathlib import Path
import sys

import torch
from hydra import compose, initialize_config_dir

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wilor.configs import dataset_config
from wilor.datasets import WiLoRDataModule
from wilor.models import WiLoR


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config_dir = str((Path(__file__).resolve().parents[1] / "configs_hydra").resolve())
    with initialize_config_dir(version_base="1.2", config_dir=config_dir):
        cfg = compose(
            config_name="train.yaml",
            overrides=["data=freihand_mini", "experiment=wilor_smoke"],
        )

    data = WiLoRDataModule(cfg, dataset_config(cfg.dataset_config))
    data.setup()
    batch = next(iter(data.train_dataloader()["img"]))
    batch = move_to_device(batch, args.device)

    model = WiLoR(cfg, init_renderer=False).to(args.device)
    # Keep gradients enabled but disable stochastic depth/dropout for this diagnostic.
    model.eval()
    optimizer = torch.optim.AdamW(
        model.get_parameters(), lr=cfg.TRAIN.LR, weight_decay=cfg.TRAIN.WEIGHT_DECAY
    )
    refine_probe = next(model.refine_net.parameters()).detach().clone()
    losses = []

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model.forward_step(batch, train=True)
            loss = model.compute_loss(batch, output, train=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().float().item())
        if step == 0 or (step + 1) % 10 == 0:
            print(f"step={step + 1:04d} loss={losses[-1]:.6f}", flush=True)

    refine_delta = (next(model.refine_net.parameters()).detach() - refine_probe).abs().max().item()
    print(f"initial_loss={losses[0]:.6f}")
    print(f"final_loss={losses[-1]:.6f}")
    print(f"loss_ratio={losses[-1] / losses[0]:.6f}")
    print(f"refine_param_max_delta={refine_delta:.9f}")


if __name__ == "__main__":
    main()
