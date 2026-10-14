"""Minimal local NCCL collective smoke test for the configured GPUs."""

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    value = torch.tensor(float(dist.get_rank() + 1), device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    print(
        f"rank={dist.get_rank()} local_rank={local_rank} "
        f"device={torch.cuda.current_device()} all_reduce={value.item()}",
        flush=True,
    )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
