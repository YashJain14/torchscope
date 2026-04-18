"""
Example: profile a multi-GPU DDP training job with torchscope.

Tests:
  - CommCollector: all-reduce timing, compute-to-communication ratio
  - Straggler detection across ranks
  - Per-rank JSON log export
  - NVTX annotations visible in nsys across all ranks

Run (single node, 4 GPUs):
    torchrun --nproc_per_node=4 examples/distributed_training.py

Run (multi-node, 2 nodes × 4 GPUs via SLURM / PBS):
    # node 0
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
             --master_addr=<node0_ip> --master_port=29500 \
             examples/distributed_training.py

    # node 1
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
             --master_addr=<node0_ip> --master_port=29500 \
             examples/distributed_training.py

Run under Nsight Systems (per-rank trace):
    nsys profile --trace=cuda,nvtx,osrt \
        --output=torchscope_rank%q{RANK} \
        torchrun --nproc_per_node=4 examples/distributed_training.py

What to look for in the report:
  - HIGH_COMM_OVERHEAD  if comm > 15% of step time
  - STRAGGLER_GPU       if one rank is consistently >10% slower
  - LOW_GPU_UTIL        if batch size is too small
  - Kernel trace shows  nccl* ops dominating when comm overhead is high
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from torchscope import Profiler
from torchscope.nvtx import annotate


# ── model ─────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim: int = 1024, heads: int = 16, mlp_ratio: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.ff(self.ln2(x))
        return x


class SmallTransformer(nn.Module):
    def __init__(self, dim: int = 1024, layers: int = 8, n_classes: int = 1000):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(dim) for _ in range(layers)])
        self.head   = nn.Linear(dim, n_classes)

    def forward(self, x):
        return self.head(self.blocks(x).mean(dim=1))


# ── training ──────────────────────────────────────────────────────────────────

def train(rank: int, world_size: int):
    # ── init process group ────────────────────────────────────────────────────
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # ── data ──────────────────────────────────────────────────────────────────
    dim        = 1024
    seq_len    = 64
    n_samples  = 512 * world_size
    batch_size = 32

    X = torch.randn(n_samples, seq_len, dim)
    y = torch.randint(0, 1000, (n_samples,))
    sampler    = DistributedSampler(TensorDataset(X, y),
                                    num_replicas=world_size, rank=rank, shuffle=True)
    loader     = DataLoader(TensorDataset(X, y), batch_size=batch_size,
                            sampler=sampler, pin_memory=True, num_workers=2)

    # ── model ─────────────────────────────────────────────────────────────────
    model     = SmallTransformer(dim=dim).to(device)
    model     = DDP(model, device_ids=[rank])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # ── torchscope ────────────────────────────────────────────────────────────
    prof = Profiler(
        interval    = 0.25,
        device      = rank,
        job_name    = "ddp_transformer",
        run_id      = os.environ.get("SLURM_JOB_ID", ""),
        rank        = rank,
        export_json = f"logs/ddp_rank{rank}.ndjson",
    )
    prof.comm.attach(model)   # register DDP comm hooks AFTER wrapping
    prof.start()

    # kernel tracing on rank 0 only (avoids redundant overhead on all ranks)
    if rank == 0:
        prof.enable_kernel_tracing(wait=2, warmup=1, active=3)

    # ── training loop ─────────────────────────────────────────────────────────
    n_epochs = 3
    for epoch in range(n_epochs):
        sampler.set_epoch(epoch)

        with prof.kernel_profiler() as kp:
            for step, (xb, yb) in enumerate(loader):
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                with prof.comm.step():
                    with annotate("forward"):
                        logits = model(xb)
                        loss   = criterion(logits, yb)

                    with annotate("backward"):
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()       # all-reduce happens here

                    with annotate("optimizer"):
                        optimizer.step()

                kp.step()

        if rank == 0:
            print(f"  epoch {epoch+1}/{n_epochs}  loss={loss.item():.4f}")

    # ── report (rank 0 writes HTML; all ranks write JSON) ────────────────────
    comm_sum = prof.comm.summary()
    if rank == 0:
        print(f"\n  comm ratio: {comm_sum.get('comm_pct', 'N/A')}% of step time")
        print(f"  stragglers: {comm_sum.get('stragglers', [])}")

    prof.report(
        output_path = f"torchscope_ddp_rank{rank}.html",
        title       = f"DDP Transformer — rank {rank}/{world_size}",
        metadata    = {
            "world_size":  world_size,
            "rank":        rank,
            "batch_size":  batch_size,
            "dim":         dim,
            "seq_len":     seq_len,
            "backend":     "nccl",
        },
    )

    dist.destroy_process_group()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    rank       = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    if world_size < 2:
        print("torchscope DDP example: only 1 GPU found.")
        print("Run with:  torchrun --nproc_per_node=<N> examples/distributed_training.py")
        print("Running in single-GPU mode for demonstration...")
        # Single-GPU fallback: still exercises CommCollector (no-op for single GPU)
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["RANK"]       = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["LOCAL_RANK"] = "0"
        world_size = 1

    train(rank, world_size)


if __name__ == "__main__":
    main()
