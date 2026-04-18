"""
Example: profile a batch inference workload (embedding generation).

Simulates the Exa-style throughput bottleneck from the implementation plan:
  - Small batch sizes → GPU underutilised
  - torchscope flags MEMORY_UNDERUSED and LOW_GPU_UTIL
  - Increasing batch size resolves both

    python examples/inference_server.py --batch 32   # underutilised
    python examples/inference_server.py --batch 128  # better
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchscope import Profiler
from torchscope.nvtx import annotate


class EmbeddingModel(nn.Module):
    def __init__(self, vocab: int = 30522, dim: int = 384, layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=8, dim_feedforward=dim * 4,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.pool = lambda x: x[:, 0]   # CLS token

    def forward(self, x):
        return self.pool(self.encoder(self.embed(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch",    type=int, default=32)
    ap.add_argument("--n",        type=int, default=1024, help="Total samples")
    ap.add_argument("--seq-len",  type=int, default=128)
    ap.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device  = args.device
    X       = torch.randint(0, 30522, (args.n, args.seq_len))
    loader  = DataLoader(TensorDataset(X), batch_size=args.batch,
                         pin_memory=(device == "cuda"), num_workers=2)

    model = EmbeddingModel().to(device)
    if device == "cuda":
        model = model.half()   # fp16

    prof = Profiler(
        interval    = 0.25,
        job_name    = "embedding_inference",
        export_json = f"logs/inference_b{args.batch}.ndjson",
    )
    prof.start()

    total = 0
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            with annotate("inference"):
                emb = model(xb)
            total += xb.shape[0]

    prof.report(
        output_path   = f"torchscope_inference_b{args.batch}.html",
        title         = f"Embedding Inference — batch={args.batch}",
        metadata      = {
            "total_samples": total,
            "batch_size":    args.batch,
            "seq_len":       args.seq_len,
            "throughput_est": f"{total / max(prof._start_ts and (1), 1):.0f} samples/s",
        },
    )


if __name__ == "__main__":
    main()
