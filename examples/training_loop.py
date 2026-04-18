"""
Example: profile a standard single-GPU PyTorch training loop.

    python examples/training_loop.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchscope import Profiler
from torchscope.nvtx import annotate


def build_model(hidden: int = 1024, layers: int = 6) -> nn.Sequential:
    parts = [nn.Linear(hidden, hidden), nn.ReLU()]
    for _ in range(layers - 1):
        parts += [nn.Linear(hidden, hidden), nn.ReLU()]
    parts.append(nn.Linear(hidden, 10))
    return nn.Sequential(*parts)


def main():
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    hidden    = 2048
    n_samples = 2048
    batch     = 64
    epochs    = 3

    X = torch.randn(n_samples, hidden)
    y = torch.randint(0, 10, (n_samples,))
    loader = DataLoader(TensorDataset(X, y), batch_size=batch,
                        pin_memory=(device == "cuda"), num_workers=2)

    model     = build_model(hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    prof = Profiler(
        interval    = 0.25,
        device      = 0 if device == "cuda" else 0,
        job_name    = "mlp_training",
        export_json = "logs/training_run.ndjson",
    )
    prof.start()

    # Enable kernel tracing for 3 steps after 2-step warmup
    prof.enable_kernel_tracing(wait=2, warmup=1, active=3)

    with prof.kernel_profiler() as kp:
        for epoch in range(epochs):
            for step, (xb, yb) in enumerate(loader):
                xb, yb = xb.to(device), yb.to(device)

                with annotate("forward"):
                    logits = model(xb)
                    loss   = criterion(logits, yb)

                with annotate("backward"):
                    optimizer.zero_grad()
                    loss.backward()

                with annotate("optimizer"):
                    optimizer.step()

                kp.step()

            print(f"  epoch {epoch + 1}/{epochs}  loss={loss.item():.4f}")

    prof.report(
        output_path = "torchscope_training.html",
        title       = f"MLP Training — hidden={hidden} batch={batch}",
        metadata    = {"epochs": epochs, "batch_size": batch,
                       "model_params": sum(p.numel() for p in model.parameters())},
    )


if __name__ == "__main__":
    main()
