"""
CommCollector — distributed training communication profiler.

Registers DDP communication hooks to measure:
  - Per all-reduce duration and tensor size
  - Compute-to-communication ratio per training step
  - Straggler GPU detection (ranks consistently slower than median)

Only active in distributed jobs (torch.distributed initialized + DDP model).
Gracefully no-ops for single-GPU workloads.

Usage:
    comm = CommCollector()
    comm.attach(ddp_model)          # register hooks once after DDP wrap

    for batch in dataloader:
        with comm.step():           # time one full forward+backward+step
            loss = model(batch)
            loss.backward()
            optimizer.step()

    print(comm.summary())
"""

from __future__ import annotations

import time
from typing import Optional


class CommCollector:
    def __init__(self):
        self._comm_events: list[dict]      = []
        self._step_times:  list[float]     = []
        self._per_rank:    dict[int, list] = {}
        self._attached     = False

    # ── attach ────────────────────────────────────────────────────────────────

    def attach(self, model) -> "CommCollector":
        """Register a communication hook on a DDP model."""
        try:
            import torch.nn as nn
            import torch.distributed as dist
        except ImportError:
            return self

        if not isinstance(model, nn.parallel.DistributedDataParallel):
            return self

        comm_events = self._comm_events

        def _hook(state, bucket):
            t0  = time.perf_counter()
            fut = dist.all_reduce(bucket.buffer(), async_op=True).get_future()

            def _cb(f):
                comm_events.append({
                    "ts":            time.time(),
                    "duration_ms":   (time.perf_counter() - t0) * 1e3,
                    "bytes_mb":      (
                        bucket.buffer().numel()
                        * bucket.buffer().element_size()
                        / 1e6
                    ),
                    "rank":          dist.get_rank(),
                })
                return bucket.buffer()

            return fut.then(_cb)

        model.register_comm_hook(state=None, hook=_hook)
        self._attached = True
        return self

    # ── step context manager ──────────────────────────────────────────────────

    def step(self) -> "_StepCtx":
        return _StepCtx(self)

    def _record_step(self, duration_ms: float):
        self._step_times.append(duration_ms)
        rank = self._rank()
        self._per_rank.setdefault(rank, []).append(duration_ms)

    @staticmethod
    def _rank() -> int:
        try:
            import torch.distributed as dist
            return dist.get_rank() if dist.is_initialized() else 0
        except Exception:
            return 0

    # ── analytics ─────────────────────────────────────────────────────────────

    def compute_comm_ratio(self) -> dict:
        if not self._step_times:
            return {}
        n         = len(self._step_times)
        avg_step  = sum(self._step_times) / n
        # average total comm per step = sum(comm durations) / n_steps
        avg_comm  = (
            sum(e["duration_ms"] for e in self._comm_events) / n
            if self._comm_events else 0.0
        )
        avg_comp  = max(avg_step - avg_comm, 0.0)
        return {
            "n_steps":        n,
            "avg_step_ms":    round(avg_step, 2),
            "avg_comm_ms":    round(avg_comm,  2),
            "avg_compute_ms": round(avg_comp,  2),
            "comm_pct":       round(avg_comm  / avg_step * 100, 1) if avg_step else 0,
            "compute_pct":    round(avg_comp  / avg_step * 100, 1) if avg_step else 0,
        }

    def detect_stragglers(self, threshold_pct: float = 10.0) -> list[dict]:
        """Return ranks whose median step time exceeds global median by threshold_pct."""
        if len(self._per_rank) < 2:
            return []
        medians = {
            rank: sorted(times)[len(times) // 2]
            for rank, times in self._per_rank.items()
        }
        global_median = sorted(medians.values())[len(medians) // 2]
        if global_median == 0:
            return []
        return [
            {
                "rank":           rank,
                "median_step_ms": round(med, 2),
                "slowdown_pct":   round((med - global_median) / global_median * 100, 2),
            }
            for rank, med in medians.items()
            if (med - global_median) / global_median * 100 > threshold_pct
        ]

    def summary(self) -> dict:
        return {
            "attached":        self._attached,
            **self.compute_comm_ratio(),
            "stragglers":      self.detect_stragglers(),
            "total_comm_ops":  len(self._comm_events),
            "total_bytes_mb":  round(sum(e["bytes_mb"] for e in self._comm_events), 2),
        }


class _StepCtx:
    def __init__(self, col: CommCollector):
        self._col = col

    def __enter__(self):
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        try:
            import torch
            torch.cuda.synchronize()
        except Exception:
            pass
        self._col._record_step((time.perf_counter() - self._t0) * 1e3)
