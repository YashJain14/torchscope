"""
MemoryCollector — tracks GPU memory allocation patterns via torch.cuda.

Captures every `interval` seconds:
  - Bytes allocated / reserved / peak allocated
  - Fragmentation %  (reserved − allocated) / reserved
  - Allocation retries and OOM events
  - Active memory segments and blocks

Fragmentation and monotonic allocation growth are the two most common
silent killers of long training jobs.
"""

from __future__ import annotations

import threading
import time
from typing import Optional


class MemoryCollector:
    """
    Background thread that snapshots torch.cuda memory stats.

    Usage:
        col = MemoryCollector(device=0)
        col.start()
        ... workload ...
        snapshots = col.stop()
        print(col.summary())
    """

    def __init__(self, interval: float = 0.5, device: int = 0):
        self.interval  = interval
        self.device    = device
        self.snapshots: list[dict] = []
        self._running  = False
        self._thread:  Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "MemoryCollector":
        try:
            import torch
            torch.cuda.reset_peak_memory_stats(self.device)
        except Exception:
            pass
        self._running = True
        self._thread  = threading.Thread(target=self._poll, daemon=True,
                                         name="torchscope.memory")
        self._thread.start()
        return self

    def stop(self) -> list[dict]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self.snapshots

    # ── polling ───────────────────────────────────────────────────────────────

    def _poll(self):
        try:
            import torch
        except ImportError:
            return

        while self._running:
            try:
                stats    = torch.cuda.memory_stats(self.device)
                alloc    = torch.cuda.memory_allocated(self.device)
                reserved = torch.cuda.memory_reserved(self.device)
                peak     = torch.cuda.max_memory_allocated(self.device)

                self.snapshots.append({
                    "ts":              time.time(),
                    "allocated_gb":    alloc    / 1e9,
                    "reserved_gb":     reserved / 1e9,
                    "peak_alloc_gb":   peak     / 1e9,
                    "fragmentation_pct": (
                        (reserved - alloc) / reserved * 100
                        if reserved > 0 else 0.0
                    ),
                    "alloc_retries":   stats.get("num_alloc_retries", 0),
                    "oom_count":       stats.get("num_ooms", 0),
                    "active_segments": stats.get("segment.all.current", 0),
                    "active_blocks":   stats.get("block.all.current", 0),
                })
            except Exception:
                pass
            time.sleep(self.interval)

    # ── analytics ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.snapshots:
            return {}
        alloc_v = [s["allocated_gb"]      for s in self.snapshots]
        res_v   = [s["reserved_gb"]       for s in self.snapshots]
        frag_v  = [s["fragmentation_pct"] for s in self.snapshots]

        try:
            import torch
            total_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
        except Exception:
            total_gb = self.snapshots[0]["reserved_gb"]

        last = self.snapshots[-1]

        return {
            "peak_alloc_gb":       round(max(alloc_v), 3),
            "avg_alloc_gb":        round(sum(alloc_v) / len(alloc_v), 3),
            "peak_reserved_gb":    round(max(res_v), 3),
            "total_gpu_gb":        round(total_gb, 2),
            "memory_used_pct":     round(max(alloc_v) / total_gb * 100, 1),
            "headroom_gb":         round(total_gb - max(res_v), 2),
            "avg_fragmentation_pct": round(sum(frag_v) / len(frag_v), 1),
            "max_fragmentation_pct": round(max(frag_v), 1),
            "alloc_retries":       last["alloc_retries"],
            "oom_count":           last["oom_count"],
            "n_snapshots":         len(self.snapshots),
        }

    def detect_leak(self, window: int = 20) -> tuple[bool, float]:
        """
        Heuristic: if allocated memory is monotonically increasing for >80%
        of the last `window` samples, flag a likely leak.
        Returns (leak_detected, monotone_ratio_pct).
        """
        if len(self.snapshots) < window:
            return False, 0.0
        recent = [s["allocated_gb"] for s in self.snapshots[-window:]]
        rises  = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        ratio  = rises / (len(recent) - 1) * 100
        return ratio > 80.0, round(ratio, 1)
