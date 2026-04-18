"""
KernelTracer — wraps torch.profiler to capture CUDA kernel timings.

Captures:
  - Per-kernel CUDA time, CPU time, call count
  - Estimated FLOPs per kernel
  - CPU↔GPU memory transfer overhead (memcpy events)
  - Input tensor shapes (for debugging)

Designed to be used for a bounded number of steps inside a warm loop,
not for the full training run. Typical use: wrap 3–5 steps after warmup.

Usage:
    tracer = KernelTracer(wait=1, warmup=1, active=3)

    with tracer.profile():
        for step, batch in enumerate(loader):
            model(batch)
            tracer.step()           # advance profiler schedule

    print(tracer.top_kernels(n=10))
    print(tracer.transfer_overhead())
"""

from __future__ import annotations

from typing import Optional

try:
    import torch
    from torch.profiler import profile, ProfilerActivity, schedule
    _TORCH = True
except ImportError:
    _TORCH = False


class KernelTracer:
    def __init__(self, wait: int = 1, warmup: int = 1, active: int = 3):
        self._wait   = wait
        self._warmup = warmup
        self._active = active
        self._events: list[dict] = []
        self._prof: Optional[object] = None

    # ── public API ────────────────────────────────────────────────────────────

    def profile(self):
        """
        Return a context manager that activates torch.profiler.

        The returned object is also a step-callable: call .step() on it
        (or on this tracer) after each workload iteration to advance the
        profiler schedule.  If you never call step(), no trace is captured.
        """
        if not _TORCH:
            return _NoopCtx()
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self._wait, warmup=self._warmup,
                active=self._active, repeat=1,
            ),
            on_trace_ready=self._handle_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
        )
        return _TracerCtx(self._prof, self)

    def profile_block(self):
        """
        Context manager for a single self-contained code block.
        Automatically steps the profiler — no manual .step() needed.

            with tracer.profile_block():
                output = model(batch)
        """
        if not _TORCH:
            return _NoopCtx()
        return _BlockCtx(self)

    def step(self):
        """Advance the profiler schedule by one step."""
        if self._prof is not None:
            try:
                self._prof.step()
            except Exception:
                pass

    # ── results ───────────────────────────────────────────────────────────────

    def top_kernels(self, n: int = 20) -> list[dict]:
        """Top N kernels by total CUDA time, deduplicated by name."""
        if not self._events:
            return []
        seen: dict[str, dict] = {}
        for ev in self._events:
            k = ev["name"]
            if k not in seen or ev["cuda_time_ms"] > seen[k]["cuda_time_ms"]:
                seen[k] = ev
        return sorted(seen.values(), key=lambda x: x["cuda_time_ms"], reverse=True)[:n]

    def transfer_overhead(self) -> dict:
        """Fraction of total CUDA time spent in CPU↔GPU memcpy operations."""
        if not self._events:
            return {}
        xfer = [
            e for e in self._events
            if any(kw in e["name"].lower()
                   for kw in ("memcpy", "memset", "h2d", "d2h", "htod", "dtoh"))
        ]
        total = sum(e["cuda_time_ms"] for e in self._events)
        xfer_t = sum(e["cuda_time_ms"] for e in xfer)
        return {
            "transfer_ms":   round(xfer_t, 3),
            "compute_ms":    round(total - xfer_t, 3),
            "transfer_pct":  round(xfer_t / total * 100, 1) if total > 0 else 0.0,
            "n_xfer_ops":    len(xfer),
        }

    def flop_summary(self) -> dict:
        """Aggregate FLOP counts across all captured events."""
        total_flops = sum(e.get("flops", 0) for e in self._events)
        return {
            "total_flops":    total_flops,
            "total_gflops":   round(total_flops / 1e9, 3),
        }

    def has_data(self) -> bool:
        return bool(self._events)

    # ── internal ─────────────────────────────────────────────────────────────

    def _handle_trace(self, prof):
        for ev in prof.key_averages():
            self._events.append({
                "name":          ev.key,
                "cuda_time_ms":  round(ev.cuda_time_total  / 1e3, 3),
                "cpu_time_ms":   round(ev.cpu_time_total   / 1e3, 3),
                "calls":         ev.count,
                "flops":         ev.flops or 0,
                "cuda_mem_mb":   ev.cuda_memory_usage / 1e6,
                "cpu_mem_mb":    ev.cpu_memory_usage  / 1e6,
            })


class _NoopCtx:
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def step(self): pass


class _TracerCtx:
    """Wraps torch.profiler and exposes .step() so callers don't need to hold two refs."""
    def __init__(self, prof, tracer: "KernelTracer"):
        self._prof   = prof
        self._tracer = tracer

    def __enter__(self):
        self._prof.__enter__()
        return self

    def __exit__(self, *args):
        self._prof.__exit__(*args)

    def step(self):
        self._tracer.step()


class _BlockCtx:
    """
    Single-block profiler: runs wait=0,warmup=0,active=1 internally.
    Fires on __exit__ so exactly one trace is captured per block.
    """
    def __init__(self, tracer: "KernelTracer"):
        self._tracer = tracer

    def __enter__(self):
        if not _TORCH:
            return self
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
            on_trace_ready=self._tracer._handle_trace,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            with_flops=True,
        )
        self._prof.__enter__()
        return self

    def __exit__(self, *args):
        if not _TORCH:
            return
        self._prof.step()   # trigger the single active step
        self._prof.__exit__(*args)
