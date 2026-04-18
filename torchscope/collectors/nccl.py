"""
NCCLCollector — per-collective NCCL operation profiler.

Two modes:
  "hook" — monkey-patches torch.distributed.* to record wall-clock latency.
            Graceful no-op if torch.distributed is not initialized.
  "log"  — tails a file at log_path written with NCCL_DEBUG=INFO (redirect
            stderr to that path). Background daemon thread.

Usage (hook mode):
    nccl = NCCLCollector(mode="hook")
    nccl.start()
    # ... distributed training ...
    nccl.stop()
    s = nccl.summary()
    # {"allreduce_ms": ..., "dominant_collective": "all_reduce", ...}

Usage (log mode):
    # Run with: NCCL_DEBUG=INFO python train.py 2>nccl.log
    nccl = NCCLCollector(mode="log", log_path="nccl.log")
    nccl.start()
    ...
    nccl.stop()
"""

from __future__ import annotations

import re
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

_COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "broadcast", "reduce")

_DTYPE_SIZES: dict[str, int] = {
    "float32": 4, "float": 4,
    "float16": 2, "half":  2,
    "bfloat16": 2,
    "float64": 8, "double": 8,
    "int32": 4, "int64": 8, "int8": 1,
}

# Matches NCCL INFO lines like:
# NCCL INFO AllReduce: ... count=1048576 datatype=float32 ... algorithm=Ring protocol=Simple
_NCCL_LOG_RE = re.compile(
    r"NCCL\s+INFO\s+(?P<op>\w+):.*?"
    r"count=(?P<count>\d+).*?"
    r"datatype=(?P<dtype>\w+)"
    r"(?:.*?algorithm=(?P<algo>\w+))?"
    r"(?:.*?protocol=(?P<proto>\w+))?",
    re.IGNORECASE,
)


@dataclass
class NCCLEvent:
    op:          str
    size_bytes:  int
    duration_ms: float
    algorithm:   str
    protocol:    str
    ts:          float = field(default_factory=time.time)


class NCCLCollector:
    """
    Profiles NCCL collective operations.

    Args:
        mode:     "hook" patches torch.distributed; "log" tails a NCCL debug log file.
        log_path: required when mode="log"; path to NCCL_DEBUG=INFO stderr output.
    """

    def __init__(self, mode: str = "hook", log_path: Optional[str] = None) -> None:
        self.mode      = mode
        self.log_path  = log_path
        self.events:   list[NCCLEvent] = []
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        self._patches: dict = {}

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "NCCLCollector":
        self._running = True
        if self.mode == "hook":
            self._install_hooks()
        elif self.mode == "log":
            self._thread = threading.Thread(
                target=self._tail_log, daemon=True, name="torchscope.nccl"
            )
            self._thread.start()
        return self

    def stop(self) -> "NCCLCollector":
        self._running = False
        if self.mode == "hook":
            self._remove_hooks()
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None
        return self

    # ── hook mode ─────────────────────────────────────────────────────────────

    def _install_hooks(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        if not dist.is_available():
            return
        for op_name in _COLLECTIVES:
            orig = getattr(dist, op_name, None)
            if orig is None:
                continue
            self._patches[op_name] = orig
            setattr(dist, op_name, self._make_wrapper(op_name, orig))

    def _make_wrapper(self, op_name: str, orig_fn):
        collector = self

        def _wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = orig_fn(*args, **kwargs)
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
            duration_ms = (time.perf_counter() - t0) * 1e3
            collector.events.append(NCCLEvent(
                op=op_name,
                size_bytes=_estimate_size(args),
                duration_ms=duration_ms,
                algorithm="",
                protocol="",
            ))
            return result

        _wrapper.__name__ = orig_fn.__name__
        return _wrapper

    def _remove_hooks(self) -> None:
        try:
            import torch.distributed as dist
        except ImportError:
            return
        for op_name, orig in self._patches.items():
            setattr(dist, op_name, orig)
        self._patches.clear()

    # ── log mode ─────────────────────────────────────────────────────────────

    def _tail_log(self) -> None:
        if not self.log_path:
            return
        try:
            with open(self.log_path) as f:
                f.seek(0, 2)
                while self._running:
                    line = f.readline()
                    if not line:
                        time.sleep(0.05)
                        continue
                    event = _parse_nccl_line(line)
                    if event:
                        self.events.append(event)
        except (OSError, IOError):
            pass

    # ── analytics ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.events:
            return {"n_events": 0}

        by_op: dict[str, list[NCCLEvent]] = {}
        for ev in self.events:
            by_op.setdefault(ev.op, []).append(ev)

        op_totals = {op: sum(e.duration_ms for e in evs) for op, evs in by_op.items()}
        total_ms  = sum(op_totals.values()) or 1.0
        dominant  = max(op_totals, key=op_totals.__getitem__) if op_totals else ""
        dom_pct   = op_totals.get(dominant, 0) / total_ms * 100

        sizes     = [e.size_bytes for e in self.events if e.size_bytes > 0]
        avg_bytes = int(sum(sizes) / len(sizes)) if sizes else 0

        return {
            "n_events":               len(self.events),
            "total_ms":               round(total_ms, 2),
            "dominant_collective":    dominant,
            "dominant_pct":           round(dom_pct, 1),
            "avg_message_size_bytes": avg_bytes,
            "allreduce_ms":           round(op_totals.get("all_reduce",    0), 2),
            "allgather_ms":           round(op_totals.get("all_gather",    0), 2),
            "reducescatter_ms":       round(op_totals.get("reduce_scatter", 0), 2),
            "events_by_op": {
                op: {
                    "count":          len(evs),
                    "total_ms":       round(sum(e.duration_ms for e in evs), 2),
                    "avg_ms":         round(sum(e.duration_ms for e in evs) / len(evs), 2),
                    "avg_size_bytes": int(sum(e.size_bytes for e in evs) / len(evs)),
                }
                for op, evs in by_op.items()
            },
        }


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_nccl_line(line: str) -> Optional[NCCLEvent]:
    m = _NCCL_LOG_RE.search(line)
    if not m:
        return None
    raw_op = m.group("op").lower()
    op = raw_op
    for canonical in _COLLECTIVES:
        if canonical.replace("_", "") in raw_op.replace("_", ""):
            op = canonical
            break
    count   = int(m.group("count"))
    dtype   = (m.group("dtype") or "float32").lower()
    elem_sz = _DTYPE_SIZES.get(dtype, 4)
    algo    = (m.group("algo")  or "").lower()
    proto   = (m.group("proto") or "").upper()
    return NCCLEvent(
        op=op,
        size_bytes=count * elem_sz,
        duration_ms=0.0,
        algorithm=algo,
        protocol=proto,
    )


def _estimate_size(args) -> int:
    try:
        import torch
        t = args[0] if args else None
        if isinstance(t, torch.Tensor):
            return t.numel() * t.element_size()
        if isinstance(t, (list, tuple)) and t and isinstance(t[0], torch.Tensor):
            return sum(x.numel() * x.element_size() for x in t)
    except Exception:
        pass
    return 0
