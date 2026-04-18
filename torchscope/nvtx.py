"""
NVTX range annotations for Nsight Systems compatibility.

When a workload is run under `nsys profile`, these annotations appear as
named coloured ranges in the Nsight Systems timeline, letting you correlate
high-level torchscope stages (pipeline name, stage type) with low-level
CUDA kernel execution.

Usage:
    from torchscope.nvtx import range_push, range_pop, annotate

    with annotate("preprocess"):
        tensor = preprocess(frames)

    # or manually:
    range_push("inference", color="green")
    out = model(tensor)
    range_pop()

Run under Nsight:
    nsys profile --trace=cuda,nvtx,osrt \
        --output=profile \
        python your_script.py
"""

from __future__ import annotations

import contextlib
from typing import Optional

# Colour palette — maps stage names to NVTX colour IDs
_COLORS: dict[str, int] = {
    "decode":      0xFF5B9BD5,  # blue
    "preprocess":  0xFFED7D31,  # orange
    "inference":   0xFF70AD47,  # green
    "postprocess": 0xFF9E6FCE,  # purple
    "forward":     0xFF70AD47,
    "backward":    0xFFE74C3C,  # red
    "optimizer":   0xFFF39C12,  # yellow
    "data":        0xFF5B9BD5,
    "comm":        0xFFE74C3C,
    "default":     0xFFAAAAAA,
}

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _color_for(name: str) -> int:
    lower = name.lower()
    for key, val in _COLORS.items():
        if key in lower:
            return val
    return _COLORS["default"]


def range_push(name: str, color: Optional[int] = None) -> None:
    """Push an NVTX range. No-op when torch.cuda is unavailable."""
    if not _HAS_TORCH:
        return
    try:
        torch.cuda.nvtx.range_push(name)
    except Exception:
        pass


def range_pop() -> None:
    """Pop the most recent NVTX range."""
    if not _HAS_TORCH:
        return
    try:
        torch.cuda.nvtx.range_pop()
    except Exception:
        pass


@contextlib.contextmanager
def annotate(name: str, color: Optional[int] = None):
    """Context manager that wraps a code block in an NVTX range."""
    range_push(name, color or _color_for(name))
    try:
        yield
    finally:
        range_pop()


class NVTXRegion:
    """
    Decorator + context manager.

    @NVTXRegion("forward_pass")
    def forward(self, x):
        ...

    or:
        with NVTXRegion("batch_decode"):
            frames = decoder.read(n)
    """

    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        range_push(self.name)
        return self

    def __exit__(self, *_):
        range_pop()

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with NVTXRegion(self.name):
                return fn(*args, **kwargs)

        return wrapper
