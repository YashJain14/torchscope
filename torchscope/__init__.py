"""
torchscope — PyTorch GPU observability toolkit.

    from torchscope import Profiler

    with Profiler() as prof:
        train(model, dataloader)

    prof.report("report.html")
"""

__version__ = "0.1.0"
__author__  = "Yash Jain"


def __getattr__(name: str):
    if name == "Profiler":
        from .profiler import Profiler
        return Profiler
    if name == "RayProfiler":
        from .ray_profiler import RayProfiler
        return RayProfiler
    raise AttributeError(f"module 'torchscope' has no attribute {name!r}")


__all__ = ["Profiler", "RayProfiler"]
