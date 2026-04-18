"""
benchmark_adapter — thin adapter between benchmark.py and torchscope.

Converts the list of StageTimings objects produced by benchmark.py's
run_benchmark() into the generic dicts torchscope.Profiler.report() expects.

This is intentionally thin — torchscope has no knowledge of benchmark.py's
internal types; the adapter is the only place that bridges the two.
"""

from __future__ import annotations

import dataclasses
from collections import defaultdict


def adapt_results(results: list) -> tuple[dict, dict]:
    """
    Convert run_benchmark() output → (custom_stages, pipeline_meta).

    custom_stages:  {stage_label: avg_seconds_across_ALL_pipelines_and_runs}
                    Used by BottleneckAnalyzer._check_custom_stages().

    pipeline_meta:  flat dict of per-pipeline avg FPS for the report metadata table.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        d = _to_dict(r)
        if not d.get("error"):
            groups[d["pipeline_name"]].append(d)

    if not groups:
        return {}, {}

    # Per-pipeline averages
    pipeline_avgs: dict[str, dict] = {}
    for name, runs in groups.items():
        n = len(runs)
        pipeline_avgs[name] = {
            "avg_fps":       round(sum(r["fps"]          for r in runs) / n, 1),
            "decode_s":      sum(r["decode_s"]       for r in runs) / n,
            "preprocess_s":  sum(r["preprocess_s"]   for r in runs) / n,
            "inference_s":   sum(r["inference_s"]    for r in runs) / n,
            "postprocess_s": sum(r["postprocess_s"]  for r in runs) / n,
            "total_s":       sum(r["total_s"]        for r in runs) / n,
        }

    # custom_stages: use the WORST pipeline (highest total_s) to surface bottlenecks
    worst = max(pipeline_avgs, key=lambda n: pipeline_avgs[n]["total_s"])
    w     = pipeline_avgs[worst]
    custom_stages = {
        "decode":      w["decode_s"],
        "preprocess":  w["preprocess_s"],
        "inference":   w["inference_s"],
        "postprocess": w["postprocess_s"],
    }

    # pipeline_meta: per-pipeline FPS for the report metadata table
    pipeline_meta = {
        f"{name} avg FPS": f"{pipeline_avgs[name]['avg_fps']}"
        for name in pipeline_avgs
    }
    pipeline_meta["bottleneck pipeline"] = worst

    return custom_stages, pipeline_meta


def _to_dict(r) -> dict:
    if isinstance(r, dict):
        return r
    if dataclasses.is_dataclass(r):
        d = dataclasses.asdict(r)
        d["fps"] = r.fps if hasattr(r, "fps") else (
            d["frames_processed"] / d["total_s"] if d.get("total_s", 0) > 0 else 0
        )
        return d
    return vars(r)
