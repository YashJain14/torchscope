"""
RayProfiler — cluster-level GPU observability for Ray Train workloads.

Each Ray Train worker runs a standard Profiler internally and periodically
ships its GPU/memory summaries to a driver-side aggregator actor. The driver
then computes cluster-level stats (imbalance, pressure, etc.) and generates
a unified HTML report with BottleneckAnalyzer cluster rules.

Usage (driver side):
    from torchscope.ray_profiler import RayProfiler
    rp = RayProfiler(num_workers=4)

Usage (inside Ray Train worker function):
    from torchscope.ray_profiler import worker_profiler_context
    with worker_profiler_context(rp, worker_id=train.get_context().get_world_rank()) as prof:
        for batch in dataloader:
            model(batch)

After training:
    report = rp.aggregate_report("cluster_report.html")
"""

from __future__ import annotations

import time
import threading
from typing import Optional, TYPE_CHECKING

try:
    import ray
    _RAY = True
except ImportError:
    _RAY = False


# ── aggregator actor ──────────────────────────────────────────────────────────

if _RAY:
    @ray.remote
    class _AggregatorActor:
        """Lives on the driver. Workers call .push.remote() to report summaries."""

        def __init__(self, num_workers: int) -> None:
            self._num_workers = num_workers
            self._summaries:  list[dict] = []

        def push(self, summary: dict) -> None:
            self._summaries.append(summary)

        def get_all(self) -> list[dict]:
            return list(self._summaries)

        def reset(self) -> None:
            self._summaries.clear()

else:
    # Stub so RayProfiler can still be imported and tested without Ray
    class _AggregatorActor:  # type: ignore[no-redef]
        def __init__(self, num_workers: int) -> None:
            self._summaries: list[dict] = []

        def push(self, summary: dict) -> None:
            self._summaries.append(summary)

        def get_all(self) -> list[dict]:
            return list(self._summaries)

        def reset(self) -> None:
            self._summaries.clear()


# ── driver-side profiler ──────────────────────────────────────────────────────

class RayProfiler:
    """
    Driver-side aggregator for cluster-level GPU metrics.

    Args:
        num_workers:     expected number of Ray Train workers
        interval:        GPU polling cadence inside each worker (seconds)
        report_interval: how often workers push summaries to aggregator (seconds)
    """

    def __init__(
        self,
        num_workers:     int   = 1,
        interval:        float = 0.5,
        report_interval: float = 10.0,
    ) -> None:
        self._num_workers     = num_workers
        self._interval        = interval
        self._report_interval = report_interval
        self._enabled         = _RAY
        self._actor_handle    = None

        if not _RAY:
            print("torchscope: ray not installed — RayProfiler will return empty "
                  "reports. Install with: pip install 'torchscope[ray]'")

    def _actor(self):
        """Lazily create the aggregator actor on first access."""
        if self._actor_handle is None:
            if _RAY:
                self._actor_handle = _AggregatorActor.remote(self._num_workers)
            else:
                self._actor_handle = _AggregatorActor(self._num_workers)
        return self._actor_handle

    def _compute_cluster_stats(self, summaries: list[dict]) -> dict:
        """
        Compute cluster-level aggregates from per-worker summary dicts.
        Pure function — no Ray dependency.
        """
        if not summaries:
            return {}

        mem_pct_threshold = 85  # matches _T["cluster_mem_pressure_pct"] in analyzer

        worker_utils: dict[int, float] = {}
        worker_mem_pcts: dict[int, float] = {}

        for s in summaries:
            wid      = s.get("worker_id", -1)
            gpu_sum  = s.get("gpu_summary", {})
            mem_sum  = s.get("mem_summary",  {})
            util     = gpu_sum.get("avg_gpu_util", -1)
            mem_pct  = mem_sum.get("memory_used_pct", -1)
            if util >= 0:
                # Keep the most recent reading per worker
                worker_utils[wid] = util
            if mem_pct >= 0:
                worker_mem_pcts[wid] = mem_pct

        if not worker_utils:
            return {"n_summaries_received": len(summaries)}

        utils = list(worker_utils.values())
        cluster_avg = sum(utils) / len(utils)
        cluster_max = max(utils)

        bottleneck_id   = min(worker_utils, key=worker_utils.__getitem__)
        bottleneck_util = worker_utils[bottleneck_id]
        imbalance_pct   = (
            (cluster_avg - bottleneck_util) / cluster_avg * 100
            if cluster_avg > 0 else 0.0
        )

        all_mem_gb = [
            s.get("gpu_summary", {}).get("avg_mem_gb", 0)
            for s in summaries
        ]
        cluster_avg_mem = sum(all_mem_gb) / len(all_mem_gb) if all_mem_gb else 0
        cluster_max_mem = max(all_mem_gb) if all_mem_gb else 0

        pressured = [
            wid for wid, pct in worker_mem_pcts.items()
            if pct > mem_pct_threshold
        ]

        return {
            "cluster_avg_gpu_util": round(cluster_avg, 1),
            "cluster_max_gpu_util": round(cluster_max, 1),
            "bottleneck_worker_id": bottleneck_id,
            "bottleneck_avg_util":  round(bottleneck_util, 1),
            "cluster_avg_mem_gb":   round(cluster_avg_mem, 2),
            "cluster_max_mem_gb":   round(cluster_max_mem, 2),
            "worker_utils":         worker_utils,
            "imbalance_pct":        round(imbalance_pct, 1),
            "pressured_workers":    pressured,
            "n_summaries_received": len(summaries),
        }

    def aggregate_report(
        self,
        output_path: str = "cluster_report.html",
        title:       str = "Cluster GPU Report",
    ) -> dict:
        """
        Pull all worker summaries from the aggregator actor, compute
        cluster-level stats, run BottleneckAnalyzer with cluster rules,
        and generate an HTML report.

        Returns the analysis dict.
        """
        if not self._enabled:
            return {}

        actor = self._actor()
        if _RAY:
            summaries = ray.get(actor.get_all.remote())
        else:
            summaries = actor.get_all()

        cluster_stats = self._compute_cluster_stats(summaries)

        # Build a representative per-worker combined gpu/mem summary
        # (use the bottleneck worker's data for single-device charts)
        bot_id  = cluster_stats.get("bottleneck_worker_id", -1)
        bot_sum = next(
            (s for s in summaries if s.get("worker_id") == bot_id),
            summaries[0] if summaries else {}
        )
        gpu_sum = bot_sum.get("gpu_summary", {})
        mem_sum = bot_sum.get("mem_summary",  {})

        from .analyzer import BottleneckAnalyzer
        analyzer = BottleneckAnalyzer(
            gpu     = gpu_sum,
            memory  = mem_sum,
            cluster = cluster_stats,
        )
        analysis = analyzer.analyze()
        analysis["_cluster_stats"] = cluster_stats

        from .exporters.html import HTMLExporter
        HTMLExporter().generate(
            output_path = output_path,
            title       = title,
            analysis    = analysis,
            metadata    = {
                "workers":        self._num_workers,
                "summaries_recv": len(summaries),
                "cluster_avg_util": f"{cluster_stats.get('cluster_avg_gpu_util', '?')}%",
            },
        )
        print(f"torchscope: cluster report → {output_path}")
        return analysis


# ── worker-side context manager ───────────────────────────────────────────────

class _WorkerProfilerContext:
    """
    Used inside a Ray Train worker function. Runs a standard Profiler
    and periodically pushes gpu/memory summaries to the aggregator actor.
    """

    def __init__(
        self,
        ray_profiler: RayProfiler,
        worker_id:    int   = 0,
        device:       int   = 0,
        interval:     float = 0.5,
    ) -> None:
        self._rp        = ray_profiler
        self._worker_id = worker_id
        self._device    = device
        self._interval  = interval
        self._prof      = None
        self._thread:   Optional[threading.Thread] = None
        self._running   = False

    def __enter__(self):
        from .profiler import Profiler
        self._prof = Profiler(interval=self._interval, device=self._device,
                              rank=self._worker_id)
        self._prof.start()
        self._running = True
        self._thread  = threading.Thread(
            target=self._report_loop, daemon=True, name="torchscope.ray_reporter"
        )
        self._thread.start()
        return self._prof

    def __exit__(self, *_) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=self._rp._report_interval + 2)
        if self._prof:
            self._prof.stop()
        # Final push
        self._push_summary()

    def _push_summary(self) -> None:
        if not self._prof or not self._rp._enabled:
            return
        summary = {
            "worker_id":   self._worker_id,
            "gpu_summary": self._prof.gpu.summary(),
            "mem_summary": self._prof.memory.summary(),
            "timestamp":   time.time(),
        }
        actor = self._rp._actor()
        if _RAY:
            actor.push.remote(summary)
        else:
            actor.push(summary)

    def _report_loop(self) -> None:
        while self._running:
            time.sleep(self._rp._report_interval)
            if self._running:
                self._push_summary()


def worker_profiler_context(
    ray_profiler: RayProfiler,
    worker_id:    int   = 0,
    device:       int   = 0,
) -> _WorkerProfilerContext:
    """
    Factory for the worker-side profiler context.

    Example:
        with worker_profiler_context(rp, worker_id=rank) as prof:
            for batch in loader:
                model(batch)
    """
    return _WorkerProfilerContext(
        ray_profiler=ray_profiler,
        worker_id=worker_id,
        device=device,
        interval=ray_profiler._interval,
    )
