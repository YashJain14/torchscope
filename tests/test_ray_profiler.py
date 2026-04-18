"""
Tests for RayProfiler and cluster-level BottleneckAnalyzer rules.
All tests run without Ray or a real GPU — synthetic summaries injected directly.
"""

import pytest
from unittest.mock import patch

from torchscope.ray_profiler import RayProfiler, _AggregatorActor, worker_profiler_context
from torchscope.analyzer import BottleneckAnalyzer


# ── helpers ───────────────────────────────────────────────────────────────────

def _worker_summary(worker_id, avg_gpu_util, memory_used_pct=50.0, avg_mem_gb=4.0):
    return {
        "worker_id":   worker_id,
        "gpu_summary": {
            "avg_gpu_util":  avg_gpu_util,
            "avg_mem_gb":    avg_mem_gb,
            "n_samples":     10,
        },
        "mem_summary":  {
            "memory_used_pct": memory_used_pct,
            "peak_alloc_gb":   avg_mem_gb,
        },
        "timestamp": 0.0,
    }


# ── cluster stats computation ─────────────────────────────────────────────────

class TestClusterStats:
    def setup_method(self):
        self.rp = RayProfiler(num_workers=4)

    def test_avg_util_computed(self):
        summaries = [
            _worker_summary(0, 60),
            _worker_summary(1, 80),
            _worker_summary(2, 70),
        ]
        stats = self.rp._compute_cluster_stats(summaries)
        assert stats["cluster_avg_gpu_util"] == pytest.approx(70.0)

    def test_max_util_computed(self):
        summaries = [_worker_summary(i, u) for i, u in enumerate([60, 80, 70])]
        stats = self.rp._compute_cluster_stats(summaries)
        assert stats["cluster_max_gpu_util"] == pytest.approx(80.0)

    def test_bottleneck_worker_identified(self):
        summaries = [
            _worker_summary(0, 80),
            _worker_summary(1, 40),
            _worker_summary(2, 75),
        ]
        stats = self.rp._compute_cluster_stats(summaries)
        assert stats["bottleneck_worker_id"] == 1
        assert stats["bottleneck_avg_util"] == pytest.approx(40.0)

    def test_imbalance_pct_calculation(self):
        # workers [80, 40, 80, 80] → mean=70, min=40
        # imbalance = (70-40)/70 * 100 ≈ 42.9%
        summaries = [
            _worker_summary(0, 80),
            _worker_summary(1, 40),
            _worker_summary(2, 80),
            _worker_summary(3, 80),
        ]
        stats = self.rp._compute_cluster_stats(summaries)
        assert stats["imbalance_pct"] == pytest.approx(42.9, abs=0.1)

    def test_no_imbalance_balanced_workers(self):
        summaries = [_worker_summary(i, 78 + i) for i in range(4)]
        stats = self.rp._compute_cluster_stats(summaries)
        # All workers within a few pct of each other
        assert stats["imbalance_pct"] < 5.0

    def test_pressured_workers_identified(self):
        summaries = [
            _worker_summary(0, 70, memory_used_pct=90.0),
            _worker_summary(1, 70, memory_used_pct=50.0),
            _worker_summary(2, 70, memory_used_pct=95.0),
        ]
        stats = self.rp._compute_cluster_stats(summaries)
        assert set(stats["pressured_workers"]) == {0, 2}

    def test_no_pressure_below_threshold(self):
        summaries = [_worker_summary(i, 70, memory_used_pct=70.0) for i in range(3)]
        stats = self.rp._compute_cluster_stats(summaries)
        assert stats["pressured_workers"] == []

    def test_empty_summaries_returns_empty(self):
        stats = self.rp._compute_cluster_stats([])
        assert stats == {}

    def test_multiple_summaries_same_worker_uses_latest(self):
        # Worker 0 appears twice — the second (later) entry should win
        summaries = [
            _worker_summary(0, 30),
            _worker_summary(0, 80),  # more recent
        ]
        stats = self.rp._compute_cluster_stats(summaries)
        assert stats["worker_utils"][0] == pytest.approx(80.0)

    def test_n_summaries_received(self):
        summaries = [_worker_summary(i, 70) for i in range(3)]
        stats = self.rp._compute_cluster_stats(summaries)
        assert stats["n_summaries_received"] == 3


# ── aggregator actor ──────────────────────────────────────────────────────────

class TestAggregatorActor:
    def test_push_and_get_all(self):
        actor = _AggregatorActor(num_workers=2)
        actor.push({"worker_id": 0, "gpu_summary": {}})
        actor.push({"worker_id": 1, "gpu_summary": {}})
        summaries = actor.get_all()
        assert len(summaries) == 2

    def test_reset_clears_summaries(self):
        actor = _AggregatorActor(num_workers=2)
        actor.push({"worker_id": 0})
        actor.reset()
        assert actor.get_all() == []


# ── analyzer cluster rules ────────────────────────────────────────────────────

class TestAnalyzerClusterRules:
    def _analyze(self, cluster):
        return BottleneckAnalyzer(cluster=cluster).analyze()

    def test_cluster_imbalance_fires(self):
        result = self._analyze({
            "imbalance_pct":       35.0,
            "bottleneck_worker_id": 2,
            "bottleneck_avg_util": 40.0,
        })
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_IMBALANCE" in types

    def test_cluster_imbalance_below_threshold(self):
        result = self._analyze({
            "imbalance_pct":       10.0,
            "bottleneck_worker_id": 0,
            "bottleneck_avg_util": 72.0,
        })
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_IMBALANCE" not in types

    def test_cluster_imbalance_severity_high(self):
        result = self._analyze({
            "imbalance_pct": 50.0, "bottleneck_worker_id": 1, "bottleneck_avg_util": 30.0
        })
        f = next(x for x in result["findings"] if x["type"] == "CLUSTER_IMBALANCE")
        assert f["severity"] == "HIGH"

    def test_cluster_low_util_fires(self):
        result = self._analyze({
            "cluster_avg_gpu_util": 25.0,
            "worker_utils": {0: 25.0, 1: 25.0},
        })
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_LOW_GPU_UTIL" in types

    def test_cluster_low_util_above_threshold(self):
        result = self._analyze({"cluster_avg_gpu_util": 70.0, "worker_utils": {0: 70.0}})
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_LOW_GPU_UTIL" not in types

    def test_cluster_low_util_severity_medium(self):
        result = self._analyze({"cluster_avg_gpu_util": 20.0, "worker_utils": {}})
        f = next(x for x in result["findings"] if x["type"] == "CLUSTER_LOW_GPU_UTIL")
        assert f["severity"] == "MEDIUM"

    def test_cluster_mem_pressure_fires(self):
        result = self._analyze({"pressured_workers": [0, 2]})
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_MEM_PRESSURE" in types

    def test_cluster_mem_pressure_empty_list(self):
        result = self._analyze({"pressured_workers": []})
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_MEM_PRESSURE" not in types

    def test_empty_cluster_no_false_positives(self):
        result = self._analyze({})
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_IMBALANCE"    not in types
        assert "CLUSTER_LOW_GPU_UTIL" not in types
        assert "CLUSTER_MEM_PRESSURE" not in types

    def test_none_cluster_no_false_positives(self):
        result = BottleneckAnalyzer().analyze()
        types = [f["type"] for f in result["findings"]]
        assert "CLUSTER_IMBALANCE"    not in types
        assert "CLUSTER_LOW_GPU_UTIL" not in types
        assert "CLUSTER_MEM_PRESSURE" not in types


# ── graceful degradation (no Ray) ─────────────────────────────────────────────

class TestRayProfilerDegradation:
    def test_no_ray_returns_empty_report(self):
        with patch("torchscope.ray_profiler._RAY", False):
            rp = RayProfiler(num_workers=2)
            rp._enabled = False
            result = rp.aggregate_report()
            assert result == {}

    def test_import_succeeds_without_ray(self):
        # Module-level import should always succeed
        from torchscope.ray_profiler import RayProfiler as RP
        assert RP is not None

    def test_compute_cluster_stats_pure_function(self):
        # Should work regardless of _RAY flag
        rp = RayProfiler(num_workers=2)
        stats = rp._compute_cluster_stats([_worker_summary(0, 60), _worker_summary(1, 80)])
        assert "cluster_avg_gpu_util" in stats
