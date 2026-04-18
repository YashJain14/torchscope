"""
CPU tests — run anywhere, no GPU required.

Tests:
  - BottleneckAnalyzer: every rule fires correctly with synthetic inputs
  - MemoryCollector: leak detection heuristic
  - CommCollector: compute/comm ratio math, straggler detection
  - KernelTracer: transfer_overhead math on synthetic events
  - HTMLExporter: renders without error, produces valid HTML
  - JSONExporter: writes correct NDJSON records
  - CLI analyze: end-to-end from CSV → report
  - Profiler: context manager lifecycle on CPU

Run:
    pytest tests/test_cpu.py -v
"""

import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import pytest

# make torchscope importable when running from the torchscope/ directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ══════════════════════════════════════════════════════════════════════════════
# BottleneckAnalyzer
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.analyzer import BottleneckAnalyzer


class TestAnalyzerRules:

    def _analyze(self, **kwargs) -> dict:
        return BottleneckAnalyzer(**kwargs).analyze()

    # ── GPU utilisation ───────────────────────────────────────────────────────

    def test_low_gpu_util_fires(self):
        result = self._analyze(gpu={"avg_gpu_util": 20, "pct_time_idle": 5})
        types = [f["type"] for f in result["findings"]]
        assert "LOW_GPU_UTIL" in types

    def test_low_gpu_util_does_not_fire_above_threshold(self):
        result = self._analyze(gpu={"avg_gpu_util": 75, "pct_time_idle": 5})
        types = [f["type"] for f in result["findings"]]
        assert "LOW_GPU_UTIL" not in types

    def test_gpu_idle_time_fires(self):
        result = self._analyze(gpu={"avg_gpu_util": 60, "pct_time_idle": 35})
        types = [f["type"] for f in result["findings"]]
        assert "GPU_IDLE_TIME" in types

    def test_gpu_util_skipped_when_unavailable(self):
        result = self._analyze(gpu={"avg_gpu_util": -1})
        types = [f["type"] for f in result["findings"]]
        assert "LOW_GPU_UTIL" not in types

    # ── memory ────────────────────────────────────────────────────────────────

    def test_memory_underused_fires(self):
        result = self._analyze(memory={
            "memory_used_pct": 20, "headroom_gb": 30,
            "peak_alloc_gb": 2, "total_gpu_gb": 40,
        })
        types = [f["type"] for f in result["findings"]]
        assert "MEMORY_UNDERUSED" in types

    def test_memory_underused_skipped_when_full(self):
        result = self._analyze(memory={
            "memory_used_pct": 80, "headroom_gb": 8,
            "peak_alloc_gb": 32, "total_gpu_gb": 40,
        })
        types = [f["type"] for f in result["findings"]]
        assert "MEMORY_UNDERUSED" not in types

    def test_memory_fragmentation_fires(self):
        result = self._analyze(memory={"avg_fragmentation_pct": 45})
        types = [f["type"] for f in result["findings"]]
        assert "MEMORY_FRAGMENTATION" in types

    def test_oom_fires(self):
        result = self._analyze(memory={"oom_count": 2})
        types = [f["type"] for f in result["findings"]]
        assert "OUT_OF_MEMORY" in types
        assert result["findings"][0]["severity"] == "HIGH"

    def test_alloc_retries_fires(self):
        result = self._analyze(memory={"alloc_retries": 5})
        types = [f["type"] for f in result["findings"]]
        assert "ALLOC_RETRIES" in types

    def test_memory_leak_fires(self):
        result = self._analyze(memory={"leak_detected": True, "leak_ratio_pct": 90})
        types = [f["type"] for f in result["findings"]]
        assert "MEMORY_LEAK" in types

    # ── transfer overhead ─────────────────────────────────────────────────────

    def test_transfer_overhead_fires(self):
        result = self._analyze(tracer={
            "transfer_overhead": {"transfer_pct": 25, "transfer_ms": 100}
        })
        types = [f["type"] for f in result["findings"]]
        assert "HIGH_TRANSFER_OVERHEAD" in types

    def test_transfer_overhead_skipped_below_threshold(self):
        result = self._analyze(tracer={
            "transfer_overhead": {"transfer_pct": 5, "transfer_ms": 10}
        })
        types = [f["type"] for f in result["findings"]]
        assert "HIGH_TRANSFER_OVERHEAD" not in types

    # ── dominant kernel ───────────────────────────────────────────────────────

    def test_dominant_kernel_fires(self):
        result = self._analyze(tracer={"top_kernels": [
            {"name": "aten::mm",  "cuda_time_ms": 80, "calls": 10},
            {"name": "aten::relu","cuda_time_ms": 20, "calls": 10},
        ]})
        types = [f["type"] for f in result["findings"]]
        assert "DOMINANT_KERNEL" in types

    def test_dominant_kernel_attention_recommendation(self):
        result = self._analyze(tracer={"top_kernels": [
            {"name": "aten::scaled_dot_product_attention",
             "cuda_time_ms": 80, "calls": 10},
            {"name": "aten::relu", "cuda_time_ms": 20, "calls": 10},
        ]})
        assert any("Flash Attention" in r for r in result["recommendations"])

    def test_dominant_kernel_skipped_below_threshold(self):
        result = self._analyze(tracer={"top_kernels": [
            {"name": "aten::mm",   "cuda_time_ms": 35, "calls": 10},
            {"name": "aten::relu", "cuda_time_ms": 65, "calls": 10},
        ]})
        types = [f["type"] for f in result["findings"]]
        assert "DOMINANT_KERNEL" not in types

    # ── thermal ───────────────────────────────────────────────────────────────

    def test_thermal_throttle_fires(self):
        result = self._analyze(gpu={"thermal_throttle_pct": 50, "max_temp_c": 88})
        types = [f["type"] for f in result["findings"]]
        assert "THERMAL_THROTTLING" in types

    # ── comm ──────────────────────────────────────────────────────────────────

    def test_high_comm_overhead_fires(self):
        result = self._analyze(comm={
            "comm_pct": 30, "avg_comm_ms": 60, "avg_step_ms": 200
        })
        types = [f["type"] for f in result["findings"]]
        assert "HIGH_COMM_OVERHEAD" in types

    def test_straggler_fires(self):
        result = self._analyze(comm={
            "stragglers": [{"rank": 2, "median_step_ms": 150, "slowdown_pct": 25}]
        })
        types = [f["type"] for f in result["findings"]]
        assert "STRAGGLER_GPU" in types

    # ── custom stages ─────────────────────────────────────────────────────────

    def test_stage_bottleneck_fires(self):
        result = self._analyze(custom_stages={
            "decode": 3.0, "preprocess": 0.1, "inference": 0.5
        })
        types = [f["type"] for f in result["findings"]]
        assert any("STAGE_BOTTLENECK" in t for t in types)

    def test_stage_bottleneck_correct_stage(self):
        result = self._analyze(custom_stages={
            "decode": 3.0, "preprocess": 0.1, "inference": 0.5
        })
        bottleneck = result["summary"]["primary_bottleneck"]
        assert "DECODE" in bottleneck

    def test_stage_bottleneck_no_false_positive(self):
        result = self._analyze(custom_stages={
            "a": 1.0, "b": 1.0, "c": 1.0  # evenly split — no bottleneck
        })
        types = [f["type"] for f in result["findings"]]
        assert not any("STAGE_BOTTLENECK" in t for t in types)

    # ── summary structure ─────────────────────────────────────────────────────

    def test_summary_keys_present(self):
        result = self._analyze()
        s = result["summary"]
        assert "total_findings"     in s
        assert "high_severity"      in s
        assert "medium_severity"    in s
        assert "low_severity"       in s
        assert "primary_bottleneck" in s

    def test_no_false_positives_on_empty_input(self):
        result = self._analyze()
        assert result["summary"]["total_findings"] == 0
        assert result["summary"]["primary_bottleneck"] == "NONE"

    def test_findings_sorted_high_first(self):
        result = self._analyze(
            gpu={"avg_gpu_util": 10},       # HIGH
            memory={"avg_fragmentation_pct": 50},  # MEDIUM
        )
        sevs = [f["severity"] for f in result["findings"]]
        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        assert sevs == sorted(sevs, key=lambda s: order[s])


# ══════════════════════════════════════════════════════════════════════════════
# MemoryCollector
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.collectors.memory import MemoryCollector


class TestMemoryCollector:

    def test_detect_leak_monotone(self):
        col = MemoryCollector()
        col.snapshots = [
            {"ts": i, "allocated_gb": 1.0 + i * 0.01,
             "reserved_gb": 2.0, "fragmentation_pct": 50,
             "alloc_retries": 0, "oom_count": 0,
             "active_segments": 0, "active_blocks": 0,
             "peak_alloc_gb": 1.0 + i * 0.01}
            for i in range(25)
        ]
        leak, ratio = col.detect_leak(window=20)
        assert leak is True
        assert ratio > 80

    def test_detect_leak_stable(self):
        import random
        rng = random.Random(42)
        col = MemoryCollector()
        col.snapshots = [
            {"ts": i, "allocated_gb": 1.0 + rng.uniform(-0.01, 0.01),
             "reserved_gb": 2.0, "fragmentation_pct": 50,
             "alloc_retries": 0, "oom_count": 0,
             "active_segments": 0, "active_blocks": 0,
             "peak_alloc_gb": 1.0}
            for i in range(25)
        ]
        leak, _ = col.detect_leak(window=20)
        assert leak is False

    def test_detect_leak_insufficient_samples(self):
        col = MemoryCollector()
        col.snapshots = [
            {"ts": i, "allocated_gb": float(i),
             "reserved_gb": float(i), "fragmentation_pct": 0,
             "alloc_retries": 0, "oom_count": 0,
             "active_segments": 0, "active_blocks": 0,
             "peak_alloc_gb": float(i)}
            for i in range(5)
        ]
        leak, ratio = col.detect_leak(window=20)
        assert leak is False
        assert ratio == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# CommCollector
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.collectors.comm import CommCollector


class TestCommCollector:

    def test_compute_comm_ratio_math(self):
        col = CommCollector()
        col._step_times   = [200.0, 200.0, 200.0]
        col._comm_events  = [
            {"ts": 0, "duration_ms": 40.0, "bytes_mb": 10, "rank": 0},
            {"ts": 0, "duration_ms": 40.0, "bytes_mb": 10, "rank": 0},
            {"ts": 0, "duration_ms": 40.0, "bytes_mb": 10, "rank": 0},
        ]
        r = col.compute_comm_ratio()
        assert r["avg_step_ms"]    == 200.0
        assert r["avg_comm_ms"]    == 40.0
        assert r["avg_compute_ms"] == 160.0
        assert r["comm_pct"]       == 20.0
        assert r["compute_pct"]    == 80.0

    def test_no_data_returns_empty(self):
        col = CommCollector()
        assert col.compute_comm_ratio() == {}

    def test_detect_stragglers_found(self):
        col = CommCollector()
        col._per_rank = {
            0: [100.0] * 10,
            1: [100.0] * 10,
            2: [130.0] * 10,   # 30% slower → straggler
        }
        stragglers = col.detect_stragglers(threshold_pct=10)
        assert len(stragglers) == 1
        assert stragglers[0]["rank"] == 2
        assert stragglers[0]["slowdown_pct"] == pytest.approx(30.0, abs=1)

    def test_detect_stragglers_none(self):
        col = CommCollector()
        col._per_rank = {
            0: [100.0] * 10,
            1: [102.0] * 10,   # within 10% — not a straggler
        }
        assert col.detect_stragglers() == []

    def test_detect_stragglers_needs_at_least_two_ranks(self):
        col = CommCollector()
        col._per_rank = {0: [100.0] * 10}
        assert col.detect_stragglers() == []

    def test_step_context_records_time(self):
        col = CommCollector()
        with col.step():
            time.sleep(0.05)
        assert len(col._step_times) == 1
        assert col._step_times[0] >= 40   # at least 40ms

    def test_summary_structure(self):
        col = CommCollector()
        s = col.summary()
        assert "attached"       in s
        assert "total_comm_ops" in s
        assert "total_bytes_mb" in s
        assert "stragglers"     in s


# ══════════════════════════════════════════════════════════════════════════════
# KernelTracer
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.tracer import KernelTracer


class TestKernelTracer:

    def _make_tracer(self, events: list[dict]) -> KernelTracer:
        t = KernelTracer()
        t._events = events
        return t

    def test_top_kernels_sorted(self):
        t = self._make_tracer([
            {"name": "aten::mm",     "cuda_time_ms": 50, "calls": 5, "flops": 0,
             "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
            {"name": "aten::relu",   "cuda_time_ms": 10, "calls": 5, "flops": 0,
             "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
            {"name": "aten::linear", "cuda_time_ms": 30, "calls": 5, "flops": 0,
             "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
        ])
        tops = t.top_kernels()
        assert tops[0]["name"] == "aten::mm"
        assert tops[1]["name"] == "aten::linear"

    def test_top_kernels_deduplicates(self):
        t = self._make_tracer([
            {"name": "aten::mm", "cuda_time_ms": 30, "calls": 5, "flops": 0,
             "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
            {"name": "aten::mm", "cuda_time_ms": 80, "calls": 5, "flops": 0,
             "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
        ])
        tops = t.top_kernels()
        assert len(tops) == 1
        assert tops[0]["cuda_time_ms"] == 80   # keeps the larger

    def test_top_kernels_empty(self):
        t = self._make_tracer([])
        assert t.top_kernels() == []

    def test_transfer_overhead_math(self):
        t = self._make_tracer([
            {"name": "Memcpy HtoD",  "cuda_time_ms": 20, "calls": 1, "flops": 0,
             "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
            {"name": "aten::conv2d", "cuda_time_ms": 80, "calls": 1, "flops": 0,
             "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
        ])
        ov = t.transfer_overhead()
        assert ov["transfer_pct"]  == pytest.approx(20.0)
        assert ov["transfer_ms"]   == pytest.approx(20.0)
        assert ov["compute_ms"]    == pytest.approx(80.0)
        assert ov["n_xfer_ops"]    == 1

    def test_transfer_overhead_empty(self):
        t = self._make_tracer([])
        assert t.transfer_overhead() == {}

    def test_flop_summary(self):
        t = self._make_tracer([
            {"name": "aten::mm", "cuda_time_ms": 10, "calls": 1,
             "flops": 1_000_000_000, "cuda_mem_mb": 0, "cpu_mem_mb": 0, "cpu_time_ms": 0},
        ])
        fs = t.flop_summary()
        assert fs["total_gflops"] == pytest.approx(1.0)

    def test_profile_block_noop_without_torch_cuda(self):
        t = KernelTracer()
        with t.profile_block():
            pass  # should not raise


# ══════════════════════════════════════════════════════════════════════════════
# HTMLExporter
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.exporters.html import HTMLExporter


class TestHTMLExporter:

    def _minimal_analysis(self) -> dict:
        return {
            "findings": [
                {"type": "LOW_GPU_UTIL", "severity": "HIGH",
                 "detail": "avg util 20%"},
                {"type": "MEMORY_UNDERUSED", "severity": "MEDIUM",
                 "detail": "30 GB free"},
            ],
            "recommendations": ["Increase batch size"],
            "summary": {
                "total_findings": 2, "high_severity": 1,
                "medium_severity": 1, "low_severity": 0,
                "primary_bottleneck": "LOW_GPU_UTIL",
            },
            "_gpu_summary": {
                "avg_gpu_util": "20", "peak_mem_gb": "8.0",
                "avg_power_w": "250", "duration_s": "30",
            },
        }

    def test_generates_html_file(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            HTMLExporter().generate(
                output_path=path, title="Test",
                analysis=self._minimal_analysis(),
            )
            content = Path(path).read_text()
            assert "<!DOCTYPE html>" in content
            assert "torchscope"      in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_findings_appear_in_html(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            HTMLExporter().generate(
                output_path=path, title="Test",
                analysis=self._minimal_analysis(),
            )
            content = Path(path).read_text()
            assert "LOW_GPU_UTIL"   in content
            assert "MEMORY_UNDERUSED" in content
            assert "Increase batch size" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_gpu_util_chart_hex_color_fix(self):
        """Regression: _chart_gpu_util used to produce broken fillcolor from hex."""
        from torchscope.exporters.html import _chart_gpu_util
        samples = [
            {"ts": float(i), "gpu_id": 0, "gpu_util": 60 + i,
             "mem_used_gb": 1.0, "temp_c": 70, "power_w": 200,
             "sm_clock_mhz": 1500, "pcie_tx_gbs": 0.5, "pcie_rx_gbs": 0.5,
             "mem_free_gb": 38.0, "mem_total_gb": 40.0, "mem_pct": 2.5,
             "mem_util": 3}
            for i in range(5)
        ]
        fig_json = _chart_gpu_util(samples)
        assert fig_json is not None
        parsed = json.loads(fig_json)
        # fillcolor must be rgba(...), not contain broken string manipulation
        fill = parsed["data"][0].get("fillcolor", "")
        assert fill.startswith("rgba("), f"Bad fillcolor: {fill!r}"


# ══════════════════════════════════════════════════════════════════════════════
# JSONExporter
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.exporters.json_log import JSONExporter


class TestJSONExporter:

    def test_writes_ndjson(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson",
                                         delete=False) as f:
            path = f.name
        try:
            exp = JSONExporter(job_name="test_job", output_path=path)
            exp.write_gpu_sample({"gpu_id": 0, "gpu_util": 55})
            exp.write_event("epoch_end", epoch=1, loss=0.42)
            exp.write_summary(
                gpu_summary={"avg_gpu_util": 55},
                analysis={
                    "findings": [], "recommendations": [],
                    "summary": {"total_findings": 0, "high_severity": 0,
                                "medium_severity": 0, "low_severity": 0,
                                "primary_bottleneck": "NONE"},
                },
            )
            exp.close()

            lines = Path(path).read_text().strip().split("\n")
            records = [json.loads(l) for l in lines]
            types = [r["record_type"] for r in records]
            assert "gpu_sample"  in types
            assert "event"       in types
            assert "run_summary" in types
        finally:
            Path(path).unlink(missing_ok=True)

    def test_job_name_propagated(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson",
                                         delete=False) as f:
            path = f.name
        try:
            exp = JSONExporter(job_name="my_job", output_path=path)
            exp.write_event("test")
            exp.close()
            record = json.loads(Path(path).read_text().strip())
            assert record["job"] == "my_job"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_context_manager_closes(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".ndjson",
                                         delete=False) as f:
            path = f.name
        try:
            with JSONExporter(output_path=path) as exp:
                exp.write_event("start")
            # file handle should be closed — writing again raises
            assert exp._fh is None
        finally:
            Path(path).unlink(missing_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# CLI analyze
# ══════════════════════════════════════════════════════════════════════════════

class TestCLIAnalyze:

    def _write_csv(self, path: Path):
        path.write_text(
            "pipeline_name,run_id,decode_s,preprocess_s,inference_s,"
            "postprocess_s,total_s,frames_processed,fps,error\n"
            "OpenCV CPU,0,1.75,2.46,3.04,0.01,7.26,1024,141.1,\n"
            "OpenCV CPU,1,1.75,2.47,3.05,0.01,7.28,1024,140.6,\n"
            "PyNvVideoCodec 2.1,0,1.50,0.11,0.91,0.01,2.53,1024,404.7,\n"
            "PyNvVideoCodec 2.1,1,1.50,0.11,0.91,0.01,2.54,1024,403.1,\n"
        )

    def test_analyze_csv_produces_report(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path    = Path(d) / "bench.csv"
            report_path = Path(d) / "report.html"
            self._write_csv(csv_path)

            from torchscope.cli import cmd_analyze

            class _Args:
                inputs = [str(csv_path)]
                report = str(report_path)
                title  = "test"

            cmd_analyze(_Args())
            assert report_path.exists()
            content = report_path.read_text()
            assert "<!DOCTYPE html>" in content

    def test_analyze_detects_stage_bottleneck(self):
        with tempfile.TemporaryDirectory() as d:
            csv_path    = Path(d) / "bench.csv"
            report_path = Path(d) / "report.html"
            self._write_csv(csv_path)

            from torchscope.cli import cmd_analyze
            from torchscope.analyzer import BottleneckAnalyzer
            from torchscope.cli import _load_csv, _stages_from_rows

            rows   = _load_csv(csv_path)
            stages = _stages_from_rows(rows)
            result = BottleneckAnalyzer(custom_stages=stages).analyze()
            types  = [f["type"] for f in result["findings"]]
            assert any("STAGE_BOTTLENECK" in t for t in types)


# ══════════════════════════════════════════════════════════════════════════════
# Profiler — lifecycle (CPU, no torch.cuda)
# ══════════════════════════════════════════════════════════════════════════════

class TestProfilerLifecycle:

    def test_context_manager_starts_and_stops(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from torchscope import Profiler

        with Profiler(interval=0.1) as prof:
            assert prof._active is True
            time.sleep(0.15)

        assert prof._active is False

    def test_report_produces_html(self):
        from torchscope import Profiler
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            with Profiler(interval=0.1) as prof:
                time.sleep(0.1)
            prof.report(output_path=path, title="lifecycle test")
            content = Path(path).read_text()
            assert "<!DOCTYPE html>" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_double_stop_is_safe(self):
        from torchscope import Profiler
        prof = Profiler(interval=0.1)
        prof.start()
        prof.stop()
        prof.stop()   # must not raise

    def test_report_returns_analysis_dict(self):
        from torchscope import Profiler
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            with Profiler(interval=0.1) as prof:
                time.sleep(0.1)
            result = prof.report(output_path=path)
            assert "findings"        in result
            assert "recommendations" in result
            assert "summary"         in result
        finally:
            Path(path).unlink(missing_ok=True)
