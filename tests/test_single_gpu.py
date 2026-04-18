"""
Single-GPU tests — require exactly one CUDA device.

Tests:
  - GPUCollector: actually polls pynvml/torch.cuda, produces valid samples
  - MemoryCollector: captures real allocation changes
  - KernelTracer: profile_block captures real kernel events
  - Profiler: full end-to-end with a real torch workload
  - BottleneckAnalyzer: fires LOW_GPU_UTIL on a deliberately idle workload
  - HTMLExporter: GPU util chart renders with real samples

Run:
    pytest tests/test_single_gpu.py -v

Skip automatically on CPU-only machines.
"""

import sys
import tempfile
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── skip whole module if no CUDA ──────────────────────────────────────────────
torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


# ══════════════════════════════════════════════════════════════════════════════
# GPUCollector
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.collectors.gpu import GPUCollector


class TestGPUCollectorGPU:

    def test_collects_samples(self):
        col = GPUCollector(interval=0.1, gpu_ids=[0])
        col.start()
        # do some GPU work so utilisation is non-zero
        x = torch.randn(2048, 2048, device="cuda")
        for _ in range(50):
            x = torch.mm(x, x.T)
        torch.cuda.synchronize()
        time.sleep(0.4)
        col.stop()
        assert len(col.samples) >= 2

    def test_sample_schema(self):
        col = GPUCollector(interval=0.05, gpu_ids=[0])
        col.start()
        time.sleep(0.2)
        col.stop()
        s = col.samples[0]
        required = ("ts", "gpu_id", "gpu_util", "mem_used_gb",
                    "mem_total_gb", "mem_pct", "temp_c", "power_w")
        for key in required:
            assert key in s, f"Missing key: {key}"

    def test_summary_keys(self):
        col = GPUCollector(interval=0.1, gpu_ids=[0])
        col.start()
        time.sleep(0.3)
        col.stop()
        s = col.summary()
        assert s["n_samples"] >= 2
        assert "avg_gpu_util"  in s
        assert "peak_mem_gb"   in s
        assert "duration_s"    in s

    def test_live_callback_fires(self):
        fired = []
        col = GPUCollector(interval=0.05, gpu_ids=[0])
        col.register_callback(fired.append)
        col.start()
        time.sleep(0.3)
        col.stop()
        assert len(fired) >= 2
        assert fired[0]["gpu_id"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# MemoryCollector
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.collectors.memory import MemoryCollector


class TestMemoryCollectorGPU:

    def test_captures_allocation(self):
        col = MemoryCollector(interval=0.05, device=0)
        col.start()
        tensors = []
        for _ in range(5):
            tensors.append(torch.zeros(256, 256, 256, device="cuda"))
            time.sleep(0.06)
        col.stop()
        del tensors

        allocs = [s["allocated_gb"] for s in col.snapshots]
        assert max(allocs) > min(allocs), "Expected allocation to grow during test"

    def test_efficiency_keys(self):
        col = MemoryCollector(interval=0.1, device=0)
        col.start()
        x = torch.zeros(512, 512, device="cuda")
        time.sleep(0.3)
        col.stop()
        del x
        e = col.efficiency()
        assert "peak_alloc_gb"    in e
        assert "memory_used_pct"  in e
        assert "headroom_gb"      in e
        assert "total_gpu_gb"     in e

    def test_fragmentation_field_present(self):
        col = MemoryCollector(interval=0.1, device=0)
        col.start()
        time.sleep(0.3)
        col.stop()
        assert all("fragmentation_pct" in s for s in col.snapshots)


# ══════════════════════════════════════════════════════════════════════════════
# KernelTracer
# ══════════════════════════════════════════════════════════════════════════════

from torchscope.tracer import KernelTracer


class TestKernelTracerGPU:

    def test_profile_block_captures_kernels(self):
        tracer = KernelTracer()
        x = torch.randn(512, 512, device="cuda")
        with tracer.profile_block():
            y = torch.mm(x, x.T)
            torch.cuda.synchronize()
        assert tracer.has_data(), "Expected kernel events after profile_block"

    def test_top_kernels_non_empty(self):
        tracer = KernelTracer()
        x = torch.randn(512, 512, device="cuda")
        with tracer.profile_block():
            y = torch.mm(x, x.T)
            torch.cuda.synchronize()
        kernels = tracer.top_kernels()
        assert len(kernels) > 0
        assert all("cuda_time_ms" in k for k in kernels)

    def test_transfer_overhead_captures_memcpy(self):
        tracer = KernelTracer()
        x = torch.randn(512, 512)           # CPU tensor
        with tracer.profile_block():
            y = x.cuda()                    # H→D transfer
            torch.cuda.synchronize()
        ov = tracer.transfer_overhead()
        assert ov.get("n_xfer_ops", 0) >= 1, "Expected at least one H2D memcpy"

    def test_profile_loop_with_manual_step(self):
        tracer = KernelTracer(wait=0, warmup=0, active=2)
        x = torch.randn(256, 256, device="cuda")
        with tracer.profile() as kp:
            for _ in range(3):
                y = torch.mm(x, x.T)
                torch.cuda.synchronize()
                kp.step()
        assert tracer.has_data()


# ══════════════════════════════════════════════════════════════════════════════
# Full Profiler — single GPU end-to-end
# ══════════════════════════════════════════════════════════════════════════════

from torchscope import Profiler


class TestProfilerGPU:

    def test_full_run_produces_report(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            with Profiler(interval=0.1, device=0) as prof:
                x = torch.randn(1024, 1024, device="cuda")
                for _ in range(20):
                    x = torch.mm(x, x.T)
                torch.cuda.synchronize()

            result = prof.report(output_path=path, title="single-GPU test")
            assert Path(path).exists()
            content = Path(path).read_text()
            assert "<!DOCTYPE html>" in content
            assert "findings"        in result
        finally:
            Path(path).unlink(missing_ok=True)

    def test_gpu_samples_captured(self):
        with Profiler(interval=0.05, device=0) as prof:
            x = torch.randn(1024, 1024, device="cuda")
            for _ in range(30):
                x = torch.mm(x, x.T)
            torch.cuda.synchronize()
            time.sleep(0.2)

        assert len(prof.gpu.samples) >= 2

    def test_memory_snapshots_captured(self):
        with Profiler(interval=0.05, device=0) as prof:
            tensors = [torch.zeros(256, 256, 256, device="cuda") for _ in range(3)]
            time.sleep(0.3)

        assert len(prof.memory.snapshots) >= 2

    def test_kernel_tracing_with_profile_block(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            prof = Profiler(interval=0.1, device=0)
            prof.start()

            x = torch.randn(512, 512, device="cuda")
            with prof.profile_block():
                y = torch.mm(x, x.T)
                torch.cuda.synchronize()

            result = prof.report(output_path=path, title="kernel tracing test")
            kernels = prof.tracer.top_kernels()
            assert len(kernels) > 0, "Expected kernel data from profile_block"
        finally:
            Path(path).unlink(missing_ok=True)

    def test_json_export_live_streaming(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ndjson", delete=False
        ) as f:
            json_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            html_path = f.name
        try:
            import json as _json
            with Profiler(interval=0.05, device=0,
                          export_json=json_path) as prof:
                x = torch.randn(512, 512, device="cuda")
                for _ in range(10):
                    x = torch.mm(x, x.T)
                torch.cuda.synchronize()
                time.sleep(0.3)

            prof.report(output_path=html_path)

            lines = [l for l in Path(json_path).read_text().strip().split("\n") if l]
            records = [_json.loads(l) for l in lines]
            gpu_samples = [r for r in records if r["record_type"] == "gpu_sample"]
            # live streaming: GPU samples should be written before the summary
            summary_idx = next(
                (i for i, r in enumerate(records) if r["record_type"] == "run_summary"),
                len(records),
            )
            assert len(gpu_samples) >= 1
            assert any(records.index(s) < summary_idx for s in gpu_samples)
        finally:
            Path(json_path).unlink(missing_ok=True)
            Path(html_path).unlink(missing_ok=True)

    def test_low_gpu_util_detected_on_idle(self):
        """A deliberately idle workload should trigger LOW_GPU_UTIL."""
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            with Profiler(interval=0.05, device=0) as prof:
                time.sleep(0.5)   # GPU sits idle

            result = prof.report(output_path=path)
            gpu_sum = prof.gpu.summary()
            # Only assert if we actually got util readings (pynvml available)
            if gpu_sum.get("avg_gpu_util", -1) >= 0:
                types = [f["type"] for f in result["findings"]]
                assert "LOW_GPU_UTIL" in types
        finally:
            Path(path).unlink(missing_ok=True)

    def test_custom_stages_in_report(self):
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            with Profiler(interval=0.1, device=0) as prof:
                time.sleep(0.1)

            result = prof.report(
                output_path   = path,
                custom_stages = {"decode": 2.0, "inference": 0.5},
                title         = "stage test",
            )
            types = [f["type"] for f in result["findings"]]
            assert any("STAGE_BOTTLENECK" in t for t in types)
        finally:
            Path(path).unlink(missing_ok=True)
