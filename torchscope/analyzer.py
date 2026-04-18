"""
BottleneckAnalyzer — generic rule engine that turns collected telemetry
into prioritised findings and actionable recommendations.

Completely decoupled from any specific workload type. Findings are
data-driven — each check receives only the summary dicts produced by
the collectors and tracer.

Adding a new check: define a method _check_<name>(findings, recs) and
call it from analyze(). No other changes needed.
"""

from __future__ import annotations

# ── thresholds (all tunable) ──────────────────────────────────────────────────

_T = {
    "low_util_pct":         50,    # avg GPU util below this → LOW_GPU_UTIL
    "idle_time_pct":        20,    # pct_time_idle above this → GPU_IDLE_TIME
    "mem_underuse_pct":     50,    # memory_used_pct below this + headroom > 2GB
    "mem_headroom_gb":       2,
    "transfer_pct":         10,    # CPU↔GPU transfer > this % of CUDA time
    "dominant_kernel_pct":  30,    # single kernel > this % of CUDA time
    "thermal_temp_c":       80,    # GPU temp above this
    "thermal_pct":          30,    # % of samples above thermal_temp_c
    "comm_pct":             15,    # communication > this % of step time
    "frag_pct":             30,    # memory fragmentation above this %
}

_SEV = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}

_ATTENTION_OPS = ("attention", "sdpa", "bmm", "baddbmm", "flash_attn", "scaled_dot")
_COMM_OPS      = ("nccl", "all_reduce", "all_gather", "broadcast", "reduce_scatter")


class BottleneckAnalyzer:
    """
    Stateless rule engine.

    All inputs are plain dicts — the summaries produced by collectors
    and the tracer. Pass only what you have; every check guards against
    missing data.

    Args:
        gpu:        GPUCollector.summary()
        memory:     MemoryCollector.summary()
        comm:       CommCollector.summary()          (optional)
        tracer:     {"top_kernels": [...], "transfer_overhead": {...}, "flop_summary": {...}}
        custom_stages: arbitrary {stage_name: seconds} dict for pipeline-level analysis
    """

    def __init__(
        self,
        gpu:           dict = None,
        memory:        dict = None,
        comm:          dict = None,
        tracer:        dict = None,
        custom_stages: dict = None,
    ):
        self.gpu    = gpu    or {}
        self.mem    = memory or {}
        self.comm   = comm   or {}
        self.tracer = tracer or {}
        self.stages = custom_stages or {}

    # ── public ────────────────────────────────────────────────────────────────

    def analyze(self) -> dict:
        findings = []
        recs     = []

        self._check_gpu_util(findings, recs)
        self._check_gpu_idle_time(findings, recs)
        self._check_memory_underuse(findings, recs)
        self._check_memory_fragmentation(findings, recs)
        self._check_memory_errors(findings, recs)
        self._check_memory_leak(findings, recs)
        self._check_transfer_overhead(findings, recs)
        self._check_dominant_kernel(findings, recs)
        self._check_thermal(findings, recs)
        self._check_comm_overhead(findings, recs)
        self._check_stragglers(findings, recs)
        self._check_custom_stages(findings, recs)

        findings.sort(key=lambda f: _SEV.get(f["severity"], 9))

        high = sum(1 for f in findings if f["severity"] == "HIGH")
        med  = sum(1 for f in findings if f["severity"] == "MEDIUM")
        low  = sum(1 for f in findings if f["severity"] == "LOW")
        return {
            "findings":        findings,
            "recommendations": recs,
            "summary": {
                "total_findings":     len(findings),
                "high_severity":      high,
                "medium_severity":    med,
                "low_severity":       low,
                "primary_bottleneck": findings[0]["type"] if findings else "NONE",
            },
        }

    # ── GPU utilisation ───────────────────────────────────────────────────────

    def _check_gpu_util(self, F, R):
        avg = self.gpu.get("avg_gpu_util", -1)
        if avg < 0:
            return
        if avg < _T["low_util_pct"]:
            F.append(_f("LOW_GPU_UTIL", "HIGH",
                f"Average GPU utilization {avg}% — well below the 50% threshold."))
            R.append(
                f"GPU util is {avg}%. Common causes: CPU-bound preprocessing, "
                f"small batch sizes, synchronous data loading. "
                f"Try: larger batches, async DataLoader workers, or GPU-native preprocessing."
            )

    def _check_gpu_idle_time(self, F, R):
        idle = self.gpu.get("pct_time_idle", -1)
        if idle < 0:
            return
        if idle > _T["idle_time_pct"]:
            F.append(_f("GPU_IDLE_TIME", "MEDIUM",
                f"GPU was idle (<10% util) for {idle}% of the profiling window."))
            R.append(
                f"GPU spent {idle}% of its time idle. "
                f"Overlap CPU work with GPU execution: "
                f"use non_blocking=True for .cuda() transfers, "
                f"pin_memory=True in DataLoader, or pipeline decode with inference."
            )

    # ── memory ────────────────────────────────────────────────────────────────

    def _check_memory_underuse(self, F, R):
        used_pct  = self.mem.get("memory_used_pct", 101)
        headroom  = self.mem.get("headroom_gb", 0)
        peak_gb   = self.mem.get("peak_alloc_gb", 0)
        total_gb  = self.mem.get("total_gpu_gb", 1)
        if used_pct < _T["mem_underuse_pct"] and headroom > _T["mem_headroom_gb"]:
            F.append(_f("MEMORY_UNDERUSED", "MEDIUM",
                f"Peak allocation {peak_gb:.1f} GB / {total_gb:.1f} GB ({used_pct}%). "
                f"{headroom:.1f} GB unused."))
            R.append(
                f"{headroom:.0f} GB GPU memory is free. "
                f"Increase batch size to improve arithmetic intensity and throughput."
            )

    def _check_memory_fragmentation(self, F, R):
        frag = self.mem.get("avg_fragmentation_pct", 0)
        if frag > _T["frag_pct"]:
            F.append(_f("MEMORY_FRAGMENTATION", "MEDIUM",
                f"Average memory fragmentation {frag}% — "
                f"reserved but not allocated memory is wasted."))
            R.append(
                f"Memory fragmentation at {frag}%. "
                f"Call torch.cuda.empty_cache() between pipeline stages, "
                f"or set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True."
            )

    def _check_memory_errors(self, F, R):
        retries = self.mem.get("alloc_retries", 0)
        ooms    = self.mem.get("oom_count", 0)
        if ooms > 0:
            F.append(_f("OUT_OF_MEMORY", "HIGH",
                f"{ooms} OOM event(s) detected during profiling."))
            R.append("OOM detected. Reduce batch size, enable gradient checkpointing, "
                     "or use mixed-precision (fp16/bf16).")
        if retries > 0:
            F.append(_f("ALLOC_RETRIES", "MEDIUM",
                f"PyTorch retried memory allocation {retries} times."))
            R.append("Allocation retries indicate fragmentation. "
                     "Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.")

    def _check_memory_leak(self, F, R):
        # Provided as a pre-computed flag by MemoryCollector.detect_leak()
        if self.mem.get("leak_detected"):
            ratio = self.mem.get("leak_ratio_pct", "?")
            F.append(_f("MEMORY_LEAK", "HIGH",
                f"Allocated memory monotonically increasing "
                f"({ratio}% of recent samples trending up)."))
            R.append("Memory leak suspected. Profile with torch.cuda.memory_snapshot() "
                     "or use memray to identify the allocation site.")

    # ── data transfers ────────────────────────────────────────────────────────

    def _check_transfer_overhead(self, F, R):
        t   = self.tracer.get("transfer_overhead", {})
        pct = t.get("transfer_pct", 0)
        if pct > _T["transfer_pct"]:
            F.append(_f("HIGH_TRANSFER_OVERHEAD", "HIGH",
                f"CPU↔GPU transfers consume {pct}% of total CUDA time "
                f"({t.get('transfer_ms', 0):.1f} ms)."))
            R.append(
                f"Transfer overhead is {pct}%. "
                f"Use pin_memory=True + non_blocking=True for DataLoader transfers. "
                f"Pre-load tensors to GPU, or move preprocessing into CUDA with DALI or CV-CUDA."
            )

    # ── kernel analysis ───────────────────────────────────────────────────────

    def _check_dominant_kernel(self, F, R):
        kernels = self.tracer.get("top_kernels", [])
        if not kernels:
            return
        total = sum(k["cuda_time_ms"] for k in kernels)
        if total == 0:
            return
        top     = kernels[0]
        top_pct = top["cuda_time_ms"] / total * 100
        if top_pct < _T["dominant_kernel_pct"]:
            return

        F.append(_f("DOMINANT_KERNEL", "HIGH",
            f"Kernel '{top['name']}' consumes {top_pct:.1f}% of total CUDA time "
            f"({top['cuda_time_ms']:.1f} ms, {top['calls']} calls)."))

        name_lower = top["name"].lower()
        if any(kw in name_lower for kw in _ATTENTION_OPS):
            R.append(
                f"Attention kernel dominates ({top_pct:.0f}%). "
                f"Use torch.nn.functional.scaled_dot_product_attention (PyTorch 2.0+) "
                f"or the flash-attn library for fused tiled attention."
            )
        elif any(kw in name_lower for kw in ("conv", "gemm", "matmul", "mm")):
            R.append(
                f"Compute kernel '{top['name']}' dominates. "
                f"Enable TF32 (torch.backends.cuda.matmul.allow_tf32=True), "
                f"use fp16/bf16, or export to TensorRT."
            )
        else:
            R.append(
                f"Kernel '{top['name']}' dominates ({top_pct:.0f}% of CUDA time). "
                f"Consider a fused CUDA implementation or check for redundant launches."
            )

    # ── thermals ──────────────────────────────────────────────────────────────

    def _check_thermal(self, F, R):
        throttle = self.gpu.get("thermal_throttle_pct", 0)
        max_t    = self.gpu.get("max_temp_c", -1)
        if throttle > _T["thermal_pct"]:
            F.append(_f("THERMAL_THROTTLING", "MEDIUM",
                f"GPU above {_T['thermal_temp_c']}°C for {throttle}% of job. "
                f"Max temp: {max_t}°C."))
            R.append("GPU is thermally throttling — SM clock is being reduced. "
                     "Check cooling, reduce power limit, or lower sustained batch size.")

    # ── distributed ───────────────────────────────────────────────────────────

    def _check_comm_overhead(self, F, R):
        comm_pct = self.comm.get("comm_pct", 0)
        avg_comm = self.comm.get("avg_comm_ms", 0)
        avg_step = self.comm.get("avg_step_ms", 0)
        if comm_pct > _T["comm_pct"]:
            F.append(_f("HIGH_COMM_OVERHEAD", "HIGH",
                f"Communication consumes {comm_pct}% of training step time "
                f"({avg_comm:.1f} ms comm / {avg_step:.1f} ms step)."))
            R.append(
                f"DDP communication is {comm_pct}% of step time. "
                f"Try: gradient compression (PowerSGD), larger bucket_cap_mb, "
                f"or FSDP/ZeRO to overlap comm with backward."
            )

    def _check_stragglers(self, F, R):
        for s in self.comm.get("stragglers", []):
            F.append(_f("STRAGGLER_GPU", "MEDIUM",
                f"Rank {s['rank']} is {s['slowdown_pct']:.1f}% slower than median "
                f"({s['median_step_ms']:.1f} ms vs global median)."))
            R.append(
                f"Straggler at rank {s['rank']}. Check: thermal throttling on that node, "
                f"uneven data distribution, or a slow NIC/NVLink port."
            )

    # ── custom pipeline stages ────────────────────────────────────────────────

    def _check_custom_stages(self, F, R):
        """
        Generic stage-level bottleneck detection for any pipeline.
        Caller passes custom_stages = {"stage_name": avg_seconds, ...}
        """
        if not self.stages:
            return
        total = sum(self.stages.values())
        if total == 0:
            return
        worst      = max(self.stages, key=self.stages.__getitem__)
        worst_pct  = self.stages[worst] / total * 100
        worst_ms   = self.stages[worst] * 1e3

        if worst_pct > 35:
            F.append(_f(
                f"STAGE_BOTTLENECK_{worst.upper().replace(' ', '_')}",
                "HIGH" if worst_pct > 50 else "MEDIUM",
                f"Stage '{worst}' is {worst_pct:.1f}% of total pipeline time "
                f"({worst_ms:.1f} ms avg).",
            ))
            R.append(
                f"'{worst}' dominates the pipeline ({worst_pct:.0f}%). "
                f"Profile this stage independently and consider parallelising, "
                f"caching, or moving it to a faster device."
            )


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(type_: str, severity: str, detail: str) -> dict:
    return {"type": type_, "severity": severity, "detail": detail}
