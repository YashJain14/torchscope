"""
Profiler — the single public entry point for torchscope.

Designed to wrap any PyTorch workload with minimal code change.

── Minimal usage ────────────────────────────────────────────────────────────

    from torchscope import Profiler

    with Profiler() as prof:
        train(model, dataloader)

    prof.report("report.html")

── With all features ────────────────────────────────────────────────────────

    prof = Profiler(
        interval   = 0.5,        # GPU polling cadence (seconds)
        device     = 0,          # CUDA device index
        export_json= "logs/run.ndjson",   # structured log for Loki/Datadog
        export_prom= True,                # start Prometheus endpoint
        prom_port  = 9000,
    )

    prof.start()

    # DDP: attach comm profiler after DDP wrap
    prof.comm.attach(ddp_model)

    for step, batch in enumerate(dataloader):
        with prof.comm.step():
            loss = model(batch)
            loss.backward()
            optimizer.step()

        # kernel tracing (bounded window of steps)
        if step == 0:
            prof.enable_kernel_tracing(wait=1, warmup=1, active=3)

    prof.report("report.html", title="Training run")

── NVTX / Nsight Systems ────────────────────────────────────────────────────

    Annotate code regions that show up in nsys GUI:

        from torchscope.nvtx import annotate

        with annotate("decode"):
            frames = decoder.read(n)

    Then run:
        nsys profile --trace=cuda,nvtx python train.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .collectors.gpu    import GPUCollector
from .collectors.memory import MemoryCollector
from .collectors.comm   import CommCollector
from .tracer            import KernelTracer
from .analyzer          import BottleneckAnalyzer
from .exporters.html    import HTMLExporter
from .exporters.json_log import JSONExporter


class Profiler:
    """
    Orchestrates all collectors, the tracer, the analyzer, and all exporters.

    Attributes exposed for direct use:
        .gpu    — GPUCollector   (hardware telemetry)
        .memory — MemoryCollector (allocation tracking)
        .comm   — CommCollector   (DDP communication hooks)
        .tracer — KernelTracer    (torch.profiler kernel tracing)
    """

    def __init__(
        self,
        interval:    float          = 0.5,
        device:      int            = 0,
        gpu_ids:     Optional[list] = None,
        export_json: Optional[str]  = None,
        export_prom: bool           = False,
        prom_port:   int            = 9000,
        job_name:    str            = "torchscope",
        run_id:      str            = "",
        rank:        int            = 0,
    ):
        self.device   = device
        self.job_name = job_name
        self.rank     = rank

        # collectors
        self.gpu    = GPUCollector(interval=interval, gpu_ids=gpu_ids)
        self.memory = MemoryCollector(interval=interval, device=device)
        self.comm   = CommCollector()
        self.tracer = KernelTracer()

        # exporters
        self._json_exp: Optional[JSONExporter] = (
            JSONExporter(job_name=job_name, run_id=run_id,
                         rank=rank, output_path=export_json)
            if export_json else None
        )
        self._prom_exp = None
        if export_prom:
            try:
                from .exporters.prometheus import PrometheusExporter
                self._prom_exp = PrometheusExporter(
                    job_name=job_name, port=prom_port, rank=rank)
                self._prom_exp.start()
            except ImportError:
                print("torchscope: prometheus-client not installed, "
                      "skipping Prometheus export")

        # wire live streaming: each GPU sample is pushed immediately
        # to Prometheus and/or the JSON log as it arrives
        def _live_push(sample: dict) -> None:
            if self._prom_exp:
                self._prom_exp.update_gpu_sample(sample)
            if self._json_exp:
                self._json_exp.write_gpu_sample(sample)

        if self._prom_exp or self._json_exp:
            self.gpu.register_callback(_live_push)

        self._active   = False
        self._start_ts: Optional[float] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> "Profiler":
        self.gpu.start()
        self.memory.start()
        self._active   = True
        self._start_ts = time.time()
        print(f"torchscope: profiling started  "
              f"[job={self.job_name} device={self.device}]")
        return self

    def stop(self) -> "Profiler":
        if not self._active:
            return self
        self.gpu.stop()
        self.memory.stop()
        self._active = False
        elapsed = time.time() - self._start_ts if self._start_ts else 0
        print(f"torchscope: profiling stopped  "
              f"({elapsed:.1f}s, {len(self.gpu.samples)} GPU samples)")
        return self

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "Profiler":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ── kernel tracing ────────────────────────────────────────────────────────

    def enable_kernel_tracing(
        self, wait: int = 1, warmup: int = 1, active: int = 3
    ) -> "Profiler":
        """Replace the default tracer with one configured for N active steps."""
        self.tracer = KernelTracer(wait=wait, warmup=warmup, active=active)
        return self

    def kernel_profiler(self):
        """
        Return a context manager for use in a training loop.
        Call .step() after each iteration to advance the profiler schedule.

            with prof.kernel_profiler() as kp:
                for batch in loader:
                    model(batch)
                    kp.step()
        """
        return self.tracer.profile()

    def profile_block(self):
        """
        Profile a single self-contained block — no .step() needed.

            with prof.profile_block():
                output = model(batch)
        """
        return self.tracer.profile_block()

    # ── report ────────────────────────────────────────────────────────────────

    def report(
        self,
        output_path:   str  = "torchscope_report.html",
        title:         str  = "PyTorch Workload",
        custom_stages: dict = None,
        metadata:      dict = None,
    ) -> dict:
        """
        Stop profiling (if still running), run the analyzer, and generate
        all configured exports (HTML + JSON + Prometheus).

        custom_stages: optional {stage_name: avg_seconds} dict for pipeline
                       stage breakdown. Example:
                           {"decode": 1.5, "preprocess": 0.12, "inference": 0.9}

        Returns the analysis dict (findings + recommendations + summary).
        """
        self.stop()

        gpu_sum  = self.gpu.summary()
        mem_sum  = self.memory.summary()
        comm_sum = self.comm.summary()

        # enrich memory summary with leak detection
        leak_detected, leak_ratio = self.memory.detect_leak()
        if leak_detected:
            mem_sum["leak_detected"]  = True
            mem_sum["leak_ratio_pct"] = leak_ratio

        tracer_data = {
            "top_kernels":       self.tracer.top_kernels(),
            "transfer_overhead": self.tracer.transfer_overhead(),
            "flop_summary":      self.tracer.flop_summary(),
        }

        analyzer = BottleneckAnalyzer(
            gpu           = gpu_sum,
            memory        = mem_sum,
            comm          = comm_sum,
            tracer        = tracer_data,
            custom_stages = custom_stages or {},
        )
        analysis = analyzer.analyze()
        analysis["_gpu_summary"] = gpu_sum   # forwarded to HTMLExporter for KPIs

        # HTML report
        HTMLExporter().generate(
            output_path   = output_path,
            title         = title,
            analysis      = analysis,
            gpu_samples   = self.gpu.samples,
            mem_snapshots = self.memory.snapshots,
            top_kernels   = tracer_data["top_kernels"],
            custom_stages = custom_stages,
            metadata      = self._build_metadata(title, metadata),
        )

        # JSON export
        if self._json_exp:
            self._json_exp.write_summary(
                gpu_summary  = gpu_sum,
                mem_summary  = mem_sum,
                comm_summary = comm_sum,
                analysis     = analysis,
            )
            self._json_exp.close()

        # Prometheus push
        if self._prom_exp:
            self._prom_exp.push_summary(gpu_sum, mem_sum, comm_sum)

        _print_summary(analysis)
        return analysis

    # ── internal ─────────────────────────────────────────────────────────────

    def _build_metadata(self, title: str, extra: Optional[dict]) -> dict:
        import platform, sys
        from torchscope import __version__
        meta = {
            "torchscope version": __version__,
            "job name":           self.job_name,
            "device":             f"cuda:{self.device}",
            "gpu backend":        self.gpu._backend,
            "python":             sys.version.split()[0],
            "platform":           platform.system(),
        }
        try:
            import torch
            meta["torch"] = torch.__version__
            if torch.cuda.is_available():
                meta["gpu model"] = torch.cuda.get_device_name(self.device)
        except Exception:
            pass
        if extra:
            meta.update(extra)
        return meta


# ── console summary ───────────────────────────────────────────────────────────

def _print_summary(analysis: dict):
    s = analysis["summary"]
    w = 58
    print("\n" + "─" * w)
    print("  torchscope summary")
    print("─" * w)
    print(f"  Findings  {s['total_findings']}  "
          f"(high={s['high_severity']}  medium={s['medium_severity']}  "
          f"low={s['low_severity']})")
    print(f"  Primary   {s['primary_bottleneck']}")
    if analysis["findings"]:
        print()
    for f in analysis["findings"]:
        sev_icon = {"HIGH": "●", "MEDIUM": "◐", "LOW": "○"}.get(f["severity"], "·")
        print(f"  {sev_icon} [{f['severity']:<6}]  {f['type']}")
        print(f"             {f['detail']}")
    if analysis["recommendations"]:
        print()
        print("  Recommendations")
        for r in analysis["recommendations"]:
            print(f"    → {r}")
    print("─" * w)
