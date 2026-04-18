"""
PrometheusExporter — exposes torchscope metrics as Prometheus gauges.

Suitable for cluster-level GPU observability. Metrics are tagged with
job_name, rank, and gpu_id so they can be aggregated across nodes in
Grafana dashboards.

Requirements:
    pip install prometheus-client

Usage:
    exporter = PrometheusExporter(job_name="training_run_42", port=9000)
    exporter.start()          # starts HTTP server in background thread

    # push metrics from a running GPUCollector
    exporter.update_gpu(gpu_collector.samples[-1])
    exporter.update_memory(memory_collector.summary())
    exporter.update_comm(comm_collector.summary())

    # or push a full snapshot at the end
    exporter.push_summary(gpu_summary, mem_summary, comm_summary)

Metrics exposed (all prefixed torchscope_):
    gpu_utilization_pct         gauge  per gpu_id
    gpu_memory_used_gb          gauge  per gpu_id
    gpu_memory_total_gb         gauge  per gpu_id
    gpu_temperature_c           gauge  per gpu_id
    gpu_power_w                 gauge  per gpu_id
    gpu_sm_clock_mhz            gauge  per gpu_id
    gpu_pcie_tx_gbs             gauge  per gpu_id
    memory_fragmentation_pct    gauge
    memory_alloc_retries_total  counter
    memory_oom_total            counter
    comm_step_ms                gauge
    comm_overhead_pct           gauge
    straggler_rank              gauge  (1 if straggler, 0 otherwise)
"""

from __future__ import annotations

import threading
from typing import Optional

try:
    from prometheus_client import (
        Gauge, Counter, start_http_server, REGISTRY,
        CollectorRegistry,
    )
    _PROM = True
except ImportError:
    _PROM = False


class PrometheusExporter:
    def __init__(
        self,
        job_name: str = "torchscope",
        port:     int = 9000,
        rank:     int = 0,
    ):
        if not _PROM:
            raise ImportError("pip install prometheus-client")

        self.job_name = job_name
        self.port     = port
        self.rank     = rank
        self._labels  = {"job": job_name, "rank": str(rank)}
        self._started = False

        # GPU metrics
        self._gpu_util  = Gauge("torchscope_gpu_utilization_pct",
                                "SM utilization %", ["job", "rank", "gpu_id"])
        self._gpu_mem   = Gauge("torchscope_gpu_memory_used_gb",
                                "GPU memory allocated GB", ["job", "rank", "gpu_id"])
        self._gpu_mem_t = Gauge("torchscope_gpu_memory_total_gb",
                                "GPU total memory GB", ["job", "rank", "gpu_id"])
        self._gpu_temp  = Gauge("torchscope_gpu_temperature_c",
                                "GPU temperature celsius", ["job", "rank", "gpu_id"])
        self._gpu_pwr   = Gauge("torchscope_gpu_power_w",
                                "GPU power draw watts", ["job", "rank", "gpu_id"])
        self._gpu_clk   = Gauge("torchscope_gpu_sm_clock_mhz",
                                "SM clock MHz", ["job", "rank", "gpu_id"])
        self._pcie_tx   = Gauge("torchscope_gpu_pcie_tx_gbs",
                                "PCIe TX GB/s", ["job", "rank", "gpu_id"])

        # Memory metrics
        self._frag      = Gauge("torchscope_memory_fragmentation_pct",
                                "Memory fragmentation %", ["job", "rank"])
        self._retries   = Counter("torchscope_memory_alloc_retries_total",
                                  "Allocation retries", ["job", "rank"])
        self._ooms      = Counter("torchscope_memory_oom_total",
                                  "OOM events", ["job", "rank"])

        # Communication metrics
        self._step_ms   = Gauge("torchscope_comm_step_ms",
                                "Avg training step ms", ["job", "rank"])
        self._comm_pct  = Gauge("torchscope_comm_overhead_pct",
                                "Comm % of step time", ["job", "rank"])
        self._straggler = Gauge("torchscope_straggler_rank",
                                "1 if this rank is a straggler", ["job", "rank"])

    def start(self) -> "PrometheusExporter":
        """Start the Prometheus HTTP server on self.port."""
        if not self._started:
            start_http_server(self.port)
            self._started = True
            print(f"torchscope: Prometheus metrics at http://0.0.0.0:{self.port}/metrics")
        return self

    # ── update methods ────────────────────────────────────────────────────────

    def update_gpu_sample(self, sample: dict) -> None:
        """Push a single raw sample from GPUCollector._poll."""
        gid = str(sample.get("gpu_id", 0))
        lbl = {**self._labels, "gpu_id": gid}
        if sample.get("gpu_util",    -1) >= 0: self._gpu_util.labels(**lbl).set(sample["gpu_util"])
        if sample.get("mem_used_gb", -1) >= 0: self._gpu_mem.labels(**lbl).set(sample["mem_used_gb"])
        if sample.get("mem_total_gb",-1) >= 0: self._gpu_mem_t.labels(**lbl).set(sample["mem_total_gb"])
        if sample.get("temp_c",      -1) >= 0: self._gpu_temp.labels(**lbl).set(sample["temp_c"])
        if sample.get("power_w",     -1) >= 0: self._gpu_pwr.labels(**lbl).set(sample["power_w"])
        if sample.get("sm_clock_mhz",-1) >= 0: self._gpu_clk.labels(**lbl).set(sample["sm_clock_mhz"])
        if sample.get("pcie_tx_gbs", -1) >= 0: self._pcie_tx.labels(**lbl).set(sample["pcie_tx_gbs"])

    def update_memory(self, mem_summary: dict) -> None:
        lbl = self._labels
        frag = mem_summary.get("avg_fragmentation_pct", 0)
        self._frag.labels(**lbl).set(frag)

    def update_comm(self, comm_summary: dict) -> None:
        lbl = self._labels
        if "avg_step_ms" in comm_summary:
            self._step_ms.labels(**lbl).set(comm_summary["avg_step_ms"])
        if "comm_pct" in comm_summary:
            self._comm_pct.labels(**lbl).set(comm_summary["comm_pct"])
        is_straggler = any(
            s["rank"] == self.rank
            for s in comm_summary.get("stragglers", [])
        )
        self._straggler.labels(**lbl).set(1 if is_straggler else 0)

    def push_summary(
        self,
        gpu_summary:  dict,
        mem_summary:  dict = None,
        comm_summary: dict = None,
    ) -> None:
        """Push end-of-run summaries to Prometheus."""
        for gid in gpu_summary.get("gpu_ids", [0]):
            lbl = {**self._labels, "gpu_id": str(gid)}
            if gpu_summary.get("avg_gpu_util", -1) >= 0:
                self._gpu_util.labels(**lbl).set(gpu_summary["avg_gpu_util"])
            if gpu_summary.get("peak_mem_gb", -1) >= 0:
                self._gpu_mem.labels(**lbl).set(gpu_summary["peak_mem_gb"])
            if gpu_summary.get("avg_temp_c", -1) >= 0:
                self._gpu_temp.labels(**lbl).set(gpu_summary["avg_temp_c"])
            if gpu_summary.get("avg_power_w", -1) >= 0:
                self._gpu_pwr.labels(**lbl).set(gpu_summary["avg_power_w"])

        if mem_summary:
            self.update_memory(mem_summary)
        if comm_summary:
            self.update_comm(comm_summary)
