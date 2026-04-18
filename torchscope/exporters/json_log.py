"""
JSONExporter — structured JSON output for log aggregators.

Produces newline-delimited JSON (NDJSON) — one record per GPU sample,
plus a final summary record — compatible with:
  - Grafana Loki  (ship with Promtail or alloy)
  - Datadog Logs  (forward via Datadog Agent)
  - Elasticsearch (Filebeat / direct ingest)
  - CloudWatch Logs, Splunk, etc.

Every record carries a consistent schema with job_name, rank, run_id,
and a record_type field so queries can filter by type.

Usage:
    exporter = JSONExporter(
        job_name="bert_finetune",
        run_id="run_20240415_001",
        output_path="logs/torchscope.ndjson",  # None → stdout
    )

    # stream live GPU samples
    exporter.write_sample(gpu_sample)

    # write end-of-run summary
    exporter.write_summary(gpu_sum, mem_sum, comm_sum, analysis)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional, TextIO


class JSONExporter:
    def __init__(
        self,
        job_name:    str = "torchscope",
        run_id:      str = "",
        rank:        int = 0,
        output_path: Optional[str] = None,
    ):
        self.job_name = job_name
        self.run_id   = run_id or f"run_{int(time.time())}"
        self.rank     = rank
        self._path    = Path(output_path) if output_path else None
        self._fh:     Optional[TextIO] = None

        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "a", encoding="utf-8")

    # ── write helpers ─────────────────────────────────────────────────────────

    def _emit(self, record: dict) -> None:
        record.setdefault("job",      self.job_name)
        record.setdefault("run_id",   self.run_id)
        record.setdefault("rank",     self.rank)
        record.setdefault("emit_ts",  time.time())
        line = json.dumps(record, default=str)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()
        else:
            print(line, file=sys.stdout)

    # ── public API ────────────────────────────────────────────────────────────

    def write_gpu_sample(self, sample: dict) -> None:
        """Write one raw GPU telemetry sample."""
        self._emit({"record_type": "gpu_sample", **sample})

    def write_memory_snapshot(self, snap: dict) -> None:
        """Write one memory snapshot."""
        self._emit({"record_type": "memory_snapshot", **snap})

    def write_comm_event(self, event: dict) -> None:
        """Write one communication event (all-reduce timing)."""
        self._emit({"record_type": "comm_event", **event})

    def write_kernel_trace(self, kernels: list[dict]) -> None:
        """Write top-kernel trace as a single record."""
        self._emit({"record_type": "kernel_trace", "kernels": kernels})

    def write_finding(self, finding: dict) -> None:
        """Write one bottleneck finding."""
        self._emit({"record_type": "finding", **finding})

    def write_summary(
        self,
        gpu_summary:   dict = None,
        mem_summary:   dict = None,
        comm_summary:  dict = None,
        analysis:      dict = None,
    ) -> None:
        """Write a consolidated end-of-run summary record."""
        record = {"record_type": "run_summary"}
        if gpu_summary:   record["gpu"]      = gpu_summary
        if mem_summary:   record["memory"]   = mem_summary
        if comm_summary:  record["comm"]     = comm_summary
        if analysis:
            record["findings"]        = analysis.get("findings", [])
            record["recommendations"] = analysis.get("recommendations", [])
            record["summary"]         = analysis.get("summary", {})
        self._emit(record)

    def write_event(self, name: str, **kwargs) -> None:
        """Write a free-form named event (e.g. epoch_start, checkpoint_saved)."""
        self._emit({"record_type": "event", "name": name, **kwargs})

    def close(self) -> None:
        if self._fh:
            self._fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
