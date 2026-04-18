"""
torchscope Kubernetes sidecar.

Standalone process — no torch required. Polls GPU via pynvml, exposes
Prometheus metrics on a single HTTP port (/metrics + /healthz).

Configuration via environment variables:
    TORCHSCOPE_PORT      (default 9000)  — HTTP port for metrics + healthz
    TORCHSCOPE_INTERVAL  (default 0.5)   — GPU polling cadence in seconds
    TORCHSCOPE_JOB_NAME  (default "torchscope")
    TORCHSCOPE_RANK      (default 0)     — pod rank label in Prometheus metrics
    TORCHSCOPE_GPU_IDS   (default "")    — comma-separated GPU indices, e.g. "0,1,2"

Usage:
    python -m torchscope.k8s.sidecar

Kubernetes liveness probe:
    livenessProbe:
      httpGet:
        path: /healthz
        port: 9000
"""

from __future__ import annotations

import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

# ── config from environment ───────────────────────────────────────────────────

PORT      = int(os.environ.get("TORCHSCOPE_PORT",     "9000"))
INTERVAL  = float(os.environ.get("TORCHSCOPE_INTERVAL", "0.5"))
JOB_NAME  = os.environ.get("TORCHSCOPE_JOB_NAME",    "torchscope")
RANK      = int(os.environ.get("TORCHSCOPE_RANK",     "0"))
_GPU_IDS_STR = os.environ.get("TORCHSCOPE_GPU_IDS", "")
GPU_IDS: Optional[list[int]] = (
    [int(x) for x in _GPU_IDS_STR.split(",") if x.strip()]
    if _GPU_IDS_STR.strip() else None
)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):
    """Serves /healthz (200 ok) and /metrics (Prometheus text) on the same port."""

    sidecar: "Sidecar" = None   # set by Sidecar before starting the server

    def do_GET(self):
        if self.path == "/healthz":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/metrics":
            try:
                from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
                body = generate_latest()
                self.send_response(200)
                self.send_header("Content-Type", CONTENT_TYPE_LATEST)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception as exc:
                body = str(exc).encode()
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_):
        pass  # suppress default access log


# ── sidecar ───────────────────────────────────────────────────────────────────

class Sidecar:
    """
    Orchestrates GPUCollector + PrometheusExporter without requiring torch.
    Designed to run as a Kubernetes sidecar container alongside a training pod.
    """

    def __init__(
        self,
        port:     int            = PORT,
        interval: float          = INTERVAL,
        job_name: str            = JOB_NAME,
        rank:     int            = RANK,
        gpu_ids:  Optional[list] = GPU_IDS,
    ) -> None:
        self._port     = port
        self._interval = interval
        self._job_name = job_name
        self._rank     = rank
        self._gpu_ids  = gpu_ids

        self._gpu      = None
        self._prom_exp = None
        self._up       = None
        self._server:  Optional[HTTPServer]    = None
        self._srv_thread: Optional[threading.Thread] = None
        self._running  = False

    def start(self) -> None:
        from ..collectors.gpu import GPUCollector

        self._gpu = GPUCollector(interval=self._interval, gpu_ids=self._gpu_ids)

        # Set up Prometheus metrics
        try:
            from ..exporters.prometheus import PrometheusExporter
            from prometheus_client import Gauge

            self._prom_exp = PrometheusExporter(
                job_name=self._job_name, port=self._port, rank=self._rank
            )
            # torchscope_up gauge — Grafana can alert on missing scrape
            self._up = Gauge(
                "torchscope_up",
                "1 if torchscope sidecar is running",
                ["job", "rank"],
            )
            self._up.labels(job=self._job_name, rank=str(self._rank)).set(1.0)

            # Wire live GPU samples to Prometheus
            self._gpu.register_callback(self._prom_exp.update_gpu_sample)

        except ImportError:
            print("torchscope-sidecar: prometheus-client not installed; "
                  "metrics endpoint will be unavailable", file=sys.stderr)

        # Start GPU polling
        self._gpu.start()
        self._running = True

        # Start HTTP server in a daemon thread
        _HealthHandler.sidecar = self
        self._server = HTTPServer(("", self._port), _HealthHandler)
        self._srv_thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="torchscope.sidecar.http",
        )
        self._srv_thread.start()
        print(f"torchscope-sidecar: listening on :{self._port}  "
              f"[job={self._job_name} rank={self._rank}]", file=sys.stderr)

    def stop(self) -> None:
        self._running = False
        if self._gpu:
            self._gpu.stop()
        if self._server:
            self._server.shutdown()
        if self._srv_thread:
            self._srv_thread.join(timeout=3)


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    sidecar = Sidecar()
    sidecar.start()

    def _shutdown(sig, frame):
        print("torchscope-sidecar: shutting down", file=sys.stderr)
        sidecar.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
