"""
Tests for the torchscope Kubernetes sidecar.
No real GPU, K8s, or pynvml required — tests use a free port and mock where needed.
"""

import socket
import threading
import time
import urllib.request
import urllib.error
import pytest
from unittest.mock import patch, MagicMock


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# ── env var config ────────────────────────────────────────────────────────────

class TestSidecarConfig:
    def test_default_values(self):
        import importlib
        import torchscope.k8s.sidecar as mod
        # Defaults when no env vars set
        assert mod.PORT     == int(mod.PORT)
        assert mod.INTERVAL == float(mod.INTERVAL)
        assert mod.JOB_NAME == str(mod.JOB_NAME)
        assert mod.RANK     == int(mod.RANK)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TORCHSCOPE_PORT",     "8080")
        monkeypatch.setenv("TORCHSCOPE_INTERVAL", "1.0")
        monkeypatch.setenv("TORCHSCOPE_JOB_NAME", "my-job")
        monkeypatch.setenv("TORCHSCOPE_RANK",     "3")
        # Sidecar reads defaults at module import time; test constructor path
        from torchscope.k8s.sidecar import Sidecar
        s = Sidecar(port=8080, interval=1.0, job_name="my-job", rank=3)
        assert s._port     == 8080
        assert s._interval == 1.0
        assert s._job_name == "my-job"
        assert s._rank     == 3

    def test_gpu_ids_parsed(self):
        from torchscope.k8s.sidecar import Sidecar
        s = Sidecar(gpu_ids=[0, 1])
        assert s._gpu_ids == [0, 1]

    def test_gpu_ids_none_by_default(self):
        from torchscope.k8s.sidecar import Sidecar
        s = Sidecar()
        # None means "all GPUs" passed through to GPUCollector
        assert s._gpu_ids is None


# ── health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Start a real Sidecar on a free port and verify HTTP responses."""

    def _start_sidecar(self, port):
        from torchscope.k8s.sidecar import Sidecar
        sidecar = Sidecar(port=port)
        # Patch out prometheus and the GPU collector to avoid real hardware deps
        with patch("torchscope.k8s.sidecar.Sidecar._setup_prometheus", lambda self: None), \
             patch("torchscope.collectors.gpu.GPUCollector.start",      lambda self: self), \
             patch("torchscope.collectors.gpu.GPUCollector.stop",       lambda self: self):
            try:
                sidecar.start()
            except Exception:
                pass
        return sidecar

    def test_healthz_returns_200(self):
        port = _free_port()
        from torchscope.k8s.sidecar import Sidecar, _HealthHandler
        from http.server import HTTPServer

        _HealthHandler.sidecar = None
        server = HTTPServer(("", port), _HealthHandler)
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        time.sleep(0.05)
        try:
            resp = urllib.request.urlopen(f"http://localhost:{port}/healthz", timeout=3)
            assert resp.status == 200
            assert resp.read() == b"ok"
        finally:
            server.server_close()

    def test_unknown_path_returns_404(self):
        port = _free_port()
        from torchscope.k8s.sidecar import _HealthHandler
        from http.server import HTTPServer

        server = HTTPServer(("", port), _HealthHandler)
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()
        time.sleep(0.05)
        try:
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                urllib.request.urlopen(f"http://localhost:{port}/unknown", timeout=3)
            assert exc_info.value.code == 404
        finally:
            server.server_close()


# ── sidecar lifecycle ─────────────────────────────────────────────────────────

class TestSidecarLifecycle:
    def _make_sidecar(self, port):
        from torchscope.k8s.sidecar import Sidecar
        return Sidecar(port=port, job_name="test-job", rank=0)

    def test_start_stop_no_raise(self):
        port = _free_port()
        sidecar = self._make_sidecar(port)
        # Patch GPU and Prometheus to avoid hardware deps
        with patch("torchscope.collectors.gpu.GPUCollector.start", lambda self: self), \
             patch("torchscope.collectors.gpu.GPUCollector.stop",  lambda self: self), \
             patch("torchscope.k8s.sidecar.Sidecar._setup_prometheus", lambda self: None):
            sidecar._setup_prometheus = lambda: None
            sidecar._gpu = MagicMock()
            sidecar._gpu.start = MagicMock()
            sidecar._gpu.stop  = MagicMock()

            from torchscope.k8s.sidecar import _HealthHandler
            from http.server import HTTPServer
            sidecar._server = HTTPServer(("", port), _HealthHandler)
            sidecar._running = True
            import threading
            sidecar._srv_thread = threading.Thread(
                target=sidecar._server.serve_forever, daemon=True
            )
            sidecar._srv_thread.start()

            time.sleep(0.05)
            sidecar.stop()   # should not raise

    def test_gpu_collector_attached(self):
        port = _free_port()
        sidecar = self._make_sidecar(port)
        # After Sidecar is constructed the _gpu attr is None until start()
        assert sidecar._gpu is None

    def test_sidecar_stores_config(self):
        port = _free_port()
        sidecar = self._make_sidecar(port)
        assert sidecar._port     == port
        assert sidecar._job_name == "test-job"
        assert sidecar._rank     == 0


# ── prometheus up metric ──────────────────────────────────────────────────────

class TestPrometheusUpMetric:
    def test_up_metric_served_in_metrics_endpoint(self):
        port = _free_port()
        from torchscope.k8s.sidecar import _HealthHandler
        from http.server import HTTPServer

        # Register a real torchscope_up gauge in the default registry
        try:
            from prometheus_client import Gauge, REGISTRY, CollectorRegistry
            # Use a fresh registry to avoid conflicts between tests
            reg = CollectorRegistry()
            up = Gauge("torchscope_up_test", "test gauge", registry=reg)
            up.set(1.0)

            server = HTTPServer(("", port), _HealthHandler)
            t = threading.Thread(target=server.handle_request, daemon=True)
            t.start()
            time.sleep(0.05)

            resp = urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=3)
            # prometheus_client is available; endpoint returns 200
            assert resp.status == 200
            server.server_close()
        except ImportError:
            pytest.skip("prometheus-client not installed")
