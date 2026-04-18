"""
GPUCollector — polls NVML hardware metrics from a background thread.

Collects per-GPU every `interval` seconds:
  - SM utilization %
  - Memory used / total / utilization %
  - Temperature (°C)
  - Power draw (W)
  - SM clock frequency (MHz)  — detects thermal throttling
  - PCIe / NVLink throughput  — detects bandwidth saturation

Gracefully degrades to torch.cuda stats when pynvml is unavailable
(e.g. non-NVIDIA hardware or missing driver bindings).
"""

from __future__ import annotations

import threading
import time
from typing import Optional

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
except Exception:
    _NVML = False


class GPUCollector:
    """
    Thread-safe background poller for GPU hardware telemetry.

    Usage:
        col = GPUCollector(interval=0.5)
        col.start()
        ... workload ...
        samples = col.stop()
        print(col.summary())
    """

    def __init__(
        self,
        interval:  float = 0.5,
        gpu_ids:   Optional[list[int]] = None,
        on_sample: Optional[callable]  = None,
    ):
        self.interval  = interval
        self.samples:  list[dict] = []
        self._running  = False
        self._thread:  Optional[threading.Thread] = None
        # Called with each sample dict immediately after it is collected.
        # Use this for live streaming to Prometheus / JSON exporters.
        self._on_sample = on_sample

        if _NVML:
            n = pynvml.nvmlDeviceGetCount()
            self.gpu_ids = gpu_ids if gpu_ids is not None else list(range(n))
            self._backend = "pynvml"
        else:
            self.gpu_ids  = gpu_ids if gpu_ids is not None else [0]
            self._backend = "torch"

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def register_callback(self, fn: callable) -> "GPUCollector":
        """
        Register a function called with every sample as it is collected.
        Used to stream live metrics to Prometheus or a JSON log without
        waiting for the full run to finish.

            col.register_callback(prom_exporter.update_gpu_sample)
            col.register_callback(json_exporter.write_gpu_sample)
        """
        self._on_sample = fn
        return self

    def start(self) -> "GPUCollector":
        self._running = True
        fn = self._poll_nvml if self._backend == "pynvml" else self._poll_torch
        self._thread = threading.Thread(target=fn, daemon=True, name="torchscope.gpu")
        self._thread.start()
        return self

    def stop(self) -> list[dict]:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        return self.samples

    # ── polling ───────────────────────────────────────────────────────────────

    def _poll_nvml(self):
        while self._running:
            ts = time.time()
            for gid in self.gpu_ids:
                try:
                    h    = pynvml.nvmlDeviceGetHandleByIndex(gid)
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
                    temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                    try:
                        pwr  = pynvml.nvmlDeviceGetPowerUsage(h) / 1e3  # mW → W
                    except pynvml.NVMLError:
                        pwr = 0.0
                    try:
                        clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    except pynvml.NVMLError:
                        clock = 0
                    try:
                        pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                            h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1e6  # KB/s → GB/s
                        pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                            h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1e6
                    except pynvml.NVMLError:
                        pcie_tx = pcie_rx = 0.0

                    sample = {
                        "ts":          ts,
                        "gpu_id":      gid,
                        "gpu_util":    util.gpu,
                        "mem_util":    util.memory,
                        "mem_used_gb": mem.used  / 1e9,
                        "mem_free_gb": mem.free  / 1e9,
                        "mem_total_gb":mem.total / 1e9,
                        "mem_pct":     mem.used  / mem.total * 100,
                        "temp_c":      temp,
                        "power_w":     pwr,
                        "sm_clock_mhz":clock,
                        "pcie_tx_gbs": pcie_tx,
                        "pcie_rx_gbs": pcie_rx,
                    }
                    self.samples.append(sample)
                    if self._on_sample:
                        try:
                            self._on_sample(sample)
                        except Exception:
                            pass
                except Exception:
                    pass
            time.sleep(self.interval)

    def _poll_torch(self):
        try:
            import torch
        except ImportError:
            return
        while self._running:
            ts = time.time()
            for gid in self.gpu_ids:
                try:
                    used  = torch.cuda.memory_allocated(gid)
                    total = torch.cuda.get_device_properties(gid).total_memory
                    sample = {
                        "ts":          ts,
                        "gpu_id":      gid,
                        "gpu_util":    -1,
                        "mem_util":    -1,
                        "mem_used_gb": used  / 1e9,
                        "mem_free_gb": (total - used) / 1e9,
                        "mem_total_gb":total / 1e9,
                        "mem_pct":     used  / total * 100,
                        "temp_c":      -1,
                        "power_w":     -1,
                        "sm_clock_mhz":-1,
                        "pcie_tx_gbs": -1,
                        "pcie_rx_gbs": -1,
                    }
                    self.samples.append(sample)
                    if self._on_sample:
                        try:
                            self._on_sample(sample)
                        except Exception:
                            pass
                except Exception:
                    pass
            time.sleep(self.interval)

    # ── analytics ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        if not self.samples:
            return {"backend": self._backend, "n_samples": 0}

        def _vals(key):
            return [s[key] for s in self.samples if s.get(key, -1) >= 0]

        util_v  = _vals("gpu_util")
        temp_v  = _vals("temp_c")
        pwr_v   = _vals("power_w")
        mem_v   = [s["mem_used_gb"] for s in self.samples]
        pcie_tx = _vals("pcie_tx_gbs")

        duration = self.samples[-1]["ts"] - self.samples[0]["ts"] if len(self.samples) > 1 else 0

        return {
            "backend":              self._backend,
            "n_samples":            len(self.samples),
            "duration_s":           round(duration, 2),
            "gpu_ids":              self.gpu_ids,
            # utilisation
            "avg_gpu_util":         round(sum(util_v) / len(util_v), 1) if util_v else -1,
            "max_gpu_util":         round(max(util_v), 1)               if util_v else -1,
            "pct_time_idle":        round(
                sum(1 for v in util_v if v < 10) / len(util_v) * 100, 1
            ) if util_v else -1,
            # memory
            "peak_mem_gb":          round(max(mem_v), 3),
            "avg_mem_gb":           round(sum(mem_v) / len(mem_v), 3),
            "mem_total_gb":         round(self.samples[0]["mem_total_gb"], 2),
            # thermals
            "avg_temp_c":           round(sum(temp_v) / len(temp_v), 1) if temp_v else -1,
            "max_temp_c":           round(max(temp_v), 1)               if temp_v else -1,
            "thermal_throttle_pct": round(
                sum(1 for t in temp_v if t > 80) / len(temp_v) * 100, 1
            ) if temp_v else 0,
            # power
            "avg_power_w":          round(sum(pwr_v) / len(pwr_v), 1)  if pwr_v else -1,
            "max_power_w":          round(max(pwr_v), 1)               if pwr_v else -1,
            # bandwidth
            "avg_pcie_tx_gbs":      round(sum(pcie_tx) / len(pcie_tx), 2) if pcie_tx else -1,
        }
