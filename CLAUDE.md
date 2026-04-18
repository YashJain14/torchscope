# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**torchscope** is a PyTorch GPU observability toolkit. It wraps any PyTorch workload to provide GPU telemetry, CUDA kernel tracing, DDP communication profiling, bottleneck analysis, and multi-format exports.

## Setup & Installation

```bash
pip install -e .                    # Basic install
pip install -e ".[nvml]"           # With real NVIDIA GPU telemetry (recommended)
pip install -e ".[full]"           # With nvml + Prometheus support
```

Requirements: Python >=3.10, plotly>=5.0, jinja2>=3.0.

## Running Tests

```bash
pytest tests/                          # All tests
pytest tests/test_single_gpu.py        # Single-GPU tests
pytest tests/test_cpu.py               # CPU-only / graceful degradation
pytest tests/test_multi_gpu.py         # DDP / multi-GPU tests
pytest tests/test_single_gpu.py::TestBottleneckAnalyzer  # Single class
pytest tests/test_single_gpu.py -k "test_html"           # Single test by name
```

## CLI

```bash
torchscope run <script.py>         # Profile a script with live telemetry
torchscope analyze <file.ndjson>   # Generate report from existing log
torchscope info                    # Print environment diagnostics
```

## Architecture

The public API is a single `Profiler` class that orchestrates four independent subsystems:

```
Profiler (profiler.py)
├── Collectors (background threads, poll independently)
│   ├── GPUCollector   (collectors/gpu.py)     — util%, temp, power, PCIe BW
│   ├── MemoryCollector (collectors/memory.py) — alloc/reserved/peak, OOM, leaks
│   └── CommCollector  (collectors/comm.py)    — DDP all-reduce timing, straggler detection
├── KernelTracer (tracer.py)                   — wraps torch.profiler, extracts top kernels
├── BottleneckAnalyzer (analyzer.py)           — rule engine over raw dicts → findings
└── Exporters
    ├── HTMLExporter    (exporters/html.py)     — self-contained Plotly + Jinja2 report
    ├── JSONLogExporter (exporters/json_log.py) — streaming NDJSON for log aggregators
    └── PrometheusExporter (exporters/prometheus.py) — live HTTP metrics for Grafana
```

**Profiler lifecycle:** `start()` → background threads begin → user workload runs → `stop()` → `report()` aggregates, runs analyzer, generates exports.

**`__init__.py`** uses `__getattr__` for lazy loading to avoid slow imports at package load time.

## Key Design Decisions

- **Collectors are decoupled from workload type** — they push raw dicts; the analyzer consumes plain dicts. Adding a new rule only requires editing `analyzer.py`.
- **12+ bottleneck detection rules** in `analyzer.py` with tunable thresholds at the top of the file (e.g., `LOW_GPU_UTIL_THRESHOLD = 0.50`). Rules cover: low GPU util, idle time, memory underuse/fragmentation, OOM, allocation retries, memory leaks, CPU↔GPU transfer overhead, dominant kernels, thermal throttling, DDP comm overhead, straggler GPUs, and custom pipeline stage bottlenecks.
- **Graceful degradation** — all GPU/CUDA paths are guarded; the library runs on CPU-only machines without errors (verified in `tests/test_cpu.py`).
- **Prometheus and NDJSON exporters stream live** (per sample), while the HTML exporter runs end-of-job.
- **DDP integration** requires explicit hook attachment after `DistributedDataParallel` wrap: `prof.comm.attach(ddp_model)`.

## Examples

`examples/` contains four runnable scripts demonstrating key use cases:
- `training_loop.py` — single-GPU training with NVTX annotations
- `inference_server.py` — batch inference (triggers LOW_GPU_UTIL detection)
- `distributed_training.py` — DDP with communication profiling
- `video_pipeline.py` — multi-stage pipeline with per-stage breakdown