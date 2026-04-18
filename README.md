# torchscope

**Lightweight PyTorch GPU observability toolkit.**

Wraps any PyTorch workload — training, inference, data pipelines — and produces:
- Real-time GPU telemetry (utilization, memory, temperature, power, PCIe bandwidth)
- CUDA kernel traces with FLOP counts and CPU↔GPU transfer overhead
- DDP communication profiling: compute-to-communication ratios, straggler detection
- Automated bottleneck analysis with prioritized, actionable recommendations
- Exports to self-contained HTML, structured JSON (Loki/Datadog), and Prometheus

---

## Install

```bash
pip install plotly jinja2              # required
pip install pynvml                     # recommended: real NVML telemetry
pip install prometheus-client          # optional: Prometheus export
```

---

## Quick start

```python
from torchscope import Profiler

with Profiler() as prof:
    train(model, dataloader)           # any PyTorch workload

prof.report("report.html")
```

---

## Full API

```python
prof = Profiler(
    interval    = 0.5,             # GPU polling cadence (seconds)
    device      = 0,               # CUDA device index
    job_name    = "bert_finetune",
    export_json = "logs/run.ndjson",   # structured NDJSON → Loki / Datadog
    export_prom = True,                # start Prometheus endpoint
    prom_port   = 9000,
)

prof.start()

# DDP: attach comm profiler after wrapping with DDP
prof.comm.attach(ddp_model)

# Time each training step
for step, batch in enumerate(dataloader):
    with prof.comm.step():
        loss = model(batch)
        loss.backward()
        optimizer.step()

# Kernel tracing (bounded window, not the full run)
prof.enable_kernel_tracing(wait=2, warmup=1, active=3)
with prof.kernel_profiler() as kp:
    for batch in dataloader:
        model(batch)
        kp.step()

# Custom pipeline stage breakdown
prof.report(
    output_path   = "report.html",
    title         = "BERT fine-tune — A100",
    custom_stages = {"data_load": 0.8, "forward": 1.2, "backward": 2.1},
)
```

---

## NVTX / Nsight Systems integration

Annotate code regions that appear as named ranges in the Nsight Systems GUI:

```python
from torchscope.nvtx import annotate, NVTXRegion

with annotate("preprocess"):
    tensor = preprocess(frames)

@NVTXRegion("forward_pass")
def forward(self, x):
    ...
```

Run under Nsight:
```bash
nsys profile --trace=cuda,nvtx,osrt \
    --output=profile \
    python train.py
```

---

## Prometheus metrics

When `export_prom=True`, torchscope starts an HTTP server and exposes:

| Metric | Labels |
|---|---|
| `torchscope_gpu_utilization_pct` | job, rank, gpu_id |
| `torchscope_gpu_memory_used_gb` | job, rank, gpu_id |
| `torchscope_gpu_temperature_c` | job, rank, gpu_id |
| `torchscope_gpu_power_w` | job, rank, gpu_id |
| `torchscope_gpu_pcie_tx_gbs` | job, rank, gpu_id |
| `torchscope_memory_fragmentation_pct` | job, rank |
| `torchscope_comm_overhead_pct` | job, rank |
| `torchscope_straggler_rank` | job, rank |

Scrape with Prometheus → visualise in Grafana across a whole cluster.

---

## Bottleneck rules

| Finding | Trigger | Default threshold |
|---|---|---|
| `LOW_GPU_UTIL` | avg util < 50% | 50% |
| `GPU_IDLE_TIME` | idle time > 20% of window | 20% |
| `MEMORY_UNDERUSED` | peak mem < 50% + >2 GB free | 50% / 2 GB |
| `MEMORY_FRAGMENTATION` | avg fragmentation > 30% | 30% |
| `OUT_OF_MEMORY` | torch reports OOM | any |
| `ALLOC_RETRIES` | torch reports alloc retries | any |
| `MEMORY_LEAK` | monotone alloc increase >80% of samples | 80% |
| `HIGH_TRANSFER_OVERHEAD` | CPU↔GPU transfer > 10% CUDA time | 10% |
| `DOMINANT_KERNEL` | single kernel > 30% CUDA time | 30% |
| `THERMAL_THROTTLING` | >80°C for >30% of samples | 30% |
| `HIGH_COMM_OVERHEAD` | DDP comm > 15% of step time | 15% |
| `STRAGGLER_GPU` | rank median > 10% above global median | 10% |
| `STAGE_BOTTLENECK_*` | any stage > 35% of total pipeline time | 35% |

---

## Architecture

```
torchscope/
├── torchscope/
│   ├── collectors/
│   │   ├── gpu.py          NVML hardware polling (util, mem, temp, power, PCIe)
│   │   ├── memory.py       torch.cuda allocation tracking + leak detection
│   │   └── comm.py         DDP comm hooks, step timing, straggler detection
│   ├── tracer.py           torch.profiler kernel tracing + FLOP counting
│   ├── nvtx.py             NVTX range annotations (Nsight Systems integration)
│   ├── analyzer.py         generic bottleneck rule engine
│   ├── exporters/
│   │   ├── html.py         self-contained Plotly HTML report
│   │   ├── json_log.py     NDJSON structured logs (Loki / Datadog / ELK)
│   │   └── prometheus.py   Prometheus gauges (cluster observability)
│   └── profiler.py         public API — Profiler class
├── examples/
│   ├── training_loop.py    single-GPU training
│   ├── inference_server.py batch embedding inference
│   └── video_pipeline.py  video decode/inference pipeline
└── integrations/
    └── benchmark_adapter.py  video-inference-benchmark → torchscope
```

The tool is **composable**: use only the collectors you need, choose which
exporters to activate, pass arbitrary `custom_stages` for any pipeline type.
No part of the code knows about video inference specifically — that's an
integration, not a core concept.

---

## Examples

```bash
# Training loop
python examples/training_loop.py

# Batch inference (small batch → LOW_GPU_UTIL finding)
python examples/inference_server.py --batch 32
python examples/inference_server.py --batch 128   # resolves it

# Video pipeline (from repo root)
python examples/video_pipeline.py \
    --video ToS.mp4 --model yolov8n.pt \
    --pipelines opencv_cpu ffmpeg_dali pynvvideocodec pynv_cvcuda
```
