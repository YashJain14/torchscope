# torchscope

**Cluster-scale GPU observability for PyTorch workloads.**

Wraps any training, inference, or data pipeline and produces:
- Real-time GPU telemetry (utilization, memory, temperature, power, PCIe bandwidth)
- CUDA kernel traces with FLOP counts and CPU↔GPU transfer overhead
- DDP / NCCL communication profiling: per-collective breakdowns, straggler detection
- Ray Train cluster aggregation: per-worker GPU metrics, imbalance detection
- Kubernetes sidecar: zero-code-change GPU telemetry via a drop-in container
- Automated bottleneck analysis with prioritized, actionable recommendations
- Exports to self-contained HTML, structured NDJSON (Loki/Datadog), and Prometheus/Grafana

---

## Install

```bash
pip install -e .                    # core (HTML + JSON export)
pip install -e ".[nvml]"           # + real NVIDIA hardware telemetry
pip install -e ".[full]"           # + nvml + Prometheus
pip install -e ".[ray]"            # + Ray Train cluster profiling
```

---

## Quick start

```python
from torchscope import Profiler

with Profiler() as prof:
    train(model, dataloader)

prof.report("report.html")
```

---

## Full API

```python
prof = Profiler(
    interval    = 0.5,                   # GPU polling cadence (seconds)
    device      = 0,                     # CUDA device index
    job_name    = "bert_finetune",
    export_json = "logs/run.ndjson",     # streaming NDJSON → Loki / Datadog
    export_prom = True,                  # start Prometheus endpoint
    prom_port   = 9000,
    nccl_mode   = "hook",               # NCCL collective tracing ("hook" or "log")
)

prof.start()

# DDP: attach comm profiler after wrapping with DDP
prof.comm.attach(ddp_model)

for step, batch in enumerate(dataloader):
    with prof.comm.step():
        loss = model(batch)
        loss.backward()
        optimizer.step()

prof.report(
    output_path   = "report.html",
    title         = "BERT fine-tune — A100",
    custom_stages = {"data_load": 0.8, "forward": 1.2, "backward": 2.1},
)
```

---

## NCCL collective tracing

torchscope profiles NCCL operations at the `torch.distributed` API level, giving
a per-collective breakdown that goes beyond DDP-level step timing.

**Hook mode** (monkey-patches `torch.distributed.*`, measures real wall-clock latency):
```python
prof = Profiler(nccl_mode="hook")
with prof:
    ddp_train(model, loader)

# summary includes: allreduce_ms, allgather_ms, dominant_collective, avg_message_size_bytes
```

**Log mode** (parse `NCCL_DEBUG=INFO` output, no patching):
```bash
NCCL_DEBUG=INFO python train.py 2>nccl.log
```
```python
from torchscope.collectors.nccl import NCCLCollector

nccl = NCCLCollector(mode="log", log_path="nccl.log")
nccl.start()
# ... training ...
nccl.stop()
print(nccl.summary())
```

New bottleneck rules fired automatically:

| Finding | Trigger |
|---|---|
| `NCCL_ALLREDUCE_DOMINANT` | AllReduce > 60% of collective time |
| `NCCL_SMALL_MESSAGE` | avg message size < 1 MB |

---

## Ray Train cluster profiling

Aggregate per-worker GPU metrics across a Ray Train cluster into a single report
with cluster-level bottleneck detection.

```python
# driver
from torchscope.ray_profiler import RayProfiler, worker_profiler_context

rp = RayProfiler(num_workers=4)

# inside each Ray Train worker function
def train_loop(config):
    rank = train.get_context().get_world_rank()
    with worker_profiler_context(rp, worker_id=rank) as prof:
        for batch in get_dataloader(config):
            model(batch)

trainer = TorchTrainer(train_loop, ...)
trainer.fit()

# after training completes
report = rp.aggregate_report("cluster_report.html")
```

New bottleneck rules:

| Finding | Trigger |
|---|---|
| `CLUSTER_IMBALANCE` | one worker's avg GPU util > 20% below cluster mean |
| `CLUSTER_LOW_GPU_UTIL` | cluster-wide avg util < 40% |
| `CLUSTER_MEM_PRESSURE` | any worker memory utilization > 85% |

---

## Kubernetes sidecar

Zero-code-change GPU telemetry: drop a sidecar container into any training pod.
No changes to training code required.

```bash
docker build -f torchscope/k8s/Dockerfile -t torchscope-sidecar:0.1.0 .
```

```yaml
# In your pod spec, add alongside the training container:
- name: torchscope-sidecar
  image: torchscope-sidecar:0.1.0
  env:
    - name: TORCHSCOPE_JOB_NAME
      value: "my-training-job"
    - name: TORCHSCOPE_RANK
      valueFrom:
        fieldRef:
          fieldPath: metadata.annotations['rank']
  ports:
    - containerPort: 9000
  livenessProbe:
    httpGet:
      path: /healthz
      port: 9000
```

Or deploy with Helm:
```bash
helm install torchscope-sidecar torchscope/k8s/helm/ \
    --set sidecar.jobName=my-training-job
```

The sidecar exposes:
- `GET /metrics` — Prometheus text (scrape with your existing Prometheus stack)
- `GET /healthz` — liveness probe (`200 ok`)
- `torchscope_up{job, rank}` gauge — Grafana can alert when a worker goes missing

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `TORCHSCOPE_PORT` | `9000` | HTTP port for metrics + healthz |
| `TORCHSCOPE_INTERVAL` | `0.5` | GPU polling cadence (seconds) |
| `TORCHSCOPE_JOB_NAME` | `torchscope` | Prometheus label |
| `TORCHSCOPE_RANK` | `0` | Pod rank label |
| `TORCHSCOPE_GPU_IDS` | _(all)_ | Comma-separated GPU indices, e.g. `0,1` |

---

## NVTX / Nsight Systems integration

```python
from torchscope.nvtx import annotate, NVTXRegion

with annotate("preprocess"):
    tensor = preprocess(frames)

@NVTXRegion("forward_pass")
def forward(self, x):
    ...
```

```bash
nsys profile --trace=cuda,nvtx,osrt --output=profile python train.py
```

---

## Prometheus metrics

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
| `torchscope_up` | job, rank |

---

## Bottleneck rules

| Finding | Trigger | Severity |
|---|---|---|
| `LOW_GPU_UTIL` | avg util < 50% | HIGH |
| `GPU_IDLE_TIME` | idle > 20% of window | MEDIUM |
| `MEMORY_UNDERUSED` | peak mem < 50% + >2 GB free | MEDIUM |
| `MEMORY_FRAGMENTATION` | avg fragmentation > 30% | MEDIUM |
| `OUT_OF_MEMORY` | OOM event detected | HIGH |
| `ALLOC_RETRIES` | allocation retries > 0 | MEDIUM |
| `MEMORY_LEAK` | monotone alloc increase in >80% of samples | HIGH |
| `HIGH_TRANSFER_OVERHEAD` | CPU↔GPU transfer > 10% of CUDA time | HIGH |
| `DOMINANT_KERNEL` | single kernel > 30% of CUDA time | HIGH |
| `THERMAL_THROTTLING` | >80°C for >30% of samples | MEDIUM |
| `HIGH_COMM_OVERHEAD` | DDP comm > 15% of step time | HIGH |
| `STRAGGLER_GPU` | rank median > 10% above global median | MEDIUM |
| `STAGE_BOTTLENECK_*` | any stage > 35% of pipeline time | HIGH/MEDIUM |
| `NCCL_ALLREDUCE_DOMINANT` | AllReduce > 60% of collective time | MEDIUM |
| `NCCL_SMALL_MESSAGE` | avg collective message < 1 MB | MEDIUM |
| `CLUSTER_IMBALANCE` | worker util > 20% below cluster mean | HIGH |
| `CLUSTER_LOW_GPU_UTIL` | cluster avg util < 40% | MEDIUM |
| `CLUSTER_MEM_PRESSURE` | any worker mem > 85% | MEDIUM |

---

## Architecture

```
torchscope/
├── collectors/
│   ├── gpu.py          NVML hardware polling (util, mem, temp, power, PCIe)
│   ├── memory.py       torch.cuda allocation tracking + leak detection
│   ├── comm.py         DDP comm hooks, step timing, straggler detection
│   └── nccl.py         NCCL collective tracing (hook + log modes)
├── exporters/
│   ├── html.py         self-contained Plotly HTML report
│   ├── json_log.py     NDJSON streaming (Loki / Datadog / ELK)
│   └── prometheus.py   Prometheus gauges (Grafana cluster observability)
├── k8s/
│   ├── sidecar.py      standalone GPU metrics process (no torch needed)
│   ├── Dockerfile      minimal image: pynvml + prometheus-client
│   └── helm/           Helm chart for sidecar injection
├── tracer.py           torch.profiler kernel tracing + FLOP counting
├── nvtx.py             NVTX annotations (Nsight Systems)
├── analyzer.py         18-rule bottleneck engine (single-GPU + cluster)
├── ray_profiler.py     Ray Train cluster aggregator
└── profiler.py         public API — Profiler class
```

**Data flow:** collectors run in background threads → push samples via callback → PrometheusExporter / JSONExporter (live, per sample) → at end, `report()` → BottleneckAnalyzer → HTMLExporter.

**Composable:** use only the collectors you need, activate only the exporters you want. Nothing assumes a specific workload type.

---

## Examples

```bash
python examples/training_loop.py                          # single-GPU training
python examples/inference_server.py --batch 32           # triggers LOW_GPU_UTIL
python examples/inference_server.py --batch 128          # resolves it
python examples/distributed_training.py                  # DDP + comm profiling
python examples/video_pipeline.py --video clip.mp4       # multi-stage pipeline
```
