# Ray Cluster Profiling

`RayProfiler` aggregates per-worker GPU metrics from a Ray Train cluster into
a single cluster-level report with bottleneck detection across workers.

## Architecture

```
Driver
└── RayProfiler
    └── _AggregatorActor (Ray remote actor)
            ▲ push(summary) every report_interval seconds
            │
    Worker 0                Worker 1              Worker N
    _WorkerProfilerContext  ...                   ...
    └── Profiler (gpu + memory collectors)
```

Each worker runs a standard `Profiler` internally. A background thread
periodically ships `gpu.summary()` + `memory.summary()` to the driver-side
`_AggregatorActor`. After training, `aggregate_report()` pulls all summaries,
computes cluster-level stats, runs `BottleneckAnalyzer` with cluster rules,
and generates an HTML report.

## Usage

```python
# driver script
import ray
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig
from torchscope.ray_profiler import RayProfiler, worker_profiler_context

ray.init()
rp = RayProfiler(num_workers=4, report_interval=30.0)

def train_loop(config):
    from ray import train
    rank = train.get_context().get_world_rank()

    with worker_profiler_context(rp, worker_id=rank, device=0) as prof:
        model = build_model().cuda()
        loader = get_dataloader(config)
        for epoch in range(config["epochs"]):
            for batch in loader:
                loss = model(batch)
                loss.backward()

trainer = TorchTrainer(
    train_loop,
    train_loop_config={"epochs": 10},
    scaling_config=ScalingConfig(num_workers=4, use_gpu=True),
)
trainer.fit()

# collect cluster-level report
report = rp.aggregate_report("cluster_report.html")
```

## `RayProfiler` API

```python
RayProfiler(
    num_workers:     int   = 1,     # expected number of Ray Train workers
    interval:        float = 0.5,   # GPU polling cadence per worker (seconds)
    report_interval: float = 10.0,  # how often workers push summaries to aggregator
)

rp.aggregate_report(
    output_path: str = "cluster_report.html",
    title:       str = "Cluster GPU Report",
) -> dict  # analysis dict with findings + recommendations + _cluster_stats
```

## `worker_profiler_context` API

```python
worker_profiler_context(
    ray_profiler: RayProfiler,
    worker_id:    int   = 0,   # typically train.get_context().get_world_rank()
    device:       int   = 0,   # CUDA device index on this worker
) -> context manager that yields a Profiler
```

## Cluster stats schema

`aggregate_report()` embeds `_cluster_stats` in the returned dict:

```python
{
    "cluster_avg_gpu_util":  float,   # mean across all workers
    "cluster_max_gpu_util":  float,
    "bottleneck_worker_id":  int,     # worker with lowest avg util
    "bottleneck_avg_util":   float,
    "cluster_avg_mem_gb":    float,
    "cluster_max_mem_gb":    float,
    "worker_utils":          {worker_id: avg_gpu_util},
    "imbalance_pct":         float,   # (mean - min) / mean * 100
    "pressured_workers":     list[int],
    "n_summaries_received":  int,
}
```

## Bottleneck rules

| Finding | Trigger | Severity |
|---|---|---|
| `CLUSTER_IMBALANCE` | one worker's avg util > 20% below cluster mean | HIGH |
| `CLUSTER_LOW_GPU_UTIL` | cluster-wide avg util < 40% | MEDIUM |
| `CLUSTER_MEM_PRESSURE` | any worker memory utilization > 85% | MEDIUM |

## Prometheus integration

Each worker's internal `Profiler` is initialized with `rank=worker_id`, so
existing Prometheus labels (`job`, `rank`, `gpu_id`) automatically slice
by worker — no changes to `PrometheusExporter` needed.

```python
prof = Profiler(
    export_prom = True,
    prom_port   = 9000 + worker_id,  # one port per worker
    rank        = worker_id,
)
```

## Without Ray installed

`RayProfiler` degrades gracefully when `ray` is not installed:

```python
rp = RayProfiler(num_workers=4)
# prints: "torchscope: ray not installed — RayProfiler will return empty reports."
rp.aggregate_report()  # returns {}
```

`_compute_cluster_stats()` is a pure function with no Ray dependency and can
be used standalone to compute cluster stats from any list of worker summary dicts.
