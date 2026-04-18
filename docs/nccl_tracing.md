# NCCL Collective Tracing

torchscope profiles NCCL operations at the `torch.distributed` API level, giving
a per-collective breakdown that goes beyond DDP-level step timing.

## Why this matters

`CommCollector` (DDP hooks) tells you *what fraction of a training step is communication*.
`NCCLCollector` tells you *which NCCL operation is responsible* — AllReduce, AllGather,
ReduceScatter, Broadcast — and how large each message is. This is the information
needed to decide whether to increase `bucket_cap_mb`, switch to FSDP, or apply
gradient compression.

## Modes

### Hook mode

Monkey-patches `torch.distributed.all_reduce`, `all_gather`, `reduce_scatter`,
`broadcast`, and `reduce`. Measures real wall-clock latency (with `cuda.synchronize()`
after each call) and estimates message size from the tensor arguments.

```python
from torchscope import Profiler

prof = Profiler(nccl_mode="hook")
prof.start()
prof.comm.attach(ddp_model)

for step, batch in enumerate(loader):
    with prof.comm.step():
        loss = model(batch)
        loss.backward()
        optimizer.step()

analysis = prof.report("report.html")
# analysis["findings"] may include NCCL_ALLREDUCE_DOMINANT or NCCL_SMALL_MESSAGE
```

Or use `NCCLCollector` directly:

```python
from torchscope.collectors.nccl import NCCLCollector

nccl = NCCLCollector(mode="hook")
nccl.start()
# ... distributed training ...
nccl.stop()
print(nccl.summary())
```

Hooks are restored on `stop()` — no permanent patching.

### Log mode

Parses `NCCL_DEBUG=INFO` stderr output using regex. Does not patch any functions.
Extracts collective type, message size, algorithm (ring/tree), and protocol (LL/LL128/Simple).
Duration is not available in log mode (set to 0) — use hook mode for timing.

```bash
NCCL_DEBUG=INFO python train.py 2>nccl.log
```

```python
nccl = NCCLCollector(mode="log", log_path="nccl.log")
nccl.start()
# tails nccl.log in a background thread
train(model, loader)
nccl.stop()
```

## Summary schema

```python
{
    "n_events":               int,     # total collective calls observed
    "total_ms":               float,   # sum of all durations
    "dominant_collective":    str,     # "all_reduce" | "all_gather" | ...
    "dominant_pct":           float,   # % of total_ms
    "avg_message_size_bytes": int,
    "allreduce_ms":           float,
    "allgather_ms":           float,
    "reducescatter_ms":       float,
    "events_by_op": {
        "all_reduce": {"count": int, "total_ms": float, "avg_ms": float, "avg_size_bytes": int},
        ...
    }
}
```

## Bottleneck rules

| Finding | Condition | Recommendation |
|---|---|---|
| `NCCL_ALLREDUCE_DOMINANT` | AllReduce > 60% of collective time | Increase `bucket_cap_mb`, use PowerSGD, switch to FSDP ZeRO-2/3 |
| `NCCL_SMALL_MESSAGE` | avg message < 1 MB | Increase `bucket_cap_mb` (default 25 MB → 100–200 MB), use gradient accumulation |

## Interaction with CommCollector

`CommCollector` and `NCCLCollector` are complementary:

- `CommCollector.attach(ddp_model)` measures at the DDP bucket level — total comm time per step
- `NCCLCollector` measures at the `torch.distributed` API level — per-collective-call timing

When both are active, `CommCollector` gives the step-level comm/compute split and
`NCCLCollector` gives the breakdown of which collective dominates within that comm budget.
