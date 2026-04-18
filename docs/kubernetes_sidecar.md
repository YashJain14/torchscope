# Kubernetes Sidecar

The torchscope sidecar is a standalone container that exports GPU metrics to
Prometheus **without any changes to training code**. Drop it alongside your
training container and get immediate GPU observability.

## How it works

The sidecar runs `GPUCollector` (pynvml-based) and `PrometheusExporter` in-process,
independent of the training workload. It does not import torch. It scrapes the GPU
via NVML at a configurable interval and exposes metrics on a single HTTP port.

```
Training Pod
├── training-container   ← your code, unchanged
└── torchscope-sidecar   ← polls NVML, exposes :9000/metrics
        │
        └── Prometheus scrapes /metrics
                │
                └── Grafana dashboard
```

## Build

```bash
docker build -f torchscope/k8s/Dockerfile -t torchscope-sidecar:0.1.0 .
```

The image is minimal: `python:3.10-slim` + `pynvml` + `prometheus-client`.
No torch, no CUDA toolkit — NVML access goes through the host driver via
the NVIDIA device plugin.

## Manual pod spec

Add to `spec.template.spec.containers` in your existing Deployment or Job:

```yaml
- name: torchscope-sidecar
  image: torchscope-sidecar:0.1.0
  env:
    - name: TORCHSCOPE_PORT
      value: "9000"
    - name: TORCHSCOPE_INTERVAL
      value: "0.5"
    - name: TORCHSCOPE_JOB_NAME
      value: "my-training-job"
    - name: TORCHSCOPE_RANK
      valueFrom:
        fieldRef:
          fieldPath: metadata.annotations['rank']
  ports:
    - name: metrics
      containerPort: 9000
  livenessProbe:
    httpGet:
      path: /healthz
      port: 9000
    initialDelaySeconds: 5
    periodSeconds: 15
  readinessProbe:
    httpGet:
      path: /healthz
      port: 9000
    initialDelaySeconds: 3
    periodSeconds: 10
  resources:
    limits:
      cpu: "200m"
      memory: "128Mi"
    requests:
      cpu: "50m"
      memory: "64Mi"
```

Add Prometheus scrape annotations to the pod metadata:
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port:   "9000"
  prometheus.io/path:   "/metrics"
```

## Helm chart

```bash
# Install with defaults (targets pods labeled app=ml-training)
helm install torchscope-sidecar torchscope/k8s/helm/

# Override job name and target labels
helm install torchscope-sidecar torchscope/k8s/helm/ \
    --set sidecar.jobName=bert-finetune \
    --set targetPodLabels.app=bert-training \
    --set targetPodLabels.env=prod
```

Key `values.yaml` knobs:

```yaml
image:
  repository: torchscope-sidecar
  tag: "0.1.0"

sidecar:
  port: 9000
  interval: "0.5"
  jobName: "torchscope"

targetPodLabels:
  app: ml-training
```

## HTTP endpoints

| Path | Response |
|---|---|
| `GET /healthz` | `200 ok` — for Kubernetes liveness/readiness probes |
| `GET /metrics` | Prometheus text format — for Prometheus scrape |

## `torchscope_up` gauge

The sidecar always sets `torchscope_up{job, rank} = 1.0` on startup.
In Grafana, alert on `absent(torchscope_up{job="my-job"})` to detect
workers that have crashed or been evicted.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `TORCHSCOPE_PORT` | `9000` | HTTP port (metrics + healthz on same port) |
| `TORCHSCOPE_INTERVAL` | `0.5` | GPU polling cadence (seconds) |
| `TORCHSCOPE_JOB_NAME` | `torchscope` | `job` label on all Prometheus metrics |
| `TORCHSCOPE_RANK` | `0` | `rank` label — set from pod annotations for multi-worker jobs |
| `TORCHSCOPE_GPU_IDS` | _(all GPUs)_ | Comma-separated indices to restrict polling, e.g. `0,1` |

## Run locally (smoke test)

```bash
TORCHSCOPE_PORT=19000 python -m torchscope.k8s.sidecar &
curl localhost:19000/healthz     # → ok
curl localhost:19000/metrics     # → Prometheus text with torchscope_up 1.0
```
