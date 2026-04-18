FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install runtime dependencies only — no torch needed for the sidecar
RUN pip install --no-cache-dir pynvml>=11.0 prometheus-client>=0.17

# Copy the torchscope package (only collectors + exporters are used at runtime)
COPY torchscope/ /app/torchscope/

ENV TORCHSCOPE_PORT=9000 \
    TORCHSCOPE_INTERVAL=0.5 \
    TORCHSCOPE_JOB_NAME=torchscope \
    TORCHSCOPE_RANK=0 \
    TORCHSCOPE_GPU_IDS=""

EXPOSE 9000

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:$TORCHSCOPE_PORT/healthz')"

ENTRYPOINT ["python", "-m", "torchscope.k8s.sidecar"]
