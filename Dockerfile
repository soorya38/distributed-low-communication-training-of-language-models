# syntax=docker/dockerfile:1
#
# One image definition, two flavours:
#   docker build -t diloco:cpu .                          # CPU only (small)
#   docker build -t diloco:gpu --build-arg TARGET=gpu .   # NVIDIA GPUs via CUDA + NCCL
# The code picks CUDA/NCCL at runtime when a GPU is visible and falls back to CPU/gloo.

ARG TARGET=cpu
ARG TORCH_VERSION=2.14.1

FROM python:3.12-slim-bookworm AS base-cpu
ARG TORCH_VERSION
RUN pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu

FROM pytorch/pytorch:${TORCH_VERSION}-cuda12.6-cudnn9-runtime AS base-gpu

FROM base-${TARGET}
# iproute2 provides `tc` for optional WAN emulation (NET_DELAY / NET_RATE).
RUN apt-get update \
    && apt-get install -y --no-install-recommends iproute2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY diloco ./diloco
RUN pip install --no-cache-dir numpy matplotlib && pip install --no-cache-dir .

COPY . .
RUN python -c "import sys; sys.path.insert(0, 'torch_diloco'); import train_lm; train_lm.load_text()" \
    && mkdir -p results && chown -R app:app /app

ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["python", "torch_diloco/train_lm.py", "--mode", "diloco", "--inner-steps", "50", "--steps", "500", "--eval-interval", "50"]
