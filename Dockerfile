# Laya as an HTTP API. Guide: docs/API.md
#
#   docker build -t laya-api .
#   docker run --rm -p 8000:8000 -v laya-models:/models laya-api
#   open http://localhost:8000/docs
#
# GPU: the CUDA wheels bundle their own runtime, so the same slim base works --
#   docker build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 -t laya-api:gpu .
#   docker run --rm --gpus all -p 8000:8000 -v laya-models:/models laya-api:gpu
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

# CPU wheels by default: they keep the image ~1 GB instead of ~6 GB.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # transformers probes for TensorFlow at import; when TF is present its abseil runtime can
    # deadlock model construction. Laya is torch-only.
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    # checkpoints land here, so mounting a volume on /models survives image rebuilds
    HF_HOME=/models

WORKDIR /app

# torch first and on its own layer: it is the slow half of the build and changes least.
RUN pip install torch --index-url ${TORCH_INDEX_URL}

COPY pyproject.toml setup.py README.md ./
COPY laya ./laya
RUN pip install ".[serve]"

RUN useradd --create-home --uid 1000 laya && mkdir -p /models && chown laya:laya /models
USER laya

EXPOSE 8000

# A cold start downloads checkpoints (~1.3-1.7 GB each), so the grace period is generous:
# failures inside it leave the container "starting" rather than marking it unhealthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["uvicorn", "laya.server:app", "--host", "0.0.0.0", "--port", "8000"]
