# obi-lookout as an OpenAI-compatible server, CPU only, weights baked in so it runs with no network after the build.
# Build from the repository root (docker compose -f docker/compose.yaml up does this for you):
#   docker build -f docker/Dockerfile -t obi-lookout .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 HOME=/tmp HF_HOME=/opt/hf
WORKDIR /app

# CPU-only torch is far smaller than the default build. The other pins come from requirements.txt.
RUN pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# The weights (about 1.2 GB) are fetched once, at the same pinned commit obi_lookout.py uses.
COPY obi_lookout.py server.py ./
RUN python -c "import obi_lookout as o; from huggingface_hub import snapshot_download; snapshot_download(o.MODEL, revision=o.REVISION)" \
 && chmod -R a+rX /opt/hf

ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN useradd --system --uid 10001 obi
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4)"
CMD ["python", "server.py", "--port", "8000"]
