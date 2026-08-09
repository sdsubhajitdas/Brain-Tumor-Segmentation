FROM python:3.14-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=1

WORKDIR /app

COPY requirements-web.txt .

# CPU-only torch, scoped to this one command so the CPU wheel index never
# influences resolution of the packages in requirements-web.txt below. Plain
# `pip install torch` on Linux pulls a CUDA-bundled build (100s of MB of
# nvidia-* packages) even though it still only runs on CPU here.
# torchvision is intentionally NOT installed -- web/inference.py's PIL/numpy
# preprocessing replaced the 3 torchvision transforms it used to need, so the
# web image no longer carries torchvision's extra import-time memory at all.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
        torch==2.13.0 \
    && pip install -r requirements-web.txt

COPY bts/ ./bts/
COPY web/ ./web/
# Copy the whole directory rather than naming the checkpoint file directly --
# its name contains literal [ ] characters, which Dockerfile COPY treats as
# glob metacharacters and fails to parse/match as a literal path.
COPY saved_models/ ./saved_models/

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')" || exit 1

CMD ["python", "-m", "uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000"]
