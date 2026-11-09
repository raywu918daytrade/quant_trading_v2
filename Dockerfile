# fade (quant_trading_v2) 多架構映像: linux/amd64 + linux/arm64 (Oracle A1 Ampere)。
# 所有相依 (pandas / pyarrow / numpy / shioaji / huggingface_hub) 都有 manylinux_2_28 aarch64 wheel, 不需編譯。
# 用法: docker run --rm -e HF_TOKEN=... ghcr.io/raywu918daytrade/fade nightly   (entrypoint 模式見 deploy/entrypoint.sh)
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONWARNINGS=ignore \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TZ=Asia/Taipei \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    HF_HOME=/app/data/hf-cache

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates tzdata curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 相依先裝, scripts 改了不用重裝 (layer cache)
COPY requirements.txt deploy/constraints.txt ./
RUN pip install -r requirements.txt shioaji -c constraints.txt

COPY scripts/ ./scripts/
COPY deploy/entrypoint.sh /usr/local/bin/fade-entrypoint

# 非 root; 程式會寫 data/ signals/ paper/ stats/ (config.py import 時 mkdir), 正式環境把 data/ signals/ paper/ 掛 PVC
RUN useradd --uid 1000 --create-home fade \
    && mkdir -p /app/data /app/signals /app/paper /app/stats \
    && chmod +x /usr/local/bin/fade-entrypoint \
    && chown -R fade:fade /app
USER fade

ENTRYPOINT ["fade-entrypoint"]
CMD ["help"]
