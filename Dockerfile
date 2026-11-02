# 星织的镜像。两段：前一段只装依赖，后一段只放代码，镜像里不留构建工具。

FROM python:3.12-slim AS deps
WORKDIR /app
COPY pyproject.toml README.md VERSION ./
COPY commands ./commands
COPY features ./features
COPY utils ./utils
COPY main.py ./
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir . \
    && pip install --no-cache-dir pillow onnx onnxruntime

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
COPY . .
RUN useradd -u 10001 -m ori && mkdir -p /data && chown -R ori /data /app
USER ori
ENV ORI_HOME=/data
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD ori doctor > /dev/null || exit 1
ENTRYPOINT ["ori"]
CMD ["--help"]
