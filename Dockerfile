# 网关/面板运行镜像；默认只装必需依赖，训练镜像可给 --build-arg ORI_EXTRAS=full,eval,export。
FROM python:3.12-slim AS deps
WORKDIR /app
ARG ORI_EXTRAS=""
COPY pyproject.toml README.md VERSION main.py ./
COPY commands ./commands
COPY features ./features
COPY utils ./utils
RUN if [ -n "$ORI_EXTRAS" ]; then pip install --no-cache-dir ".[${ORI_EXTRAS}]"; else pip install --no-cache-dir .; fi

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin
RUN useradd -u 10001 -m ori && mkdir -p /data && chown ori /data
USER ori
ENV ORI_HOME=/data
EXPOSE 8080 8765
# 健康检查随 Compose 的具体服务声明；doctor 缺可选训练包不代表 HTTP 服务故障。
ENTRYPOINT ["ori"]
CMD ["--help"]
