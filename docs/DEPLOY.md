# 部署

一条 `docker compose up -d` 起两件：网关（8080）和面板（8765）。产物与请求日志都在 `./data` 卷里。

## 前一次准备

```bash
mkdir -p data/bolt data/model      # 产物目录和模型目录
echo 'ORI_TOKEN=换成你的令牌' > .env
cp 训练好的权重 data/model/        # 或者先 down 掉，不带模型起面板看数据
docker compose build
docker compose up -d
curl -H "Authorization: Bearer $ORI_TOKEN" 127.0.0.1:8080/v1/models
```

`.env` 里的 `ORI_TOKEN` 是必填项，没给的话 compose 会在启动前就报错（`${ORI_TOKEN:?...}`）。

## 两件的分工

| 服务 | 干什么 | 端口 | 卷 |
| --- | --- | --- | --- |
| serve | OpenAI 兼容网关，带鉴权、限流、请求日志 | 8080 | `./data` 可写 |
| loom | 只读面板，看数据集卡、样本、拒收、进度 | 8765 | `./data` 只读 |

面板挂在 `serve` 的健康检查之后起来，网关没就绪时不会先开面板。

## 硬约束

- 镜像里非 root（uid 10001），`/data` 是唯一可写处。
- 健康检查用 `ori doctor`，它只查本机依赖，不联网。
- 网关的 `--host 0.0.0.0` 只在容器里有意义；宿主机上默认还是 127.0.0.1。

## 没容器的时候

同一条命令直接跑，只是 `--host` 换回默认：

```bash
ori serve data/model --port 8080 --token "$ORI_TOKEN" --log data/gateway.jsonl
ori loom data/bolt --port 8765
```

## 反代

放公网前面一定要加一层。Caddy 两行：

```
ori.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

面板同理，或者只在内网开。

## 日志与轮转

请求日志是 `data/gateway.jsonl`，一行一条。交给外面的 logrotate 或者采集器，别指望它自己转。
