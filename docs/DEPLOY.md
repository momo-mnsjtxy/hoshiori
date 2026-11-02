# 部署

Compose 启动网关（宿主机 127.0.0.1:8080）和认证面板（127.0.0.1:8765），两者都以容器内 uid 10001 运行。
默认网关明确使用 `echo` 后端，供接口连通性验收；模型推理需将 command 改成 `--backend cmd --answer-cmd ...`，并把推理程序安装到镜像中。

## 准备与启动

```bash
mkdir -p data/bolt data/model
# 复制 ori weave 生成的 card.json/bolt.jsonl 等产物到 data/bolt。
# 容器用户要能写网关日志与面板元数据，请按宿主机情况设置 data 的属主/ACL。
sudo chown -R 10001:10001 data
# 在 .env 中设置两项不同的随机令牌（不要提交 .env）：
# ORI_TOKEN=...
# ORI_LOOM_TOKEN=...
docker compose build
docker compose up -d --wait
docker compose ps
```

`ORI_TOKEN` 与 `ORI_LOOM_TOKEN` 都必填。网关从环境变量读取令牌；显式 `--token` 可以覆盖。
`ORI_DATA_DIR` 可将绑定目录从 `./data` 改到别处。目录需提前存在，面板需要有效 `card.json`。

浏览器打开面板后，用户名任意，密码填写 `ORI_LOOM_TOKEN`。首页可编辑数据集名称、许可与描述。
HTTP API 使用 `Authorization: Bearer <对应令牌>`；未认证的页面、指标和模型清单均返回 401。
数据卷可写，只有通过认证的卡片元数据写操作会修改 card.json；样本统计和指纹不能经该接口改写。

健康检查实际请求带认证的 `/v1/models` 和面板首页。面板等待网关健康才启动。
`ori doctor` 会检查可选训练依赖，不能用来判断一个轻量 HTTP 镜像是否健康，因此不再作为容器探针。

## 镜像范围

默认镜像只装网关与面板所需基本依赖。可按需增加训练、HF 题库、ONNX 能力：

```bash
docker build --build-arg ORI_EXTRAS=full,eval,export -t hoshiori:full .
```

`full` 包含 torch、NumPy、blake3、safetensors、Pillow、tokenizers；`eval` 增加 datasets；`export` 增加 ONNX/ONNX Runtime。
默认镜像不包含 llama.cpp、ollama 或 GPU 驱动。构建不会把 .env、数据、仓库历史或本机工具配置复制进镜像。

## 反向代理

宿主机映射默认只绑定 loopback。向公网提供访问时，用 HTTPS 反代并保留认证：

```caddy
ori.example.com {
    reverse_proxy 127.0.0.1:8080
}
loom.example.com {
    reverse_proxy 127.0.0.1:8765
}
```

网关请求日志位于数据目录的 `gateway.jsonl`，可交给宿主机 logrotate 或日志采集器。
关闭服务用 `docker compose down`；它保留绑定目录中的模型、样本和卡片。

## 无容器运行

```bash
ori serve data/model --backend echo --port 8080 --log data/gateway.jsonl
ori loom data/bolt --port 8765
```

同样从 `ORI_TOKEN` 和 `ORI_LOOM_TOKEN` 读取令牌。面板没有令牌时仅允许本机只读，公网监听会拒绝。
