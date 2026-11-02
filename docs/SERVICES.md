# 评测、迁移与本地服务

这些入口保留原来的目录和文件格式。新增选项保持旧配方可用；答题命令现按独立参数执行，原来依赖隐式 shell 管道的模板需要改为脚本（见下文）。

## HF 题库与 few-shot

安装 `pip install '.[eval]'` 后，`ori judge --import-lm-eval task.yaml` 支持 Hub 数据集：

```yaml
task: hub-demo-one-shot
dataset_path: lhoestq/demo1
test_split: train[:3]
fewshot_split: train[3:5]
num_fewshot: 1
fewshot_seed: 17
doc_to_text: 'Review: {{review}} Stars:'
doc_to_target: '{{star}}'
metric_list: [exact_match]
```

评分 split 优先 `test_split`、其次 `validation_split`、再 `training_split`，默认 `test`。
`dataset_name` 选择子配置，`dataset_kwargs.revision` 可以固定数据版本；Hub 鉴权沿用 datasets/HF 自己的配置。
本地 JSONL 不依赖 datasets；多文件配置和 Hub 由 datasets 装载。

示范池优先 `fewshot_config.samples` 或顶层 `fewshot_samples/fewshot`，其次 `fewshot_split/training_split`。
没有独立示范池时从评分池抽，但每题都排除相同题面。示范不足明确报错，不能偷偷少拼或泄露当前题答案。
`target_delimiter` 默认空格，`fewshot_delimiter` 默认两个换行；前缀确实写进导出套件的 prompt，所以后端会收到它。

这仍是星织轻量导入器：模板支持 `{{字段}}` 和 lower/upper/strip，不等于完整 lm-evaluation-harness。
未知 Jinja 语句、指标会进入导入报告，不声称复现它们。
实现依据：[HF Datasets 装载文档](https://huggingface.co/docs/datasets/loading)。

## 答题命令的参数边界

`ori judge --answer-cmd`、擂台两侧后端和网关 `--answer-cmd` 使用同一规则：先用 `shlex.split` 解析可信命令模板，再在每个参数内一次性替换 `{prompt}`，最终 `subprocess.run(..., shell=False)`。网关还会替换 `{model}`；judge 和擂台需要在模板中写明模型名。引号、换行、`$()` 和反引号属于题面数据，不能改变已确定的参数边界，也不会被系统 shell 求值。

可以继续使用这些模板：

```text
ollama run my-model {prompt}
echo {prompt}
python backend.py --model {model} --prompt {prompt}
python -c 'import sys; print(sys.argv[1])' {prompt}
```

命令模板不再隐式支持管道、重定向、shell 变量或 `&&`。复杂后端请写成固定脚本，使用 `sys.argv` / 脚本位置参数接收独立输入。不要使用 `python -c 'print("{prompt}")'` 或 `sh -c '{prompt}'` 这类把题面变成程序的模板；直接 Python/shell 的这类 `-c` 插值会明确拒绝。固定 `-c` 程序加独立题面参数仍可用。

可执行程序与脚本必须由运营者控制。这里保证的是数据不经系统 shell 解释，不审计任意执行器、包装命令或脚本内部逻辑；后端脚本不得再次 `eval` 或执行输入参数。外部数据中的 `{model}` 等字面字符串不会触发第二次占位替换。

## Axolotl 与 LLaMA-Factory

```bash
ori import axolotl train.yaml --out imported
ori import llama-factory train.yaml --out imported
ori validate imported/train.warp.yaml
```

两家均映射基座、批量、梯度累积、学习率、epoch、精度、LoRA 参数等。Axolotl 读取 `datasets[].path`；
LLaMA-Factory 按 `dataset_dir/dataset_info.json` 将 `dataset` 名称解析为本地文件。
相对语料路径按原配置文件的位置解析，迁出后改成绝对路径；原文件不会被改写。

迁移同时写 `.warp.yaml` 和逐字段 `.diff.md`。缺基座、缺本地数据、未知训练阶段会返回 1。
未支持的模板/数据字段转换、调度器和其他选项原值进入差异清单；复杂 Alpaca/ShareGPT 模板需要先转换成星织的 text/messages JSONL，不能仅凭校验通过就宣称训练语义完全等价。
映射依据：[Axolotl 配置](https://docs.axolotl.ai/docs/config-reference.html)、[LLaMA-Factory 数据注册](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html)。

## MCP 服务端

```bash
ori mcp serve --root /absolute/workspace
```

使用 stdio，一行一帧 JSON-RPC；客户端可以配置 `command: ori`、`args: [mcp, serve, --root, /absolute/workspace]`。
支持 initialize/initialized、ping、tools/list、tools/call；不把启动日志混进 stdout。

工具 `read_card` 读取根内 card.json；`validate_recipe` 校验根内完整配方；`list_suites` 返回内置套件名。
前两个工具参数是 `{"path":"相对路径"}`，第三个是 `{}`。路径逃逸和符号链接越界都拒绝。
配方校验不展开 extends，避免隐式访问根外文件；需要预先展开为完整配方。
协议依据：[MCP 2025-11-25 生命周期](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)。

## 认证面板

设置 `ORI_LOOM_TOKEN` 后运行 `ori loom bolt`，所有页面与指标接口都需要认证。
浏览器输入任意用户名、令牌作密码；API 使用 `Authorization: Bearer <token>`。
`--token-env` 可改环境变量名。未配置令牌时只允许本机只读模式，对外监听会提前报错。

首页支持修改名称、许可、描述，对应 `POST /api/card` 的 JSON 字段 `name/license/description`。
其他字段不能修改，避免伪造语料指纹与统计；更新采用锁和原子替换。写操作始终需要令牌，跨源写、非 JSON 表单和超大请求会拒绝。
反向代理向公网提供服务时应配置 HTTPS，示例见 [DEPLOY.md](DEPLOY.md)。

## 擂台参数和停止

```bash
ori judge model.warp.yaml --suite zh-lite --answer-cmd 'a {prompt}' --arena 'b {prompt}' \
  --elo-k 24 --elo-initial 1000 --arena-min-matches 30 --arena-window 10 --arena-tolerance 0.5
```

默认 K=32、初始 1500、容差 0（不早停）。开启容差后，至少完成 min-matches 场，且最近 window 场的 A 方 Elo 最大值与最小值之差不超过容差就停止。
报告记录全部参数、实际对局、是否稳定和停止原因。这是分数稳定准则，不是统计显著性或排名置信区间。

## 本轮验证记录（2026-11-02）

- 服务相关回归 101 项通过，覆盖真实 MCP 子进程握手/工具调用、面板 HTTP 登录与写入、网关环境令牌、两框架 CLI 迁移、few-shot 与早停；记录：[tests.log](evidence/services/tests.log)。随后新增“缺 test split 不得回退 train”回归，与非 HTTP 服务组共 75 项通过。
- HF Hub 真下载 `lhoestq/demo1`，固定实测 revision `87ecf163bedca9d80598b528940a9c4f99e14c11`，3 个评分样本、独立 2 条示范池、1-shot，逐题核对提示词含一条示范；记录：[hf-import.json](evidence/services/hf-import.json)。离线 mock 测试只用于边界行为，未替代此次联网结果。
- Docker 28.5.1、Compose 2.40.3 实际构建并启动两项服务；记录：[构建日志](evidence/services/docker-build.log)、[镜像标识](evidence/services/docker-image.txt)、[启动日志](evidence/services/docker-up.log)、[健康状态](evidence/services/docker-ps.json)。最终镜像标识及大小以记录文件为准；版本命令与 MCP 子命令在已安装 wheel 内正常运行。
- HTTP 实测匿名访问两服务都是 401，认证后模型清单/面板 200，echo chat 200，面板卡片编辑 200 且绑定目录读回新值；记录：[docker-run.json](evidence/services/docker-run.json)。这是服务栈验收，echo 不计作模型推理验收。

最终镜像已包含外部命令参数隔离修复；修复后的全量回归 **451 passed / 1 deselected**（14 条 ONNX 弃用警告），见 [全量日志](evidence/integration.log)。最终 HTTP 记录时间晚于镜像构建，摘要见 [validation.json](evidence/services/validation.json)。

源码哈希见 [source-hashes.json](evidence/services/source-hashes.json)，后续变更应重新跑受影响的验证。运行用随机令牌未进入仓库证据。
