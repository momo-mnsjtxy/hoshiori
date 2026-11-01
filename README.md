# 星织 Hoshiori

数据是线，模型是布 —— 织布的人该拿到梭子。

从语料到上线的全链路训练与评测台。收数据、洗、去重、打指纹出卡 → 训练 → 评测 → 导出 GGUF/ONNX/MLX → 起 OpenAI 兼容服务。每段都带产物，都能单独跑。

**线在你手里。**

---

## 现在能跑什么

| 能力 | 状态 |
| --- | --- |
| `ori weave`：收 JSONL/TXT/目录/stdin/链接，规范化、MinHash 去重（索引可落盘分片）、指纹、数据集卡 | 可用（百万级语料过百万条后转冷慢，见下） |
| `ori spin`：真训练，七种任务族（sft/pretrain/distill/dpo/grpo/ppo/rloo）、LoRA、nf4/int8、ddp/fsdp2、健康停止、断点续训 | 可用 |
| `ori import lmlab`：把 lmlab 的 yaml 转成配方，逐条差异清单 | 可用 |
| `ori validate`：配方校验，报错指到字段并给候选 | 可用 |
| `ori judge`：跑评测套件、三格式报告、对基线走回归门，掉分退出 1 | 可用 |
| `ori judge --import-lm-eval`：把 lm-eval 任务导成自己的套件 | 可用 |
| `ori forge`：safetensors → GGUF（f16/q8_0/q6_k/q5_k_s/q5_k_m 含 _M 抬升）、词表导出、LoRA 合并、onnx | 可用（mlx 待 macOS 复验） |
| `ori serve`：OpenAI 兼容网关，流式、限流、请求日志、令牌鉴权、真分词 usage | 可用（后端 echo / 外部命令） |
| `ori loom`：只读面板，五页看卡、样本、拒收、进度、对比 | 可用 |
| `ori spin --dry-run`：训练计划表与显存估算 | 可用 |
| `ori doctor`：环境自检 | 可用 |

## 装

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .
ori doctor
```

只用标准库也能跑，`numpy` 让去重快一档，`blake3` 换指纹算法时再装。

## 八分钟跑通第一条数据

```bash
# 1. 收一份语料，去重、出卡
ori weave --in 你的数据.jsonl --out bolt --name zh-mix --license cc0

# 2. 写配方：extends 继承、白名单校验
ori validate sft.warp.yaml

# 3. 学一个词表，再出账看显存放不放得下
ori vocab --in bolt/bolt.jsonl --out vocab.json --size 32000
ori spin sft.warp.yaml --dry-run

# 4. 从别家迁过来
ori import lmlab ../configs/sft_lora.yaml --out imported

# 5. 评测，跟基线比
ori judge sft.warp.yaml --suite zh-lite --answer-cmd 'ollama run {model} {prompt}' --baseline out/base/report.json

# 6. 导出、起服务、看面板
ori forge sft.warp.yaml --out out/gguf --format gguf --quant q8_0
ori serve out/merged --port 8080
ori loom bolt
```

`weave` 出来四样东西，都在 `--out` 里：`bolt.jsonl`（收下的正文）、`rejects.jsonl`（拒收的带原因码）、`fingerprint.json`（整份指纹）、`card.json`（数据集卡）。训练只吃 `bolt.jsonl`。

## 目录

```
main.py              入口：认命令、接错误、按类型给退出码
commands/            一个命令一个文件
  weave.py           收语料
  import_lmlab.py    迁外部配置
  validate.py        验配方
  spin.py            训练前出账
  judge.py           跑评测与回归门
  forge.py           导出与量化
  serve.py           起兼容网关
  loom.py            起面板
  doctor.py          自检
features/            主体，平铺
  sample.py          一条样本：认正文、收干净、拒收带原因
  duplicate.py       判重：视图、指纹、分带、回算
  source.py          来源：文件/目录/stdin/链接
  fingerprint.py     整份数据的指纹
  dataset.py         产物目录与卡片，只有它写这些文件
  recipe.py          配方读取、extends 展开、校验
  plan.py            训练计划表
  migration.py       lmlab 预设转配方
  woof.py            混配比例
  suite.py           评测套件装载
  scorer.py          判分规则
  report.py          评测报告三格式
  gate.py            回归门
  forge.py           GGUF 写入与量化
  merge.py           adapter 合并
  gateway.py         兼容网关
  loom.py            只读面板
utils/               跨主体的小工具
  fail.py            带类型和退出码的错误
  logs.py            终端与 loom.jsonl
  yaml_io.py         yaml 读写
tests/               测试
examples/data/       示例语料
docs/                命名与 PRD
```

`features/` 里加一个主体就是加一个文件。`commands/` 里加一个命令就在 `main.py` 的 `COMMANDS` 里加一行。

## 判重怎么工作

三步，缺一步就废。先用视图剔掉空白和标点（`view`），再取 5 字窗口压成 128 位指纹（`sign`），最后按段比：任意一段全等是候选，候选回算真实重合度，过 `--threshold` 才算重复。

`--threshold 0.8` 是常规档。段内行数会按阈值自动推，阈值越低行数越小，否则真重复撞不上同一段。判掉的都在 `rejects.jsonl` 里带 `duplicate` 原因，`--keep-duplicates` 可以让它们留在 `bolt.jsonl` 并标上 `dup_of`。

## 已知慢点

去重索引在百万条以上、而且开了 `--store` 的时候，温层转冷那一段会明显拖慢落盘速度（一百一十万条要四十分钟以上）。二十万以内没有影响，内存比在 11.9 上下。转冷是后台线程，主流程不被它挡住，慢的是总时长。

## 退出码

| 码 | 含义 |
| --- | --- |
| 0 | 成了 |
| 1 | 程序故障，或迁移后仍缺字段 |
| 2 | 输入写得不对，改参数 |
| 3 | 配方不合规，报错里有字段位置 |
| 4 | 机器上缺东西，先装 |

## 许可

Apache-2.0。文档 CC-BY-4.0。
