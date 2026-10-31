# 星织 Hoshiori

数据是线，模型是布 —— 织布的人该拿到梭子。

从语料到上线的全链路训练与评测台。收数据、洗、去重、打指纹出卡 → 训练 → 评测 → 导出 GGUF/ONNX/MLX → 起 OpenAI 兼容服务。每段都带产物，都能单独跑。

**线在你手里。**

---

## 现在能跑什么

| 能力 | 状态 |
| --- | --- |
| `ori weave`：收 JSONL/TXT/目录/stdin/链接，规范化、MinHash 去重、指纹、数据集卡 | 可用 |
| `ori import lmlab`：把 lmlab 的 yaml 转成配方，逐条差异清单 | 可用 |
| `ori validate`：配方校验，报错指到字段并给候选 | 可用 |
| `ori spin --dry-run`：训练计划表与显存估算 | 可用（实跑接训练后端） |
| `ori doctor`：环境自检 | 可用 |
| `ori judge` / `forge` / `serve` / `loom` | 参数面已定，后端接入中 |

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

# 3. 出账，看显存放不放得下
ori spin sft.warp.yaml --dry-run

# 4. 从别家迁过来
ori import lmlab ../configs/sft_lora.yaml --out imported
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
