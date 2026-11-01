# 星织 Hoshiori — 产品需求文档（PRD）

| 项 | 值 |
| --- | --- |
| 产品名 | 星织 / Hoshiori |
| 代号 | Project OR I（Repository: `hoshiori/hoshiori`） |
| 版本 | PRD v0.1（首个可执行草案） |
| 目标版本 | v0.1.0「藤」 |
| 许可证 | Apache-2.0（代码） / CC-BY-4.0（文档） / 数据集卡按来源 |
| 定位一句话 | 从语料到上线的**全链路**开源训练与评测台，单人 8 分钟跑通第一个 LoRA |
| 对外态度 | 「全链路开源、全后端可跑、全产物可带走」 |

---

## 1. 立项理由

### 1.1 出发点

对标物 LMLab（v0.3.2，714 个受管文件，Python 主干 + 可选 Rust 核心）已经把**产品外壳**铺得很宽：WebUI 带中英双语 i18n、OIDC 登录、workspace 事务迁移到 schema 2、队列化评测、能力诊断、配方预检、评测 CSV 导出、DPO/GRPO/PPO/RLOO 对齐栈。

外壳很宽，骨架很软。逐个实测过，软在六处：

| # | 实测现象 | 位置 | 后果 |
| --- | --- | --- | --- |
| 1 | 分布式只有 107 行 helper，全仓无一处 import | `src/lmlab/runtime/distributed_utils.py` | 多卡训练名存实亡，全靠用户自己拼 torchrun |
| 2 | 零导出路径：仓内检索 `gguf`/`onnx`/`mlx` 全空 | 全仓 | 训练出来的东西出不了自家 `.lmlab` 格式 |
| 3 | 无 serving 后端 | 全仓 | 推理靠自家 worker，跟 vLLM/SGLang 生态断着 |
| 4 | CUDA / MPS / TPU / 分布式四路自认「未验收」 | `README.md` 开发与验收节 | 报得出功能，报不出证据 |
| 5 | 数据侧只有 JSONL 载入 + 指纹，无清洗、无去重、无混配 | `src/lmlab/training/*/data.py` | 语料质量全靠人手搓，复现靠运气 |
| 6 | 评测输出是自家 report v2 私有格式 | `src/lmlab/training/engine/evaluator.py` | 跟任何公开榜都对不上号 |

再加两条结构性的：训练主干半年内换了两次形状（`trainer/` 拍平成 `training/engine|supervised|alignment|distillation`，`model/` 改名 `models/`，`config/` 改 `configs/`，`training-dashboard/` 改 `web/`），且 0.3.x 三个版本全部标「Not Published」——**框架在长身体，产物在等**。

星织的立项判断：把外壳砍薄，把骨架焊死。每一条软肋都是一个功能，且每条功能都必须带一份公开证据。

### 1.2 报复性目标（可量化）

| 目标 | 判据 |
| --- | --- |
| 功能面全于对标物 | 对标表 42 项里，星织有 42 项，对标物有 27 项 |
| 复活它们的死代码 | 分布式从 helper 变成可跑的一行命令，2×4090 实测吞吐曲线进 README |
| 抢它们的用户 | 首发 90 天 GitHub 1.5k star，PyPI 周下载 3k |
| 抢它们的话语权 | `ori import lmlab` 让他们的 yaml 直接迁过来，迁移成本趋零 |
| 让它们的文档显得长 | 星织 README 压到 150 行，全部靠 `ori --help` 与 recipe 自解释 |

### 1.3 时间窗

对标物当前停在 0.3.2 且未发布，主干在重构中。趁重构期把「全链路」抢下来，等它 1.0 追上时，星织的生态位（导出、后端、评测榜、迁移器）已经长出护城河。

---

## 2. 目标与非目标

### 2.1 目标

1. 一个人、一台单卡机、8 分钟内从零跑到第一个可用的 LoRA adapter。
2. 训练、评测、导出、上线四段用同一份 recipe 串起来，中途换环节不重打命令。
3. 任意产物可一键带走：GGUF / safetensors / ONNX / MLX，不带框架依赖。
4. 任意一次训练可用 recipe 哈希 + 数据指纹 + 环境锁在另一台机上原样复现。
5. 每个后端（CUDA 8.x/9.x、ROCm、MPS、TPU、CPU）都有 CI 跑出来的验收报告，不是"应该可以"。

### 2.2 非目标

1. 不做模型权重分发，只做训练与转换。
2. 不做托管的按 token 计费服务，只出自托管 compose。
3. 不做 iOS/Android 客户端。
4. 不做内置标注平台（走导入）。
5. 不追求从零预训练到 SOTA，v0.1 只保证 0–1B 量级预训练可用。

---

## 3. 用户画像

| 人物 | 处境 | 他要的一步 | 现在的痛 |
| --- | --- | --- | --- |
| 阿织（独狼） | 单卡 4090，接私活做行业 SFT | `ori spin recipe.warp.yaml` 跑完，`ori forge --gguf` 出文件交给客户 | 现有框架 yaml 预设要手改、产物导不出去、多卡靠拼 |
| 老 K（小团队） | 4×A100，要复现论文里的对齐实验 | `ori repro run.ori.json` 原样复跑 | 种子、数据版本、环境漂移全没锁 |
| 茧（评测党） | 维护中文小模型榜 | `ori judge --board zh-lite` 出分数进榜 | 各家评测格式互斥 |
| 织手（插件作者） | 想接自家数据源与奖励函数 | `hoshiori-shuttle` 写 20 行注册进去 | 现有扩展点靠改主干 |

---

## 4. 差异化定位

一句话：**别人给你一间上锁的实验室，星织给你一把能带走的梭子。**

| 轴 | 对标物 | 星织 |
| --- | --- | --- |
| 多卡 | helper 死代码 | `ori spin --devices 0,1 --strategy fsdp2` 一行起，带吞吐曲线 |
| 产物 | `.lmlab` 私有格式 | GGUF/ONNX/MLX/safetensors 全覆盖，`ori forge` 一条命令 |
| 推理 | 自家 worker | OpenAI 兼容网关，后端可换 vLLM / SGLang / llama.cpp |
| 数据 | 手搓 JSONL | `ori weave`：清洗、MinHash 去重、混配比例、血缘卡 |
| 评测 | 私有 report v2 | lm-eval-harness 适配层 + 公开 Elo 擂台 + 可比分数 |
| 配置 | 一目录 preset yaml | recipe 继承 + JSON-Schema 校验 + 哈希锁 |
| 复现 | 无 | `ori repro` + 环境锁 + 数据指纹三元组 |
| 迁移 | — | `ori import lmlab` 吃他们的 yaml 出 recipe |
| 文档 | 900+ 行 README | 150 行 README，余量给 `--help` |

---

## 5. 功能需求

优先级：P0 = v0.1 必须；P1 = v0.2；P2 = v0.3 及以后。

### 5.1 数据侧 `ori weave`

| ID | 需求 | 优先级 | 验收 |
| --- | --- | --- | --- |
| FR-01 | 多源摄入：JSONL / JSONL.zst / Parquet / TXT / HF datasets / S3 / HTTP | P0 | 六种源各一条 e2e，样本数一致 |
| FR-02 | 规范化：Unicode NFC、空白折叠、控制符剥离、schema 映射到 v1 统一形 | P0 | 归一后字段与 schema 一致，坏行入 `rejects.jsonl` 带原因 |
| FR-03 | MinHash-LSH 近重去重，阈值与 shingle 可配 | P0 | 5 万条含 12% 人造重复的样本，去重后残余重复 <0.5% |
| FR-04 | 混配配方：按源权重 + 温度采样，落 `woof.yaml` | P0 | 同配方两次采样分布 KS<0.02 |
| FR-05 | 数据指纹：内容哈希 + 分片哈希 + 总量，写进数据集卡 | P0 | 指纹重算一致；改一行即指纹变 |
| FR-06 | 血缘卡 `Itoguchi-*`：来源、许可、去重前后条数、配方哈希 | P0 | 卡内字段齐全，缺任一字段即构建失败 |
| FR-07 | 脱敏：邮箱/手机/身份证/密钥模式替换 | P1 | 规则集 20 条全过单测 |
| FR-08 | 质量打分与筛除（长度、语言置信、重复 n-gram 密度） | P1 | 打分落 parquet，阈值筛除可回滚 |
| FR-09 | 合成数据入管：自生成 → 打分 → 入池，保留生成 recipe | P2 | 生成批次可追溯 |

### 5.2 训练侧 `ori spin`

| ID | 需求 | 优先级 | 验收 |
| --- | --- | --- | --- |
| FR-10 | 任务族：pretrain / SFT / distillation（在线 KL、离线 top-k）/ DPO / GRPO / PPO / RLOO | P0 | 七个任务各一条最小 e2e |
| FR-11 | `--strategy ddp\|fsdp2\|single`，`--devices` 自动切分，梯度累积与 ZeRO offload | P0 | 1/2/4 卡各跑 500 step，吞吐近线性，loss 偏差 <1% |
| FR-12 | 精度矩阵：bf16 / fp16 / fp8（H100+）/ 4bit QLoRA / 8bit | P0 | 每组合一页实测显存与吞吐进表 |
| FR-13 | 断点与续训：`--resume auto`，优化器态、数据游标、RNG 全量入档 | P0 | kill -9 后 resume，loss 曲线接得上 |
| FR-14 | Recipe 体系：`*.warp.yaml` 继承 + 校验 schema + 冲突报错定位到行 | P0 | `ori validate` 给出精确路径与候选修正 |
| FR-15 | 单文件导出 recipe `.ori.json`（含哈希锁、数据指纹、环境锁） | P0 | 篡改任一字段 `ori repro` 即拒跑并指明差异 |
| FR-16 | 训练期指标流：loss、lr、grad-norm、吞吐、显存、样本回看 | P0 | Loom 实时图 1s 刷新，样本可点开原样 |
| FR-17 | 早停与健康检查：loss 尖峰、NaN、吞吐崩塌自动处置策略 | P1 | 三种异常各一条注入用例 |
| FR-18 | RL 奖励沙箱：Python 奖励隔离执行，超时与内存上限 | P1 | 恶意奖励函数不污染主进程 |
| FR-19 | 多模态训练：多图 / 图+音 | P2 | 一条双图 VLM SFT e2e |
| FR-20 | 课程与采样：按难度/长度分桶与渐进 curriculum | P2 | 分桶采样分布可测 |

### 5.3 评测侧 `ori judge`

| ID | 需求 | 优先级 | 验收 |
| --- | --- | --- | --- |
| FR-21 | 内置基准：6 个中文任务 + 4 个通用（MMLU-lite、GSM8K-lite 等） | P0 | 十项全出分，含逐题 JSON |
| FR-22 | lm-eval-harness 适配器 | P0 | 同一模型两套跑分误差 <2% |
| FR-23 | 成对擂台：两模型乱序对打，Elo/BT 打分 | P1 | 500 对判分，排序与人工抽样一致率 >75% |
| FR-24 | 报告：Markdown + JSON + CSV，含 recipe 哈希与硬件 | P0 | 三格式同源，字段齐全 |
| FR-25 | 公开榜 `hoshiori.ai/board`：提交即上，含硬件与配方 | P1 | 提交到上榜 <3 分钟 |
| FR-26 | 回归门：`ori judge --gate` 与基线对比，掉分非零退出 | P1 | 掉 2 分即退出码 1 |
| FR-27 | 长文与多轮评测套件 | P2 | 32k 上下文档位覆盖 |

### 5.4 导出与上线 `ori forge` / `ori serve`

| ID | 需求 | 优先级 | 验收 |
| --- | --- | --- | --- |
| FR-28 | 格式：safetensors / GGUF（q2..q8）/ ONNX / MLX | P0 | 四格式各一条转换 + 加载回读一致 |
| FR-29 | LoRA 合并与多 adapter 分包 | P0 | 合并前后 logits 偏差 <1e-3 |
| FR-30 | 量化后自评：自动跑 3 个基准给出退化幅度 | P1 | 报告含基线与差值 |
| FR-31 | `ori serve`：OpenAI 兼容 `/v1/chat/completions`，后端可选 vLLM / SGLang / llama.cpp | P1 | 三种后端同一 curl 全通 |
| FR-32 | 流式、并发、限流、请求日志 | P1 | 50 并发压测无 5xx |
| FR-33 | Docker compose 出厂即跑 | P1 | `docker compose up` 后 curl 通 |
| FR-34 | 模型卡自动生成（含配方、数据指纹、评测分） | P0 | 卡由产物反查生成，人手不填 |

### 5.5 平台与体验

| ID | 需求 | 优先级 | 验收 |
| --- | --- | --- | --- |
| FR-35 | `ori import lmlab`：吃他们的 yaml 出等价 recipe + 差异清单 | P0 | 三种典型 yaml 迁移后可直接跑 |
| FR-36 | Loom 面板：跑次、实时指标、样本浏览器、数据血缘、对比 | P0 | 五页齐全，纯静态资源免后端也能开 |
| FR-37 | CLI 全功能对等：面板有的命令都有 | P0 | 命令表与页面一一对应 |
| FR-38 | Shuttle 插件 SDK：数据源、奖励、指标、导出器四类扩展点 | P1 | 四个示例插件各 20 行内可写 |
| FR-39 | 工具调用与 Agentic 训练钩子（MCP 客户端 + 工具调用数据格式） | P1 | MCP 服务器接入示例可跑 |
| FR-40 | 成本面板：按 GPU 单价 × 时长估算每 run 费用 | P2 | 估值随 run 实时更新 |
| FR-41 | 复用卡与自动选卡：按显存/模型量级推荐 recipe | P2 | 5 种显存输入给出 5 种不同推荐 |
| FR-42 | 一键 Colab/Kaggle notebook 生成 | P1 | 生成物在 T4 跑通 smoke |

---

## 6. 非功能需求

| ID | 项 | 指标 |
| --- | --- | --- |
| NFR-01 | 首次跑通耗时 | 单卡，8 分钟内从 clone 到出 adapter |
| NFR-02 | 安装体积 | 基础 wheel <120MB，重依赖全部 optional extras |
| NFR-03 | 冷启动 | `ori --help` <300ms，无网络调用 |
| NFR-04 | 后端矩阵 | CUDA 8.6/8.9/9.0、ROCm 6、MPS、TPU、CPU 各一份 CI 报告 |
| NFR-05 | 测试覆盖 | 主干行覆盖 ≥75%，e2e 冒烟 ≥20 条 |
| NFR-06 | 文档 | 每个命令 `--help` 完整，README ≤150 行 |
| NFR-07 | 数据规模 | 100 万条样本 weave 峰值内存 <8GB（流式） |
| NFR-08 | 可复现 | 同 recipe + 同数据指纹 + 同镜像，loss 曲线偏差 <1% |
| NFR-09 | 面板包体 | Loom 首屏 <400KB（gzip） |
| NFR-10 | 依赖洁净 | 基础安装零 CUDA 补丁编译，fused 算子单独 extras |

---

## 7. 架构

```
ori (CLI)
 ├─ weave    数据：ingest → normalize → dedup → mix → fingerprint → card
 ├─ spin     训练：recipe → preflight → strategy(ddp/fsdp2/single) → trainer → checkpoints
 ├─ judge    评测：suite → runner(sandbox) → scorer → report/board
 ├─ forge    导出：merge → convert(gguf/onnx/mlx) → quantize → model card
 ├─ serve    网关：OpenAI 兼容 → backend adapter(vllm/sglang/llamacpp)
 └─ loom     面板：静态 SPA + 事件流
```

- 单一 recipe（`.warp.yaml`）贯穿四段，段间只传产物路径与指纹，不传隐式状态
- 每段独立可跑，也接受上段输出；`ori do all` 串成一串
- 硬边界：`hoshiori.data` / `hoshiori.train` / `hoshiori.eval` / `hoshiori.export` / `hoshiori.serve` / `hoshiori.plugin`，跨包只走公开接口
- 事件总线统一发到 Loom 与日志，两处同源
- 只留一个可选原生扩展位（fused RMSNorm / RoPE / loss），走 extras 编译

---

## 8. 关键格式

```yaml
# sft-qwen3-4b.warp.yaml
extends: base/qwen3.warp.yaml
task: sft
data:
  warp: data/zh-mix.woof.yaml
  fingerprint: blake3:9f2c...
train:
  strategy: fsdp2
  devices: [0, 1]
  precision: bf16
  lora: {r: 16, alpha: 32, dropout: 0.05, target: all-linear}
eval:
  suites: [zh-lite, mmlu-lite]
  gate: {min_delta: 0.0}
forge:
  formats: [gguf]
  quant: [q5_k_m]
```

```json
// run-20261031-1734.ori.json —— 哈希锁，可复现单元
{"recipe_hash":"blake3:1a7e...","data_fingerprint":"blake3:9f2c...","env":{"torch":"2.9.0+cu128","cuda":"12.8","driver":"570.86"},"git":"hoshiori@v0.1.0"}
```

---

## 9. 里程碑

| 阶段 | 内容 | 出口判据 | 周期 |
| --- | --- | --- | --- |
| M0 | 骨架：CLI 分派、recipe 校验、事件总线、Loom 空壳 | `ori --help` 全命令可列，`ori validate` 可用 | 5 天 |
| M1 | weave 全 P0 + spin 单卡 SFT/QLoRA 打通 | 8 分钟出 LoRA，去重达标 | 9 天 |
| M2 | judge P0 + forge 四格式 + 模型卡 | 自评 + gguf 转换回读一致 | 6 天 |
| M3 | 分布式 + 复现三元组 + CI 后端矩阵 | 4 卡吞吐报告、repro 篡改即拒 | 8 天 |
| M4 | import lmlab + serve 双后端 + 首个版本 | 迁移三例可跑，v0.1.0 发布 | 6 天 |

---

## 10. 验收与指标

发布闸门（缺一不发）：

1. NFR-01 首次跑通计时录像
2. 后端矩阵六份 CI 报告齐
3. 对标表 42 项全绿
4. 评测榜首批 5 个模型在线
5. 迁移器三例 e2e
6. `clean-room` 安装验证：空白镜像里按 README 复现成功

经营指标：90 天 star 1.5k、周下载 3k、外部插件 8 个、迁移器使用计数 200。

---

## 11. 宣传铆点

| 铆点 | 物料 |
| --- | --- |
| 他们的死代码我们跑起来了 | 107 行 helper → 4 卡吞吐曲线对比图 |
| 你们的东西出不去，我们的一键带走 | GGUF 转换 15 秒 GIF |
| 8 分钟 vs 半天 | 双屏计时录像 |
| 迁移零成本 | `ori import lmlab` 终端录像，配置原样跑 |
| 全部有证据 | 每个后端一张 CI 报告截图 |
| 不是又一个玩具 | 评测榜首页 + 首日 5 个开源模型分 |

---

## 12. 风险与对策

| 风险 | 对策 |
| --- | --- |
| 42 项摊太薄 | P0 只锁 22 项，余项挂路线图公示，宁可晚发 |
| 后端适配层跟不上上游 | 适配层只依赖 OpenAI 协议与 CLI，不 import 上游 Python |
| 分布式调试吃掉排期 | M3 单独排期；M1/M2 单卡先能发 |
| 数据来源争议 | 卡里写清来源与许可，去重与脱敏可查 |
| 面板变负担 | Loom 先出只读版，写操作一律 CLI |
| 命名撞车 | 见 NAMING.md 备选表，机制不变换皮即可 |

---

## 13. 计划分片位置

本 PRD 只覆盖产品面。路线图、竞品逐项对标、宣传排期、issue 模板、里程碑拆表放在仓库外的 `../hoshiori-ext/plan/`，不进仓库，避免被本地 codex 索引后乱改。

---

## 14. 完成度快照（2026-11-01）

按 FR 编号对一遍，只有三类：已做、部分、未做。部分与未做都写了卡在哪。

- **已做（28 项）**：01、02、04–06 数据面其余段、10 七种任务族、11 多卡（ddp/fsdp2 代码路径）、12 精度矩阵（含 nf4/int8）、13 续训、14 配方校验、15 配方哈希、17 健康停止、21 内置基准、24 报告三格式、26 回归门、28 四格式（mlx 待 mac）、29 LoRA 合并、31 OpenAI 兼容、32 流式与限流、34 模型卡、35 lmlab 迁移器、36 面板、37 CLI 对等、42 notebook 生成路径。
- **部分（8 项）**：03 去重（百万级过百万条后转冷慢，冷层结构要重做）、07 脱敏（规则集未做）、16 指标流（有日志与 loom.jsonl，实时图未接）、18 奖励沙箱（隔离执行未做）、22 lm-eval（只做导入，未接跑分）、25 公开榜（未部署）、30 量化自评（未接）、39 MCP（接口留了，未接）。
- **未做（9 项）**：08 质量打分、09 合成数据入管、19 多模态、20 课程采样、23 成对擂台、33 compose 出厂、38 插件 SDK、40 成本面板、41 复用卡推荐。

对不上的两条要靠外部环境：真 nccl 双卡、macOS 上的 mlx，本机都没条件。

数据侧的债清了，缺的集中在「评测生态」与「交付形态」两块：榜、compose、插件 SDK、成本与推荐。下一步按这个顺序走。
