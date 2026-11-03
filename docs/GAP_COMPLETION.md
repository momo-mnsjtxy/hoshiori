# 功能缺口补齐记录

目标来源：2026-11-02 用户附件「功能缺口清单」。所有必须补、建议补和可选深度均在范围内；未验证不视为完成。实现遵守 `~/HOP/SKILL.md`，保留现有主体目录与旧存档读取能力。

| 编号 | 验收目标 | 当前状态与证据 |
| --- | --- | --- |
| 03 | 百万性能达标或前缀分片+带片号位置表 | v5前缀分片+带片号位置表已实现，旧布局可读且显式迁移，见 INDEX.md；110万实测1844.40秒、50.1MiB峰值，未达10分钟，按新布局分支验收；证据 evidence/index-v5-1100k.json |
| 25 | 外网地址与首批5个模型 | 已通过Sites公开部署5个真实模型；[公开地址](https://hoshiori-model-board.mnsjtxy.chatgpt.site)，匿名HTTP已验证5模型，原始报告、响应头及部署ID见 evidence/board |
| 30 | llama.cpp 真跑 q8_0 自评 | 修正分词后真实f16/q8_0跑12题：0.7500→0.6667，退化0.0833；自评门实际退出码1，最终报告和CLI日志见 evidence/quantization |
| 18 | 安全隔离或明确不是安全边界 | sandbox.py 与 PLUGIN.md 已明确不是安全边界，说明文件/网络/环境与后代限制 |
| 19 | 真实视觉塔、跨批负例、检索分 | 已实现卷积视觉塔与跨批负例队列；6色受控留出检索双向R@1达到1.0，原始结果见 evidence/vision-cpu.json；不代表自然图像能力 |
| 10 | 异构教师 | 教师按自己的config加载层数/维度，异构几何回归通过；师生仍须共享词表与分词语义 |
| 12 | fp8及sm_86/sm_89精度真机 | FP8已在5090D实际调用FP8矩阵乘法并完成3→5步续训；NF4/INT8也已真跑；sm86/sm89仍缺硬件，不能标全矩阵通过 |
| 13 | 真双卡NCCL续训 | 已提供 tests/benchmark_multigpu.py，校验两张不同GPU UUID、NCCL及连续/续训权重一致性；现有远端仅1张物理卡，未执行双卡验收 |
| 22 | HF远端数据集与few-shot | 已实际读取Hub的lhoestq/demo1并为3道评分题拼接独立split的1-shot；导入规则与边界见 SERVICES.md |
| 35 | Axolotl、LLaMA-Factory迁移 | 两种迁移入口、字段差异报告与行为测试已完成；未支持的模板语义明确列出，见 SERVICES.md |
| 17 | 健康阈值与eval退化 | 已支持可配阈值、验证集token准确率、掉分耐心与断点状态；行为回归通过 |
| 41 | 复用卡真机校准 | 已真实测量5090D的BF16 batch1/2/4，严格匹配模型/配方，见 COST.md 与 evidence/cost-5090d-profile.json |
| 40 | 真实云价与实测吞吐 | 已接Runpod公共实时价与实测吞吐；5090价不冒用为5090D，见 COST.md 与 evidence/cost-matched-estimate.json |
| 39 | MCP服务端 | stdio JSON-RPC握手、工具清单/调用与工作目录约束已实现并通过协议测试；见 SERVICES.md |
| 36 | 面板鉴权、写操作 | 认证读写、跨源拒绝及元数据原子保存已实现；Compose真服务验证匿名401、认证200和绑定卷写入，见 evidence/services/docker-run.json |
| 08 | 人工标注校准权重 | 已实际用 OASST1 人工 quality 投票校准143条独立对话样本；115训练/28留出，MSE 0.09823→0.08841；证据 evidence/quality/oasst1-profile.json |
| 07 | 地址、姓名、银行卡脱敏 | 标签上下文的姓名/地址、Luhn卡号已实现并通过行为测试；无标签自然语言实体识别不在正则能力范围 |
| 20 | 难度课程 | 支持人工difficulty与词元稀有度排序；weave保留并校验difficulty，课程顺序回归通过 |
| 23 | Elo参数、收敛停止 | K、初始分、最少场数、窗口与容差均可配置，稳定窗口实际停止并写入报告；见 SERVICES.md |
| 33 | Compose真实构建 | 镜像真实构建、两个容器均healthy，鉴权HTTP与绑定卷写入均通过；已安装wheel的版本和MCP入口也可用，见 evidence/services |
| 29 | DoRA、LoHa | 两变体训练、存读、合并已接入，量化基座与BF16挂件回归已补齐 |

## 数据清洗的实际边界

`ori quality fit labels.jsonl --annotator <标注团队或来源> --out quality.json` 接受每行 `text` 与 `score`（0–1）的人工标注。至少10篇独立正文，重复正文在切分前合并打分。固定种子按80/20划分，权重仅由训练集选择；档案保存训练/留出MSE、旧权重的留出MSE、标注来源和数据摘要。生成标签测试只证明程序行为，不能替代人工校准。

`ori weave --in raw.jsonl --quality-profile quality.json --min-score 0.5` 使用该档案。没有档案时继续使用明确的启发式默认值。姓名和地址规则依赖「姓名：」「联系人：」「地址：」等字段标签；银行卡采用13–19位长度和Luhn校验。它们不能保证自由文本里所有人名地址都被识别，也不验证卡号是否真实存在。

## 干净环境复现与录像

- 复现脚本 `scripts/cleanroom.sh`：九步（建 venv 装依赖 → 自检 → 造语料 → 学词表 → 收数据 → 配方校验 → 训练二十步 → 评测 → 导出 GGUF），每步都断言产物，断在哪就退哪一步。
- 本机首次完整跑通三次：5 分 26 秒、5 分 56 秒、全通各一次（八分钟目标内）。干净环境当场暴露三个真 bug 并已修：numpy 未进基础依赖、`spin` 顶层 `import torch`、随机起权重挂 LoRA 时没有基座可合并。
- 录像 `scripts/record-demo.sh`：`script -f` 实时刷盘，输出放在仓库外（本机 `/tmp` 会被清）。GPU 机（5090 D）上跑通一遍完整九步，证据 `docs/evidence/cleanroom/run-demo.typescript.gz`，用 `ORI_USE_SYSTEM=1` 复用已装环境，故这一段不含拉依赖的耗时。
- 本机 pypi 拉 torch 很慢（多次超过十分钟），"含安装的完整首录"仍没录成；换网络或在 GPU 机上先 `pip install -e ".[full]"` 再录可解。

## 当前资源

用户提供的 `python ~/rd.py '<命令>'` 可用。远端实际探测为1张 RTX 5090 D、PyTorch 2.8.0+cu128。本机解除沙箱后也可读到RTX5060 Laptop（sm120），但本地torch是CPU构建，公开榜使用CPU-only llama.cpp。系统 ssh 的密码认证失败不能据此断言服务器不可用；已按用户提供的 Paramiko 通道连接。凭据不进入仓库、日志和产物。

## 人工标注校准实测

使用 [OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1)（Apache-2.0）的公开人工 quality 标签：验证集前1000行中每个对话树取首个非合成、未删除、至少两票的助手消息，共143篇。115篇拟合、28篇独立留出，权重为 length=0.2、repeat=0.58、punctuation=0.22；留出MSE从0.098230859降至0.088409992。该样本是公开多语言对话子集，不能保证在用户中文业务语料上的改进，因此保留默认启发式，使用 `--quality-profile docs/evidence/quality/oasst1-profile.json` 显式选用。下载分页URL、摘要、消息ID和真实误差均在档案中；复现脚本是 `tests/calibrate_quality.py`。

## 量化自评实测

实际运行 llama.cpp b12040，使用批处理入口 `llama-completion`，SmolLM2-135M-Instruct 固定revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`。修正预分词后，`en-smoke` 12题的f16为0.7500、q8_0为0.6667，退化0.0833，超过默认0.05门槛；验收的是检测并报告真实退化，不能把此模型量化标为无损通过。报告保留两边逐题原始答案、运行时、硬件、模型及GGUF摘要。修复了新版交互UI污染标准输出、缺聊天模板、SmolLM数字预分词，以及HF Llama Q/K的RoPE布局转换；后者对照 [llama.cpp 官方转换器](https://github.com/ggml-org/llama.cpp/blob/b12040/conversion/llama.py)，并用旋转后数值等价测试验证。数值/空白/中文emoji三组文本的HF与GGUF token IDs完全相同，见 `evidence/quantization/tokenizer-parity.json`。

## 公开榜

[星织公开榜](https://hoshiori-model-board.mnsjtxy.chatgpt.site) 已由 Sites 部署且访问权限为 public，五个模型的固定revision、12道公开题、逐题答案与硬件信息均在站点。源检出目录 `.work/board-site`，项目ID与已推送commit在 `evidence/board/deployment.json`。使用 `tests/seed_board.py` 重新生成实测榜；不要手工补分数。当前是小规模英语启动题，不能充当通用能力排名或中文能力结论。

## 当前集成回归

最终完整本机回归：`python -m pytest`，451 passed / 1 deselected，14条已有ONNX弃用警告；持久日志 [integration.log](evidence/integration.log)。默认跳过的1项是slow压力用例，百万压测另有独立记录。Compose已实际构建并通过HTTP/鉴权/写卷验收；本机回归仍不能替代真双卡与sm86/sm89硬件验证。

## 外部命令调用的兼容变化

评测、擂台与网关的答题模板现在先解析参数，再填入题面，以 `shell=False` 执行。题面里的引号、换行、命令替换保持数据语义；原先依赖隐式管道或重定向的模板需要改为接收独立参数的后端脚本。固定命令示例继续可用，详见 [服务说明](SERVICES.md#答题命令的参数边界)。这项变更解决了远端题面或HTTP消息被当作shell代码执行的问题。
