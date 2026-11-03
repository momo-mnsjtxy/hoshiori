> 主推：**星织 / Hoshiori**（`hoshiori`）
> 一句话：数据是线，模型是布。织布的人，本来就该拿到梭子。

## 名字为什么是它

「星织」= 星を織る，把散落的星点织成整匹布。
训练框架干的事就是这个：一堆碎语料 → 一条条样本 → 一遍遍迭代 → 一匹能用的织物。
星是原始数据，织是训练管线，布是产物。三层都落在语义里，不是硬贴的二次元皮。

二次元侧的味：织女、机织、梭子、线轴、星屑。视觉天然——深夜织机、发光经线、指尖缠线。
技术侧的味：`ori` 一行命令的短促感、`weave`/`spin`/`forge` 的动词干净、CLI 念起来像口癖。

对标 git 那种名字的运气：短、能当二进制名、能在句子里当动词用（"你 ori 一下就跑起来了"）。

## 生态命名全套（一次定死，别中途改）

| 面向 | 名字 | 说明 |
| --- | --- | --- |
| 仓库 / PyPI / 包名 | `hoshiori` | 全小写，无连字符，包导入即 `import hoshiori` |
| CLI 二进制 | `ori` | 三字母，和 `git`/`uv` 同级的手感 |
| 数据管线 | `ori weave` | 清洗、去重、混配、打指纹，全部走这一支 |
| 训练入口 | `ori spin` | 预训练/SFT/蒸馏/对齐共用，靠 recipe 分派 |
| 评测入口 | `ori judge` | 基准、Elo 擂台、对照报告 |
| 导出入口 | `ori forge` | GGUF / ONNX / MLX / safetensors / 量化 |
| 一键迁移 | `ori import lmlab` | 吃他们的 yaml，吐我们的 recipe |
| Recipe 文件 | `*.warp.yaml` | warp=经线。配 `woof.yaml`（纬线）做覆盖层 |
| 仪表盘 | **Loom**（织机） | 独立子命令 `ori loom`，也吃 `loom.toml` |
| 插件 SDK | **Shuttle**（梭子） | `pip install hoshiori-shuttle`，插件作者=织手 Weaver |
| 模型卡前缀 | `Sagimono-` | 織物。`Sagimono-3B-SFT-v1` |
| 数据集卡前缀 | `Itoguchi-` | 糸口。带血缘与配方哈希 |
| 版本代号 | 花名表：藤 / 麻 / 絹 / 木綿 / 綾 / 錦 | 0.1 藤，0.5 絹，1.0 錦 |
| 默认工作区 | `~/.hoshiori/` | 日志 `loom.log`，产物 `bolt/`（匹） |

## Slogan

- 中文主标语：**线在你手里。**
- 英文：**The thread is yours.**
- 副标语（README 第二行）：`数据是线，模型是布 —— 织布的人该拿到梭子。`
- 挑衅式长标语（宣传物料用）：`别人的实验室把线锁在柜子里，我们把织机搬到大街上。`

## 视觉调性

- 主色：深夜底 `#0B0A12`，星屑 `#F2E9FF`，经线亮 `#C9A6FF`，织机铜 `#C98B5E`
- 图形：单一发光经线穿过织机筘齿，筘齿打散成 token 方块
- logo 一版：`おり` 两字横排 + 筘齿竖纹底，负空间藏一个 `o`
- 不画萌妹。星织的二次元感走"深夜工坊"，不走"立绘小姐姐"，避免和目标产品的文档气质撞车
- README 顶图：织机运转长曝光，线是唯一光源

## 域名 / 商标占位（发布前 5 分钟核一遍）

- `hoshiori.dev` → 文档站
- `hoshiori.ai` → 落地页 + Elo 擂台
- `get.hoshiori.dev/install.sh` → 安装器
- PyPI `hoshiori`（可用，先占位）
- npm `@hoshiori/loom`（scope 可用，先占位）
- GitHub 仓库 `momo-mnsjtxy/hoshiori`（已建，私有待发布）；`github.com/hoshiori` 已被一个零仓库个人号占据，故组织名改走 `hoshiori-labs`
- 名称撞车备选（真被占了就换，机制不变）：`tsumugi`（紡ぎ，纺）、`shizuku`（雫）、`kagari`（篝）
- `ori` 若撞系统命令，退 `hshi` 或 `ori-cli`；别名永远保留 `ori`

## 备选名（同机制，不同味型）

1. **紡 / Tsumugi** —— 纺线。最贴"从原始数据纺出样本"这一层
2. **雫 / Shizuku** —— 水滴。冷、干净、适合做评测向子项目
3. **篝 / Kagari** —— 篝火。最燃，适合宣传物料
4. **綾 / Aya** —— 斜纹织物。最文气，适合做文档与规范子项目
5. **機 / Hakari** —— 直白的"织机"。最不二次元，最好懂
6. **糸屋 / Itoya** —— 线屋。像老铺子，适合做数据集市集子项目

不选的理由都记在这：tsumugi 太长、shizuku 太软、kagari 撞二次元作品太多、aya 太短且重名极多、hakari 没味、itoya 只适合做子品牌。
