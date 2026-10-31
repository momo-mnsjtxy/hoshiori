# 架构

四段一条线：`weave` → `spin` → `judge` → `forge`/`serve`。每段吃上一段的产物路径，产出自己的产物，段与段之间只走文件，不走内存里的隐式状态。

## 为什么这么切

每段的失败方式不同，混在一起就互相拖：数据脏会污染训练，训练崩会丢评测，评测慢会卡导出。分开以后每段能单独重跑，前面那段坏了不用从头收数据。

## 调用方向

```
main.py                   入口：认命令、接错误、给退出码
  └ commands/*.py          一个动作一个文件：检查输入、调主体、拼结果
      └ features/*.py      主体：自己那件事的全部逻辑与数据
          └ utils/*.py     跨主体的小工具：报错、日志、yaml
```

依赖只能向下，同层之间按需互调，都是普通函数调用，没有回调、没有订阅、没有字符串分发。

## 文件职责

| 文件 | 管什么 | 谁掉数据 |
| --- | --- | --- |
| features/source.py | 按地址吐原始记录 | 无 |
| features/sample.py | 一条样本的规范化与拒收 | 无 |
| features/duplicate.py | 重复判定索引 | 内存中的签名与窗口 |
| features/fingerprint.py | 整份数据的摘要 | 无 |
| features/dataset.py | 产物目录的全部写入 | bolt.jsonl、rejects.jsonl、card.json、fingerprint.json |
| features/recipe.py | 配方读取、继承展开、校验 | 无 |
| features/plan.py | 训练计划与显存估算 | 无 |
| features/migration.py | 外部配置转配方 | 无 |
| features/woof.py | 混配比例 | 无 |
| features/forge.py | 权重转换与量化 | 产物文件 |
| features/merge.py | adapter 合并 | 无 |
| features/suite.py | 评测套件装载 | 无 |
| features/scorer.py | 判分规则 | 无 |
| features/report.py | 评测报告三格式 | 报告文件 |
| features/gate.py | 回归门比对 | 无 |
| features/gateway.py | OpenAI 兼容服务 | 请求日志 |
| features/loom.py | 只读面板 | 无 |
| utils/fail.py | 带类型和退出码的错误 | 无 |
| utils/logs.py | 终端输出与 loom.jsonl | loom.jsonl |
| utils/yaml_io.py | yaml 读写 | 无 |

规则：一份数据只由一张文件写。产物只有 dataset 与 report 碰，日志只有 logs 碰。

## 失败怎么走

只有 main.py 接错误。`utils/fail.py` 抛的 Failure 带 kind，退出码按 kind 定：输入 2、配方 3、缺依赖 4、程序故障 1。业务里出现拒收（数据不合格）不算失败，写进 rejects.jsonl 继续跑。

## 边界

检查只做一次，就在命令层的开头：参数在不在、范围对不对。过了这一层，内部函数直接信任入参，不查类型、不判空、不铺 try。

## 后端

训练、转换、网关各自留一个可换的后端函数。眼下网关与评测都支持"外部命令"后端，不把任何模型库钉进依赖。

## 不做的事

- 不做插件热加载，加功能就是加文件、加一行注册。
- 不做事件总线，进度就是三个直调函数。
- 不做基类继承，共用逻辑靠 utils 里的函数。
