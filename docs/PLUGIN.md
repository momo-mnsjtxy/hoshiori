# 插件

四类扩展点，放一个文件就生效，删掉就消失。作者不用注册、不用改仓库任何别的文件。

## 四类位置

| 目录 | 干什么 | `run` 的签名 |
| --- | --- | --- |
| `plugins/sources/` | 数据来源 | `run(path) -> list[str]` |
| `plugins/metrics/` | 判分指标 | `run(answer, reference) -> float` |
| `plugins/rewards/` | 奖励函数 | `run(answer, **opts) -> float` |
| `plugins/exporters/` | 报告导出 | `run(report, path) -> str` |

## 一个插件长什么样

每个文件必须有三个模块级名字：`NAME`（不许和文件名冲突）、`SUMMARY`（一句话）、`run`。抄一个最简单的：

```python
"""按行读一个 UTF-8 文本文件。"""

NAME = "lines"                     # 调用的名字
SUMMARY = "按行读取一个文本文件"    # 列表里显示的那句

def run(path: str) -> list:
    """按行返回，空行丢掉。"""
    with open(path, encoding="utf-8") as handle:
        return [one.rstrip("\n") for one in handle if one.strip()]
```

## 用

```bash
ori plugin list                                     # 四类都列出来
ori plugin run metrics char_hit --args '{"answer":"星织","reference":"星织布"}'
ori plugin run sources lines --args '{"path":"README.md"}'
```

`--args` 必须是 JSON 对象，键就是 `run` 的参数名；解析不了退出 2。

## 坏的插件不会连累别人

导入失败、缺 `NAME`/`SUMMARY`/`run`、签名不对，都只打一条警告然后跳过，`list` 里看不到它，别的插件照跑。想确认某个插件自己挂没挂，直接 `ori plugin run` 它一次。

## 边界

- 奖励插件在训练里是隔离子进程跑的（见 `features/sandbox.py` 那条路），写崩了只影响那一步。
- 插件不要改项目里的文件，只读参数、只回值。
- 名字用小写加下划线，和文件名一致。

## 奖励代码的信任边界

`features.sandbox.run` **不是安全边界**。`python -I`、独立进程组、内存与时间限制用于控制普通故障，不阻止恶意代码访问文件、环境变量或网络。不要把第三方上传的 Python 直接交给奖励沙箱。只运行你信任的奖励文件；不可信代码应放在另外配置的无网络容器或虚拟机中，移除凭据，只挂载必需的只读输入。子进程创建新会话也可能逃离进程组清理，当前机制不保证终止恶意后代。
