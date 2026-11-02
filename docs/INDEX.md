# 去重索引 v5

新建的落盘索引默认采用“按哈希前缀分片 + 位置表带片号”布局。已有 v4 目录仍按原来的三层格式读写，不会自动迁移。纯内存模式保持原来的行为。

```python
from pathlib import Path
from features.index import Index

index = Index(Path("my-index"), key_size=8, per_shard=500_000)
index.put(b"abcdefgh", 7, [1, 2, 3], {4, 5})
assert index.get(b"abcdefgh") == [7]
assert index.record(7) == ([1, 2, 3], {4, 5})
index.save()
index.close()
```

`Duplicate` 与 `ori weave` 无须修改调用方式，新建索引会自动选 v5。

## 数据如何定位

1. 键计算 `BLAKE2b(digest_size=8)`，取最高 4 位作为前缀。
2. 16 个前缀各自对应一张 `keys_00` 到 `keys_0f` 的 SQLite B-tree。`get(key)` 只查询其中一张，按完整键匹配；前缀碰撞不会造成误报。这些逻辑键片存放在同一个 `prefix.sqlite3` 文件中，共享事务与 8 MiB 页缓存，不是 16 份独立数据库。
3. `positions` 表保存 `sub → (shard, offset, length)`。签名与窗口正文顺序追加到 `records-NNNNNN.rec`，默认每 50 万条换一个记录片。`record(sub)` 读取位置后只 `pread` 指定片中的这一条。

键片与记录片分别定位，记录增加、记录片增加都不会让 `get` 遍历更多片。片内查询仍有 B-tree 的对数查找成本，命中很多候选时仍须返回全部命中。单库页缓存有界，文件句柄池最多保留 16 个记录片；不再保存随样本数量增长的 Python 位置数组，也没有温片转冷和全片键链扫描。

旧参数 `hot`、`warm` 保留用于调用兼容，在 v5 中不控制存储。`per_shard` 只控制记录文件滚片，`out()` 在新布局中提交检查点。新布局的重复 `(key, sub)` 登记是幂等的，候选列表按下标排列；去重调用方本来就按集合使用候选。同一下标可以在任意时间补充键，首次登记的签名和窗口不被覆盖。

## 保存与故障边界

累计 65,536 次键登记后，在下一个样本开始前自动提交检查点；同一样本连续登记的全部分带键不会被自动提交切开。显式 `save` / `out` / `close` 也会提交。先同步记录正文与目录，再在同一个 SQLite 事务中提交所有前缀键、位置和记录数。进程在未提交批次中断时，SQLite 回滚整个批次；记录文件尾部可能留下未引用的字节，不会成为可查询样本，也不会影响继续追加。

一份索引只支持一个写进程；同实例多线程访问有锁保护。它不是多进程数据库服务。`close()` 会保存所有已经登记的内容，关闭后不能继续读写。文件被外部删除或损坏时不会自动重建成空索引。

`save(other_directory)` 要求目标为空，复制记录文件并使用 SQLite 一致快照备份，最后发布清单。复制中断会留下一个未完成的目标目录；它不覆盖源索引，调用方应检查失败原因并使用新的空目录重试。

## 显式迁移旧索引

```python
from pathlib import Path
from features.index import Index, migrate

migrate(Path("old-index-v4"), Path("new-index-v5"))
check = Index(Path("new-index-v5"))
print(check.size())
check.close()
```

源目录必须是已保存的 v4 索引，迁移期间不要让其他进程写源目录。迁移流式读取旧记录与全部键，目标须为空；不打开会重写偏移表的旧索引，不改源目录的任何字节。目标验证通过后，旧目录的保留或删除由用户决定。中途失败的目标可保留检查，用另一个空目录重试。

## 运行验收

```bash
python -m pytest tests/test_prefix_index.py tests/test_index.py tests/test_data.py
python tests/stress_index.py 200000 --directory .bench --json /tmp/index-200k.json
python tests/stress_index.py 1100000 --disk-only --directory .bench --json /tmp/index-1100k.json
```

测试包含真实去重结果与内存版一致、跨 100 个记录片仍只查询一个前缀 B-tree、保存后续写、稀疏下标、碰撞键、多线程、未提交进程崩溃回滚，以及迁移源文件 SHA-256 不变。

压测耗时包含 `save` 与 `close`，结束后重开检查命中。`--disk-only` 不运行纯内存基线，因此其结果不能独立证明“内存比 ≥16”。需求第 03 项采用新布局的验收分支；性能数字应使用当前机器的运行结果，不沿用旧版本的真机数据。

2026-11-02 本机 20 万条真实 MinHash 去重压测：纯内存 56.87 秒、3543.5 MiB 峰值；v5 落盘 164.61 秒、46.8 MiB 峰值，内存比 75.7。详见 [index-v5-200k.json](evidence/index-v5-200k.json)。这只证明 20 万规模，不把比例外推到百万，也不作为“110 万 ≤10 分钟”的证据。

同日 **110 万条已完整跑完**：1844.40 秒（30 分 44.40 秒）、峰值 50.1 MiB、596 条/秒；包含保存与关闭，所有 110 万条均按预期保留，重开后首条语料能正确判重。原始结果见 [index-v5-1100k.json](evidence/index-v5-1100k.json)。本机没有运行这个规模的纯内存基线，JSON 中相应数值为 `null`，因此百万内存比未知。

本次耗时**未达到 10 分钟**。第 03 项按清单明确允许的“哈希前缀分片 + 带片号位置表”新布局分支完成，不能改写成性能门槛已通过。这些数据来自本次开发验证期间的运行快照，不是隔离其他负载后的跨版本性能对照；压测启动后的最后一处源码修正仅将自动检查点移到记录之间，最终崩溃一致性回归另行通过。
