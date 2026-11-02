"""v5 前缀索引的行为验收：单片查键、跨记录片续写、崩溃回滚与只读迁移。

运行：python -m pytest tests/test_prefix_index.py
旧 v4 的三层布局细节继续由 test_index.py 覆盖；这里验证公开 Index 默认路径。
"""

import hashlib                              # 检查迁移前后源文件字节完全不变
import json                                 # 检查实际落盘格式版本
import subprocess                           # 崩溃要在独立进程发生
import sys                                  # 使用当前解释器运行子进程
import threading                            # 同实例读写应安全

import pytest                               # 临时目录与错误断言

from features import duplicate              # 用真实去重判定检查两种布局一致
from features.index import Index, LegacyIndex, PrefixIndex, migrate


def key(number):
    """稳定的八字节假键，不需要昂贵的 MinHash。"""
    return number.to_bytes(8, "big")


def test_default_lookup_routes_once_across_many_record_shards(tmp_path):
    index = Index(tmp_path, key_size=8, per_shard=3)  # 100 个记录片，逼出旧式扇出问题
    for sub in range(300):
        index.put(key(sub), sub, [sub], {sub + 1})
    queries = []
    index._database.set_trace_callback(queries.append)  # 记录真正执行的 SQL，不猜调用路径
    assert index.get(key(137)) == [137]
    assert len(queries) == 1                     # 全部候选只需一个前缀片查询
    assert "WHERE key=" in queries[0]
    plan = index._database.execute(f"EXPLAIN QUERY PLAN SELECT sub FROM keys_{index.prefix(key(137)):02x} WHERE key=?", (key(137),)).fetchone()
    assert "SEARCH" in plan[3]                  # 库实际使用 B-tree 查找，不能变成表扫描
    assert index.record(137) == ([137], {138})
    assert len(index._files) <= 16               # 记录片很多时句柄缓存仍然封顶
    index.close()


def test_same_prefix_collision_is_not_same_key(tmp_path):
    index = Index(tmp_path, key_size=8)
    first = key(1)
    collision = next(key(n) for n in range(2, 1000) if index.prefix(key(n)) == index.prefix(first))
    index.put(first, 7, [11], {12})
    index.put(collision, 8, [21], {22})
    index.put(first, 9, [31], {32})                # 一个键允许指向多个不同样本
    index.put(first, 7)                           # 重复登记同键同样本是幂等
    assert index.get(first) == [7, 9]
    assert index.get(collision) == [8]
    assert index.size() == 3
    index.close()


def test_sparse_sub_and_late_keys_survive_reload_and_append(tmp_path):
    index = Index(tmp_path, key_size=8, per_shard=2)
    for sub in (3, 9, 10**12):                    # 不按最大下标分配内存
        index.put(key(sub), sub, [sub % 17], {sub % 23})
    index.put(key(123), 3)                       # 已落盘记录以后补键，不重复计数或覆盖签名
    index.save()
    index.close()
    back = Index(tmp_path)                       # 键宽与片容量来自清单，不依赖调用参数
    assert isinstance(back, PrefixIndex)
    assert back.size() == 3
    assert back.get(key(123)) == [3]
    assert back.record(3) == ([3], {3})
    assert back.record(10**12) == ([10**12 % 17], {10**12 % 23})
    back.put(key(4), 4, [4], {5})                 # 重开后继续写，不覆盖旧记录
    back.close()
    again = Index(tmp_path)
    assert again.size() == 4
    assert again.record(4) == ([4], {5})
    again.close()


def test_save_copy_is_independent(tmp_path):
    original = Index(tmp_path / "original", key_size=8)
    original.put(key(1), 1, [1], {2})
    original.save(tmp_path / "copy")
    original.put(key(2), 2, [2], {3})
    original.close()
    copy = Index(tmp_path / "copy")
    assert copy.size() == 1
    assert copy.get(key(2)) == []
    assert copy.record(1) == ([1], {2})
    with pytest.raises(ValueError, match="空目录"):
        copy.save(tmp_path / "original")         # 另存不覆盖既有索引
    copy.close()


def test_uncommitted_crash_preserves_last_checkpoint(tmp_path):
    index = Index(tmp_path, key_size=8)
    index.put(key(1), 1, [1], {1})
    index.close()
    snippet = "from features.index import Index; import os; i=Index(%r); i.put((2).to_bytes(8, 'big'), 2, [2], {2}); os._exit(0)" % str(tmp_path)
    subprocess.run([sys.executable, "-c", snippet], check=True)  # 不调用 close，不提交本批键
    back = Index(tmp_path)
    assert back.size() == 1                       # SQLite 回滚位置、候选和计数
    assert back.get(key(2)) == []
    assert back.record(1) == ([1], {1})
    back.put(key(3), 3, [3], {3})                 # 未引用的记录尾部不妨碍后续追加
    assert back.record(3) == ([3], {3})
    back.close()


def test_legacy_open_is_not_an_implicit_migration(tmp_path):
    old = LegacyIndex(tmp_path, key_size=8, hot=2, per_shard=4, warm=8)
    for sub in range(20):
        old.put(key(sub), sub, [sub], {sub})
    old.save()
    old.close()
    back = Index(tmp_path)
    assert isinstance(back, LegacyIndex)          # 自动识别旧格式，但不改写它的布局
    assert back.get(key(0)) == [0]
    back.save()
    back.close()
    assert json.loads((tmp_path / "index.json").read_text())["version"] == 4
    assert not (tmp_path / "prefix.sqlite3").exists()


def test_explicit_migration_is_read_only_on_source(tmp_path):
    source, target = tmp_path / "v4", tmp_path / "v5"
    old = LegacyIndex(source, key_size=8, hot=1, per_shard=3, warm=3)
    for sub in range(21):
        for band in range(2):
            old.put(key(sub * 2 + band), sub, [sub], {sub, sub + 1})
    old.save()
    old.close()
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    migrate(source, target)
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}
    back = Index(target)
    assert back.size() == 21
    for sub in range(21):
        assert back.get(key(sub * 2 + 1)) == [sub]
        assert back.record(sub) == ([sub], {sub, sub + 1})
    back.close()


def test_concurrent_reader_keeps_record_and_key_consistent(tmp_path):
    index = Index(tmp_path, key_size=8, per_shard=7)
    failures = []
    def reader():
        try:
            for _ in range(5):
                for sub in range(200):
                    for found in index.get(key(sub)):
                        assert index.record(found) == ([found], {found})
        except BaseException as error:
            failures.append(error)
    thread = threading.Thread(target=reader)
    thread.start()
    for sub in range(200):
        index.put(key(sub), sub, [sub], {sub})
    thread.join()
    assert not failures
    index.close()


def test_default_duplicate_matches_memory_after_reopen(tmp_path):
    texts = [f"第 {n} 份语料，认真检查索引在保存与重开之后的真实去重效果。" for n in range(30)]
    memory = duplicate.Duplicate(duplicate.Rule())
    disk = duplicate.Duplicate(duplicate.Rule(), store=tmp_path)
    assert [memory.check(t) for t in texts] == [disk.check(t) for t in texts]
    disk.save()
    disk.close()
    disk = duplicate.Duplicate(duplicate.Rule(), store=tmp_path)
    assert [memory.check(t) for t in texts[::-1]] == [disk.check(t) for t in texts[::-1]]
    disk.close()


def test_input_errors_and_missing_record(tmp_path):
    index = Index(tmp_path, key_size=8)
    with pytest.raises(ValueError):
        index.put(b"wrong", 0)
    with pytest.raises(ValueError):
        index.put(key(1), -1)
    with pytest.raises(KeyError):
        index.record(999)
    index.close()
    index.close()                                # 重复释放连接保持无副作用


def test_missing_database_does_not_create_empty_index(tmp_path):
    import sqlite3  # SQLite 以只允许读写既有库的模式打开
    index = Index(tmp_path, key_size=8)
    index.put(key(1), 1)
    index.close()
    (tmp_path / "prefix.sqlite3").unlink()  # 模拟外部丢失数据库
    with pytest.raises(sqlite3.OperationalError):
        Index(tmp_path)
    assert not (tmp_path / "prefix.sqlite3").exists()


def test_auto_checkpoint_does_not_split_one_samples_bands(tmp_path):
    snippet = """
import os
from features import index
index.CHECKPOINT_KEYS = 2
idx = index.Index(%r, key_size=8)
for band in range(3):
    idx.put(band.to_bytes(8, 'big'), 7, [7], {7})
os._exit(0)
""" % str(tmp_path)
    subprocess.run([sys.executable, "-c", snippet], check=True)  # 同一条记录超过提交阈值后崩溃
    back = Index(tmp_path)
    assert back.size() == 0                      # 要么所有分带键保存，要么整个未提交样本回滚
    assert back.get(key(0)) == []
    back.close()
