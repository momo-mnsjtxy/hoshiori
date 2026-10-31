"""forge 的测试：GGUF 写出去能按格式读回来，量化块字节数对，LoRA 合并偏差够小。

约定：权重全是几行几列的小矩阵，量化只考字节布局和还原误差，不考压缩率。
"""

import json
import struct

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

import main
from features import forge, merge
from features.forge import Choice, Export
from features.merge import Adapter


# ---------- 造数据 ----------

def ckpt(tmp_path):
    """造一个最小权重目录：一个 fp32 权重文件加一份结构配置。"""
    folder = tmp_path / "ckpt"
    folder.mkdir()
    rng = np.random.default_rng(7)
    save_file(
        {
            "model.embed_tokens.weight": rng.normal(size=(4, 32)).astype(np.float32),  # 词表，能映射
            "model.layers.0.self_attn.q_proj.weight": rng.normal(size=(4, 32)).astype(np.float32),  # 层内权重
            "model.norm.weight": rng.normal(size=(32,)).astype(np.float32),  # 一维范数
            "lm_head.weight": rng.normal(size=(4, 32)).astype(np.float32),  # 输出层
        },
        str(folder / "model.safetensors"),
    )
    (folder / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": 1, "hidden_size": 32, "torch_dtype": "float32"}),
        encoding="utf-8",
    )
    return folder


def warp(tmp_path, rel="ckpt"):
    """造一份能过校验的配方，model.path 指向 rel。"""
    path = tmp_path / "sft.warp.yaml"
    path.write_text(f"version: 1\ntask: sft\nmodel:\n  path: {rel}\nname: forge-test\n", encoding="utf-8")
    return path


def read_gguf(path):
    """最小的 GGUF v3 reader，只看这一份文件，跟写入器不共用代码。"""
    raw = path.read_bytes()
    assert raw[:4] == b"GGUF"  # magic
    version = struct.unpack_from("<I", raw, 4)[0]  # 版本
    pos = 8
    count = struct.unpack_from("<Q", raw, pos)[0]  # tensor 数
    pos += 8
    kv_count = struct.unpack_from("<Q", raw, pos)[0]  # kv 数
    pos += 8
    kv = {}
    for _ in range(kv_count):  # kv 段
        size = struct.unpack_from("<Q", raw, pos)[0]
        key = raw[pos + 8 : pos + 8 + size].decode()
        pos += 8 + size
        kind = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if kind == 8:  # 字符串
            size = struct.unpack_from("<Q", raw, pos)[0]
            kv[key] = raw[pos + 8 : pos + 8 + size].decode()
            pos += 8 + size
        elif kind == 4:
            kv[key] = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
        elif kind == 6:
            kv[key] = struct.unpack_from("<f", raw, pos)[0]
            pos += 4
        else:
            raise AssertionError(f"kv 类型没见过 {kind}")
    infos = []
    for _ in range(count):  # tensor 信息段
        size = struct.unpack_from("<Q", raw, pos)[0]
        name = raw[pos + 8 : pos + 8 + size].decode()
        pos += 8 + size
        dims = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        shape = [struct.unpack_from("<Q", raw, pos + 8 * i)[0] for i in range(dims)]
        pos += 8 * dims
        kind = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        at = struct.unpack_from("<Q", raw, pos)[0]
        pos += 8
        infos.append({"name": name, "shape": tuple(reversed(shape)), "kind": kind, "at": at})
    start = (pos + 31) // 32 * 32  # 数据段起点，32 字节对齐
    for one in infos:  # 按偏移切数据
        size = forge.block_bytes(one["kind"], int(np.prod(one["shape"])))
        one["data"] = raw[start + one["at"] : start + one["at"] + size]
    return {"version": version, "kv": kv, "infos": infos, "data_len": len(raw) - start}


def run_forge(tmp_path, *extra):
    """跑一次命令，返回退出码。权重目录没有就先造一份。"""
    if not (tmp_path / "ckpt").exists():
        ckpt(tmp_path)
    return main.run(["forge", str(warp(tmp_path)), *extra])


# ---------- GGUF 写入与回读 ----------

def test_gguf_roundtrip(tmp_path):
    out = tmp_path / "m.gguf"
    assert run_forge(tmp_path, "--out", str(out), "--format", "gguf", "--quant", "q8_0") == 0
    got = read_gguf(out)
    assert got["version"] == 3  # 写的是 v3
    assert len(got["infos"]) == 4  # 张量一个不少
    assert got["kv"]["general.alignment"] == 32  # 对齐写在 kv 里
    assert got["kv"]["general.architecture"] == "qwen3"  # 结构名从 model_type 来
    assert got["kv"]["qwen3.block_count"] == 1  # 层数映射上了
    kinds = {one["name"]: one["kind"] for one in got["infos"]}
    assert kinds["token_embd.weight"] == forge.GGML_Q8_0  # 二维权重被量化
    assert kinds["output_norm.weight"] == forge.GGML_F32  # 一维范数留 fp32
    assert {"token_embd.weight", "blk.0.attn_q.weight"} <= set(kinds)  # 名字按 ggml 的来
    expected = 0
    for one in got["infos"]:  # 自己按对齐规则算一遍数据段该多长
        expected = (expected + 31) // 32 * 32
        expected += forge.block_bytes(one["kind"], int(np.prod(one["shape"])))
    assert got["data_len"] == expected  # 数据段长度严丝合缝
    assert not got["kv"].get("general.tokenizer")  # 词表没导，别假装有


def test_q8_0_block_bytes(tmp_path):
    out = tmp_path / "q.gguf"
    assert run_forge(tmp_path, "--out", str(out), "--format", "gguf", "--quant", "q8_0") == 0
    got = read_gguf(out)
    seen = 0
    for one in got["infos"]:  # 每个 q8_0 张量都是 34 字节一块
        if one["kind"] != forge.GGML_Q8_0:
            continue
        blocks = int(np.prod(one["shape"])) // 32  # 32 个值一块
        assert len(one["data"]) == blocks * 34
        seen += 1
    assert seen


def test_q8_0_block_layout(tmp_path):
    """解一个块出来，比例和 int8 都要对得上原始值。"""
    out = tmp_path / "q.gguf"
    assert run_forge(tmp_path, "--out", str(out), "--format", "gguf", "--quant", "q8_0") == 0
    got = read_gguf(out)
    one = next(one for one in got["infos"] if one["name"] == "token_embd.weight")
    block = one["data"][:34]
    d = struct.unpack("<e", block[:2])[0]  # 前两个字节是 fp16 比例
    q = np.frombuffer(block[2:], dtype=np.int8).astype(np.float32)  # 后 32 个是 int8
    restored = q * d  # 按定义还原
    source = load_file(tmp_path / "ckpt" / "model.safetensors")["model.embed_tokens.weight"][0, :32]
    assert np.abs(restored - source).max() < 0.02  # 8 位的还原误差很小


def test_q5_k_superblock(tmp_path):
    """q5_k_m 要按 256 个值 176 字节的超块写，还原误差也别太大。"""
    folder = tmp_path / "big"
    folder.mkdir()
    ramp = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(1, 256)  # 斜坡，好量误差
    save_file({"model.layers.0.mlp.up_proj.weight": ramp}, str(folder / "m.safetensors"))
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3", "num_hidden_layers": 1}), encoding="utf-8")
    job = Export(Choice(source=folder, out=tmp_path / "k.gguf", quant="q5_k_m"))
    job.run()
    got = read_gguf(tmp_path / "k.gguf")
    one = next(one for one in got["infos"] if one["kind"] == forge.GGML_Q5_K)
    assert len(one["data"]) == 176  # 256 个值正好一个超块
    d, dmin = struct.unpack("<e", one["data"][:2])[0], struct.unpack("<e", one["data"][2:4])[0]  # 两个 fp16 基准
    scales = one["data"][4:16]  # 12 字节的 6bit 比例
    qh, qs = one["data"][16:48], one["data"][48:176]  # 第 5 位位图与低 4 位
    restored = np.zeros(256)
    for j in range(8):  # 8 个子块，照 get_scale_min_k4 的排法解比例
        if j < 4:
            sc, m = scales[j] & 63, scales[j + 4] & 63
        else:
            sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4)
            m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4)
        group = j // 2  # 第几组 64 个值
        high_half = j % 2  # 0 取低 4 位，1 取高 4 位
        bit = 1 << (2 * group + high_half)  # 对应的第 5 位
        for l in range(32):
            pair = qs[group * 32 + l]  # 两个值挤一个字节
            q = pair & 0xF if high_half == 0 else pair >> 4
            restored[group * 64 + high_half * 32 + l] = d * sc * (q + (16 if qh[l] & bit else 0)) - dmin * m
    err = np.abs(restored - ramp.ravel())
    assert err.max() < 0.12  # 5 位 + 子块比例的粗误差上限


def test_q5_k_falls_back_on_narrow_rows(tmp_path):
    """行宽不是 256 的倍数就退回 q8_0，不硬凑超块。"""
    out = tmp_path / "n.gguf"
    assert run_forge(tmp_path, "--out", str(out), "--format", "gguf", "--quant", "q5_k_m") == 0
    got = read_gguf(out)
    kinds = {one["kind"] for one in got["infos"] if not one["name"].endswith("output_norm.weight")}
    assert forge.GGML_Q5_K not in kinds  # 32 宽的行上不了超块
    assert forge.GGML_Q8_0 in kinds  # 退到 8 位


def test_dropped_fields_listed(tmp_path):
    """config 里映射不上的字段要进 dropped，不能悄悄丢。"""
    folder = ckpt(tmp_path)
    job = Export(Choice(source=folder, out=tmp_path / "d.gguf", config=folder / "config.json", quant="q8_0"))
    report = job.run()
    assert any("torch_dtype" in one for one in report["dropped"])  # 没映射的字段留了名
    assert not any("num_hidden_layers" in one for one in report["dropped"])  # 映射上的不算丢


# ---------- LoRA 合并 ----------

def adapter_dir(tmp_path, name, a, b, alpha=2.0, r=2):
    """写一个 adapter 目录，含配置和一对矩阵。"""
    folder = tmp_path / name
    folder.mkdir()
    save_file({"w.lora_A.weight": a, "w.lora_B.weight": b}, str(folder / "a.safetensors"))
    (folder / "adapter_config.json").write_text(json.dumps({"r": r, "lora_alpha": alpha}), encoding="utf-8")
    return folder


def test_merge_keeps_logits_close(tmp_path):
    """合并后的 logits 要跟 fp64 参考算出来的几乎一样。"""
    rng = np.random.default_rng(11)
    weight = rng.normal(size=(8, 8)).astype(np.float32) * 0.2
    left = rng.normal(size=(2, 8)).astype(np.float32) * 0.1  # A
    right = rng.normal(size=(8, 2)).astype(np.float32) * 0.1  # B
    x = rng.normal(size=(4, 8)).astype(np.float32)
    scale = 2.0 / 2  # alpha / r
    folder = adapter_dir(tmp_path, "bad", left, right)

    merged = merge.merge({"w": weight}, [Adapter(folder, alpha=2.0, r=2)])
    reference = weight.astype(np.float64) + (right.astype(np.float64) @ left.astype(np.float64)) * scale  # fp64 参考
    worse = np.abs(x @ merged["w"].T.astype(np.float64) - x.astype(np.float64) @ reference.T).max()
    base = np.abs(x @ merged["w"].T.astype(np.float64) - x.astype(np.float64) @ weight.T.astype(np.float64)).max()
    assert worse < 1e-3  # 偏差过线
    assert base > 1e-3  # 确实改了，不是原样返回


def test_merge_stacks_two_adapters(tmp_path):
    """两个 adapter 依次叠，等于两个增量分别加。"""
    rng = np.random.default_rng(13)
    weight = rng.normal(size=(8, 8)).astype(np.float32) * 0.2
    pairs = []
    folders = []
    for index in (1, 2):  # 两个 adapter
        a = rng.normal(size=(2, 8)).astype(np.float32) * 0.05
        b = rng.normal(size=(8, 2)).astype(np.float32) * 0.05
        pairs.append((a, b))
        folders.append(adapter_dir(tmp_path, f"a{index}", a, b))
    merged = merge.merge({"w": weight}, [Adapter(one) for one in folders])
    want = weight.astype(np.float64) + sum((b.astype(np.float64) @ a.astype(np.float64)) * 1.0 for a, b in pairs)
    assert np.abs(merged["w"].astype(np.float64) - want).max() < 1e-4  # 两份增量都加上了


def test_merge_unknown_tensor_reports(tmp_path):
    """基座里没有的名字要当场报错。"""
    weight = np.zeros((8, 8), np.float32)
    a = np.zeros((2, 8), np.float32)
    b = np.zeros((8, 2), np.float32)
    folder = adapter_dir(tmp_path, "x", a, b)
    with pytest.raises(Exception) as bad:
        merge.merge({"别的": weight}, [Adapter(folder)])
    assert "w" in str(bad.value)


def test_command_merges_adapter(tmp_path):
    """model.path 指 adapter 目录时命令先合并再导，导出来的得是合并后的值。"""
    base = ckpt(tmp_path)
    folder = tmp_path / "adapter"
    folder.mkdir()
    rng = np.random.default_rng(17)
    left = rng.normal(size=(2, 32)).astype(np.float32) * 0.2  # A
    right = rng.normal(size=(4, 2)).astype(np.float32) * 0.2  # B
    name = "model.layers.0.self_attn.q_proj.weight"
    save_file({name + ".lora_A.weight": left, name + ".lora_B.weight": right}, str(folder / "a.safetensors"))
    (folder / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 20, "base_model_name_or_path": "../ckpt"}), encoding="utf-8")
    recipe_path = tmp_path / "m.warp.yaml"
    recipe_path.write_text("version: 1\ntask: sft\nmodel:\n  path: adapter\n", encoding="utf-8")
    out = tmp_path / "m.gguf"
    assert main.run(["forge", str(recipe_path), "--out", str(out), "--format", "gguf", "--quant", "q8_0"]) == 0
    one = next(one for one in read_gguf(out)["infos"] if one["name"] == "blk.0.attn_q.weight")
    got = []
    for start in range(0, len(one["data"]), 34):  # 按 q8_0 的块解开
        d = struct.unpack("<e", one["data"][start : start + 2])[0]
        got.append(np.frombuffer(one["data"][start + 2 : start + 34], dtype=np.int8) * d)
    want = load_file(base / "model.safetensors")[name] + right @ left * (20 / 2)  # 手算的合并结果
    assert np.abs(np.concatenate(got).reshape(want.shape) - want).max() < 0.02  # 合并没有漏（0.02 是 q8_0 自己的误差）


# ---------- 命令层 ----------

def test_missing_out_is_usage_error(tmp_path):
    with pytest.raises(SystemExit) as stop:
        main.run(["forge", str(warp(tmp_path))])
    assert stop.value.code == 2  # argparse 的用法错误码


def test_unknown_format(tmp_path):
    assert run_forge(tmp_path, "--out", str(tmp_path / "o"), "--format", "tflite") == 2


def test_onnx_needs_package(tmp_path):
    code = run_forge(tmp_path, "--out", str(tmp_path / "o"), "--format", "onnx")
    assert code == 4  # 缺后端按 exit 4


def test_safetensors_roundtrip(tmp_path):
    out = tmp_path / "m.safetensors"
    assert run_forge(tmp_path, "--out", str(out), "--format", "safetensors") == 0
    got = load_file(str(out))
    assert len(got) == 4  # 张量数一致
    source = load_file(tmp_path / "ckpt" / "model.safetensors")
    assert np.allclose(got["output_norm.weight"], source["model.norm.weight"])  # 数值没变


def test_missing_weights(tmp_path):
    path = tmp_path / "sft.warp.yaml"
    path.write_text("version: 1\ntask: sft\nmodel:\n  path: nope\n", encoding="utf-8")
    assert main.run(["forge", str(path), "--out", str(tmp_path / "o"), "--format", "gguf"]) == 2
