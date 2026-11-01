"""forge 的测试：GGUF 写出去能按格式读回来，量化块字节数对，档位分派对，词表 kv 落盘，LoRA 合并偏差够小。

约定：权重全是几行几列的小矩阵，量化只考字节布局和还原误差，不考压缩率。
缺包的用例用 monkeypatch 假掉 find_spec，机器上装没装 onnx / mlx_lm 都跑得出同一个结果。
onnx 的参考值一律拿 torch 那一侧的模块算，不拿 onnx 自己的输出当答案；past 一律用空的或零的。
"""

import importlib.util
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
    return tokenizer_files(folder)  # 词表跟着权重放，多数用例都要它


def one_tensor(tmp_path, name, array):
    """只放一个张量的目录，单独考某种形状或某个名字的量化分派。"""
    folder = tmp_path / "one"
    folder.mkdir()
    save_file({name: array.astype(np.float32)}, str(folder / "m.safetensors"))
    (folder / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    return folder


def full_ckpt(tmp_path):
    """造一套几何自洽的小权重：层数、头数、头宽、前馈宽度、输出层全在，够导一张完整的因果图。

    heads × head_dim 正好等于 hidden，和 features.model 的约束一致，所以这张图的形状是真结构。
    """
    folder = tmp_path / "full"
    folder.mkdir()
    rng = np.random.default_rng(31)
    hidden, heads, head_dim, layers, vocab, wide = 32, 2, 16, 2, 32, 64

    def one(*shape):  # 一份小方差权重，量级和真训练出来的差不多
        return (rng.normal(size=shape) * 0.2).astype(np.float32)

    tensors = {"model.embed_tokens.weight": one(vocab, hidden), "model.norm.weight": one(hidden)}
    tensors["lm_head.weight"] = one(vocab, hidden)  # 不共用词表的输出层
    for index in range(layers):  # 每层的九个张量一个不少
        at = f"model.layers.{index}."
        tensors[at + "self_attn.q_proj.weight"] = one(hidden, hidden)
        tensors[at + "self_attn.k_proj.weight"] = one(hidden, hidden)
        tensors[at + "self_attn.v_proj.weight"] = one(hidden, hidden)
        tensors[at + "self_attn.o_proj.weight"] = one(hidden, hidden)
        tensors[at + "mlp.gate_proj.weight"] = one(wide, hidden)
        tensors[at + "mlp.up_proj.weight"] = one(wide, hidden)
        tensors[at + "mlp.down_proj.weight"] = one(hidden, wide)
        tensors[at + "input_layernorm.weight"] = one(hidden) + 1
        tensors[at + "post_attention_layernorm.weight"] = one(hidden) + 1
    save_file(tensors, str(folder / "model.safetensors"))
    (folder / "config.json").write_text(
        json.dumps(
            {
                "model_type": "hoshiori",
                "hidden_size": hidden,
                "num_hidden_layers": layers,
                "num_attention_heads": heads,
                "head_dim": head_dim,
                "vocab_size": vocab,
                "max_position_embeddings": 16,
                "rope_theta": 10000.0,
                "tie_word_embeddings": False,
            }
        ),
        encoding="utf-8",
    )
    return folder


def tokenizer_files(folder):
    """给权重目录补一份词表和一份特殊 token 配置。"""
    (folder / "tokenizer.json").write_text(
        json.dumps(
            {
                "model": {"type": "BPE", "vocab": {"a": 0, "b": 1, "c": 2}, "merges": ["a b", ["b", "c"]]},  # 两种 merges 写法都认
                "added_tokens": [
                    {"id": 3, "content": "<s>", "special": True},
                    {"id": 4, "content": "</s>", "special": True},
                    {"id": 5, "content": "<pad>", "special": True},
                ],
            }
        ),
        encoding="utf-8",
    )
    (folder / "tokenizer_config.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "pad_token": {"content": "<pad>"}}),
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
        if kind == 9:  # 数组：元素类型、个数、逐个值
            item_kind = struct.unpack_from("<I", raw, pos)[0]
            pos += 4
            size = struct.unpack_from("<Q", raw, pos)[0]
            pos += 8
            items = []
            for _ in range(size):
                if item_kind == 8:
                    one = struct.unpack_from("<Q", raw, pos)[0]
                    items.append(raw[pos + 8 : pos + 8 + one].decode())
                    pos += 8 + one
                else:
                    items.append(struct.unpack_from({4: "<I", 5: "<i"}[item_kind], raw, pos)[0])
                    pos += 4
            kv[key] = items
        elif kind == 8:  # 字符串
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
    for one in infos:  # 按偏移切数据，宽度按补齐后算
        size = forge.tensor_bytes(one["kind"], one["shape"])
        one["data"] = raw[start + one["at"] : start + one["at"] + size]
    return {"version": version, "kv": kv, "infos": infos, "data_len": len(raw) - start}


def run_forge(tmp_path, *extra):
    """跑一次命令，返回退出码。权重目录没有就先造一份。"""
    if not (tmp_path / "ckpt").exists():
        ckpt(tmp_path)
    return main.run(["forge", str(warp(tmp_path)), *extra])


def decode_q6(data: bytes, rows: int) -> np.ndarray:
    """按 q6_k 的字节布局解开，返回 (行, 256) 的还原值。低 4 位、高 2 位、子块比例三处都要拼回来。"""
    out = np.zeros((rows, 256))
    for index in range(rows):  # 一块 210 字节，一行一块
        block = data[index * 210 : (index + 1) * 210]
        ql, qh = block[:128], block[128:192]  # 低 4 位与高 2 位
        scales = np.frombuffer(block[192:208], dtype=np.int8)  # 16 个 int8 子块比例
        d = struct.unpack("<e", block[208:210])[0]  # 一个 fp16 超块比例
        for half in range(2):  # 两个 128 值的大半块
            for l in range(32):
                at = half * 128 + l
                pair = [  # 每个字节里摆着 4 个值的一片，照 ggml 的排法解
                    (ql[half * 64 + l] & 0xF) | ((qh[half * 32 + l] & 3) << 4),
                    (ql[half * 64 + 32 + l] & 0xF) | (((qh[half * 32 + l] >> 2) & 3) << 4),
                    (ql[half * 64 + l] >> 4) | (((qh[half * 32 + l] >> 4) & 3) << 4),
                    (ql[half * 64 + 32 + l] >> 4) | (((qh[half * 32 + l] >> 6) & 3) << 4),
                ]
                for k in range(4):  # 第 k 片对应 at + 32k
                    out[index, at + 32 * k] = d * float(scales[half * 8 + 2 * k + l // 16]) * (pair[k] - 32)
    return out


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
        expected += forge.tensor_bytes(one["kind"], one["shape"])
    assert got["data_len"] == expected  # 数据段长度严丝合缝
    assert not got["kv"].get("general.tokenizer")  # 没有 general.tokenizer 这条，词表走 tokenizer.ggml.*


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
    ramp = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(1, 256)  # 斜坡，好量误差
    folder = one_tensor(tmp_path, "model.layers.0.mlp.down_proj.weight", ramp)
    Export(Choice(source=folder, out=tmp_path / "k.gguf", quant="q5_k_m")).run()
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


def test_q6_k_block_bytes(tmp_path):
    """q6_k 一块 256 个值 210 字节：低 4 位 128 + 高 2 位 64 + 16 个 int8 比例 + 1 个 fp16 比例。"""
    ramp = np.linspace(-1.0, 1.0, 512, dtype=np.float32).reshape(2, 256)  # 两行，正好两个块
    folder = one_tensor(tmp_path, "model.layers.0.mlp.down_proj.weight", ramp)
    Export(Choice(source=folder, out=tmp_path / "k6.gguf", quant="q6_k")).run()
    got = read_gguf(tmp_path / "k6.gguf")
    one = next(one for one in got["infos"] if one["kind"] == forge.GGML_Q6_K)
    assert forge.BLOCK[forge.GGML_Q6_K] == 256  # 一块 256 个值
    assert forge.BLOCK_BYTES[forge.GGML_Q6_K] == 210  # 一块 210 字节
    assert len(one["data"]) == 2 * 210  # 两行两个块


def test_q6_k_block_layout(tmp_path):
    """解一个 q6_k 块出来，6 位有符号 + 子块比例的还原误差要小。"""
    ramp = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(1, 256)
    folder = one_tensor(tmp_path, "model.layers.0.mlp.down_proj.weight", ramp)
    Export(Choice(source=folder, out=tmp_path / "k6.gguf", quant="q6_k")).run()
    one = next(one for one in read_gguf(tmp_path / "k6.gguf")["infos"] if one["kind"] == forge.GGML_Q6_K)
    ql, qh = one["data"][:128], one["data"][128:192]  # 低 4 位与高 2 位
    scales = np.frombuffer(one["data"][192:208], dtype=np.int8)  # 16 个 int8 子块比例
    d = struct.unpack("<e", one["data"][208:210])[0]  # 一个 fp16 超块比例
    restored = np.zeros(256)
    for half in range(2):  # 两个 128 值的大半块
        for l in range(32):
            at = half * 128 + l
            pair = [  # 每个字节里摆着 4 个值的一片，按 ggml 的排法解
                (ql[half * 64 + l] & 0xF) | ((qh[half * 32 + l] & 3) << 4),
                (ql[half * 64 + 32 + l] & 0xF) | (((qh[half * 32 + l] >> 2) & 3) << 4),
                (ql[half * 64 + l] >> 4) | (((qh[half * 32 + l] >> 4) & 3) << 4),
                (ql[half * 64 + 32 + l] >> 4) | (((qh[half * 32 + l] >> 6) & 3) << 4),
            ]
            for k in range(4):  # 第 k 片对应 at + 32k
                restored[at + 32 * k] = d * float(scales[half * 8 + 2 * k + l // 16]) * (pair[k] - 32)
    assert np.abs(restored - ramp.ravel()).max() < 0.02  # 6 位在 1/32 步长量级


def test_q6_k_rmse_iteration_beats_approx():
    """同一份随机张量：rmse 迭代版的重构误差要比 -max/32 近似版低 15% 以上。

    张量造的是「一条窄带 + 一成远值」：远值把 -max/32 顶得很大，窄带那 15 个值全被压成一格；
    rmse 迭代会把比例缩下来、让远值去截断，窄带才回得来。这是子块内极值不占主导时最吃亏的样子。
    """
    rng = np.random.default_rng(57)
    want = (rng.normal(0, 0.02, size=(8, 256)) + 1.37).astype(np.float32)  # 窄带
    want[rng.random((8, 256)) < 0.1] = 3.0  # 一成的远值，把极值抬走
    approx = decode_q6(forge.q6_k(want, refine=False), 8)  # 旧的一拍脑袋版
    refined = decode_q6(forge.q6_k(want), 8)  # rmse 迭代版

    def error(got):  # 还原值和原值的 rmse，越小越好
        return float(np.sqrt(np.mean((got - want) ** 2)))

    assert error(refined) <= error(approx) * 0.85  # 低 15% 以上；实测这一份是 25% 出头
    assert error(refined) < error(approx)  # 至少方向对，别出现偶然相等


def test_q5_k_m_lifts_output_layers(tmp_path):
    """_M 把 output.weight 与 token_embd.weight 升到 q6_k，其余仍 q5_k；q5_k_s 一层都不抬。"""
    ramp = np.linspace(-1.0, 1.0, 256, dtype=np.float32).reshape(1, 256)
    folder = tmp_path / "pair"
    folder.mkdir()
    save_file(
        {
            "lm_head.weight": ramp,  # 输出层
            "model.embed_tokens.weight": ramp,  # 词表
            "model.layers.0.mlp.down_proj.weight": ramp,  # 普通层
        },
        str(folder / "m.safetensors"),
    )
    Export(Choice(source=folder, out=tmp_path / "m.gguf", quant="q5_k_m")).run()
    kinds = {one["name"]: one["kind"] for one in read_gguf(tmp_path / "m.gguf")["infos"]}
    assert kinds["output.weight"] == forge.GGML_Q6_K  # 输出层被抬一档
    assert kinds["token_embd.weight"] == forge.GGML_Q6_K  # 词表也被抬
    assert kinds["blk.0.ffn_down.weight"] == forge.GGML_Q5_K  # 普通层还是 5 位
    Export(Choice(source=folder, out=tmp_path / "s.gguf", quant="q5_k_s")).run()
    plain = {one["name"]: one["kind"] for one in read_gguf(tmp_path / "s.gguf")["infos"]}
    assert set(plain.values()) == {forge.GGML_Q5_K}  # 非 _M 档一层都不抬


def test_q5_k_pads_narrow_rows(tmp_path):
    """行宽不是 256 的倍数就补零到块宽，不再整体退 fp32。"""
    out = tmp_path / "n.gguf"
    assert run_forge(tmp_path, "--out", str(out), "--format", "gguf", "--quant", "q5_k_m") == 0
    got = read_gguf(out)
    kinds = {one["name"]: one["kind"] for one in got["infos"]}
    assert kinds["blk.0.attn_q.weight"] == forge.GGML_Q5_K  # 32 宽的行补到 256 再量化
    one = next(one for one in got["infos"] if one["name"] == "blk.0.attn_q.weight")
    assert len(one["data"]) == 4 * 176  # 4 行，每行补成一个超块
    assert kinds["output_norm.weight"] == forge.GGML_F32  # 一维范数仍然 fp32


def test_narrow_rows_decode_close(tmp_path):
    """30 宽的行补到 32 之后能解回近似值，尾巴那两格是补的零。"""
    rng = np.random.default_rng(23)
    source = rng.normal(size=(4, 30)).astype(np.float32)
    folder = one_tensor(tmp_path, "model.layers.0.mlp.gate_proj.weight", source)
    Export(Choice(source=folder, out=tmp_path / "w.gguf", quant="q8_0")).run()
    one = next(one for one in read_gguf(tmp_path / "w.gguf")["infos"] if one["kind"] == forge.GGML_Q8_0)
    assert one["shape"] == (4, 30)  # dims 照原样写，补的零不算进去
    assert len(one["data"]) == 4 * 34  # 每行补成 32 个值一块
    restored = []
    for start in range(0, len(one["data"]), 34):
        d = struct.unpack("<e", one["data"][start : start + 2])[0]
        restored.append(np.frombuffer(one["data"][start + 2 : start + 34], dtype=np.int8) * d)
    got = np.stack(restored)[:, :30]  # 只取真值那 30 格
    assert np.abs(got - source).max() < 0.02  # 补零没把真值搅坏


def test_three_d_tensor_quantized(tmp_path):
    """三维张量按最外层拆成行量化，dims 原样写回。"""
    rng = np.random.default_rng(29)
    source = rng.normal(size=(2, 3, 32)).astype(np.float32)
    folder = one_tensor(tmp_path, "model.layers.0.mlp.up_proj.weight", source)
    report = Export(Choice(source=folder, out=tmp_path / "d3.gguf", quant="q8_0")).run()
    one = next(one for one in read_gguf(tmp_path / "d3.gguf")["infos"] if one["name"] == "blk.0.ffn_up.weight")
    assert one["kind"] == forge.GGML_Q8_0  # 三维也走量化，不退 fp32
    assert one["shape"] == (2, 3, 32)  # dims 不变
    assert len(one["data"]) == 6 * 34  # 最外层摊成 6 行，行行一块
    assert not any("不是二维" in one for one in report["dropped"])  # 不再记"留 fp32"那条


# ---------- 词表 ----------

def test_tokenizer_kv_written(tmp_path):
    """词表和 merges 要按 tokenizer.ggml.* 写进 kv，特殊 token 的 id 从配置里换。"""
    out = tmp_path / "t.gguf"
    assert run_forge(tmp_path, "--out", str(out), "--format", "gguf", "--quant", "q8_0") == 0
    save = read_gguf(out)["kv"]
    assert save["tokenizer.ggml.model"] == "gpt2"  # BPE 词表就是这么标
    assert save["tokenizer.ggml.tokens"] == ["a", "b", "c", "<s>", "</s>", "<pad>"]  # 词按 id 就位
    assert save["tokenizer.ggml.merges"] == ["a b", "b c"]  # 两种 merges 写法都归一成字符串
    assert save["tokenizer.ggml.token_type"] == [1, 1, 1, 3, 3, 3]  # 3 是控制符
    assert save["tokenizer.ggml.bos_token_id"] == 3  # 配置里的 <s> 换成 id
    assert save["tokenizer.ggml.eos_token_id"] == 4  # </s>
    assert save["tokenizer.ggml.padding_token_id"] == 5  # <pad>


def test_tokenizer_special_ids_default_when_missing(tmp_path):
    """配置里没给特殊 token 就写默认，并在 dropped 里点名缺了哪条。"""
    folder = ckpt(tmp_path)
    (folder / "tokenizer.json").write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}}), encoding="utf-8"
    )
    report = Export(Choice(source=folder, out=tmp_path / "t.gguf", quant="q8_0")).run()
    save = read_gguf(tmp_path / "t.gguf")["kv"]
    assert save["tokenizer.ggml.bos_token_id"] == 1  # bos 默认 1
    assert save["tokenizer.ggml.eos_token_id"] == 2  # eos 默认 2
    assert save["tokenizer.ggml.padding_token_id"] == forge.NO_ID  # 没有 pad 就是 -1 的补码
    assert any("bos_token_id" in one for one in report["dropped"])  # 缺哪条说哪条


def _varint(value):
    """protobuf 变长整数，造假的 tokenizer.model 用。"""
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _piece(text, kind=1):
    """一条 SentencePiece piece：字段 1 是文本，字段 3 是类型。"""
    body = b"\x0a" + _varint(len(text.encode())) + text.encode()
    if kind != 1:
        body += b"\x18" + _varint(kind)
    return b"\x0a" + _varint(len(body)) + body


def test_tokenizer_model_file(tmp_path):
    """SentencePiece 的 tokenizer.model 也要能解出词表和类型。"""
    folder = one_tensor(tmp_path, "model.norm.weight", np.zeros(32, np.float32))
    model = b"".join([_piece("▁hello"), _piece("<s>", 3), _piece("</s>", 3), _piece("<pad>", 3)])
    (folder / "tokenizer.model").write_bytes(model)
    Export(Choice(source=folder, out=tmp_path / "t.gguf", quant="f16")).run()
    kv = read_gguf(tmp_path / "t.gguf")["kv"]
    assert kv["tokenizer.ggml.model"] == "llama"  # SentencePiece 就是这套
    assert kv["tokenizer.ggml.tokens"] == ["▁hello", "<s>", "</s>", "<pad>"]  # 词按顺序解出来
    assert kv["tokenizer.ggml.token_type"] == [1, 3, 3, 3]  # 类型跟着 piece 走
    assert kv["tokenizer.ggml.bos_token_id"] == 1  # <s> 的位置
    assert kv["tokenizer.ggml.padding_token_id"] == 3  # <pad> 的位置


def test_tokenizer_missing_is_dropped(tmp_path):
    """基座目录没词表就把产物标成不能独立跑，不假装有。"""
    folder = one_tensor(tmp_path, "model.norm.weight", np.zeros(32, np.float32))
    report = Export(Choice(source=folder, out=tmp_path / "t.gguf", quant="q8_0")).run()
    assert any("tokenizer.json / tokenizer.model" in one for one in report["dropped"])  # 缺什么点名什么
    assert "tokenizer.ggml.tokens" not in read_gguf(tmp_path / "t.gguf")["kv"]  # kv 里确实没词表


def test_no_tokenizer_switch(tmp_path):
    """--no-tokenizer 时一条词表 kv 都不写。"""
    assert run_forge(tmp_path, "--out", str(tmp_path / "t.gguf"), "--format", "gguf", "--no-tokenizer") == 0
    kv = read_gguf(tmp_path / "t.gguf")["kv"]
    assert not any(key.startswith("tokenizer.") for key in kv)


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


def test_onnx_needs_package(tmp_path, monkeypatch):
    """缺 onnx 时按缺包报 exit 4。"""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name in ("onnx", "torch") else real(name))
    assert run_forge(tmp_path, "--out", str(tmp_path / "o"), "--format", "onnx") == 4


def export_with_past(tmp_path):
    """导一张带 past 的完整图，返回（torch, session, 模块, 假输入, 输入名, 输出名）。

    假输入里已经是空的 past（长度 0）；要对数的参考值一律拿模块那边算。
    """
    torch = pytest.importorskip("torch")
    runtime = pytest.importorskip("onnxruntime")
    folder = full_ckpt(tmp_path)
    out = tmp_path / "m.onnx"
    Export(Choice(source=folder, out=out, format="onnx", config=folder / "config.json")).run()
    tensors = forge.rename(forge.load(folder), [])
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    module, feeds = forge.onnx_graph(tensors, config, torch)  # 同一份权重再搭一遍，只用来出参考值
    names, out_names = forge.onnx_names(len(feeds[1]))
    session = runtime.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    return torch, session, module, feeds, names, out_names


def feed(names, feeds, ids, past_k=None, past_v=None):
    """按输入名拼一份 onnxruntime 的输入；不给 past 就用假输入里那份空的。都换成 numpy。"""
    _, empty_k, empty_v = feeds
    parts = [ids, *(empty_k if past_k is None else past_k), *(empty_v if past_v is None else past_v)]
    return dict(zip(names, [np.asarray(one) for one in parts]))


def test_onnx_writes_full_graph(tmp_path):
    """整张因果图要能导出来：checker 过、输入输出对、权重落旁边同名文件。"""
    pytest.importorskip("torch")
    onnx = pytest.importorskip("onnx")
    folder = full_ckpt(tmp_path)
    out = tmp_path / "m.onnx"
    report = Export(Choice(source=folder, out=out, format="onnx", config=folder / "config.json")).run()
    model = onnx.load(str(out))
    onnx.checker.check_model(model)  # 图本身要合法
    assert [one.name for one in model.graph.input] == ["input_ids", "past_k.0", "past_k.1", "past_v.0", "past_v.1"]  # 每层一对 past，加 token id
    assert [one.name for one in model.graph.output] == ["logits", "present_k.0", "present_k.1", "present_v.0", "present_v.1"]  # 打分，加各层回出去的 present
    nodes = {one.op_type for one in model.graph.node}  # 图里真的有点积、softmax、查表、拼 past 这些算子
    assert {"MatMul", "Softmax", "Gather", "Concat"} <= nodes
    assert len(model.graph.input) == 5  # 两层就是四份 past，一份不少
    assert sorted(load_file(str(tmp_path / "m.safetensors")))  # 权重整份落在旁边
    assert any("onnx 对拍" in one for one in report["dropped"])  # 自检结论进了出账


def test_onnx_graph_matches_torch_logits(tmp_path):
    """整图导出的 logits 要和 torch 那边的 logits 对得上，相对误差 < 1e-4。"""
    torch, session, module, feeds, names, _ = export_with_past(tmp_path)
    import onnx

    onnx.checker.check_model(onnx.load(str(tmp_path / "m.onnx")))
    ids, past_k, past_v = feeds
    with torch.no_grad():
        want = module(ids, past_k, past_v)[0].numpy()
    got = session.run(None, feed(names, feeds, ids.numpy()))[0]
    assert want.shape == got.shape == (1, 8, 32)  # 一批、8 步、词表 32
    error = float(np.max(np.abs(got - want)) / float(np.max(np.abs(want))))  # 相对最大值
    assert error < 1e-4  # 导出来的图算得对，不是只是长得像


def test_onnx_past_len_zero_matches_whole_forward(tmp_path):
    """past_len 为 0 时就是整段前向：相对误差 < 1e-4，而且前几位不受后面的 token 影响。"""
    torch, session, module, feeds, names, _ = export_with_past(tmp_path)
    ids = torch.randint(0, 32, (1, 8), dtype=torch.long)  # 随便挑一串 token
    with torch.no_grad():
        want = module(ids, *feeds[1:])[0].numpy()  # 空 past 喂进去就是整段前向
    got = session.run(None, feed(names, feeds, ids.numpy()))[0]
    error = float(np.max(np.abs(got - want)) / float(np.max(np.abs(want))))
    assert error < 1e-4  # 走 past 这条岔路，结果不能变
    short = session.run(None, feed(names, feeds, ids[:, :5].numpy()))[0]  # 只喂前 5 位，past 仍是 0
    head = float(np.max(np.abs(short - want[:, :5])) / float(np.max(np.abs(want[:, :5]))))
    assert head < 1e-4  # 前 5 位的 logits 和整段跑出来的前 5 位一样，因果没被 past 弄坏


def test_onnx_split_decode_matches_whole(tmp_path):
    """先喂 5 再喂 5、第二次带上第一次的 present：最后一位 logits 和一次喂 10 的差在 1e-3 以内。"""
    torch, session, module, feeds, names, _ = export_with_past(tmp_path)
    ids = torch.randint(0, 32, (1, 10), dtype=torch.long)
    with torch.no_grad():
        want = module(ids, *feeds[1:])[0].numpy()  # 一次喂 10 的参考值
    layer = len(feeds[1])
    first = session.run(None, feed(names, feeds, ids[:, :5].numpy()))  # 前半段自己跑
    past_k, past_v = first[1 : 1 + layer], first[1 + layer :]  # 回出来的 present 当第二段的 past
    got = session.run(None, feed(names, feeds, ids[:, 5:].numpy(), past_k, past_v))[0]
    assert got.shape == (1, 5, 32)  # 第二段只管这 5 步
    want_last, got_last = want[0, -1], got[0, -1]
    error = float(np.max(np.abs(got_last - want_last)) / float(np.max(np.abs(want_last))))
    assert error < 1e-3  # 分段解码累积误差够小，KV 复用是真能用


def test_onnx_present_shape_grows_along_past(tmp_path):
    """present 和 past 同构，只有长度那维加上本步：past 长 4、喂 3 步，回出来是 7，前 4 位就是喂进去的 past。"""
    torch, session, module, feeds, names, out_names = export_with_past(tmp_path)
    layer, heads, head_dim = len(feeds[1]), feeds[1][0].shape[1], feeds[1][0].shape[3]
    past_k = [torch.zeros(1, heads, 4, head_dim) for _ in range(layer)]  # 零 past，拼出来的前 4 位也该是零
    past_v = [torch.zeros(1, heads, 4, head_dim) for _ in range(layer)]
    ids = torch.randint(0, 32, (1, 3), dtype=torch.long)
    got = session.run(None, feed(names, feeds, ids.numpy(), past_k, past_v))
    assert [one.name for one in session.get_outputs()] == out_names  # 输出顺序和声明的一致
    assert [one.shape for one in got[1:]] == [(1, heads, 7, head_dim)] * (2 * layer)  # 每份 present 同构加长
    assert all(float(np.abs(one[:, :, :4]).max()) == 0.0 for one in got[1:])  # 前 4 位原样是喂进去的 past


def test_mlx_needs_package(tmp_path, monkeypatch):
    """缺 mlx_lm 时按缺包报 exit 4。"""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "mlx_lm" else real(name))
    assert run_forge(tmp_path, "--out", str(tmp_path / "o"), "--format", "mlx") == 4


def test_mlx_print_only(tmp_path, monkeypatch, capsys):
    """--mlx-print-only 只把命令、参数、目录布局打出来，不跑子进程，也不要求装 mlx_lm。"""
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "mlx_lm" else real(name))
    out = tmp_path / "mlx"
    assert run_forge(tmp_path, "--out", str(out), "--format", "mlx", "--mlx-print-only") == 0
    printed = capsys.readouterr().out
    assert "mlx_lm.convert" in printed  # 命令名打出来了，能照抄
    assert "--mlx-path" in printed  # 参数也打出来了
    assert str(out) in printed  # 产物目录写清了
    assert str(tmp_path / "ckpt") in printed  # 输入从哪个基座目录来也写清了
    assert "print-only" in printed  # 出账里点明这次没执行
    assert not list(out.glob("*.safetensors"))  # 一个文件都没真转出来


def test_mlx_runs_convert(tmp_path, monkeypatch):
    """有 mlx_lm 时走子进程，转出来的目录里得有东西。"""
    stub = tmp_path / "stub"
    (stub / "mlx_lm").mkdir(parents=True)
    (stub / "mlx_lm" / "__init__.py").write_text("", encoding="utf-8")
    (stub / "mlx_lm" / "convert.py").write_text(
        "import argparse, json\n"
        "from pathlib import Path\n"
        "ap = argparse.ArgumentParser()\n"
        "ap.add_argument('--input')\n"
        "ap.add_argument('--mlx-path')\n"
        "args = ap.parse_args()\n"
        "out = Path(args.mlx_path)\n"
        "out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'config.json').write_text(json.dumps({'input': args.input}))\n"
        "(out / 'weights.safetensors').write_bytes(b'x')\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(stub))  # 进程内找得到 mlx_lm
    monkeypatch.setenv("PYTHONPATH", str(stub))  # 子进程也找得到
    out = tmp_path / "mlx"
    assert run_forge(tmp_path, "--out", str(out), "--format", "mlx") == 0
    assert (out / "weights.safetensors").is_file()  # 子进程真写了产物
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["input"].endswith("ckpt")  # 收到的是基座目录


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


def test_onnx_accepts_any_sequence_length(tmp_path):
    """图要能吃不同长度：1、7、13 三种都跟 torch 对得上，长度维是动态的。"""
    torch, session, module, feeds, names, _ = export_with_past(tmp_path)  # 同一个 session 反复喂不同长度
    for length in (1, 7, 13):
        ids = torch.randint(0, 32, (1, length), dtype=torch.long)  # 随便挑一串 token
        with torch.no_grad():
            want = module(ids, *feeds[1:])[0].numpy()
        got = session.run(None, feed(names, feeds, ids.numpy()))[0]
        assert got.shape == want.shape == (1, length, 32)  # 输出长度跟着输入走
        error = float(np.max(np.abs(got - want)) / max(float(np.max(np.abs(want))), 1e-9))
        assert error < 1e-4  # 每个长度都算得对
