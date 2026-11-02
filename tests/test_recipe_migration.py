"""配方校验、extends 装载、迁移器、命令入口的测试。"""

import json

import pytest

import main
from features import migration, recipe
from utils.fail import Failure


def _recipe(**over):
    base = {"version": 1, "task": "sft", "model": {"path": "base"}, "train": {"batch_size": 2}}
    base.update(over)
    return base


# ---------- 配方校验 ----------

def test_good_recipe_passes():
    recipe.check(_recipe())


def test_unknown_field_has_candidate():
    with pytest.raises(Failure) as bad:
        recipe.check(_recipe(train={"batch_sizes": 2}))
    assert "batch_size" in bad.value.hint
    assert bad.value.where == "train"


def test_bad_type_locates():
    with pytest.raises(Failure) as bad:
        recipe.check(_recipe(train={"batch_size": "八"}))
    assert bad.value.where == "train.batch_size"


def test_bad_enum():
    with pytest.raises(Failure):
        recipe.check(_recipe(train={"strategy": "mpi"}))


def test_finetune_needs_base():
    with pytest.raises(Failure) as bad:
        recipe.check({"version": 1, "task": "sft"})
    assert bad.value.where == "model.path"


def test_multi_card_needs_devices():
    with pytest.raises(Failure) as bad:
        recipe.check(_recipe(train={"strategy": "fsdp2"}))
    assert bad.value.where == "train.devices"


def test_forge_needs_out():
    with pytest.raises(Failure) as bad:
        recipe.check(_recipe(forge={"formats": ["gguf"]}))
    assert bad.value.where == "forge.out"


def test_nested_field_checked():
    with pytest.raises(Failure) as bad:
        recipe.check(_recipe(train={"lora": {"r": "十六"}}))
    assert bad.value.where == "train.lora.r"


# ---------- extends ----------

def _put(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_extends_merges_child_over_parent(tmp_path):
    _put(tmp_path / "base.warp.yaml", "version: 1\ntask: sft\ntrain:\n  batch_size: 4\n  lr: 0.1\n")
    child = _put(tmp_path / "c.warp.yaml", "extends: base.warp.yaml\ntrain:\n  lr: 0.2\n")
    merged = recipe.load(child)
    assert merged["train"] == {"batch_size": 4, "lr": 0.2}
    assert "extends" not in merged


def test_extends_loop(tmp_path):
    _put(tmp_path / "a.warp.yaml", "extends: b.warp.yaml\n")
    _put(tmp_path / "b.warp.yaml", "extends: a.warp.yaml\n")
    with pytest.raises(Failure) as bad:
        recipe.load(tmp_path / "a.warp.yaml")
    assert "圈" in bad.value.message


def test_extends_missing_parent(tmp_path):
    child = _put(tmp_path / "c.warp.yaml", "extends: nope.warp.yaml\n")
    with pytest.raises(Failure) as bad:
        recipe.load(child)
    assert "nope.warp.yaml" in bad.value.message


def test_stamp_ignores_key_order():
    assert recipe.stamp({"a": 1, "b": {"c": 2}}) == recipe.stamp({"b": {"c": 2}, "a": 1})
    assert recipe.stamp({"a": 1}) != recipe.stamp({"a": 2})


# ---------- 迁移 ----------

PRESET = {
    "project_name": "demo",
    "output_root": "out",
    "model": {"d_model": 2560, "max_seq_len": 1024},
    "train": {"mode": "finetune", "batch_size": 4, "learning_rate": 0.0002, "seed": 42},
    "peft": {"rank": 8, "alpha": 16.0, "target_modules": ["w_q"]},
    "finetune": {"base_checkpoint_path": "../merged_model", "method": "lora"},
}


def test_convert_moves_core_fields():
    job = migration.convert(PRESET, where="demo.yaml")
    assert job.recipe["task"] == "sft"
    assert job.recipe["train"]["lr"] == 0.0002
    assert job.recipe["train"]["lora"]["r"] == 8
    assert job.recipe["model"]["path"] == "../merged_model"


def test_convert_records_geometry_as_dropped():
    job = migration.convert(PRESET, where="demo.yaml")
    dropped = [one["path"] for one in job.dropped]
    assert "model.d_model" in dropped
    assert "project_name" in dropped


def test_convert_flags_unknown():
    job = migration.convert(dict(PRESET, 神秘开关=True), where="x.yaml")
    assert any(one["path"] == "神秘开关" for one in job.unknown)


def test_convert_blocks_without_base():
    preset = dict(PRESET)
    preset["finetune"] = {"method": "lora"}
    assert migration.convert(preset, where="x.yaml").blocked


def test_convert_full_drops_lora():
    preset = dict(PRESET)
    preset["finetune"] = {"base_checkpoint_path": "b", "method": "full"}
    assert "lora" not in migration.convert(preset, where="x.yaml").recipe.get("train", {})


def test_report_lists_sections():
    text = migration.convert(PRESET, where="demo.yaml").report()
    assert "已搬运" in text and "丢弃" in text


def test_read_bad_yaml(tmp_path):
    bad = _put(tmp_path / "b.yaml", "a: [\n")
    with pytest.raises(Failure):
        migration.read_preset(bad)


# ---------- 命令 ----------

def test_version_exits():
    with pytest.raises(SystemExit):  # --version 由 argparse 直接收尾
        main.run(["--version"])


def test_unknown_command_exits():
    with pytest.raises(SystemExit):
        main.run(["没这个命令"])


def test_weave_without_input():
    with pytest.raises(SystemExit):  # argparse 的 required 先拦
        main.run(["weave"])


def test_weave_bad_range(tmp_path, capsys):
    src = tmp_path / "a.jsonl"
    src.write_text('{"text": "甲"}\n', encoding="utf-8")
    code = main.run(["weave", "--in", str(src), "--min-chars", "9", "--max-chars", "3", "--out", str(tmp_path / "o")])
    assert code == 2
    assert "max-chars" in capsys.readouterr().err


def test_validate_command(tmp_path, capsys):
    good = tmp_path / "a.warp.yaml"
    good.write_text("version: 1\ntask: sft\nmodel:\n  path: base\n", encoding="utf-8")
    assert main.run(["validate", str(good)]) == 0
    assert "过了" in capsys.readouterr().out


def test_validate_command_bad(tmp_path, capsys):
    bad = tmp_path / "b.warp.yaml"
    bad.write_text("version: 1\ntask: sft\n", encoding="utf-8")
    assert main.run(["validate", str(bad)]) == 3
    assert "model.path" in capsys.readouterr().err


def test_import_command(tmp_path, capsys):
    preset = tmp_path / "p.yaml"
    preset.write_text("task:\n  mode: finetune\nfinetune:\n  base_checkpoint_path: b\n", encoding="utf-8")
    code = main.run(["import", "lmlab", str(preset), "--out", str(tmp_path / "out")])
    assert code == 0
    made = json.loads("{}") or (tmp_path / "out" / "p.warp.yaml").exists()
    assert made


def test_import_blocks_by_exit_code(tmp_path):
    preset = tmp_path / "p.yaml"
    preset.write_text("train:\n  mode: finetune\n", encoding="utf-8")
    assert main.run(["import", "lmlab", str(preset), "--out", str(tmp_path / "o")]) == 1


def test_spin_dry_run_missing_data(tmp_path, capsys):
    bad = tmp_path / "a.warp.yaml"
    bad.write_text("task: sft\nmodel:\n  path: b\ndata:\n  files: [没有这个.jsonl]\n", encoding="utf-8")
    assert main.run(["spin", str(bad)]) == 2
    assert "训练数据不在" in capsys.readouterr().err


def test_doctor_runs(capsys):
    main.run(["doctor"])
    assert "python" in capsys.readouterr().out


@pytest.mark.parametrize("framework", ["axolotl", "llama-factory"])
def test_framework_migration_cli_produces_valid_recipe(framework, tmp_path):
    """两家配置真实经过命令入口，迁出的配方能校验，未知项不丢失。"""
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.jsonl").write_text('{"text":"training sample"}\n')
    if framework == "axolotl":
        body = {"base_model": "base", "adapter": "lora", "micro_batch_size": 2, "num_epochs": 1,
                "learning_rate": "2e-4", "lora_r": 4, "lora_target_modules": ["q_proj"],
                "datasets": [{"path": "data/train.jsonl"}], "bf16": True, "custom_field": 42}
    else:
        (data / "dataset_info.json").write_text(json.dumps({"demo": {"file_name": "train.jsonl"}}))
        body = {"model_name_or_path": "base", "stage": "sft", "finetuning_type": "lora", "lora_rank": 4,
                "lora_target": "all", "dataset": "demo", "dataset_dir": "data", "per_device_train_batch_size": 2,
                "learning_rate": "2e-4", "bf16": True, "custom_field": 42}
    source = tmp_path / "source.yaml"
    source.write_text(json.dumps(body))
    out = tmp_path / "imported"
    assert main.run(["import", framework, str(source), "--out", str(out)]) == 0
    imported = recipe.load(out / "source.warp.yaml")
    recipe.check(imported)
    assert imported["train"]["lr"] == 0.0002 and imported["train"]["precision"] == "bf16"
    assert imported["data"]["files"] == [str(data / "train.jsonl")]
    assert "custom_field" in (out / "source.diff.md").read_text()


def test_unresolved_factory_dataset_blocks_instead_of_guessing(tmp_path):
    """注册表缺失时不把数据集名字当作本地文件，报告标为阻塞。"""
    job = migration.convert({"model_name_or_path": "base", "dataset": "not-registered"}, str(tmp_path / "task.yaml"), "llama-factory")
    assert job.blocked
    assert "dataset_info.json" in job.report()
