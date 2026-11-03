from pathlib import Path

import pytest

from zsmedseg.config import Config, load_config


def test_defaults_are_the_method_settings():
    cfg = Config()
    assert cfg.prompting.num_points == 5
    assert cfg.prompting.suppression_frac == 0.10
    assert cfg.saliency.logit_scale == 40.0
    assert cfg.selection.num_objects == 1


def test_dataset_file_sets_only_dataset_values(repo_root: Path):
    cfg = load_config([repo_root / "configs/datasets/busi.yaml"])
    assert cfg.dataset.name == "busi"
    assert cfg.dataset.target_prompts == ["benign tumor", "malignant tumor"]
    # untouched sections keep their defaults
    assert cfg.saliency.short_side == 336


def test_lung_configs_keep_two_objects(repo_root: Path):
    for name in ("lung_xray", "lung_ct"):
        cfg = load_config([repo_root / f"configs/datasets/{name}.yaml"])
        assert cfg.selection.num_objects == 2
        assert cfg.evaluation.nsd_tolerance == 2  # inherited from the shared protocol file


def test_overrides_are_typed():
    cfg = load_config(overrides=["prompting.num_points=3", "dataset.target_prompts=[polyp, adenoma]"])
    assert cfg.prompting.num_points == 3
    assert cfg.dataset.target_prompts == ["polyp", "adenoma"]


def test_unknown_key_fails_loudly():
    with pytest.raises(KeyError, match="num_point"):
        load_config(overrides=["prompting.num_point=3"])


def test_inherit_chain(tmp_path: Path):
    (tmp_path / "base.yaml").write_text("prompting:\n  num_points: 2\n")
    (tmp_path / "child.yaml").write_text("inherit: base.yaml\nselection:\n  num_objects: 2\n")
    cfg = load_config([tmp_path / "child.yaml"])
    assert cfg.prompting.num_points == 2
    assert cfg.selection.num_objects == 2


def test_all_shipped_configs_load(repo_root: Path):
    for path in sorted((repo_root / "configs").rglob("*.yaml")):
        if path.parent.name == "sam2":
            continue
        load_config([path])
