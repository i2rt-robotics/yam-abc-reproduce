from pathlib import Path

from scripts.generate_depth_annotation import default_output_root


def test_default_output_root_uses_depth_root(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "lerobot" / "owner" / "dataset"
    depth_root = tmp_path / "depth"
    monkeypatch.setenv("LEROBOT_DEPTH_DATA_ROOT", str(depth_root))

    assert (
        default_output_root(dataset_root, "owner/dataset", None)
        == depth_root / "owner" / "dataset"
    )


def test_default_output_root_falls_back_to_suffixed_neighbor(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "lerobot" / "dataset"
    monkeypatch.delenv("LEROBOT_DEPTH_DATA_ROOT", raising=False)

    assert default_output_root(dataset_root, "dataset", None) == dataset_root.with_name("dataset_depth")


def test_default_output_root_honors_explicit_output_dir(monkeypatch, tmp_path: Path):
    dataset_root = tmp_path / "lerobot" / "dataset"
    output_root = tmp_path / "custom"
    monkeypatch.setenv("LEROBOT_DEPTH_DATA_ROOT", str(tmp_path / "depth"))

    assert default_output_root(dataset_root, "dataset", str(output_root)) == output_root.resolve()
