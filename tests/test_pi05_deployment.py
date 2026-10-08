"""Check environment patch boundaries without installing or importing a model runtime."""

import os
from pathlib import Path

import pytest

from owac.backends.pi05 import copy_patch_files


def test_patch_preserves_shared_cache_file(tmp_path: Path) -> None:
    source = tmp_path / "patches"
    source.mkdir()
    (source / "model.py").write_text("patched")
    environment = tmp_path / ".venv"
    destination = environment / "transformers"
    destination.mkdir(parents=True)
    cache = tmp_path / "cached_model.py"
    cache.write_text("original")
    os.link(cache, destination / "model.py")

    copy_patch_files(source, destination, environment)

    assert cache.read_text() == "original"
    assert (destination / "model.py").read_text() == "patched"
    assert cache.stat().st_ino != (destination / "model.py").stat().st_ino


def test_patch_rejects_external_environment(tmp_path: Path) -> None:
    source = tmp_path / "patches"
    source.mkdir()
    (source / "model.py").write_text("patched")
    external = tmp_path / "external"
    external.mkdir()
    (external / "model.py").write_text("original")

    with pytest.raises(ValueError, match="inside the selected virtual environment"):
        copy_patch_files(source, external, tmp_path / ".venv")

    assert (external / "model.py").read_text() == "original"


def test_patch_rejects_nested_symlink_escape(tmp_path: Path) -> None:
    source = tmp_path / "patches"
    (source / "models").mkdir(parents=True)
    (source / "models/model.py").write_text("patched")
    environment = tmp_path / ".venv"
    destination = environment / "transformers"
    destination.mkdir(parents=True)
    external = tmp_path / "external"
    external.mkdir()
    (external / "model.py").write_text("original")
    (destination / "models").symlink_to(external, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the selected virtual environment"):
        copy_patch_files(source, destination, environment)

    assert (external / "model.py").read_text() == "original"
