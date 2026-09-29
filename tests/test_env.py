import pytest

from recant import env


def test_parse_wheel_name():
    assert env.parse_wheel_name("flash_linear_attention-0.4.2-py3-none-any.whl") == ("flash-linear-attention", "0.4.2")
    assert env.parse_wheel_name("torch-2.10.0+cu128-cp312-cp312-manylinux_2_28_x86_64.whl") == ("torch", "2.10.0+cu128")
    assert env.parse_wheel_name("numpy-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl") == ("numpy", "2.1.0")
    assert env.parse_wheel_name("notes.txt") is None


def test_choose_tmp_dir_prefers_roomiest_and_fails_when_full(tmp_path):
    root = env.choose_tmp_dir(str(tmp_path), min_free_gb=0.0)
    assert root == tmp_path / "recant" and root.is_dir()
    with pytest.raises(RuntimeError, match="free"):
        env.choose_tmp_dir(str(tmp_path), min_free_gb=10**9)


def test_setup_process_env_redirects_caches(tmp_path, monkeypatch):
    for k in ("TRITON_CACHE_DIR", "HF_HOME", "TMPDIR", "PYTHONPYCACHEPREFIX"):
        monkeypatch.delenv(k, raising=False)
    got = env.setup_process_env(tmp_path)
    assert got["TRITON_CACHE_DIR"].startswith(str(tmp_path)) and got["HF_HOME"].startswith(str(tmp_path))
    assert got["PYTHONDONTWRITEBYTECODE"] == "1"


def test_resolve_input_zip(tmp_path):
    import zipfile

    z = tmp_path / "w.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("a.txt", "hi")
    out = env.resolve_input(str(z), tmp_path / "scratch", "wheel_dir")
    assert (out / "a.txt").read_text() == "hi"
    assert env.resolve_input(str(tmp_path), tmp_path, "x") == tmp_path
    with pytest.raises(FileNotFoundError):
        env.resolve_input(str(tmp_path / "missing"), tmp_path, "x")
