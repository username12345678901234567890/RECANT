"""Runtime environment helpers for Kaggle: probes, scratch dir, caches, wheel install.

The repo is expected to run read-only from /kaggle/input, so every cache and
bytecode file is redirected under a scratch directory (never the repo or
/kaggle/working).
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TMP_CANDIDATES = ("/tmp", "/kaggle/temp", "/dev/shm")
PROBE_PATHS = ("/tmp", "/kaggle/temp", "/kaggle/working", "/kaggle/input", "/dev/shm")


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (out.stdout or out.stderr).strip()
    except Exception as e:  # noqa: BLE001 - probe must never crash the run
        return f"<unavailable: {e}>"


def disk_info(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        u = shutil.disk_usage(path)
    except OSError:
        return None
    return {
        "path": path,
        "total_gb": round(u.total / 2**30, 2),
        "free_gb": round(u.free / 2**30, 2),
        "writable": os.access(path, os.W_OK),
    }


def probe_env(extra_paths: tuple[str, ...] = ()) -> dict:
    """Collect everything worth knowing about the box; written to env.json."""
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "disks": [d for d in (disk_info(p) for p in PROBE_PATHS + tuple(extra_paths)) if d],
        "nvidia_smi": _run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,power.limit",
                            "--format=csv,noheader"]),
        "nvcc": _run(["nvcc", "--version"]).splitlines()[-1:] or None,
        "pip_versions": {},
    }
    try:
        import psutil

        vm = psutil.virtual_memory()
        info["ram_total_gb"] = round(vm.total / 2**30, 2)
        info["ram_available_gb"] = round(vm.available / 2**30, 2)
    except ImportError:
        info["ram_total_gb"] = None
    from importlib import metadata

    for name in ("torch", "triton", "numpy", "transformers", "tokenizers", "safetensors",
                 "flash-linear-attention", "flash-attn", "nvidia-cutlass-dsl", "tilelang",
                 "pyarrow", "huggingface-hub", "psutil"):
        try:
            info["pip_versions"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            info["gpu_arch_list"] = torch.cuda.get_arch_list()
    except Exception:  # noqa: BLE001
        pass
    return info


def choose_tmp_dir(requested: str | None, min_free_gb: float = 5.0,
                   candidates: tuple[str, ...] = DEFAULT_TMP_CANDIDATES) -> Path:
    """Pick the scratch dir. An explicit request wins; otherwise the writable
    candidate with the most free space. Fails loudly if nothing has room."""
    if requested:
        cands = (requested,)
    else:
        cands = candidates
    best, best_free = None, -1.0
    for c in cands:
        try:
            os.makedirs(c, exist_ok=True)
        except OSError:
            continue
        d = disk_info(c)
        if d and d["writable"] and d["free_gb"] > best_free:
            best, best_free = c, d["free_gb"]
    if best is None:
        raise RuntimeError(f"no writable scratch dir among {cands}")
    if best_free < min_free_gb:
        raise RuntimeError(f"scratch dir {best} has only {best_free:.1f} GB free (< {min_free_gb} GB)")
    root = Path(best) / "recant"
    root.mkdir(parents=True, exist_ok=True)
    return root


def setup_process_env(tmp_dir: Path) -> dict[str, str]:
    """Redirect every cache/bytecode location under tmp_dir. Call before importing
    triton / torch.compile / huggingface_hub so they see the new paths."""
    cache = tmp_dir / "cache"
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
        "TMPDIR": str(tmp_dir / "tmp"),
        "XDG_CACHE_HOME": str(cache / "xdg"),
        "HF_HOME": str(cache / "hf"),
        "HF_HUB_CACHE": str(cache / "hf" / "hub"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "inductor"),
        "TILELANG_CACHE_DIR": str(cache / "tilelang"),
        "CUDA_CACHE_PATH": str(cache / "cuda"),
        "TOKENIZERS_PARALLELISM": "true",
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    }
    for k, v in env.items():
        os.environ.setdefault(k, v)
        if k not in ("PYTHONDONTWRITEBYTECODE", "TOKENIZERS_PARALLELISM", "PYTORCH_CUDA_ALLOC_CONF"):
            os.makedirs(os.environ[k], exist_ok=True)
    sys.dont_write_bytecode = True
    return {k: os.environ[k] for k in env}


# --------------------------------------------------------------------------- wheels

_WHEEL_RE = re.compile(r"^(?P<name>[A-Za-z0-9_.]+?)-(?P<ver>[0-9][^-]*)(?:-\d[^-]*)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_wheel_name(filename: str) -> tuple[str, str] | None:
    m = _WHEEL_RE.match(filename)
    return (_norm(m["name"]), m["ver"]) if m else None


def _version_tuple(v: str) -> tuple:
    parts = re.findall(r"\d+", v.split("+")[0])
    return tuple(int(p) for p in parts[:4])


def install_wheels(wheel_dir: str | os.PathLike, tmp_dir: Path, log=print) -> list[str]:
    """Install wheels that are missing or older than the wheelhouse copy into
    tmp_dir/site (pip --target, --no-deps, --no-index) and prepend it to sys.path.

    We deliberately never touch packages that are already new enough, so the
    image's torch/triton/numpy stay untouched.
    """
    from importlib import metadata

    wheel_dir = Path(wheel_dir)
    wheels = sorted(wheel_dir.rglob("*.whl"))
    site = tmp_dir / "site"
    todo: list[Path] = []
    for w in wheels:
        parsed = parse_wheel_name(w.name)
        if not parsed:
            continue
        name, ver = parsed
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have is None or _version_tuple(have) < _version_tuple(ver):
            todo.append(w)
    if todo:
        site.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet",
               "--disable-pip-version-check", "--target", str(site), *map(str, todo)]
        log(f"[env] installing {len(todo)} wheel(s) into {site}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"wheel install failed:\n{r.stdout}\n{r.stderr}")
    if site.exists() and str(site) not in sys.path:
        sys.path.insert(0, str(site))
    return [w.name for w in todo]


def resolve_input(path: str, tmp_dir: Path, what: str) -> Path:
    """Accept a directory or a .zip; a zip is extracted under tmp_dir once."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"--{what} not found: {path}")
    if p.is_file() and p.suffix == ".zip":
        import zipfile

        dst = tmp_dir / "unzipped" / p.stem
        marker = dst / ".done"
        if not marker.exists():
            dst.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(p) as z:
                z.extractall(dst)
            marker.write_text("ok")
        return dst
    return p


def write_json(path: str | os.PathLike, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
