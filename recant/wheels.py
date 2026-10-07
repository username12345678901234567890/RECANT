from __future__ import annotations
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
PROTECTED = re.compile('^(torch|torchvision|torchaudio|triton|nvidia-.*|pytorch-triton.*)$')
CORE_REQS = ['flash-linear-attention', 'einops', 'transformers', 'tokenizers', 'safetensors', 'huggingface_hub', 'pyarrow', 'psutil', 'nvidia-ml-py', 'numpy', 'pytest']
EXTRA_REQS = {'cutlass': ['nvidia-cutlass-dsl'], 'tilelang': ['tilelang']}
LINUX_PLATFORMS = ['manylinux_2_28_x86_64', 'manylinux_2_17_x86_64', 'manylinux2014_x86_64', 'linux_x86_64']

def norm(name: str) -> str:
    return re.sub('[-_.]+', '-', name).lower()

def pip(*args: str, check: bool=True) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, '-m', 'pip', '--disable-pip-version-check', *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"pip {' '.join(args[:4])} failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return r

def plan_install(reqs: list[str], workdir: Path) -> list[tuple[str, str]]:
    report = workdir / 'pip_report.json'
    pip('install', '--dry-run', '--quiet', '--report', str(report), *reqs)
    items = json.loads(report.read_text()).get('install', [])
    out = []
    for it in items:
        md = it['metadata']
        name = norm(md['name'])
        if PROTECTED.match(name):
            continue
        out.append((name, md['version']))
    return sorted(set(out))

def download(pins: list[tuple[str, str]], dest: Path, py_version: str | None, platforms: list[str] | None, log=print) -> tuple[list[str], list[str]]:
    ok, failed = ([], [])
    for name, ver in pins:
        cmd = ['download', '--no-deps', '--only-binary=:all:', '-d', str(dest), f'{name}=={ver}']
        if py_version:
            nodot = py_version.replace('.', '')
            cmd += ['--python-version', py_version, '--implementation', 'cp', '--abi', f'cp{nodot}', '--abi', 'abi3', '--abi', 'none']
            for p in platforms or LINUX_PLATFORMS:
                cmd += ['--platform', p]
        r = pip(*cmd, check=False)
        if r.returncode == 0:
            ok.append(f'{name}=={ver}')
        else:
            failed.append(f'{name}=={ver}')
            log(f'[wheels] FAILED {name}=={ver}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}')
    return (ok, failed)

def find_flash_attn(dest: Path, py_tag: str, torch_tag: str, cuda_tag: str, log=print) -> str | None:
    api = 'https://api.github.com/repos/mjun0812/flash-attention-prebuild-wheels/releases?per_page=100'
    try:
        req = urllib.request.Request(api, headers={'User-Agent': 'recant-prep'})
        releases = json.load(urllib.request.urlopen(req, timeout=30))
    except Exception as e:
        log(f'[wheels] flash-attn lookup skipped: {e!r}')
        return None
    best = None
    for rel in releases:
        for asset in rel.get('assets', []):
            n = asset['name']
            if n.startswith('flash_attn-') and py_tag in n and (f'torch{torch_tag}' in n) and (cuda_tag in n) and ('linux' in n) and ('x86_64' in n):
                ver = re.match('flash_attn-([0-9.]+)', n)
                key = tuple((int(x) for x in ver.group(1).split('.') if x.isdigit())) if ver else ()
                if best is None or key > best[0]:
                    best = (key, asset)
    if not best:
        log(f'[wheels] no prebuilt flash-attn for {py_tag}/torch{torch_tag}/{cuda_tag}')
        return None
    url, name = (best[1]['browser_download_url'], best[1]['name'])
    dst = dest / name.replace('%2B', '+')
    urllib.request.urlretrieve(url, dst)
    log(f'[wheels] flash-attn wheel: {dst.name}')
    return dst.name

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()

def write_manifest_and_zip(wheelhouse: Path, out_zip: Path, meta: dict) -> dict:
    wheels = sorted(wheelhouse.glob('*.whl'))
    manifest = {**meta, 'wheels': [{'file': w.name, 'sha256': sha256(w), 'mb': round(w.stat().st_size / 2 ** 20, 2)} for w in wheels]}
    (wheelhouse / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_STORED, allowZip64=True) as z:
        for f in sorted(wheelhouse.iterdir()):
            z.write(f, f.name)
    return manifest