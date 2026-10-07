from __future__ import annotations
import re
import shutil
import stat
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
URL_RE = re.compile('https://[a-z0-9-]+\\.trycloudflare\\.com')
DOWNLOAD = 'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64'

def find_binary(explicit: str | None, scratch: Path, log=print) -> str:
    if explicit:
        return explicit
    found = shutil.which('cloudflared')
    if found:
        return found
    dst = scratch / 'cloudflared'
    if not dst.exists():
        log(f'[tunnel] downloading cloudflared to {dst} (needs internet)')
        urllib.request.urlretrieve(DOWNLOAD, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IEXEC)
    return str(dst)

def start(port: int, binary: str, timeout: float=90.0, log=print):
    proc = subprocess.Popen([binary, 'tunnel', '--no-autoupdate', '--url', f'http://127.0.0.1:{port}'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found: list[str] = []

    def pump():
        for line in proc.stdout:
            m = URL_RE.search(line)
            if m and (not found):
                found.append(m.group(0))
    threading.Thread(target=pump, daemon=True).start()
    t0 = time.time()
    while not found and time.time() - t0 < timeout and (proc.poll() is None):
        time.sleep(0.5)
    if not found:
        proc.terminate()
        raise RuntimeError('cloudflared did not report a public URL (internet off, or the binary failed to start)')
    return (proc, found[0])