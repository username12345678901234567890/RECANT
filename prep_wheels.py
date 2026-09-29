#!/usr/bin/env python
"""Download the extra libraries as wheels and zip them (CPU session, internet ON).

  !python /kaggle/input/recant/prep_wheels.py

The GPU session installs them offline (`train.py --wheel_dir`). Anything already
present in the image at a new-enough version (torch, numpy, ...) is left alone.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
    p.add_argument("--out_dir", default=default_out)
    p.add_argument("--tmp_dir", default=None)
    p.add_argument("--python_version", default="3.12",
                   help="target (GPU image) Python; the default matches the confirmed Kaggle image. "
                        "Use 'native' to download for the running interpreter instead.")
    p.add_argument("--extras", default="", help="comma list: cutlass,tilelang")
    p.add_argument("--extra_req", action="append", default=[], help="additional pip requirement (repeatable)")
    p.add_argument("--flash_attn", action=argparse.BooleanOptionalAction, default=True,
                   help="try to fetch a prebuilt community flash-attn wheel")
    p.add_argument("--torch_tag", default="2.10", help="torch minor of the GPU image")
    p.add_argument("--cuda_tag", default="cu128")
    a = p.parse_args(argv)

    from recant import wheels
    from recant.env import choose_tmp_dir

    tmp = choose_tmp_dir(a.tmp_dir, min_free_gb=2.0)
    work = Path(tempfile.mkdtemp(dir=tmp))
    house = work / "wheelhouse"
    house.mkdir()

    reqs = list(wheels.CORE_REQS)
    for e in filter(None, a.extras.split(",")):
        reqs += wheels.EXTRA_REQS[e]
    reqs += a.extra_req
    native = f"{sys.version_info.major}.{sys.version_info.minor}"
    target = native if a.python_version == "native" else a.python_version
    print(f"[wheels] running Python {native}, target {target}; requirements: {reqs}", flush=True)
    if target != native:
        print(f"[wheels] NOTE: resolving with Python {native} but downloading for {target}", flush=True)

    pins = wheels.plan_install(reqs, work)
    print(f"[wheels] {len(pins)} package(s) to fetch (protected packages excluded):", flush=True)
    print("         " + ", ".join(f"{n}=={v}" for n, v in pins), flush=True)
    ok, failed = wheels.download(pins, house, None if target == native else target, None)

    fa = None
    if a.flash_attn:
        fa = wheels.find_flash_attn(house, "cp" + target.replace(".", ""), a.torch_tag, a.cuda_tag)

    meta = {"target_python": target, "resolved_with_python": native, "requirements": reqs,
            "downloaded": ok, "failed": failed, "flash_attn_wheel": fa,
            "expects_image": {"torch": f"{a.torch_tag}.x+{a.cuda_tag}"}}
    out = Path(a.out_dir) / "recant_wheels.zip"
    manifest = wheels.write_manifest_and_zip(house, out, meta)
    total = sum(w["mb"] for w in manifest["wheels"])
    print(f"[wheels] wrote {out}: {len(manifest['wheels'])} wheels, {total:.1f} MB; failed: {failed or 'none'}",
          flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
