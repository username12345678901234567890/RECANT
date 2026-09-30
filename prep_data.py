#!/usr/bin/env python
"""Download nvidia/Open-SWE-Traces, tokenize, and write recant_data.zip (CPU session, internet ON).

  !python /kaggle/input/recant/prep_data.py --token_budget 400000000 --max_seq_len 98304
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = "/kaggle/working/recant_data" if os.path.isdir("/kaggle/working") else "./recant_data_out"
    p.add_argument("--out_dir", default=default_out, help="final zip + prep_report.json (Kaggle working)")
    p.add_argument("--tmp_dir", default=None, help="scratch; default: auto-pick the roomiest of /tmp,/kaggle/temp,/dev/shm")
    p.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B",
                   help="tokenizer dir or HF repo id (tokenizer.json is byte-identical across Qwen3.5 sizes)")
    p.add_argument("--token_budget", type=int, default=400_000_000)
    p.add_argument("--max_seq_len", type=int, default=98304)
    p.add_argument("--overlength", choices=["drop", "truncate"], default="drop")
    p.add_argument("--harnesses", default="openhands,sweagent")
    p.add_argument("--models", default="", help="comma list to restrict teacher models (default all)")
    p.add_argument("--sources", default="", help="comma list: swe-rebench-v2,scale-swe (default all)")
    p.add_argument("--resolved_values", default="0,1",
                   help="resolved labels to keep. -1 = unlabeled (e.g. all of openhands/qwen36_27b and "
                        "openhands/deepseek_v4_flash); if you add it, the trainer must give those no CE loss")
    p.add_argument("--dead_after", type=int, default=3,
                   help="skip the rest of a source after this many files with no usable rows")
    p.add_argument("--max_per_instance", type=int, default=4)
    p.add_argument("--repo_frac", type=float, default=0.05, help="max share of the budget per repo")
    p.add_argument("--source_frac", type=float, default=0.35, help="max share per harness/model/source")
    p.add_argument("--class_frac", type=float, default=0.60, help="max share per resolved class")
    p.add_argument("--val_pct", type=int, default=1)
    p.add_argument("--num_proc", type=int, default=0, help="0 = min(cpus, 6)")
    p.add_argument("--ram_frac", type=float, default=0.70, help="target max system RAM use of the shuffle buffer")
    p.add_argument("--shard_tokens", type=int, default=1 << 27)
    p.add_argument("--hint_template", default=None, help="text file with a {lines} placeholder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_files", type=int, default=0, help="debug: cap number of parquet files")
    p.add_argument("--verify_template", type=int, default=20,
                   help="compare N rows of the first file against transformers' chat template (0 = off)")
    p.add_argument("--inspect", action="store_true", help="smoke test on a few rows per source, then exit")
    p.add_argument("--inspect_rows", type=int, default=60)
    p.add_argument("--no_zip", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from recant.env import choose_tmp_dir, setup_process_env

    tmp_dir = choose_tmp_dir(a.tmp_dir)
    env_vars = setup_process_env(tmp_dir)
    print(f"[prep] scratch dir: {tmp_dir}", flush=True)
    from recant.data import prep

    prep.run(a, tmp_dir, env_vars)


if __name__ == "__main__":
    main()
