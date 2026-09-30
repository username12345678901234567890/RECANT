#!/usr/bin/env python
"""Train the RECANT heads + LoRA on a frozen NVFP4 Qwen3.5-9B backbone (Kaggle GPU session).

  !python /kaggle/input/recant/train.py \\
      --model_path /kaggle/input/qwen35-9b --wheel_dir /kaggle/input/recant-wheels \\
      --data_dir /kaggle/input/recant-data --time_limit_hours 10

Only the final adapter, heads, logs and metrics are written to --out_dir (Kaggle working).
Everything else (caches, scratch) lives under a scratch dir picked automatically (/tmp by default).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("paths")
    g.add_argument("--model_path", required=True, help="HF Qwen3.5-9B dir (bf16 safetensors); quantized to NVFP4 on load")
    g.add_argument("--data_dir", required=True, help="prepared dataset dir or recant_data.zip (from prep_data.py)")
    g.add_argument("--wheel_dir", default=None, help="wheelhouse dir or recant_wheels.zip (from prep_wheels.py)")
    g.add_argument("--out_dir", default="/kaggle/working/recant_out" if os.path.isdir("/kaggle/working") else "./recant_out")
    g.add_argument("--tmp_dir", default=None, help="scratch (default: roomiest of /tmp,/kaggle/temp,/dev/shm)")
    g.add_argument("--min_tmp_gb", type=float, default=5.0)
    g = p.add_argument_group("time / schedule")
    g.add_argument("--time_limit_hours", type=float, default=10.0, help="hard limit incl. setup and final save")
    g.add_argument("--save_reserve_min", type=float, default=12.0, help="stop training this long before the limit")
    g.add_argument("--max_steps", type=int, default=None, help="debug: stop after N optimizer steps")
    g.add_argument("--warmup_steps", type=int, default=20)
    g.add_argument("--min_lr_ratio", type=float, default=0.1)
    g = p.add_argument_group("model")
    g.add_argument("--device", default="cuda")
    g.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    g.add_argument("--no_quantize", action="store_true", help="debug: dense bf16 backbone instead of NVFP4")
    g.add_argument("--lora_r", type=int, default=128)
    g.add_argument("--lora_alpha", type=float, default=128.0)
    g.add_argument("--head_hidden", type=int, default=1024)
    g.add_argument("--n_bins", type=int, default=51)
    g.add_argument("--y_max", type=float, default=8.0, help="log1p(KL) bin range if calibration is off")
    g = p.add_argument_group("optimization")
    g.add_argument("--lr_lora", type=float, default=1e-4, help="~10x a full-finetune LR")
    g.add_argument("--lr_heads", type=float, default=5e-4)
    g.add_argument("--grad_clip", type=float, default=1.0)
    g.add_argument("--tokens_per_step", type=int, default=100_000)
    g.add_argument("--lambda_corr", type=float, default=1.0)
    g.add_argument("--lambda_keep", type=float, default=0.1)
    g.add_argument("--lambda_mag", type=float, default=0.5)
    g.add_argument("--head_warmup_steps", type=int, default=500, help="h_mix stop-grad until this step ...")
    g.add_argument("--head_warmup_frac", type=float, default=0.15, help="... or this fraction of the time window")
    g.add_argument("--head_gain", type=float, default=0.1, help="head->backbone gradient gain after warm-up")
    g.add_argument("--head_gain_ramp", type=int, default=100)
    g = p.add_argument_group("branch pass / memory")
    g.add_argument("--max_branch_turns", type=int, default=16, help="commit turns branched per trajectory per step")
    g.add_argument("--n_keep", type=int, default=256, help="L_keep positions per trajectory")
    g.add_argument("--topk", type=int, default=256)
    g.add_argument("--max_chunk", type=int, default=16384, help="tokens per no-grad forward chunk")
    g.add_argument("--vram_target", type=float, default=0.90)
    g.add_argument("--bytes_per_token_prior", type=float, default=0.6 * 2**20)
    g = p.add_argument_group("calibration / eval / logging")
    g.add_argument("--calib_trajs", type=int, default=8, help="0 disables calibration")
    g.add_argument("--calib_turns", type=int, default=8)
    g.add_argument("--eval_every_min", type=float, default=45.0, help="0 disables periodic eval")
    g.add_argument("--eval_trajs", type=int, default=8)
    g.add_argument("--final_eval", action=argparse.BooleanOptionalAction, default=True)
    g.add_argument("--shadow_every_min", type=float, default=30.0)
    g.add_argument("--log_every", type=int, default=1)
    g.add_argument("--sys_every", type=int, default=10)
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    from recant import trainer

    trainer.run(a)


if __name__ == "__main__":
    main()
