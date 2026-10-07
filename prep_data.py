import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_out = '/kaggle/working/recant_data' if os.path.isdir('/kaggle/working') else './recant_data_out'
    p.add_argument('--out_dir', default=default_out)
    p.add_argument('--tmp_dir', default=None)
    p.add_argument('--tokenizer', default='Qwen/Qwen3.5-2B')
    p.add_argument('--token_budget', type=int, default=400000000)
    p.add_argument('--max_seq_len', type=int, default=98304)
    p.add_argument('--overlength', choices=['drop', 'truncate'], default='drop')
    p.add_argument('--harnesses', default='openhands,sweagent')
    p.add_argument('--models', default='')
    p.add_argument('--sources', default='')
    p.add_argument('--resolved_values', default='0,1')
    p.add_argument('--dead_after', type=int, default=3)
    p.add_argument('--max_per_instance', type=int, default=4)
    p.add_argument('--repo_frac', type=float, default=0.05)
    p.add_argument('--source_frac', type=float, default=0.35)
    p.add_argument('--class_frac', type=float, default=0.6)
    p.add_argument('--val_pct', type=int, default=1)
    p.add_argument('--num_proc', type=int, default=0)
    p.add_argument('--ram_frac', type=float, default=0.7)
    p.add_argument('--shard_tokens', type=int, default=1 << 27)
    p.add_argument('--hint_template', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max_files', type=int, default=0)
    p.add_argument('--verify_template', type=int, default=20)
    p.add_argument('--inspect', action='store_true')
    p.add_argument('--inspect_rows', type=int, default=60)
    p.add_argument('--no_zip', action='store_true')
    return p.parse_args(argv)

def main(argv=None):
    a = parse_args(argv)
    from recant.env import choose_tmp_dir, setup_process_env
    tmp_dir = choose_tmp_dir(a.tmp_dir)
    env_vars = setup_process_env(tmp_dir)
    print(f'[prep] scratch dir: {tmp_dir}', flush=True)
    from recant.data import prep
    prep.run(a, tmp_dir, env_vars)
if __name__ == '__main__':
    main()