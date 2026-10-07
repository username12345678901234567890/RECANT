import argparse
import os
import secrets
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model_path', required=True)
    p.add_argument('--checkpoint_dir', required=True)
    p.add_argument('--wheel_dir', default=None)
    p.add_argument('--tmp_dir', default=None)
    p.add_argument('--min_tmp_gb', type=float, default=2.0)
    p.add_argument('--host', default='0.0.0.0')
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--api_key', default=None)
    p.add_argument('--served_model_name', default='recant')
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--max_ctx', type=int, default=98304)
    p.add_argument('--max_chunk', type=int, default=16384)
    p.add_argument('--snapshots', type=int, default=6)
    p.add_argument('--no_fast', action='store_true')
    p.add_argument('--fast_disable', default='')
    g = p.add_argument_group('initial RECANT knobs (changeable at runtime via /v1/recant/config)')
    g.add_argument('--gate_mode', choices=['learned', 'fixed', 'off'], default='learned')
    g.add_argument('--gate_fixed', type=float, default=1.0)
    g.add_argument('--delta_scale', type=float, default=1.0)
    p.add_argument('--tunnel', action='store_true')
    p.add_argument('--tunnel_bin', default=None)
    return p.parse_args(argv)

def main(argv=None):
    a = parse_args(argv)
    from recant.env import choose_tmp_dir, find_root, install_wheels, resolve_input, setup_process_env
    tmp = choose_tmp_dir(a.tmp_dir, min_free_gb=a.min_tmp_gb)
    setup_process_env(tmp)
    log = lambda *m: print(*m, flush=True)
    if a.wheel_dir:
        log(f"[serve] installed wheels: {install_wheels(resolve_input(a.wheel_dir, tmp, 'wheel_dir'), tmp, log=log)}")
    import torch
    from recant import fast
    from recant.data.render import RenderError
    from recant.serve.chat import ChatRenderer
    from recant.serve.engine import Engine, Knobs
    from recant.serve.server import RecantServer
    fast.configure(a.no_fast, a.fast_disable)
    ckpt = find_root(resolve_input(a.checkpoint_dir, tmp, 'checkpoint_dir'), 'adapter.safetensors')
    model_path = find_root(resolve_input(a.model_path, tmp, 'model_path'), 'config.json')
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[a.dtype]
    engine = Engine.load(model_path, ckpt, a.device, dtype, a.max_chunk, a.max_ctx, a.snapshots, log)
    engine.knobs = Knobs(mode=a.gate_mode, gate_fixed=a.gate_fixed, delta_scale=a.delta_scale)
    from tokenizers import Tokenizer
    chat = ChatRenderer(Tokenizer.from_file(str(model_path / 'tokenizer.json')))
    key = a.api_key or secrets.token_urlsafe(24)
    server = RecantServer(engine, chat, key, a.served_model_name)
    httpd = server.serve(a.host, a.port)
    log(f'[serve] listening on http://{a.host}:{a.port}  model={a.served_model_name}')
    log(f'[serve] API key: {key}')
    tunnel = None
    if a.tunnel:
        from recant.serve import tunnel as tn
        try:
            tunnel, url = tn.start(a.port, tn.find_binary(a.tunnel_bin, tmp, log), log=log)
            log(f'[serve] public URL: {url}   (OpenAI base_url: {url}/v1)')
        except Exception as e:
            log(f'[serve] tunnel failed: {e!r}')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        if tunnel is not None:
            tunnel.terminate()
if __name__ == '__main__':
    main()