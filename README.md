# RECANT

**RE**versal **C**orrection from **AN**swer-conditioned **T**eacher — 번복 예측 보정 헤드
(Qwen3.5-9B, NVFP4 고정 백본 + bf16 LoRA r=128, 보정 헤드 δ / 크기 헤드 m̂ / 게이트 g).

이 저장소는 Kaggle 데이터셋으로 올려서 `/kaggle/input/...`에서 **복사 없이** 실행하도록 만들었다
(레포 디렉터리는 읽기 전용으로 취급하고, 모든 캐시는 scratch로 돌린다).

## 현재 구현 상태

| 구성 | 상태 |
| --- | --- |
| `prep_wheels.py`, `prep_data.py` | 구현·실데이터 검증 (30M 토큰 end-to-end) |
| 모델 (`recant/model/`) | GDN+attention 하이브리드, LoRA, NVFP4 백본. **HF `Qwen3_5ForCausalLM`과 로짓 일치**(테스트) |
| 분기 통합 pass + 수동 층별 역전파 (`recant/branch.py`) | 구현. `p_full`이 힌트 붙인 전체 시퀀스 forward와 일치, 역전파가 autograd와 일치(테스트) |
| 헤드·손실·스케줄·저장·플래너·트레이너 (`train.py`) | 구현. CPU 작은 모델로 end-to-end(시간 제한 종료, 모의 OOM 복구, 크래시 시 저장 포함) |
| **GPU 실행** | **미검증** — 이 저장소를 만든 환경에는 GPU가 없다. NVFP4 GEMM(`torch._scaled_mm`), fla GDN, sm_120 SDPA 경로는 시작 시 자동 점검(`backends.json`)으로 처음 통과한 후보를 쓰고 실패하면 참조 구현으로 내려간다 |
| 미구현 | CUTLASS-DSL FP4 GEMM, Transformer Engine, FlashQLA, layer-synchronous 융합 분기(대신 캐시 청크 pass가 체크포인트 저장을 겸함), 별도 벤치 스크립트 |

## 1. 휠 준비 (인터넷 ON, CPU 세션)

```
!python /kaggle/input/recant/prep_wheels.py            # -> /kaggle/working/recant_wheels.zip
```

현재 이미지(`pip install --dry-run --report`)에 **이미 있는 패키지는 받지 않고**, torch·triton·nvidia-* 는 절대 받지 않는다.
대상 Python은 기본 3.12(확인된 Kaggle GPU 이미지). 결과 zip을 Kaggle 데이터셋으로 올린다.
학습 스크립트는 `--wheel_dir`(폴더 또는 zip)에서 `pip install --no-index --no-deps --target <scratch>/site`로 필요한 것만 설치한다.

## 2. 데이터 준비 (인터넷 ON, CPU 세션)

```
!python /kaggle/input/recant/prep_data.py --token_budget 400000000 --max_seq_len 98304
# -> /kaggle/working/recant_data.zip + prep_report.json
```

`nvidia/Open-SWE-Traces`에서 **Python · resolved∈{0,1} · openhands+sweagent**만 골라 Qwen3.5 템플릿으로 렌더링·토큰화한다.

- **빠른 토큰화**: jinja 없이 자체 렌더러 + `tokenizers.encode_batch`. 렌더러는 모델의 채팅 템플릿과 토큰 ID 단위로 일치한다
  (`--verify_template`, 기본 20행). 주의: `transformers` 5.x의 `Qwen2Tokenizer`는 `tokenizer.json`의 pre-tokenizer 정규식
  (`\p{M}` 포함)을 옛 정규식으로 바꿔 쓰는 경우가 있어, 우리는 **원본 `tokenizer.json`** 기준으로 토큰화한다(리포트의
  `hf_tokenizer_divergence`가 그 차이를 세어 준다).
- **RAM 적응형 셔플러**: 버퍼 크기가 실시간 가용 RAM(`--ram_frac`)을 따라간다.
- **선택 정책**: 이슈당/레포당/소스당/라벨 클래스당 상한(`--max_per_instance`, `--repo_frac`, `--source_frac`, `--class_frac`),
  `instance_id` 해시로 train/val 분할, 라벨이 전부 −1인 소스는 자동으로 건너뜀.
- **길이**: 궤적 p10/p50/p90 = 38K/63K/94K 토큰. 기본 `--max_seq_len 98304 --overlength drop`(≈92% 보존).
  `--overlength truncate`는 마지막 완결 assistant 턴에서 자른다.

### 산출물 형식 (`recant/data/writer.py`, 읽기는 `recant/data/store.py`)

`tokens_XXX.bin`(uint32) · `hints.bin` · `index.parquet` · `manifest.json` · `prep_report.json`.
`index.parquet`의 `turns`는 assistant 턴마다 8개 int (`recant/data/render.py`의 `T_*`): 턴 시작, CE 시작, CE 끝,
툴 호출 시작·끝, 앵커(툴 호출 직전 토큰), 플래그(커밋/finish/추론 유무), 호출 수.
힌트는 파일 경로와 `@@` 헤더의 함수·클래스 이름만 담는다(diff 본문 없음).

## 테스트

```
RECANT_TOKENIZER_DIR=/path/to/Qwen3.5-9B-tokenizer-dir python -m pytest tests -q
```

## 3. 학습 (GPU 세션)

```
!python /kaggle/input/recant/train.py \
    --model_path /kaggle/input/qwen35-9b --wheel_dir /kaggle/input/recant-wheels \
    --data_dir /kaggle/input/recant-data --time_limit_hours 10
```

- `--model_path`/`--data_dir`는 중첩 디렉터리여도 `config.json`/`manifest.json`을 자동으로 찾는다. 휠·데이터는 zip 경로도 가능.
- 작업은 scratch(`/tmp` 등 가장 여유 있는 곳)에서 하고 `--out_dir`(Kaggle working)에는 **최종 어댑터·헤드와 로그·지표만** 쓴다:
  `adapter.safetensors`, `heads.safetensors`, `config.json`, `trainer_state.json`, `train.jsonl`, `eval.jsonl`,
  `env.json`, `backends.json`, `calibration.json`, `summary.json`, `stdout.log`.
- 시간 기반 스케줄: `--time_limit_hours`(설정·저장 포함 하드 제한) − `--save_reserve_min` 까지 학습하고 저장한다.
  종료 신호·예외·OOM 후에도 `finally`에서 저장하며, 30분마다 호스트 RAM에 그림자 복사본을 갱신해 GPU 컨텍스트가
  망가져도 저장할 수 있다.
- 첫 GPU 실행은 짧은 스모크 런으로: `--max_steps 2 --calib_trajs 2 --time_limit_hours 0.5`.
  `backends.json`이 어떤 커널 경로를 골랐는지, `train.jsonl`의 `mem_c0_gb`/`mem_c1_mb_per_tok`/`tok_per_s`가 메모리·속도를 보여 준다.

한 trajectory의 흐름(`recant/branch.py`): 캐시 청크 forward로 본 스트림을 돌리며 층 입력을 기록(=활성 체크포인트) →
샘플한 커밋 턴마다 캐시를 fork해 `힌트+턴`을 돌려 `p_full`(top-256+꼬리)만 저장 → 기록한 최종 hidden에서 `p_plain`·
크기 타깃 계산 → 헤드/손실 → 층별로 입력에서 재계산하며 역전파. 본 forward를 별도로 한 번 더 돌지 않는다.
