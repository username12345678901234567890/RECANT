# RECANT

**RE**versal **C**orrection from **AN**swer-conditioned **T**eacher — 번복 예측 보정 헤드
(Qwen3.5-9B, NVFP4 고정 백본 + bf16 LoRA r=128, 보정 헤드 δ / 크기 헤드 m̂ / 게이트 g).

이 저장소는 Kaggle 데이터셋으로 올려서 `/kaggle/input/...`에서 **복사 없이** 실행하도록 만들었다
(레포 디렉터리는 읽기 전용으로 취급하고, 모든 캐시는 scratch로 돌린다).

## 현재 구현 상태

| 단계 | 상태 |
| --- | --- |
| `prep_wheels.py` 오프라인 휠 준비 | 구현·검증 |
| `prep_data.py` 데이터 다운로드+토큰화+zip | 구현·검증 (실데이터 30M 토큰 end-to-end) |
| `train.py`, `bench.py`, 모델·손실·분기 pass·패커 | 미구현 (다음 단계) |

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
