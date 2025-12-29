# RunPod Web Terminal에서 Qwen2-VL LoRA 파인튜닝 과정을 “기록 + 재현”하는 템플릿

RunPod 웹터미널에서 이것저것 직접 치면, 나중에 **무슨 명령을 어떤 순서로 실행했는지** 남기기 어렵습니다.
이 폴더는 **setup → train → infer**를 전부 스크립트/로그로 남겨서,

- RunPod에서 다시 들어와도
- 다른 컴퓨터에서

동일한 과정을 그대로 실행/설명할 수 있게 만드는 템플릿입니다.

---

## 0) 권장 워크플로우(요약)

```bash
# 0) (완전 처음) RunPod 새 Pod에서 레포부터 받기
# / 에서 바로 치지 말고, /workspace 같은 작업 폴더로 이동해서 진행하세요.
cd /workspace
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_NAME>

# (중요) 공용 RunPod workspace에서 내 결과물 폴더를 분리하고 싶으면
export RUNPOD_USER="your_name_or_id"  # 예: kimjunyoung / 20210001

# 2) 환경 구축
bash runpod/setup.sh

# 3) 학습(로그 파일로 저장)
bash runpod/train_qwen2vl_lora.sh 2>&1 | tee runpod/logs/train_$(date +%Y%m%d_%H%M%S).log

# 4) 추론 확인(학습 결과 재현)
bash runpod/infer_qwen2vl_lora.sh

# 5) 로컬로 다운로드할 tar.gz 생성(학습 스크립트에서 자동 생성되지만, 필요하면 다시 생성 가능)
bash runpod/export_artifacts.sh
```

### 0-0) “아무것도 없는 상태”에서 한 번에 시작하고 싶다면

1) RunPod 새 Pod 터미널에서 아래처럼 실행

```bash
cd /workspace
export REPO_URL=<YOUR_REPO_URL>
git clone $REPO_URL
cd $(basename $REPO_URL .git)
bash runpod/setup.sh
```

2) (선택) bootstrap 스크립트 사용

레포를 받은 뒤에는 아래가 가능합니다:

```bash
export REPO_URL=<YOUR_REPO_URL>
bash runpod/bootstrap_from_scratch.sh
```

> 참고: RunPod 환경에서 레포를 받기 전에는 `runpod/bootstrap_from_scratch.sh` 파일이 존재하지 않아서 실행할 수 없습니다.
> 먼저 `git clone`을 해야 합니다.

---

## 0.1) 웹터미널에서 “내가 쳤던 명령”도 남기고 싶다면(선택)

### 옵션 A: 전체 세션 로그 남기기
```bash
mkdir -p runpod/logs
script -q -f runpod/logs/terminal_session_$(date +%Y%m%d_%H%M%S).log

# 이제부터 터미널에 치는 모든 것이 log에 기록됨
# ...작업...

exit  # script 종료
```

### 옵션 B: tmux로 세션 유지
```bash
tmux new -s train
# 여기서 학습 실행
# detach: Ctrl+b 누르고 d
tmux attach -t train
```

---

## 1) 학습에 필요한 것

### (A) 데이터
이 템플릿은 JSONL을 가정합니다.

`train.jsonl` 예시:
```json
{"image":"/workspace/data/img001.jpg","question":"스티커가 있나요? 있으면 번호와 색을 말해줘.","answer":"{\"is_target\":true,\"has_sticker\":true,\"number\":\"42\",\"color\":\"초록색\"}"}
```

**중요:** `image`는 RunPod 컨테이너 내부의 실제 파일 경로여야 합니다.

#### (A-1) 라벨이 아직 없다면(권장 워크플로우)

> 결론: 전부 수작업으로 라벨링하는 방식은 비추천입니다.
> 먼저 base 모델로 **자동 라벨링(pseudo-label)** 을 하고, 사람이 틀린 것만 빠르게 고치는 방식이 가장 빠릅니다.

0) (권장) 자동 라벨링(pseudo-label)로 초안 만들기

```bash
source .venv/bin/activate
python runpod/pseudo_label_qwen2vl.py --load_4bit --image_dir data/motor_checker --out runpod/labels.auto.csv
```

만약 그래도 VRAM이 부족하면(Out of memory):

```bash
# 더 작은 모델로 pseudo-label (예: 2B/3B 계열이 있다면 그걸 사용)
python runpod/pseudo_label_qwen2vl.py --load_4bit --model <SMALLER_MODEL_ID> --image_dir data/motor_checker --out runpod/labels.auto.csv
```

1) `runpod/labels.auto.csv`를 열어서
- `has_sticker/color/number`가 틀린 행만 수정
- 잘 모르겠는 행은 `raw` 컬럼(모델 원문 출력)을 보고 판단

2) 수정한 파일을 `runpod/labels.csv`로 저장(또는 `--labels_csv`에 그대로 사용)

1) 라벨 템플릿 CSV 생성

```bash
source .venv/bin/activate
python runpod/make_label_template.py --image_dir data/motor_checker --out runpod/labels.csv
```

2) `runpod/labels.csv`를 엑셀/구글시트로 열어서 아래 컬럼을 채우기

- `has_sticker`: true/false
- `color`: 초록색/노란색/빨간색 (스티커 없으면 비움)
- `number`: 숫자만 (스티커 없으면 비움)

3) 학습용 JSONL 생성

```bash
python runpod/make_train_jsonl.py --labels_csv runpod/labels.csv --out train.jsonl --image_root /workspace
```

4) RunPod에서 학습 실행 시 `TRAIN_JSONL=/workspace/train.jsonl` 로 지정

---

#### (A-2) 왜 2단계(스티커 유무→번호/색) 대신 단일 JSON을 쓰나?

이번 과제/서버(worker.py)는 1회 호출로 `{has_sticker, number, color}`를 받으면 가장 단순합니다.
단, 실제 운영에서 "먼저 스티커 유무만 빠르게 판별"하고 싶다면 2단계(게이트)도 가능하고,
그때는 JSONL의 question/answer 스키마를 2단계로 바꾸면 됩니다.

### (B) 모델
- 기본값: `Qwen/Qwen2-VL-7B-Instruct`

---

## 2) 결과물(어디에 저장되나)

- LoRA 어댑터(기본): `outputs/users/<RUNPOD_USER>/qwen2vl-lora/`
- 로컬 다운로드용 아카이브: `outputs/users/<RUNPOD_USER>/exports/*.tar.gz`
- 로그: `runpod/logs/`

RunPod Pod가 종료되면 컨테이너 파일이 날아갈 수 있으니, 결과물은 반드시:
- RunPod Volume(영구 볼륨)
- HuggingFace Hub
- S3/GDrive

중 하나로 복사/업로드하세요.

---

## 3) “내가 뭘 했는지” 증빙하기 팁

1) 학습 시작 커맨드는 항상 `tee`로 로그를 남기기
2) `outputs/` 폴더 전체를 아카이브해서 보관
3) (선택) W&B를 붙이면 웹에서 학습 곡선을 그대로 보여줄 수 있음

---

## 4) 다른 컴퓨터에서 “진행했던 모든 과정” 보여주는 방법

다른 PC에서는 아래 3가지를 같이 보여주면 가장 깔끔합니다.

1) **스크립트 자체**: `runpod/setup.sh`, `runpod/train_qwen2vl_lora.sh`, `runpod/train_qwen2vl_lora.py`, `runpod/infer_qwen2vl_lora.*`
2) **로그 파일**: `runpod/logs/train_*.log` (학습 step/loss가 남음)
3) **결과물**: `outputs/qwen2vl-lora/` (LoRA adapter)

이렇게 남겨두면, RunPod가 아니라도 동일한 커맨드로 재현이 가능합니다.

---

## 5) 회전/뒤집힘/왜곡(deformation) 대응 증강(augmentation)

숫자가 기울거나 뒤집히면 잘못 읽는 문제를 줄이기 위해 학습 중 **온더플라이 증강**을 적용합니다.

- 적용 위치: `runpod/train_qwen2vl_lora.py` → `runpod/augment.py`
- 포함: Rotate(최대 180도), Perspective, Affine(shear), Blur/Noise, Brightness/Contrast

학습 시 환경변수로 제어:

```bash
export AUGMENT=true
export AUG_P=0.85
export AUG_MAX_ROTATE=180
```

---

## 6) Hugging Face Hub로 모델 저장(업로드)

### 6-1) 결론: "가중치만" 가져와서 저장하는 건가?

LoRA 파인튜닝이면 보통 **"추가된 LoRA 어댑터 가중치"만 업로드**합니다(권장).

- 업로드 결과물(기본): `outputs/users/<RUNPOD_USER>/qwen2vl-lora/` 폴더(LoRA adapter + tokenizer/processor)
- 추론 시 필요: base 모델(`Qwen/Qwen2-VL-7B-Instruct`) + 어댑터를 합쳐서 로드

전체 모델(7B)을 통째로 올리고 싶다면, LoRA를 base에 **merge**해서 올릴 수도 있지만,
용량/업로드 시간/스토리지 비용이 커집니다.

### 6-2) RunPod에서 업로드 절차

1) HF 토큰 준비(개인계정 토큰)

```bash
export HF_TOKEN=hf_xxx
export HF_REPO_ID=yourname/motor-sticker-qwen2vl-lora
```

2) (권장) 어댑터만 업로드

```bash
export MODE=adapter
bash runpod/push_to_hf.sh
```

3) (선택) merge한 전체 모델 업로드

```bash
export MODE=merged
export BASE_MODEL=Qwen/Qwen2-VL-7B-Instruct
bash runpod/push_to_hf.sh
```

---

## 7) 다른 환경(서버/로컬)에서 사용하기

### 7-1) LoRA 어댑터를 가져와서 추론하기

`runpod/infer_qwen2vl_lora.py`는 로컬 폴더 어댑터를 읽게 되어있습니다.
Hub에서 바로 쓰려면 아래처럼 "어댑터 repo"를 `ADAPTER_DIR`로 지정해도 됩니다.

```bash
export MODEL_NAME=Qwen/Qwen2-VL-7B-Instruct
export ADAPTER_DIR=yourname/motor-sticker-qwen2vl-lora
export TEST_IMAGE=data/motor_checker/20240817_000105.jpg
python runpod/infer_qwen2vl_lora.py
```

### 7-2) student_template에 적용(중요)

현재 `student_template/worker.py`는 OpenAI 호환 API로 추론합니다.
즉, **(A) vLLM 같은 서버로 배포해서 OpenAI API 형태로 호출**하거나,
**(B) worker.py를 수정해서 로컬 HF 모델을 직접 로드**하는 2가지 옵션이 있습니다.

원하면 (B)로 바꾸는 패치(로컬 모델 추론)도 같이 만들어드릴게요.
