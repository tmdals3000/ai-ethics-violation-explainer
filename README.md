# ai-ethics-violation-explainer

텍스트(챗봇 답변·댓글)의 AI 윤리 규칙 위반 판정이 틀렸을 때, **규칙 찾기 · 규칙 문장 · 판정 모델 · 사람 라벨** 중 어디가 문제인지 밝히고, 규칙을 고치면 판정이 얼마나 나아지는지 확인하는 시스템 (Project 4 Baseline).

흐름: 데이터 준비 → P1 규칙 찾기 → P2 판정 비교 → P3 오류 진단 → P4 수정 효과 → Streamlit 화면

- 모델 가중치는 학습하지 않는다 (실행 + 팀 판정기 프롬프트·임계값만 개발).
- 임계값·프롬프트는 dev 에서만 정한다. test 지표는 `--allow-test` 를 줄 때만 계산한다.
- 임시로 정한 내용: [docs/ASSUMPTIONS.md](docs/ASSUMPTIONS.md) · 결과 요약: [docs/BASELINE_REPORT.md](docs/BASELINE_REPORT.md)

## 폴더

| 폴더 | 내용 |
|---|---|
| `configs/baseline.yaml` | 모든 설정 (경로, 표본 크기, τ, 임계값, 모델) |
| `src/` | `data` 데이터 준비 · `retrieval` P1 · `judges` 판정 · `metrics` P2 · `diagnose` P3 · `effects` P4 · `adapters` 조교 파일 읽기 |
| `scripts/run.py` | 전체 실행기 |
| `app/app.py` | Streamlit 화면 |
| `tests/` | pytest. 가짜 데이터는 `tests/fixtures` 에만 |
| `results/` | 지표·판정 결과 (원문 없음, text_id 만) |

원본 데이터·가공 데이터·캐시는 저장소 밖(`C:/ml/...`)에 두며 커밋하지 않는다.

## 재현

```bash
# 1) 환경 (OneDrive 밖에 venv)
python -m venv C:/ml/aiethics/.venv
C:/ml/aiethics/.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env        # HF_TOKEN= 채우기

# 2) 토큰·라이선스
#  - HF 에서 meta-llama/Llama-Guard-3-1B, google/shieldgemma-2b 라이선스 동의
#  - Kaggle 대회 jigsaw-agile-community-rules 참가 동의 + ~/.kaggle/kaggle.json

# 3) 원본 데이터 (C:/ml/raw)
git clone --depth 1 https://github.com/libr-ai/do-not-answer C:/ml/raw/dna
kaggle competitions download -c jigsaw-agile-community-rules -p C:/ml/raw/jigsaw
cd C:/ml/raw/jigsaw && unzip jigsaw-agile-community-rules.zip && cd -
# ProsocialDialog 는 실행 중 HF 에서 자동으로 받음

# 4) 실행 (PowerShell 에서는 $env:PYTHONUTF8=1)
export PYTHONUTF8=1
PY=C:/ml/aiethics/.venv/Scripts/python
$PY -m pytest -q tests
$PY scripts/run.py --steps data,p1,judge,p2,p3,p4
$PY scripts/run.py --steps judge --models team   # 일부 모델만 다시

# 5) 화면
C:/ml/aiethics/.venv/Scripts/streamlit run app/app.py
```

조교 파일(`similarity.csv`, `coords.csv`, `judgments.csv`, `splits.csv`)이 오면 `C:/ml/raw/ta/` 에 두면 자동으로 우선 사용한다 (`src/adapters.py`).

## 실험 결과

### 기록 규칙

- 원자료는 `results/experiments.csv` 하나다. 컬럼: `version, run_id, date, commit, split, table, source, model, metric, value, note`.
- 새 실험은 **행을 추가만** 한다. 이전 결과는 지우거나 고치지 않는다.
  - `version`: 결과 묶음 이름 (예: `baseline-v0`, `baseline-v1`). 같은 version 은 다시 기록할 수 없다.
  - `run_id`: 판정기 하나의 설정 — 기준 모델 `B-LG`(Llama Guard) · `B-QG`(Qwen3Guard) · `B-SG`(ShieldGemma), 팀 판정기 `T1`, `T2`… (프롬프트·임계값이 바뀌면 새 번호), 규칙 찾기 `R1`, `R2`…, 모델 쌍 κ 는 `B-LG×B-QG` 형식.
  - `commit`: 그 결과를 만든 코드의 커밋 해시.
- 순서: 실행 → 결과 커밋 → `python scripts/record_run.py record --version <v> --date <YYYY-MM-DD> --commit <hash>` → `python scripts/record_run.py readme --version <v>` → 커밋 · 태그.
- README 표는 손으로 고치지 않는다. 위 명령이 `experiments.csv` 에서 만든다. 새 version 은 아래 자리표시 바로 밑(맨 위)에 쌓인다.
- 수치는 dev 결과만 적는다. test 결과는 `split=test` 행으로 따로 기록하고 표 제목에 `test` 를 쓴다.

### 알려진 문제 (Baseline v0)

1. **임시 split·규칙 트리:** 조교 `splits.csv`·similarity·coords 가 없어 고정 시드 split, roberta 유사도, PCA coords 를 임시로 만들었다. 규칙 트리는 DNA·Jigsaw 를 한 트리에 넣었고, DNA 텍스트 298건(13.2%)의 top-1 이 Jigsaw 규칙이다 → [BASELINE_REPORT 질문 1](docs/BASELINE_REPORT.md#7-멘토링조교-질문).
2. **Jigsaw·ProsocialDialog 는 dev 50건 표본:** 지표의 변동이 크다. DNA 만 dev 전체(2,256건)다.
3. **팀 v1·ShieldGemma 의 DNA 판정 대상 의심:** DNA 에서 AUC 가 팀 v1 0.434, ShieldGemma 0.516 으로 무작위 수준 이하다. DNA 응답 대부분이 거절문인데, 거절문 속 유해 주제 언급을 위반으로 보는 것으로 의심된다(질문·응답 중 무엇을 판정하는지 확인 필요).
4. **최댓값 집계 문제:** base 는 상위 5 후보 규칙의 위반 확률 최댓값을 쓴다. P1 top-1 이 0.32 라 후보 대부분이 엉뚱한 규칙이고, DNA 에서는 정답 규칙을 뺀 rule_excluded AUC 가 base 보다 높다(ShieldGemma 0.583 vs 0.516, 팀 0.535 vs 0.434).
5. **PD 팀 임계값이 0.01 에 걸림:** 출처별 dev F1 최대 기준으로 고르면, 위반 비율이 72% 인 ProsocialDialog 에서는 '거의 전부 위반'이 F1 을 최대로 만들어 임계값이 격자 최솟값 0.01 이 된다.

<!-- results:new -->

<!-- results:baseline-v0 -->
### Baseline v0 (dev, 2026-11-08)

> ⚠ **모든 수치는 dev 결과다. test 는 아직 보지 않았다.** 검증 상태: `IS만` (임계값·τ 를 정한 dev 로 평가).
> 표본: DNA dev 전체 2,256건, Jigsaw·ProsocialDialog dev 각 50건. split·규칙 트리는 임시([ASSUMPTIONS](docs/ASSUMPTIONS.md)).
> 원자료: `results/experiments.csv` (version = `baseline-v0`). 이 표는 `python scripts/record_run.py readme --version baseline-v0` 로 만든다.

**실험 설정**

| run_id | 모델 | prompt_version | threshold | threshold_rule | tau | aggregation | 커밋 |
|---|---|---|---|---|---|---|---|
| B-LG | meta-llama/Llama-Guard-3-1B | official | 0.5 | 고정 0.5 | - (규칙 안 받음) | - (1회 판정) | `1e2c384` |
| B-QG | Qwen/Qwen3Guard-Gen-0.6B | official | 0.5 | 고정 0.5 | - (규칙 안 받음) | - (1회 판정) | `1e2c384` |
| B-SG | google/shieldgemma-2b | official | 0.5 | 고정 0.5 | 0.2 | 상위 5후보 max | `1e2c384` |
| T1 | google/shieldgemma-2b | team_v1 | DNA 0.6 / Jigsaw 0.15 / PD 0.01 | 출처별 dev F1(위반) 최대, 격자 0.01 | 0.2 | 상위 5후보 max | `1e2c384` |

**P1 규칙 찾기** (R1: all-roberta-large-v1)

| 출처 | n | top1_acc | top3_acc | efficiency_mean | n_boundary |
|---|---|---|---|---|---|
| DNA | 2,256 | 0.321 | 0.479 | 0.353 | 181 |
| Jigsaw | 50 | 0.620 | 0.680 | 0.289 | 3 |
| PD | 50 | - | - | 0.316 | 4 |

**P2 판정 비교** (base 조건)

| 출처 | run_id | n | n_pos | roc_auc | pr_auc | f1 | kappa_human |
|---|---|---|---|---|---|---|---|
| DNA | B-LG | 2,256 | 65 | 0.763 | 0.138 | 0.177 | 0.154 |
| DNA | B-QG | 2,256 | 65 | 0.794 | 0.254 | 0.304 | 0.284 |
| DNA | B-SG | 2,256 | 65 | 0.516 | 0.153 | 0.242 | 0.229 |
| DNA | T1 | 2,256 | 65 | 0.434 | 0.046 | 0.085 | 0.042 |
| Jigsaw | B-LG | 50 | 26 | 0.647 | 0.690 | 0.630 | 0.196 |
| Jigsaw | B-QG | 50 | 26 | 0.659 | 0.670 | 0.654 | 0.279 |
| Jigsaw | B-SG | 50 | 26 | 0.722 | 0.731 | 0.558 | 0.250 |
| Jigsaw | T1 | 50 | 26 | 0.716 | 0.744 | 0.739 | 0.304 |
| PD | B-LG | 50 | 36 | 0.757 | 0.898 | 0.400 | 0.157 |
| PD | B-QG | 50 | 36 | 0.806 | 0.925 | 0.733 | 0.371 |
| PD | B-SG | 50 | 36 | 0.660 | 0.846 | 0.318 | 0.076 |
| PD | T1 | 50 | 36 | 0.591 | 0.806 | 0.850 | 0.279 |

**모델 간 Cohen's κ** (base 조건 예측)

| run_id | DNA | Jigsaw | PD |
|---|---|---|---|
| B-LG×B-QG | 0.370 | 0.598 | 0.302 |
| B-LG×B-SG | 0.295 | 0.114 | 0.363 |
| B-LG×T1 | 0.100 | 0.100 | 0.058 |
| B-QG×B-SG | 0.433 | 0.487 | 0.178 |
| B-QG×T1 | 0.137 | 0.223 | 0.146 |
| B-SG×T1 | 0.155 | 0.254 | 0.051 |

**P3 원인별 잘못된 판정 건수** (규칙 찾기는 규칙을 받는 모델에만 적용)

| 출처 | run_id | 규칙 찾기 | 라벨 확인 | 규칙 문장 | 판정 모델 | unclassified |
|---|---|---|---|---|---|---|
| DNA | B-LG | n/a | 47 | 6 | 49 | 0 |
| DNA | B-QG | n/a | 47 | 6 | 34 | 0 |
| DNA | B-SG | 50 | 13 | 4 | 2 | 0 |
| DNA | T1 | 109 | 13 | 4 | 133 | 0 |
| Jigsaw | B-LG | n/a | 4 | 1 | 15 | 0 |
| Jigsaw | B-QG | n/a | 4 | 2 | 12 | 0 |
| Jigsaw | B-SG | 6 | 0 | 3 | 10 | 0 |
| Jigsaw | T1 | 7 | 0 | 2 | 8 | 0 |

**실행 시간** (RTX 3060 8GB)

| run_id | 작업 | 시간 | 비고 |
|---|---|---|---|
| R1 | all-roberta-large-v1 | 1.6분 (93초) | similarity 2,356 텍스트 × 81 노드; timings.csv p1 |
| B-LG | llama_guard | 1.5분 (91초) | 판정; n_forward=2355; 실행 로그 |
| B-QG | qwen3guard | 2.8분 (170초) | 판정; n_forward=2206; 실행 로그 (DNA dev 전체 첫 계산) |
| B-SG | shieldgemma | 56.3분 (3,380초) | 판정; n_forward=15507; 5번 나눠 실행한 로그 합산(15+7+18+14+2.4분), 근사 |
| T1 | team | 22.7분 (1,359초) | 판정; n_forward=15507; model_status.csv (길이 기반 배치 적용 후) |
<!-- /results:baseline-v0 -->
