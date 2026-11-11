# Baseline 정리: 텍스트 AI 윤리 규칙 위반 판정 모델을 설명하는 시스템

Project 4 · Baseline v0 (`baseline-v0`, 2026-11-08)
판정이 틀렸을 때 **규칙 찾기 · 규칙 문장 · 판정 모델 · 사람 라벨** 중 어디가 문제인지 밝히고, 고치면 얼마나 나아지는지 확인한다

> 이 문서는 Baseline 을 **연구계획서 형식으로 한 곳에 정리한 것**입니다. 무엇을 문제로 잡았고,
> 무엇을 비교했고, 무엇을 근거로 정했는지가 중심입니다. 단계별 상세는 아래 표의 문서에 있습니다.
>
> | | 단계 | 코드 | 상태 | 기록 |
> |---|---|---|---|---|
> | P0 | 데이터 준비 | [`src/data.py`](src/data.py) · [`scripts/eda_v1.py`](scripts/eda_v1.py) | ✅ | [EDA](docs/EDA.md) · [ASSUMPTIONS](docs/ASSUMPTIONS.md) (A-split, A-pd-label …) |
> | P1 | 규칙 찾기 | [`src/retrieval.py`](src/retrieval.py) | ✅ | [ASSUMPTIONS · τ 결정 기록](docs/ASSUMPTIONS.md#τ-결정-기록) |
> | P2 | 판정 비교 | [`src/judges.py`](src/judges.py) · [`src/metrics.py`](src/metrics.py) | ✅ 4개 모델 | [BASELINE_REPORT §2](docs/BASELINE_REPORT.md#2-dev-결과) |
> | P3 | 오류 진단 | [`src/diagnose.py`](src/diagnose.py) | ✅ | [BASELINE_REPORT §2](docs/BASELINE_REPORT.md#p3-원인별-잘못된-판정-base) |
> | P4 | 수정 효과 | [`src/effects.py`](src/effects.py) | ⏸ 코드만, 4개 모델로 미실행 | [BASELINE_REPORT §6](docs/BASELINE_REPORT.md#6-다음-할-일) |
> | 화면 | Streamlit | [`app/app.py`](app/app.py) | ✅ 탭 4개 | — |
>
> ⚠ **모든 수치는 dev 결과입니다. test 는 아직 보지 않았습니다.** 검증 상태: `IS만` — In-Sample: 임계값·τ 를 정한 바로 그 dev 로 평가한 값이라
> 실제 성능보다 좋게 나올 수 있습니다. 따로 떼어 둔 test 로 확인하면 `OOS 통과`(Out-of-Sample) 로 바뀝니다.
> split·유사도·좌표는 조교 파일이 없어 **임시**로 만들었습니다.

---

## 저장소 지도 · 바로 실행하기

> 1장부터는 연구 정리입니다. 코드를 보러 왔다면 이 절부터 읽으면 됩니다.

| 폴더 | 무엇 |
|---|---|
| `configs/baseline.yaml` | 모든 설정 한 곳 — 경로, 표본 크기, τ, 임계값, 모델, 집계 방식. 모든 단계가 이 파일 하나를 읽는다 |
| `src/` | `data` 데이터 준비 · `retrieval` P1 · `judges` 판정 · `metrics` P2 · `diagnose` P3 · `effects` P4 · `adapters` 조교 파일 읽기 · `common` 공통 |
| `scripts/` | `run.py` 전체 실행기 · `record_run.py` 실험 결과 기록·README 표 생성 · `trial_analysis.py` 시험 실행 분석 |
| `app/app.py` | Streamlit 화면 4탭. 유해 원문은 기본으로 가리고 눌렀을 때만 보인다 |
| `results/` | 지표·판정 결과 (**원문 없음**, `text_id` 만). 실험 원장은 `results/experiments.csv` |
| `docs/` | [`ASSUMPTIONS.md`](docs/ASSUMPTIONS.md) 임시 결정 · [`BASELINE_REPORT.md`](docs/BASELINE_REPORT.md) 결과 보고서 · [`EDA.md`](docs/EDA.md) 데이터 탐색 · [`TEAM_SYNC.md`](docs/TEAM_SYNC.md) 팀원 브랜치와 차이 |
| `tests/` | pytest 17개. 가짜 데이터는 `tests/fixtures` 에만 |

원본 데이터·가공 데이터·모델 캐시는 저장소 밖(`C:/ml/...`)에 두며 커밋하지 않는다.

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

조교 파일(`similarity.csv`, `coords.csv`, `judgments.csv`, `splits.csv`)이 오면 `C:/ml/raw/ta/` 에 두면 자동으로 우선 사용한다 ([`src/adapters.py`](src/adapters.py)).

---

## 1. Problem Definition

### 어떤 task인가

챗봇 답변·커뮤니티 댓글 같은 텍스트가 AI 윤리 규칙을 어겼는지 판정하는 모델은 많지만,
**판정이 틀렸을 때 왜 틀렸는지**는 알려 주지 않습니다. 이 과제는 판정 결과를 설명하는
시스템을 만드는 것입니다 — 틀린 판정 하나를 받아서 그 원인을 네 곳 중 하나로 짚습니다.

| 원인 | 뜻 | 고치는 곳 |
|---|---|---|
| **규칙 찾기** | 텍스트에 맞는 규칙을 찾지 못했다 | 검색(P1) |
| **라벨 확인** | 모든 모델이 같은 답인데 사람 라벨만 다르다 | 사람 라벨 재검토 |
| **규칙 문장** | 맞는 규칙을 줘도 규칙을 받는 모델이 모두 틀린다 | 규칙 문장 수정 |
| **판정 모델** | 같은 규칙으로 다른 모델은 맞고 이 모델만 틀린다 | 모델·프롬프트·임계값 |

### Baseline 의 목표

**정확도가 아니라 흐름이 끝까지 도는 것**이 목표입니다. 데이터 준비 → P1 → P2 → P3 → P4 →
화면이 한 번에 실행되고, 각 단계가 다음 단계에 넘길 표를 만드는지 확인합니다.

### 지키는 제약

- 모델 가중치는 **학습·미세조정하지 않는다** — 실행과 팀 판정기의 프롬프트·임계값만 개발한다.
- XAI, LLM 설명문, 별도 규칙 엔진은 만들지 않는다.
- 임계값·프롬프트·τ 는 **dev 에서만** 정한다. test 지표는 `--allow-test` 를 줄 때만 계산한다.
- 데이터나 결과를 지어내지 않는다. 실행하지 못한 모델·데이터는 실패와 원인을 기록한다.

---

## 2. Background & Baseline

### 2.1 대상 데이터

| 출처 | 규모 | 판정 대상 | 비고 |
|---|---|---|---|
| **Do-Not-Answer (DNA)** | 5,634행 | 응답 (`response` 모드) | 질문 939개 × 응답 LLM 6개를 한 행씩 펼침. 위반 2.9% |
| **Jigsaw ACRC** | 2,029행 | 댓글 (`prompt` 모드) | 행마다 붙은 예시 8,116개를 분리 |
| **ProsocialDialog (PD)** | 165,681행 | 발화 (`prompt` 모드) | 영어만. 안전 라벨 → 위반/정상 변환은 config |

**dev 표본:** DNA dev 전체 2,256건(위반 65), Jigsaw 50건, PD 50건(라벨 비율 유지).
DNA 는 위반이 드물어 표본을 줄이면 PR-AUC·F1 이 불안정해서 전체를 씁니다.

### 2.2 규칙 트리

| 단계 | DNA | Jigsaw |
|---|---|---|
| 대분류 | risk area 5개 | `J` 1개 |
| 중분류 | harm type 12개 | 규칙 J1(광고 금지) · J2(법률 조언 금지) |
| 소분류 | specific harm 61개 | — |

총 81노드. 두 출처를 **한 트리**에 넣었고, 이것이 아래 알려진 문제 1번이 됩니다.

### 2.3 Baseline 모델

| 역할 | 모델 | 규칙을 받는가 |
|---|---|---|
| 규칙 검색 | `sentence-transformers/all-roberta-large-v1` | — |
| 기준 판정기 | `meta-llama/Llama-Guard-3-1B` | ✗ (자체 분류 체계) |
| 기준 판정기 | `Qwen/Qwen3Guard-Gen-0.6B` | ✗ |
| 기준 판정기 | `google/shieldgemma-2b` | ✓ |
| **팀 판정기** | ShieldGemma-2B 가중치 + 팀 프롬프트 `team_v1` | ✓ |

### 2.4 조교 파일이 없어 임시로 정한 것

| 무엇 | 임시 결정 | 기록 |
|---|---|---|
| split | 고정 시드 42, train/dev/test = 0.2/0.4/0.4. DNA 는 질문 단위, Jigsaw 는 같은 본문 그룹 단위로 묶음 | A-split, A-split-dup |
| similarity | roberta 임베딩 코사인 유사도 | A-sim |
| coords | 임베딩 PCA 2차원 | A-coords |
| PD 라벨 | `probably_needs_caution` 이상 = 위반 | A-pd-label |

조교 파일이 오면 어댑터가 자동으로 그쪽을 읽고, 위 결과는 다시 계산합니다.

---

## 3. Proposed Method

### 3.1 P1 규칙 찾기 — 트리를 위에서부터 걸러 내려간다

텍스트와 81개 노드의 유사도를 구한 뒤 BFS 로 내려갑니다. 중분류 단계에서 유사도가 **τ 미만인
가지는 잘라** 비교 횟수를 줄입니다(전부 잘리면 최고점 하나는 남김). 지표는 top-k 정확도,
**효율**(비교한 노드 수 / 81), 처음 갈라진 단계, 경계 사례(τ±0.02 에서 top-1 이 바뀜)입니다.

### 3.2 P2 판정 — 위반 확률은 판정 토큰의 확률로 잰다

생성 결과 문자열을 파싱하지 않고, **판정 토큰 자리의 확률을 한 번의 forward 로 직접** 읽습니다.

| 모델 | p(위반) |
|---|---|
| ShieldGemma · 팀 | softmax(`Yes`, `No`) 의 `Yes` |
| Llama Guard | softmax(`safe`, `unsafe`) 의 `unsafe` |
| Qwen3Guard | `Unsafe` + w·`Controversial` (w = 1.0, strict) |

규칙을 받는 모델은 네 조건으로 판정합니다.

| 조건 | 받는 규칙 | 묻는 것 |
|---|---|---|
| `base` | P1 상위 5 후보, 최댓값 집계 | 실제 운영 성능 |
| `rule_excluded` | 정답 위해 유형을 뺀 상위 5 | 처음 보는 규칙에서 얼마나 떨어지나 |
| `rule_only` | 정답 규칙 하나 | 검색이 완벽하면 맞히나 |
| `rule_plus_examples` | 정답 규칙 + 위반·정상 예시 2개씩 | 예시가 도움이 되나 |

### 3.3 P3 오류 진단 — 순서대로 첫 번째로 걸리는 원인

틀린 판정마다 아래 규칙을 **위에서부터** 적용합니다. 결과가 빠진 모델이 있으면 2–4번은
적용하지 않고 `unclassified` 로 둡니다.

1. **규칙 찾기** — (규칙 모델만) 정답 규칙 순위 > 5, 또는 `rule_only` 에서는 맞음
2. **라벨 확인** — 4개 모델의 base 답이 모두 같음
3. **규칙 문장** — `rule_only` 에서 ShieldGemma 와 팀 둘 다 틀림
4. **판정 모델** — 나머지

### 3.4 P4 수정 효과

임계값(0.3 / 0.7), τ(0.15 / 0.25), 집계(top1) 를 바꿨을 때 뒤집히는 판정을 세고, 규칙 문장
하나를 고쳐 다시 판정합니다. 화면에서 임계값 슬라이더와 "다시 판정" 으로 직접 해 볼 수 있습니다.

---

## 4. Experiment Design

### 4.1 무엇을 비교했는가

**현재 설정값은 아래 측정으로 정했습니다.** 원본 수치는 `results/` 에 있습니다.

| 비교한 것 | 결론 | 근거 |
|---|---|---|
| τ 격자 0.10 – 0.40 (DNA dev 전체) | **τ = 0.20** | 효율 ≤ 0.4 인 τ 중 top-1 최대 (0.321, 효율 0.353). 0.15 는 top-1 0.344 지만 효율 0.417 로 기준 초과 → [`tau_comparison.csv`](results/p1/tau_comparison.csv) |
| Jigsaw 를 트리에 넣기 전·후 | τ 가 0.15 → 0.20 으로 이동 | DNA top-1 0.372 → 0.344 (τ = 0.15). DNA 298건이 Jigsaw 규칙으로 찾아짐 → 멘토링 질문 1 |
| 팀 임계값: 출처별 vs 공통 | **출처별** (DNA 0.60 / Jigsaw 0.15 / PD 0.01) | 출처별 dev F1 최대. 공통 하나로는 0.01 (평균 F1 0.539) → [`team_thresholds.csv`](results/team_thresholds.csv) |
| 후보 집계 max vs top1 | **max** (잠정) | P4 에서 비교 예정 |
| 판정 토큰 확률 검증 | 통과 | 판정 토큰 확률 합 ≥ 0.9992. 생성 결과와 다른 것은 p = 0.5 동률뿐 (A-token-check) |
| Gemma2 attention eager vs sdpa | **eager 유지** | softcapping 때문에 sdpa 는 결과가 달라짐 |
| 배치 방식 (고정 vs 길이 기반) | **길이 정렬 + 토큰 예산 배치** | 배치 × 최대 길이² ≤ 2,000,000. 같은 양이 약 2.4배 빨라짐 (팀 v1 실행 11.4건/초) |

### 4.2 Evaluation Metrics

- **P1 규칙 찾기**: top-1 / top-3 정확도, 효율(비교 노드 비율), 경계 사례 수
- **P2 판정**: ROC-AUC, PR-AUC, F1(위반), 사람 라벨과의 Cohen's κ, 모델 쌍 κ
- **P3 진단**: 원인별 잘못된 판정 건수, `unclassified` 건수

### 4.3 측정 원칙

1. **dev 와 test 를 가른다.** 임계값·τ·프롬프트는 dev 에서만 정하고, test 는 최종 설정을 고정한 뒤 1회만 본다.
2. **수치에는 표본을 같이 적는다.** v0 의 Jigsaw·PD 는 50건이었다. v1 부터 Jigsaw dev 전체(783건), PD 500건이다.
3. **같은 본문은 같은 split 에.** Jigsaw 에서 같은 본문 43그룹 188행이 split 을 넘나드는 누수를 찾아 105행을 옮겼다.
4. **임시로 정한 것은 전부 적는다.** 조교 파일이 없어 정한 것은 [ASSUMPTIONS](docs/ASSUMPTIONS.md) 에 번호를 붙여 남긴다.

---

## 5. 실험 결과

### 5.1 기록 규칙

- 원자료는 `results/experiments.csv` 하나다. 컬럼: `version, run_id, date, commit, split, table, source, model, metric, value, note`.
- 새 실험은 **행을 추가만** 한다. 이전 결과는 지우거나 고치지 않는다.
  - `version`: 결과 묶음 이름 (예: `baseline-v0`, `baseline-v1`). 같은 version 은 다시 기록할 수 없다.
  - `run_id`: 판정기 하나의 설정 — 기준 모델 `B-LG`(Llama Guard) · `B-QG`(Qwen3Guard) · `B-SG`(ShieldGemma), 팀 판정기 `T1`, `T2`… (프롬프트·임계값이 바뀌면 새 번호), 규칙 찾기 `R1`, `R2`…, 모델 쌍 κ 는 `B-LG×B-QG` 형식.
  - `commit`: 그 결과를 만든 코드의 커밋 해시.
- 순서: 실행 → 결과 커밋 → `python scripts/record_run.py record --version <v> --date <YYYY-MM-DD> --commit <hash>` → `python scripts/record_run.py readme --version <v>` → 커밋 · 태그.
- README 표는 손으로 고치지 않는다. 위 명령이 `experiments.csv` 에서 만든다. 새 version 은 아래 자리표시 바로 밑(맨 위)에 쌓인다.
- 수치는 dev 결과만 적는다. test 결과는 `split=test` 행으로 따로 기록하고 표 제목에 `test` 를 쓴다.

### 5.2 알려진 문제

검증 상태 `IS만` = In-Sample. 임계값·τ 를 고른 dev 와 평가한 dev 가 같아서 낙관적인 값이다. test(Out-of-Sample) 로 확인하기 전까지 최종 성능으로 쓰지 않는다.

1. **임시 split·규칙 트리:** 조교 `splits.csv`·similarity·coords 가 없어 고정 시드 split, roberta 유사도, PCA coords 를 임시로 만들었다. 규칙 트리는 DNA·Jigsaw 를 한 트리에 넣었고, DNA 텍스트 298건(13.2%)의 top-1 이 Jigsaw 규칙이다 → [BASELINE_REPORT 질문 1](docs/BASELINE_REPORT.md#7-멘토링조교-질문).
2. ~~Jigsaw·ProsocialDialog 는 dev 50건 표본~~ → **Baseline v1 에서 해결:** Jigsaw dev 전체 783건, PD dev 500건(라벨 비율 유지)으로 늘렸다. Baseline v0 표는 50건 기준 그대로 남겨 둔다.
3. **DNA 에서 팀 v1·ShieldGemma AUC 가 무작위 수준 (팀 v1 0.434, ShieldGemma 0.516) — 확인 완료:**
   - 판정 형식은 문제가 아니었다. 네 모델 모두 질문과 응답을 함께 넣고 **응답**을 판정한다. ShieldGemma 프롬프트는 토크나이저에 내장된 공식 chat template 과 문구 3곳(관사·빈 줄·공백)만 다르고, 공식 템플릿으로 바꿔도 DNA 위반 65 + 정상 65건 AUC 가 0.507 → 0.497 (확률 상관 0.998) 로 같았다 (`results/trial/sg_template_check.csv`).
   - 실제 원인 ① **팀 v1 이 거절문에 높은 확률을 준다.** 정상 응답 중 확률 상위 50건의 32건이 거절문이고, 거절문 평균 확률 0.257 이 실제로 지시를 따른 응답(위반 58/64) 0.207 보다 높다.
   - 실제 원인 ② ShieldGemma 는 거절문이 아니라 **주의 문구를 붙이고 내용을 준 응답**에 높은 확률을 주고(정상 상위 50건 중 32건), 기관 기밀 추측·거짓 전제를 그대로 받은 위반 응답에는 0에 가까운 확률을 준다.
   - 근거: `results/dna_check/`, `scripts/dna_check.py`, 응답 유형 분포는 [EDA](docs/EDA.md).
4. **후보 집계:** base 는 상위 5 후보 규칙의 위반 확률 최댓값을 쓴다. DNA 에서 정답 규칙을 뺀 rule_excluded AUC 가 base 보다 높다(ShieldGemma 0.583 vs 0.516, 팀 0.535 vs 0.434). 다만 top-1 후보만 쓰거나 유사도 가중 평균을 써도 AUC 가 0.5 를 넘지 않아(`results/dna_check/aggregation_compare.csv`) 집계 방식은 주원인이 아니다 → 최댓값 유지.
5. **PD 팀 임계값이 0.01 에 걸림:** 출처별 dev F1 최대 기준으로 고르면, 위반 비율이 72% 인 ProsocialDialog 에서는 '거의 전부 위반'이 F1 을 최대로 만들어 임계값이 격자 최솟값 0.01 이 된다.

### 5.3 결과 표

<!-- results:new -->

<!-- results:baseline-v1 -->
### Baseline v1 (dev, 2026-11-12)

> ⚠ **모든 수치는 dev 결과다. test 는 아직 보지 않았다.** 검증 상태: `IS만` (In-Sample: 임계값·τ 를 정한 dev 로 평가해 낙관적).
> 표본(dev): DNA 2,256건, Jigsaw 783건, PD 500건. split·규칙 트리는 임시([ASSUMPTIONS](docs/ASSUMPTIONS.md)).
> 원자료: `results/experiments.csv` (version = `baseline-v1`). 이 표는 `python scripts/record_run.py readme --version baseline-v1` 로 만든다.

**실험 설정**

| run_id | 모델 | prompt_version | threshold | threshold_rule | tau | aggregation | 커밋 |
|---|---|---|---|---|---|---|---|
| B-LG | meta-llama/Llama-Guard-3-1B | official | 0.5 | 고정 0.5 | - (규칙 안 받음) | - (1회 판정) | `0d13a7a` |
| B-QG | Qwen/Qwen3Guard-Gen-0.6B | official | 0.5 | 고정 0.5 | - (규칙 안 받음) | - (1회 판정) | `0d13a7a` |
| B-SG | google/shieldgemma-2b | official | 0.5 | 고정 0.5 | 0.2 | 상위 5후보 max | `0d13a7a` |
| T1 | google/shieldgemma-2b | team_v1 | DNA 0.6 / Jigsaw 0.11 / PD 0.01 | 출처별 dev F1(위반) 최대, 격자 0.01 | 0.2 | 상위 5후보 max | `0d13a7a` |

**P1 규칙 찾기** (R1: all-roberta-large-v1)

| 출처 | n | top1_acc | top3_acc | efficiency_mean | n_boundary |
|---|---|---|---|---|---|
| DNA | 2,256 | 0.321 | 0.479 | 0.353 | 181 |
| Jigsaw | 783 | 0.558 | 0.600 | 0.309 | 72 |
| PD | 500 | - | - | 0.317 | 38 |

**P2 판정 비교** (base 조건)

| 출처 | run_id | n | n_pos | roc_auc | pr_auc | f1 | kappa_human |
|---|---|---|---|---|---|---|---|
| DNA | B-LG | 2,256 | 65 | 0.763 | 0.138 | 0.177 | 0.154 |
| DNA | B-QG | 2,256 | 65 | 0.794 | 0.254 | 0.304 | 0.284 |
| DNA | B-SG | 2,256 | 65 | 0.516 | 0.153 | 0.242 | 0.229 |
| DNA | T1 | 2,256 | 65 | 0.434 | 0.046 | 0.085 | 0.042 |
| Jigsaw | B-LG | 783 | 400 | 0.522 | 0.536 | 0.533 | 0.012 |
| Jigsaw | B-QG | 783 | 400 | 0.653 | 0.632 | 0.663 | 0.270 |
| Jigsaw | B-SG | 783 | 400 | 0.690 | 0.670 | 0.529 | 0.219 |
| Jigsaw | T1 | 783 | 400 | 0.691 | 0.677 | 0.709 | 0.207 |
| PD | B-LG | 500 | 358 | 0.667 | 0.853 | 0.352 | 0.126 |
| PD | B-QG | 500 | 358 | 0.741 | 0.877 | 0.689 | 0.291 |
| PD | B-SG | 500 | 358 | 0.733 | 0.875 | 0.421 | 0.148 |
| PD | T1 | 500 | 358 | 0.735 | 0.869 | 0.842 | 0.254 |

**모델 간 Cohen's κ** (base 조건 예측)

| run_id | DNA | Jigsaw | PD |
|---|---|---|---|
| B-LG×B-QG | 0.370 | 0.325 | 0.308 |
| B-LG×B-SG | 0.295 | 0.207 | 0.478 |
| B-LG×T1 | 0.100 | 0.061 | 0.047 |
| B-QG×B-SG | 0.433 | 0.374 | 0.405 |
| B-QG×T1 | 0.137 | 0.171 | 0.164 |
| B-SG×T1 | 0.155 | 0.172 | 0.071 |

**P3 원인별 잘못된 판정 건수** (규칙 찾기는 규칙을 받는 모델에만 적용)

| 출처 | run_id | 규칙 찾기 | 라벨 확인 | 규칙 문장 | 판정 모델 | unclassified |
|---|---|---|---|---|---|---|
| DNA | B-LG | n/a | 47 | 6 | 49 | 0 |
| DNA | B-QG | n/a | 47 | 6 | 34 | 0 |
| DNA | B-SG | 50 | 13 | 4 | 2 | 0 |
| DNA | T1 | 109 | 13 | 4 | 133 | 0 |
| Jigsaw | B-LG | n/a | 60 | 6 | 320 | 0 |
| Jigsaw | B-QG | n/a | 60 | 4 | 221 | 0 |
| Jigsaw | B-SG | 137 | 0 | 11 | 160 | 0 |
| Jigsaw | T1 | 130 | 4 | 9 | 163 | 0 |

**P4 처음 보는 규칙** (정답 규칙이 있는 텍스트, 하락폭 = base − 규칙 제외, 회복폭 = 조건 − 규칙 제외)

| 출처 | run_id | auc_base | auc_rule_excluded | auc_rule_only | auc_rule_plus_examples | auc_drop | recovery_rule_only | recovery_rule_plus_examples |
|---|---|---|---|---|---|---|---|---|
| DNA | B-SG | 0.516 | 0.583 | 0.568 | 0.581 | -0.068 | -0.015 | -0.002 |
| DNA | T1 | 0.434 | 0.535 | 0.448 | 0.450 | -0.101 | -0.087 | -0.086 |
| Jigsaw | B-SG | 0.690 | 0.667 | 0.735 | 0.720 | 0.023 | 0.067 | 0.053 |
| Jigsaw | T1 | 0.691 | 0.674 | 0.698 | 0.695 | 0.016 | 0.024 | 0.021 |

**P4 설정 변경 시 뒤집힌 판정** (dev 전체 출처 합계, 맞게 = fixed, 틀리게 = broken)

| run_id | 변경 | n_flipped | n_fixed | n_broken |
|---|---|---|---|---|
| B-LG | 임계값 → 0.3 | 146 | 55 | 91 |
| B-LG | 임계값 → 0.7 | 142 | 90 | 52 |
| B-QG | 임계값 → 0.3 | 132 | 76 | 56 |
| B-QG | 임계값 → 0.7 | 108 | 41 | 67 |
| B-SG | 임계값 → 0.3 | 250 | 135 | 115 |
| B-SG | 임계값 → 0.7 | 154 | 49 | 105 |
| B-SG | 집계 → top1 | 77 | 25 | 52 |
| T1 | τ → 0.15 | 77 | 16 | 61 |
| T1 | τ → 0.25 | 60 | 46 | 14 |
| T1 | 임계값 → 0.3 | 919 | 165 | 754 |
| T1 | 임계값 → 0.7 | 896 | 395 | 501 |
| T1 | 집계 → top1 | 147 | 81 | 66 |

**실행 시간** (RTX 3060 8GB)

| run_id | 작업 | 시간 | 비고 |
|---|---|---|---|
| step:data | data | 0.2분 (11초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 00:46:49; do_not_answer=all;jigsaw=all;prosocial=500 |
| step:p1 | p1 | 1.9분 (116초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 00:46:49; do_not_answer=all;jigsaw=all;prosocial=500 |
| step:judge | judge | 17.7분 (1,061초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 01:05:25; do_not_answer=all;jigsaw=all;prosocial=500 |
| step:p2 | p2 | 0.0분 (1초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 01:05:25; do_not_answer=all;jigsaw=all;prosocial=500 |
| step:p3 | p3 | 0.0분 (1초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 01:05:25; do_not_answer=all;jigsaw=all;prosocial=500 |
| step:p4 | p4 | 8.1분 (485초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 01:14:31; do_not_answer=all;jigsaw=all;prosocial=500 |
| B-LG | llama_guard | 0.5분 (28초) | 판정 (캐시에 없던 것만 새로 계산); n_forward=1100; model_status.csv; 전체 프롬프트 3,539 |
| B-QG | qwen3guard | 0.5분 (33초) | 판정 (캐시에 없던 것만 새로 계산); n_forward=1100; model_status.csv; 전체 프롬프트 3,539 |
| B-SG | shieldgemma | 5.9분 (353초) | 판정 (캐시에 없던 것만 새로 계산); n_forward=5354; model_status.csv; 전체 프롬프트 30,323 |
| T1 | team | 10.0분 (601초) | 판정 (캐시에 없던 것만 새로 계산); n_forward=5400; model_status.csv; 전체 프롬프트 30,323 |
<!-- /results:baseline-v1 -->

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

### 5.4 해석 (dev 기준, 잠정)

- **팀 판정기 v1 은 DNA 에서 실패했다** (AUC 0.434 < 0.5). DNA 응답 대부분이 거절문인데, v1 프롬프트가 거절문 속 유해 주제 언급을 위반으로 보는 것으로 의심된다. v2 프롬프트("거절·경고·안전한 안내는 위반이 아니다")가 이 가설을 직접 시험한다.
- **정답 규칙을 빼도 성능이 오른다.** ShieldGemma·팀 모두 DNA 에서 `rule_excluded` AUC 가 `base` 보다 높다 → 최댓값 집계가 엉뚱한 후보 규칙 때문에 판정을 흐린다.
- **사람 라벨 재검토 후보 DNA 47건**: 4개 모델이 모두 같은 답인데 사람 라벨만 다르다.
- **Jigsaw·PD 에서는 판정기 간 차이가 작다.** 50건 표본이라 순위를 정하지 않는다.

상세 해석과 조건별 AUC 는 [`docs/BASELINE_REPORT.md`](docs/BASELINE_REPORT.md).

---

## 6. 기술 스택과 핵심 결정

| 영역 | 선택 | 판단 근거 |
|---|---|---|
| 실행 환경 | Python + PyTorch (CUDA 12.6), RTX 3060 8GB | 로컬 GPU 한 장에서 4개 모델을 돌릴 수 있는 크기만 사용 |
| 모델 로딩 | Hugging Face `transformers`, bf16 | 학습 없이 추론만. `HF_HOME=C:/ml/hf-cache` |
| 위반 확률 | 판정 토큰 확률 (teacher forcing, 마지막 위치 logit 만) | 문자열 파싱보다 안정적, ROC/PR-AUC 계산 가능. 메모리 초과도 해결 |
| 판정 캐시 | SQLite (512건마다 저장) | 긴 실행이 중간에 끊겨도 이어서 돌림 |
| 지표 | scikit-learn | 표준 구현 (`roc_auc`, `average_precision`, `f1`, `cohen_kappa`) |
| 화면 | Streamlit + Altair | 최소 화면. 탭 간 선택 공유, 좌표 brush 선택 |
| 실험 기록 | `results/experiments.csv` (long format) + `record_run.py` | README 표를 손으로 쓰지 않고 원장에서 생성 → 수치 불일치 방지 |

---

## 7. 산출물과 다음 할 일

| | 무엇 |
|---|---|
| **파이프라인** | `scripts/run.py --steps data,p1,judge,p2,p3,p4` 한 번으로 전 단계 실행 |
| **결과** | P1 검색 결과·τ 비교, 4개 모델 판정표, P2 지표·κ, P3 원인표 (`results/`, 원문 없음) |
| **화면** | Streamlit 4탭 — 규칙 찾기 · 판정 비교 · 오류 진단 · 수정 효과 |
| **문서** | [ASSUMPTIONS](docs/ASSUMPTIONS.md) 임시 결정 · [BASELINE_REPORT](docs/BASELINE_REPORT.md) 보고서와 멘토링 질문 7개 |
| **테스트** | pytest 17개 |

**다음 할 일**

1. 팀 프롬프트 v2·v3 를 dev 에서 비교 (`T2`, `T3`) — DNA 거절문 문제
2. P4 를 4개 모델 결과로 실행
3. Jigsaw dev 전체(783건) + PD dev 500건 판정 (약 20–35분 추정)
4. 조교 파일이 오면 어댑터로 교체 후 재실행
5. 최종 프롬프트·임계값 고정 뒤 test 1회 평가
