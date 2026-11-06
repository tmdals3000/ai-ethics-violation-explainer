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
