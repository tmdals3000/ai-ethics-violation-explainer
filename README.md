# 데모: 텍스트 AI 윤리 규칙 위반 판정 설명 시스템

Project 4 개인 데모 버전입니다. Do-Not-Answer 데이터 하나로 PDF의 전체 흐름(전처리 → 규칙 찾기 → 판정 → 분석 → 화면)을 작게 구현했습니다.

## 결과 요약 (dev 2,256개)

| 항목 | 값 |
|---|---|
| P1 규칙 찾기 (τ = 0.20) | top-1 0.268 · top-3 0.463 · 효율 0.399 · 경계 사례 336개 |
| P2 ROC-AUC | Qwen3Guard 0.800 · Llama Guard 0.758 · ShieldGemma(규칙 문장만) 0.563 |
| P2 PR-AUC | Qwen3Guard 0.262 · Llama Guard 0.140 · ShieldGemma 0.149 |
| P3 잘못된 판정 (모델별, 260건) | Llama Guard 96 · Qwen3Guard 76 · ShieldGemma 88 |
| P3 원인 | 라벨 확인 필요 138 · 규칙 문장 64 · 판정 모델 58 · 규칙 찾기 0 (데모 한계, 아래 참고) |

## 파일 구성

| 단계 | 파일 | 하는 일 | 실행 위치 |
|---|---|---|---|
| 1 | `preprocess.py` | 응답 6개를 한 행씩 펼치기 (939 → 5,634행), 텍스트 ID `dna-{질문번호}-{LLM}` | 노트북 |
| 2 | `make_assets.py` | 규칙 트리(78개 노드), dev/test 분할, 유사도(all-roberta-large-v1), 2차원 좌표(UMAP) | 노트북 (CPU 약 1~2시간) |
| 3 | `judge_colab.ipynb` | 판정 모델 3개로 위반 확률 계산 → `judgments.csv` | Colab T4 GPU (약 1시간) |
| 4 | `analyze.py` | P1 BFS 규칙 찾기·τ 비교·경계 사례, P2 지표·모델 간 κ·오판 행렬, P3 원인 분류 → `results/` | 노트북 |
| 4 | `diagnose.py` | P3 원인 규칙 1–4 (analyze.py·app.py 가 같이 씀) | — |
| 5 | `app.py` | Streamlit 화면 4탭 + 공통 필터 + 선택 연동 + 원문 가리기 | 노트북 |

## 실행 방법

```
pip install -r requirements.txt
python preprocess.py          # data_en.csv 필요 (아래 '데이터' 참고)
python make_assets.py
# judge_colab.ipynb 를 Colab 에서 실행 → judgments.csv 를 이 폴더에 넣기
python analyze.py
streamlit run app.py
```

## 데이터 (저장소에 올리지 않음)

유해한 원문이 들어 있어서 아래 파일은 올리지 않았습니다. 실행하면 다시 만들어집니다.

- `data_en.csv`: Do-Not-Answer 원본 (`github.com/libr-ai/do-not-answer` 의 `datasets/data_en.csv`)
- `dna_long.csv`: `preprocess.py` + `make_assets.py` 결과
- `text_emb.npy`: `make_assets.py` 결과 (임베딩)

## 설정과 가정

- **분할:** 조교 `splits.csv` 가 아직 없어서, 승민 님 baseline 과 같은 dev(판정된 376개 질문 × 6 = 2,256개)를 쓰고 나머지를 test 로 둠. 같은 질문의 응답 6개는 항상 같은 쪽.
- **BFS:** 대분류·중분류는 모두 비교, 유사도 ≥ τ 인 중분류만 남기고(없으면 최고 1개) 그 아래 소분류를 모두 비교. 효율 = 비교 노드 수 / 78.
- **판정 확률:** 판정 단어 바로 앞까지 프롬프트를 넣고 다음 토큰 확률을 봄.
  - Qwen3Guard: P(Unsafe) + 0.5 × P(Controversial) — 승민 님은 Controversial 을 1.0 으로 셈 (팀에서 통일 필요)
  - Llama Guard 3 1B: P(unsafe)
  - ShieldGemma-2B: 정답 소분류 규칙 문장을 넣고 P(Yes) (PDF 의 '규칙 문장만' 조건)
- **임계값:** 0.5 (화면에서 바꿀 수 있음)

## 데모의 한계

- 출처는 Do-Not-Answer 만 (Jigsaw, ProsocialDialog 없음)
- 팀 판정기 없음 → P3 원인 3번(규칙 문장)은 ShieldGemma(규칙 문장만) 하나로 판단
- P3 는 모델마다 따로 진단. 원인 1(규칙 찾기)은 규칙을 입력받는 모델이 BFS 후보 규칙으로 판정한 결과(base)가 있어야 세는데, 데모에서는 이 판정을 돌리지 않아 0건. 대신 정답 규칙이 상위 5개 후보 밖인 잘못된 판정(161건)을 화면 지도에 ◆로 표시
- P4 는 '임계값 변경 효과'만 구현. '규칙 제외'·'규칙+예시' 조건과 '규칙 문장 편집 → 다시 판정'은 GPU 판정이 더 필요해서 제외
- 모든 수치는 dev 결과. test 는 아직 보지 않음
