# 데이터 스키마

파이프라인의 모든 파일과 열을 정리한 문서입니다. **원문 포함** 표시가 있는 파일은 유해한 글이 들어 있어 저장소에 올리지 않습니다.

## 공통 규칙

### 텍스트 ID (`text_id`)

| 출처 | 형식 | 예 | 한 행의 단위 |
|---|---|---|---|
| Do-Not-Answer | `dna-{질문 id}-{응답 LLM}` | `dna-13-GPT4` | 질문 1개 × 응답 LLM 1개 (질문 939 × 6 = 5,634) |
| Jigsaw | `jig-{row_id}` | `jig-1021` | 댓글 1개 (`train.csv` 의 한 행) |
| ProsocialDialog | `pd-validation-{dialogue_id}-{response_id}` | `pd-validation-112-0` | 사용자 발화 1개 |

### 판정 모델 이름 (`judge`)과 조건 (`condition`)

화면과 결과 파일에서는 둘을 합쳐 `judge / condition` (예: `ShieldGemma-2B / base`)으로 씁니다.

| judge | 규칙 입력 | 가능한 condition |
|---|---|---|
| `LlamaGuard3-1B` | 안 받음 (자체 범주) | `default` |
| `Qwen3Guard-0.6B` | 안 받음 (자체 범주) | `default` |
| `ShieldGemma-2B` | 받음 | `base`, `rule_only`, `rule_excluded` |
| `Team-team_v3` | 받음 (ShieldGemma + 팀 원칙 문장) | `rule_only` |

| condition | 뜻 |
|---|---|
| `default` | 규칙 없이 모델 자체 기준으로 판정 |
| `base` (기본) | BFS 상위 5개 후보 규칙마다 판정 → 최댓값 |
| `rule_only` (규칙 문장만) | 정답 규칙 문장 하나만 넣고 판정 |
| `rule_excluded` (규칙 제외) | 기본 조건의 후보 중 정답 규칙과 같은 위해 유형(DNA 중분류)·같은 규칙(Jigsaw)을 빼고 최댓값. 남은 후보가 없으면 0 |
| (미구현) `rule_plus_examples` | 정답 규칙 + 위반·비위반 예시 |

### 규칙 노드 ID (`node_id`)

| 트리 | 단계 | 형식 | 개수 |
|---|---|---|---|
| dna | 대분류 | `L1:{위험 영역}` | 5 |
| dna | 중분류 | `L2:{위해 유형}` | 12 |
| dna | 소분류 (잎) | `L3:{세부 위해}` | 61 |
| jigsaw | 루트 | `J1:Reddit community rules` | 1 |
| jigsaw | 규칙 (잎) | `J2:{규칙 이름}` | 2 |

ProsocialDialog 는 정답 규칙이 없고, 규칙 찾기에는 dna 트리를 씁니다.

---

## 1단계 `preprocess.py`

### `texts.csv` — 원문 포함

| 열 | 형식 | 설명 |
|---|---|---|
| `text_id` | str | 위 규칙 |
| `source` | `do_not_answer` · `jigsaw` · `prosocial` | 출처 |
| `split` | `dev` · `test` | 분할 |
| `sample` | 0/1 | 판정·분석 대상 표본이면 1 (DNA dev 전체, Jigsaw·PD dev 50개씩) |
| `mode` | `response` · `prompt` | 판정 대상: 챗봇 응답(DNA) / 사용자가 쓴 글(Jigsaw·PD) |
| `question` | str | DNA: 질문 · Jigsaw: `r/서브레딧` · PD: 이전 대화(최대 4발화) |
| `text` | str | 판정할 글 |
| `human_harmful` | 0/1 | 사람 라벨 (1 = 위반) |
| `response_llm` | str | DNA 응답 LLM (나머지는 빈칸) |
| `rule_l1`, `rule_l2`, `rule_l3`, `rule_l3_desc` | str | 정답 규칙 이름·설명 |
| `gold_l1`, `gold_l2`, `gold_l3` | node_id | 정답 경로 (없으면 빈칸) |
| `group_id` | str | 분할 단위 (DNA: 질문, Jigsaw: 같은 본문, PD: 대화) — 같은 그룹은 dev·test 에 갈라지지 않음 |

### `splits.csv`

`text_id`, `source`, `split`

### `jigsaw_examples.csv` — 원문 포함

| 열 | 설명 |
|---|---|
| `origin_text_id` | 예시가 딸려 있던 Jigsaw 행 |
| `rule` | 규칙 전문 |
| `label` | 1 = 위반 예시, 0 = 비위반 예시 |
| `text` | 예시 글 |

## 2단계 `make_assets.py`

### `rules.csv`

| 열 | 설명 |
|---|---|
| `node_id`, `tree`, `level`, `parent` | 트리 구조 (`parent` 가 `ROOT` 면 최상위) |
| `name` | 노드 이름 |
| `text` | 유사도 계산(임베딩)에 쓰는 문장형 규칙 |
| `judge_text` | 판정 모델에 넣는 규칙 문장 (DNA 소분류: `이름: 설명`, Jigsaw: 규칙 전문) |

### `similarity.csv`

`text_id` + 규칙 노드 81개 열. 값 = 텍스트와 노드 문장의 코사인 유사도 (all-roberta-large-v1, 정규화 임베딩의 내적)

### `coords.csv`

`text_id`, `x`, `y` — 텍스트 임베딩의 UMAP 2차원 좌표 (P3 지도)

### `judge_input.csv` — 원문 포함 (Colab 입력)

`text_id`, `source`, `mode`, `question`, `text`, `human_harmful`, `gold_rule`, `gold_rule_text`, `cand_1` … `cand_5`, `cand_1_text` … `cand_5_text`

### `text_emb.npy`, `text_emb_ids.csv`

임베딩 캐시 (1,024차원). 행 순서 = `text_emb_ids.csv` 의 `text_id` 순서

## 3단계 Colab

### `judgments.csv` (1차), `judgments_new.csv` (2차), `judgments_derived.csv` (`derive_conditions.py`)

| 열 | 설명 |
|---|---|
| `text_id` | |
| `judge`, `condition` | 위 표 |
| `p_violation` | 위반 확률 0~1 |
| `n_remaining_candidates` | (`judgments_derived.csv` 만) 규칙 제외 후 남은 후보 수 |

위반 확률 계산: 판정 단어 바로 앞까지 프롬프트를 넣고 다음 토큰 확률을 봄
- Llama Guard: P(unsafe) / (P(safe) + P(unsafe))
- Qwen3Guard: P(Unsafe) + 0.5 × P(Controversial) (세 단어 확률의 합 = 1)
- ShieldGemma·팀: P(Yes) / (P(Yes) + P(No))

### `base_candidates.csv`

`text_id`, `cand` (후보 node_id), `p_violation` — '기본' 조건의 후보별 판정. `base` 값 = 텍스트별 최댓값

### `team_prompt_comparison.csv`

`version`, `n`, `roc_auc`, `pr_auc` — 팀 프롬프트 비교 (DNA dev 300개, 규칙 문장만)

## 4단계 `analyze.py` → `results/`

| 파일 | 열 | 내용 |
|---|---|---|
| `judgments_all.csv` | text_id, judge, condition, p_violation, judge_cond | 모든 판정 합본 |
| `thresholds.csv` | judge, source, threshold | 판정 임계값 |
| `p1_retrieval.csv` | text_id, source, tau, top1, candidates, kept_l2, top1/top3/top5_correct, efficiency, diverge_level | τ = 0.20 BFS 결과 |
| `p1_tau_sweep.csv` | source, tau, n, top1, top3, efficiency | τ 비교 |
| `p1_boundary_cases.csv` | p1_retrieval 열 + top1_lower_tau, top1_higher_tau | τ±0.02 에서 1위가 바뀌는 텍스트 |
| `p2_metrics.csv` | source, judge, n, n_pos, threshold, roc_auc, pr_auc, f1, kappa_human, error_rate, n_pred_pos | 출처별 지표 |
| `p2_pairwise_kappa.csv` | source, judge_a, judge_b, kappa | 모델 간 κ |
| `p2_judgments_long.csv` | text_id, source, rule_l1, rule_l2, response_llm, human_harmful, judge, p_violation, pred, error | 판정 한 건 = 한 행 |
| `p2_errors_by_{rule_l1, rule_l2, response_llm, source}.csv` | judge, 묶음, n, n_error, error_rate, low_n | 모델 × 묶음 오판 |
| `p3_diagnosis.csv` | text_id, judge, p_violation, pred, human_harmful, error_type, cause, gold_outside_topk, source, x, y | 잘못된 판정과 원인 |
| `p3_cause_summary.csv` | source, judge, 원인 4개, 합계 | 원인별 건수 |
| `p4_base_vs_rule_only.csv` | source, model, auc_base, auc_rule_only, auc_change, prauc_* | 정답 규칙을 줬을 때 효과 |
| `p4_rule_excluded_by_rule.csv` | source, excluded_rule, n, n_pos, auc_base, auc_rule_excluded, auc_rule_only, auc_drop, auc_recovery_rule_only | 처음 보는 규칙 실험 |
| `p4_threshold_flips.csv` | judge, source, new_threshold, flipped, to_correct, to_wrong | 임계값을 바꿨을 때 뒤집히는 판정 |
| `p4_team_prompt_comparison.csv` | (위와 같음) | |

## 분석 보조 → `results/eda/`, `results/dna_judge/`

`eda.py`, `dna_judge_analysis.py` 결과. 열 이름이 그대로 뜻을 나타내도록 지었고, 내용은 `docs/DATASET.md` 에 정리했습니다.
