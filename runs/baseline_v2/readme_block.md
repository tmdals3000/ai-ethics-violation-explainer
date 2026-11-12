<!-- results:baseline-v2 -->
### Baseline v2 (dev, 2026-11-12)

> ⚠ **모든 수치는 dev 결과다. test 는 아직 보지 않았다.** 검증 상태: `IS만` (In-Sample: 임계값·τ 를 정한 dev 로 평가해 낙관적).
> 표본(dev): DNA 2,256건, Jigsaw 783건, PD 500건. split·규칙 트리는 임시([ASSUMPTIONS](docs/ASSUMPTIONS.md)).
> 원자료: `runs/baseline_v2/experiments.csv` (version = `baseline-v2`). 이 표는 `python scripts/record_run.py readme --version baseline-v2` 로 만든다.

**실험 설정**

| run_id | 모델 | prompt_version | threshold | threshold_rule | tau | aggregation | 커밋 |
|---|---|---|---|---|---|---|---|
| B-LG | meta-llama/Llama-Guard-3-1B | official | 0.5 | 고정 0.5 | - (규칙 안 받음) | - (1회 판정) | `99da523428e387237c687cab047f42233e6c8801+workingtree` |
| B-QG | Qwen/Qwen3Guard-Gen-0.6B | official | 0.5 | 고정 0.5 | - (규칙 안 받음) | - (1회 판정) | `99da523428e387237c687cab047f42233e6c8801+workingtree` |
| B-SG | google/shieldgemma-2b | official | 0.5 | 고정 0.5 | 0.2 | 상위 5후보 max | `99da523428e387237c687cab047f42233e6c8801+workingtree` |
| T2-κ | google/shieldgemma-2b | team_v2 | DNA 0.63 / Jigsaw 0.44 / PD 0.05 | 출처별 dev κ(사람 라벨) 최대, 격자 0.01 | 0.2 | 상위 5후보 max | `99da523428e387237c687cab047f42233e6c8801+workingtree` |

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
| DNA | T2-κ | 2,256 | 65 | 0.483 | 0.054 | 0.116 | 0.083 |
| Jigsaw | B-LG | 783 | 400 | 0.522 | 0.536 | 0.533 | 0.012 |
| Jigsaw | B-QG | 783 | 400 | 0.653 | 0.632 | 0.663 | 0.270 |
| Jigsaw | B-SG | 783 | 400 | 0.690 | 0.670 | 0.529 | 0.219 |
| Jigsaw | T2-κ | 783 | 400 | 0.684 | 0.668 | 0.646 | 0.263 |
| PD | B-LG | 500 | 358 | 0.667 | 0.853 | 0.352 | 0.126 |
| PD | B-QG | 500 | 358 | 0.741 | 0.877 | 0.689 | 0.291 |
| PD | B-SG | 500 | 358 | 0.733 | 0.875 | 0.421 | 0.148 |
| PD | T2-κ | 500 | 358 | 0.750 | 0.876 | 0.799 | 0.370 |

**모델 간 Cohen's κ** (base 조건 예측)

| run_id | DNA | Jigsaw | PD |
|---|---|---|---|
| B-LG×B-QG | 0.370 | 0.325 | 0.308 |
| B-LG×B-SG | 0.295 | 0.207 | 0.478 |
| B-LG×T2-κ | 0.127 | 0.153 | 0.157 |
| B-QG×B-SG | 0.433 | 0.374 | 0.405 |
| B-QG×T2-κ | 0.213 | 0.230 | 0.351 |
| B-SG×T2-κ | 0.272 | 0.509 | 0.238 |

**P3 원인별 잘못된 판정 건수** (규칙 찾기는 규칙을 받는 모델에만 적용)

| 출처 | run_id | 규칙 찾기 | 라벨 확인 | 규칙 문장 | 판정 모델 | unclassified |
|---|---|---|---|---|---|---|
| DNA | B-LG | n/a | 48 | 5 | 49 | 0 |
| DNA | B-QG | n/a | 48 | 5 | 34 | 0 |
| DNA | B-SG | 50 | 13 | 4 | 2 | 0 |
| DNA | T2-κ | 74 | 13 | 4 | 61 | 0 |
| Jigsaw | B-LG | n/a | 85 | 46 | 255 | 0 |
| Jigsaw | B-QG | n/a | 85 | 11 | 189 | 0 |
| Jigsaw | B-SG | 137 | 22 | 56 | 93 | 0 |
| Jigsaw | T2-κ | 131 | 22 | 56 | 79 | 0 |

**P4 처음 보는 규칙** (정답 규칙이 있는 텍스트, 하락폭 = base − 규칙 제외, 회복폭 = 조건 − 규칙 제외)

| 출처 | run_id | auc_base | auc_rule_excluded | auc_rule_only | auc_rule_plus_examples | auc_drop | recovery_rule_only | recovery_rule_plus_examples |
|---|---|---|---|---|---|---|---|---|
| DNA | B-SG | 0.516 | 0.583 | 0.568 | 0.581 | -0.068 | -0.015 | -0.002 |
| DNA | T2-κ | 0.483 | 0.565 | 0.515 | 0.434 | -0.082 | -0.049 | -0.131 |
| Jigsaw | B-SG | 0.690 | 0.667 | 0.735 | 0.720 | 0.023 | 0.067 | 0.053 |
| Jigsaw | T2-κ | 0.684 | 0.672 | 0.691 | 0.688 | 0.013 | 0.019 | 0.017 |

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
| T2-κ | τ → 0.15 | 77 | 31 | 46 |
| T2-κ | τ → 0.25 | 55 | 30 | 25 |
| T2-κ | 임계값 → 0.3 | 733 | 102 | 631 |
| T2-κ | 임계값 → 0.7 | 518 | 174 | 344 |
| T2-κ | 집계 → top1 | 179 | 68 | 111 |

**P4 규칙 문장 수정 후보** (dev, 출처별 임계값 고정, test 미확인)

| 출처 | run_id | 규칙 | before_n | before_roc_auc | after_roc_auc | before_f1 | after_f1 | n_fixed | n_broken |
|---|---|---|---|---|---|---|---|---|---|
| DNA | T2-κ | S25 (rule_edit_S25_team_v2) | 2,256 | 0.483 | 0.483 | 0.116 | 0.106 | 1 | 1 |
| DNA | T2-κ | S47 (rule_edit_S47_team_v2) | 2,256 | 0.483 | 0.486 | 0.116 | 0.116 | 0 | 0 |
| DNA | T2-κ | S61 (rule_edit_S61_team_v2) | 2,256 | 0.483 | 0.464 | 0.116 | 0.116 | 0 | 1 |

**실행 시간** (RTX 3060 8GB)

| run_id | 작업 | 시간 | 비고 |
|---|---|---|---|
| step:p2 | p2 | 0.0분 (3초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 10:08:47; do_not_answer=all;jigsaw=all;prosocial=500 |
| step:p3 | p3 | 0.0분 (1초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 10:08:47; do_not_answer=all;jigsaw=all;prosocial=500 |
| step:p4 | p4 | 25.2분 (1,513초) | 단계 전체 (마지막 실행); timings.csv 2026-11-12 10:08:47; do_not_answer=all;jigsaw=all;prosocial=500 |
| T2 | team | 27.6분 (1,659초) | 판정 (캐시에 없던 것만 새로 계산); n_forward=11229; model_status.csv; 전체 프롬프트 30,323 |
<!-- /results:baseline-v2 -->