"""P2·P3·집계 테스트 (가짜 판정표)."""
import pandas as pd

from src.diagnose import diagnose
from src.judges import aggregate
from src.metrics import binary_metrics, main_rows


def _m(rows):
    """rows: (text_id, model, condition, prediction), 사람 라벨은 모두 1."""
    df = pd.DataFrame(rows, columns=["text_id", "model", "condition", "prediction"])
    return df.assign(human_label=1, source="do_not_answer", gold_rule_id="S01", rule_group="H01",
                     p_violation=df.prediction * 1.0, correct=(df.prediction == 1).astype(int))


def test_metrics_single_class_gives_reason():
    r = binary_metrics(pd.DataFrame({"human_label": [0, 0], "p_violation": [.1, .9], "prediction": [0, 1]}))
    assert r["roc_auc"] != r["roc_auc"] and "AUC" in r["note"]


def test_aggregate_max_and_top1():
    j = pd.DataFrame({"text_id": ["t"] * 2, "model": "team", "condition": "base", "prompt_version": "v",
                      "candidate_rule_id": ["S02", "S01"], "p_violation": [.2, .9]})
    assert aggregate(j, "max").p_violation.item() == .9
    assert aggregate(j, "top1").p_violation.item() == .2  # 유사도 1위 후보(첫 행)
    assert len(main_rows(pd.concat([j, aggregate(j, "max")]))) == 1


def _all(t, base, ro):
    """base: 모델별 base 예측, ro: 규칙 모델 rule_only 예측."""
    rows = [(t, mo, "base", p) for mo, p in base.items()]
    rows += [(t, mo, "rule_only", p) for mo, p in ro.items()]
    return rows


def test_diagnose_priority_order():
    per = pd.DataFrame({"text_id": ["a", "b", "c", "d", "e"], "gold_rank": [9, 1, 1, 1, 1]})
    four = dict(llama_guard=0, qwen3guard=0, shieldgemma=0, team=0)
    rows = (_all("a", four, dict(shieldgemma=0, team=0))                                     # 정답 규칙 못 찾음
            + _all("b", four | dict(llama_guard=1), dict(shieldgemma=1, team=0))             # rule_only 맞음
            + _all("c", four, dict(shieldgemma=0, team=0))                                   # 모두 같은 답
            + _all("d", four | dict(llama_guard=1), dict(shieldgemma=0, team=0))             # 규칙 모델 둘 다 틀림
            + _all("e", four | dict(llama_guard=1), dict(shieldgemma=1, team=0)))            # 팀만 틀림
    d = diagnose(_m(rows), per, top_k=5).set_index(["text_id", "model"]).cause
    assert d["a", "team"] == "retrieval"
    assert d["b", "shieldgemma"] == "retrieval"
    assert d["c", "qwen3guard"] == "label_check"
    assert d["d", "qwen3guard"] == "rule_text"
    assert d["e", "qwen3guard"] == "judge_model"   # rule_only 에서 shieldgemma·llama 는 맞음
    assert d["e", "team"] == "judge_model"


def test_diagnose_missing_models_not_forced_to_label_check():
    per = pd.DataFrame({"text_id": ["a"], "gold_rank": [1]})
    d = diagnose(_m([("a", "qwen3guard", "base", 0)]), per, top_k=5)
    assert d.cause.item() == "unclassified"
    assert not d.rule2_applicable.item()
    assert d.missing_models.item() == "llama_guard|shieldgemma|team"
