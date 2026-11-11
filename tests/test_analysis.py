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


def test_diagnose_does_not_skip_to_rule4_when_models_missing():
    per = pd.DataFrame({"text_id": ["a"], "gold_rank": [1]})
    d = diagnose(_m([("a", "qwen3guard", "base", 0), ("a", "llama_guard", "base", 1)]), per, top_k=5)
    q = d.set_index("model").loc["qwen3guard"]
    assert q.cause == "unclassified" and "적용 불가" in q.decided_by


def test_thresholds_per_source_and_common():
    from src.judges import choose_thresholds
    # 출처 A 는 낮은 확률대(0.1/0.2), B 는 높은 확률대(0.7/0.9) 에서 위반이 갈린다
    j = pd.DataFrame({"text_id": list("abcdefgh"), "p_violation": [.1, .2, .1, .2, .7, .9, .7, .9]})
    texts = pd.DataFrame({"text_id": list("abcdefgh"), "source": list("AAAABBBB"), "split": "dev",
                          "human_label": [0, 1, 0, 1, 0, 1, 0, 1]})
    th = choose_thresholds({"judge": {"team_threshold_step": 0.01}}, j, texts)
    assert 0.1 < th["per_source"]["A"][0] <= 0.2 and th["per_source"]["A"][1] == 1.0
    assert 0.7 < th["per_source"]["B"][0] <= 0.9 and th["per_source"]["B"][1] == 1.0
    assert th["common"][1] < 1.0  # 공통 임계값 하나로는 두 출처를 모두 맞힐 수 없다


def test_token_budget_batches():
    from src.judges import token_budget_batches
    b = token_budget_batches([100, 100, 100, 600, 900, 1600], max_batch=8, budget=2_000_000)
    assert b == [[0, 1, 2, 3], [4], [5]]  # 4×600²=1.44e6 ≤ 2e6, 5×900² > 2e6, 2×1600² > 2e6
    assert sum(map(len, b)) == 6 and all(len(x) <= 8 for x in b)


def test_thresholds_kappa_criterion_differs_from_f1():
    from src.judges import choose_thresholds
    # 위반이 많은 출처: F1 은 '전부 위반' 쪽(낮은 임계값)이 유리, κ 는 실제로 가르는 0.5 근처를 고른다
    p = [.1, .2, .3, .35, .6, .7, .8, .9, .95, .97]
    y = [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]  # 전부 위반: F1 0.889 > 0.46 기준 F1 0.857 / κ 는 0 vs 0.545
    j = pd.DataFrame({"text_id": list("abcdefghij"), "p_violation": p})
    texts = pd.DataFrame({"text_id": list("abcdefghij"), "source": "A", "split": "dev", "human_label": y})
    cfg = {"judge": {"team_threshold_step": 0.01}}
    f1 = choose_thresholds(cfg, j, texts, "f1")
    ka = choose_thresholds(cfg, j, texts, "kappa")
    assert f1["metric"] == "f1" and ka["metric"] == "kappa"
    assert f1["per_source"]["A"][0] <= 0.1 and 0.35 < ka["per_source"]["A"][0] <= 0.6


def test_team_v4_uses_compliance_for_response_and_v2_for_prompt():
    from src.judges import TeamJudge

    def build(pv, mode):
        j = TeamJudge.__new__(TeamJudge)
        j.prompt_version = pv
        return j.build({"text": "T", "context": "C", "rule": "R", "mode": mode})
    assert build("team_v4", "prompt") == build("team_v2", "prompt")
    r = build("team_v4", "response")
    assert "actually complies with or assists the request" in r and r != build("team_v2", "response")
