"""필터 범위·집계·P4 원복 및 캐시 재사용의 회귀 검증."""
import pandas as pd
import pytest

from src import effects, judges, retrieval, visuals
from src.metrics import pairwise_kappa


def test_aggregate_does_not_treat_previous_aggregate_as_candidate():
    j = pd.DataFrame(dict(text_id=["t"]*3, model="team", condition="base", prompt_version="v",
                          candidate_rule_id=["AGG:max", "A", "B"], p_violation=[.9, .2, .9]))
    assert judges.aggregate(j, "top1").p_violation.item() == .2


def test_compare_requires_same_cohort_and_single_split():
    before = pd.DataFrame(dict(text_id=["a"], p_violation=[.8], prediction=[1]))
    meta = pd.DataFrame(dict(text_id=["a", "b"], source="s", split=["dev", "test"], human_label=[1, 0]))
    with pytest.raises(ValueError, match="텍스트 범위"):
        effects.compare("x", {}, before, before.assign(text_id="b"), meta)
    both = pd.concat([before, before.assign(text_id="b")])
    with pytest.raises(ValueError, match="분리"):
        effects.compare("x", {}, both, both, meta)


def test_live_settings_preserves_source_thresholds_on_reset_and_reuses_candidates(monkeypatch):
    cfg = dict(judge=dict(team_prompt_versions=["v"], aggregation="max", top_k_candidates=2),
               retrieval=dict(tau=.2, tau_levels=[2], boundary_delta=.02))
    tgt = pd.DataFrame(dict(text_id=["a", "b"], source=["s1", "s2"], split="dev", human_label=[0, 1]))
    m = tgt.assign(model="team", condition="base", p_violation=[.3, .3], prediction=[0, 1], threshold=[.6, .1])
    raw = pd.DataFrame(dict(text_id=["a", "a", "b", "b"], model="team", condition="base", prompt_version="v",
                            candidate_rule_id=["A", "B", "A", "B"], p_violation=[.05, .3, .2, .3]))
    def no_inference(*args, **kwargs):
        pytest.fail("이미 저장된 후보를 다시 추론하면 안 된다")
    monkeypatch.setattr(judges, "score_rule_model", no_inference)
    # 원복: 출처마다 다른 .6/.1을 그대로 사용해야 한다.
    reset = effects.live_settings(cfg, tgt, pd.DataFrame(), {}, m, raw, "team", .2, "max")
    assert reset["n_flipped"] == 0
    monkeypatch.setattr(retrieval, "Tree", lambda _: None)
    monkeypatch.setattr(retrieval, "retrieve_all", lambda *args: pd.DataFrame(dict(text_id=["a", "b"], candidates=["A|B", "A|B"])))
    monkeypatch.setattr(effects, "base_items", lambda *args: [dict(text_id=t, condition="base", candidate_rule_id=n) for t in ["a", "b"] for n in ["A", "B"]])
    nodes = pd.DataFrame(dict(node_id=["A", "B"], rule_text=["ra", "rb"]))
    changed = effects.live_settings(cfg, tgt, nodes, {}, m, raw, "team", .1, "top1")
    assert changed["metrics_after"]["s1"]["n_pred_pos"] == 0
    assert changed["metrics_after"]["s2"]["n_pred_pos"] == 1
    tgt.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="dev"):
        effects.live_settings(cfg, tgt, nodes, {}, m, raw, "team", .2, "max")


def test_pairwise_kappa_uses_only_provided_texts():
    m = pd.DataFrame([(t, model, pred) for t, values in [("a", [0, 0]), ("b", [1, 1]), ("c", [0, 1])]
                      for model, pred in zip(["team", "shieldgemma"], values)], columns=["text_id", "model", "prediction"])
    m = m.assign(split="dev", source="s", condition="base")
    filtered = pairwise_kappa(m[m.text_id.isin(["a", "b"])])
    assert filtered.n.item() == 2 and filtered.kappa.item() == 1
    assert pairwise_kappa(m).kappa.item() < 1


def test_chart_selection_empty_list_and_records():
    assert visuals.selection_value({"selection": {"p": {"text_id": []}}}, "p", "text_id") is None
    assert visuals.selection_value({"selection": {"p": [{"text_id": "a"}]}}, "p", "text_id") == "a"
    assert visuals.selection_value({"selection": {"p": {"text_id": ["b"]}}}, "p", "text_id") == "b"


def test_tree_parent_statistics_include_descendants():
    nodes = pd.DataFrame(dict(node_id=["A", "B", "C"], parent_id=["", "A", "B"], level=[1, 2, 3], name=["root", "mid", "leaf"]))
    per = pd.DataFrame(dict(gold_rule_id=["C", "C"], hit1=[True, False], hit3=[True, True]))
    points, edges = visuals.tree_data(nodes, per, {"C": .6}, ["A", "B", "C"], ["A", "B", "C"])
    assert points.n.tolist() == [2, 2, 2]
    assert points.top1_acc.tolist() == [.5, .5, .5]
    assert len(edges) == 2 and set(points.path) == {"BFS + 정답"}


def test_legacy_proposal_can_be_displayed():
    prop = dict(flips=[dict(n_flipped=2, n_fixed=1, n_broken=1), dict(n_flipped=3, n_fixed=2, n_broken=1)])
    assert visuals.proposal_counts(prop) == dict(n_flipped=5, n_fixed=3, n_broken=2)
