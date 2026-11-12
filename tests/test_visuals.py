"""Visual summaries must preserve hierarchy membership and actual case counts."""
import pandas as pd
import pytest

from src import visuals


def test_icicle_parent_spans_children_and_focus_keeps_coordinates():
    nodes = pd.DataFrame(dict(node_id=["A", "B", "C", "D", "E", "F"],
                              parent_id=["", "A", "A", "B", "B", "C"],
                              level=[1, 2, 2, 3, 3, 3]))
    blocks = visuals.hierarchy_rectangles(nodes, nodes).set_index("node_id")
    assert blocks.loc["A", ["low", "high"]].tolist() == [0, 3]
    assert blocks.loc["B", ["low", "high"]].tolist() == [0, 2]
    assert blocks.loc["C", ["low", "high"]].tolist() == [2, 3]
    assert blocks.loc[["D", "E", "F"], "high"].tolist() == [1, 2, 3]
    focused = visuals.hierarchy_rectangles(nodes, nodes, "B").set_index("node_id")
    assert set(focused.index) == {"B", "D", "E"}
    pd.testing.assert_frame_equal(focused, blocks.loc[focused.index])


def test_outcome_flow_includes_unchanged_cases_and_uses_human_labels():
    baseline = pd.DataFrame(dict(text_id=list("abcd"), human_label=[1, 1, 0, 0], prediction=[1, 0, 0, 1]))
    result = visuals.outcome_transitions(baseline, [dict(text_id="b", after=1), dict(text_id="c", after=1)])
    assert result.transition.tolist() == ["unchanged_correct", "fixed", "broken", "unchanged_wrong"]
    assert result.after.tolist() == [1, 1, 1, 1]
    assert baseline.prediction.tolist() == [1, 0, 0, 1]
    unchanged = visuals.outcome_transitions(baseline, [])
    assert unchanged.transition.value_counts().to_dict() == {"unchanged_correct": 2, "unchanged_wrong": 2}


def test_outcome_flow_rejects_mismatched_cohorts():
    baseline = pd.DataFrame(dict(text_id=["a"], human_label=[1], prediction=[1]))
    with pytest.raises(ValueError):
        visuals.outcome_transitions(pd.concat([baseline, baseline]), [])
    with pytest.raises(ValueError):
        visuals.outcome_transitions(baseline, [dict(text_id="outside", after=0)])


def test_flow_ribbon_widths_preserve_counts_and_endpoints_cover_cohort():
    frame = pd.DataFrame(dict(transition=["unchanged_correct"] * 5 + ["fixed"] * 2 + ["broken"] + ["unchanged_wrong"] * 3))
    ribbons = visuals.transition_ribbons(frame)
    assert (ribbons.high - ribbons.low == ribbons.n).all()
    assert ribbons.groupby("x").n.sum().eq(len(frame)).all()
    for x in [0, 1]:
        edge = ribbons[ribbons.x == x].sort_values("low")
        assert edge.low.iloc[0] == 0 and edge.high.iloc[-1] == len(frame)
        assert edge.low.iloc[1:].tolist() == edge.high.iloc[:-1].tolist()
    assert visuals.transition_ribbons(frame.iloc[:0]).empty
