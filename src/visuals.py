"""화면용 규칙 트리와 차트 선택 해석. 추론·파일 저장은 하지 않는다."""
import pandas as pd

from src.retrieval import Tree


def selection_value(event, name, field):
    """Streamlit의 point 선택(dict/list)을 같은 방식으로 읽는다."""
    selection = event.get("selection", {}) if event else {}
    value = selection.get(name, {})
    if isinstance(value, list):
        return value[0].get(field) if value else None
    if isinstance(value, dict):
        value = value.get(field)
        return (value[0] if value else None) if isinstance(value, list) else value
    return None


def tree_data(nodes, per, similarity=None, predicted=(), gold=(), divergence=""):
    """각 노드의 위치·경로·정답 가지별 검색 정확도와 연결선을 만든다."""
    tree = Tree(nodes)
    positions, next_leaf = {}, 0

    def place(nid):
        nonlocal next_leaf
        children = tree.children.get(nid, [])
        if children:
            y = sum(place(c) for c in children) / len(children)
        else:
            y, next_leaf = next_leaf, next_leaf + 1
        positions[nid] = y
        return y

    for nid in tree.roots:
        place(nid)
    rows = nodes.copy()
    rows["x"] = rows.level
    rows["y"] = rows.node_id.map(positions)
    rows["label"] = rows.node_id + " · " + rows.name.astype(str).str.slice(0, 27)
    rows["similarity"] = rows.node_id.map(similarity or {})
    rows["path"] = ["BFS + 정답" if n in predicted and n in gold else "BFS" if n in predicted
                    else "정답" if n in gold else "기타" for n in rows.node_id]
    rows["divergence"] = rows.node_id == divergence
    rows["n"] = 0
    rows["top1_acc"] = float("nan")
    rows["top3_acc"] = float("nan")
    if len(per) and "gold_rule_id" in per:
        for i, row in rows.iterrows():
            g = per[per.gold_rule_id.isin(tree.subtree(row.node_id))]
            rows.loc[i, "n"] = len(g)
            if len(g):
                rows.loc[i, "top1_acc"] = g.hit1.mean()
                rows.loc[i, "top3_acc"] = g.hit3.mean()
    edges = []
    for r in rows.itertuples():
        if r.parent_id:
            edges.append(dict(x=tree.level[r.parent_id], y=positions[r.parent_id], x2=r.x, y2=r.y,
                              path=r.path if r.parent_id in set(predicted) | set(gold) else "기타"))
    return rows, pd.DataFrame(edges, columns=["x", "y", "x2", "y2", "path"])


def proposal_counts(prop):
    """P4 표준 저장 형식과 exp-v1의 R1 저장 형식을 모두 지원한다."""
    flips = prop.get("flips", [])
    return {key: prop.get(key, sum(int(row.get(key, 0)) for row in flips))
            for key in ("n_flipped", "n_fixed", "n_broken")}


def hierarchy_rectangles(nodes, points, focus=None):
    """Icicle overview: parent spans its leaves; focus preserves the same hierarchy."""
    tree = Tree(nodes)
    bounds = {}
    next_leaf = 0

    def visit(nid):
        nonlocal next_leaf
        children = tree.children.get(nid, [])
        if children:
            spans = [visit(child) for child in children]
            low, high = spans[0][0], spans[-1][1]
        else:
            low, high = next_leaf, next_leaf + 1
            next_leaf += 1
        bounds[nid] = (low, high)
        return low, high

    for nid in tree.roots:
        visit(nid)
    result = points.copy()
    if focus in bounds:
        result = result[result.node_id.isin(tree.subtree(focus))].copy()
    result["left"] = result.level.astype(float)
    result["right"] = result.left + .92
    result["low"] = result.node_id.map(lambda n: bounds[n][0])
    result["high"] = result.node_id.map(lambda n: bounds[n][1])
    result["center_x"] = (result.left + result.right) / 2
    result["center_y"] = (result.low + result.high) / 2
    return result


TRANSITION_NAMES = {"unchanged_correct": "맞음 유지", "fixed": "고쳐진 판정", "broken": "새로 틀린 판정", "unchanged_wrong": "오류 유지"}
TRANSITION_COLORS = {"unchanged_correct": "#afc8d3", "fixed": "#429b85", "broken": "#db8a80", "unchanged_wrong": "#a5afbf"}


def outcome_transitions(before, flipped):
    """Keep the baseline cohort fixed; apply only recorded prediction changes."""
    required = ["text_id", "human_label", "prediction"]
    frame = before[required].copy().rename(columns={"prediction": "before"})
    if not frame.text_id.is_unique:
        raise ValueError("변화 흐름에는 텍스트별 기본 판정이 하나씩 필요합니다.")
    changes = {row["text_id"]: int(row["after"]) for row in flipped}
    if set(changes) - set(frame.text_id):
        raise ValueError("기본 판정에 없는 변경 사례가 있습니다.")
    frame["after"] = [changes.get(t, int(p)) for t,p in zip(frame.text_id, frame.before)]
    good_before = frame.before == frame.human_label
    good_after = frame.after == frame.human_label
    frame["transition"] = ["unchanged_correct" if b and a else "broken" if b else "fixed" if a else "unchanged_wrong"
                           for b,a in zip(good_before,good_after)]
    frame["변화"] = frame.transition.map(TRANSITION_NAMES)
    return frame


def transition_ribbons(frame):
    """Smooth ribbons with width proportional to real cohort counts, not scores."""
    counts = frame.transition.value_counts().to_dict()
    good_before = counts.get("unchanged_correct",0) + counts.get("broken",0)
    good_after = counts.get("unchanged_correct",0) + counts.get("fixed",0)
    starts = dict(unchanged_correct=0, broken=counts.get("unchanged_correct",0), fixed=good_before,
                  unchanged_wrong=good_before+counts.get("fixed",0))
    ends = dict(unchanged_correct=0, fixed=counts.get("unchanged_correct",0), broken=good_after,
                unchanged_wrong=good_after+counts.get("broken",0))
    rows=[]
    for name in TRANSITION_NAMES:
        n=counts.get(name,0)
        if not n:
            continue
        for step in range(31):
            x=step/30
            smooth=x*x*(3-2*x)
            low=starts[name]*(1-smooth)+ends[name]*smooth
            rows.append(dict(x=x,low=low,high=low+n,transition=name,label=TRANSITION_NAMES[name],n=n))
    return pd.DataFrame(rows,columns=["x","low","high","transition","label","n"])
