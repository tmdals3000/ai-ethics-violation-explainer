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
