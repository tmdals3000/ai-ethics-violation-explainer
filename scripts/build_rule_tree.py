"""규칙 트리(대·중·소분류) 만들기 → rule_tree.csv, gold_paths.csv

texts_all.csv에 들어 있는 규칙으로 트리를 만듦 (combine_texts.py 다음에 실행).

트리 구조
  Do-Not-Answer : 대분류(risk_area 5) → 중분류(types_of_harm 12) → 소분류(specific_harms 61)
  Jigsaw        : 대분류("Community rules" 1) → 규칙(rule, 잎 노드)
  ProsocialDialog는 정답 규칙이 없어서 트리에 넣지 않음

rule_tree.csv 열
  node_id, source, level(1·2·3), parent_id, name, node_text(임베딩할 문장), is_leaf, n_gold(이 규칙이 정답인 텍스트 수)
gold_paths.csv 열
  text_id, gold_l1, gold_l2, gold_l3, gold_leaf  (경로가 2단계인 Jigsaw는 gold_l3가 비어 있음)

    python scripts/build_rule_tree.py
"""
import pandas as pd

from common import PROCESSED


def main():
    t = pd.read_csv(PROCESSED / "texts_all.csv", keep_default_na=False)
    nodes, gold = [], []

    # ---- Do-Not-Answer: 3단계 ----
    d = t[t.source == "dna"]
    if len(d):
        l1 = {v: f"dna_L1_{i + 1}" for i, v in enumerate(sorted(d.rule_l1.unique()))}
        l2 = {v: f"dna_L2_{i + 1}" for i, v in enumerate(sorted(d.rule_l2.unique()))}
        l3 = {v: f"dna_L3_{i + 1}" for i, v in enumerate(sorted(d.rule_text.unique()))}
        for v, nid in l1.items():
            nodes.append(dict(node_id=nid, source="dna", level=1, parent_id="", name=v, node_text=v))
        for v, nid in l2.items():
            parent = d[d.rule_l2 == v].rule_l1.iloc[0]
            nodes.append(dict(node_id=nid, source="dna", level=2, parent_id=l1[parent], name=v, node_text=v))
        for v, nid in l3.items():
            parent = d[d.rule_text == v].rule_l2.iloc[0]
            name = d[d.rule_text == v].gold_rule.iloc[0]
            nodes.append(dict(node_id=nid, source="dna", level=3, parent_id=l2[parent], name=name, node_text=v))
        gold.append(pd.DataFrame({"text_id": d.text_id, "gold_l1": d.rule_l1.map(l1),
                                  "gold_l2": d.rule_l2.map(l2), "gold_l3": d.rule_text.map(l3)}))

    # ---- Jigsaw: 2단계 ----
    j = t[t.source == "jigsaw"]
    if len(j):
        root = "jigsaw_L1_1"
        nodes.append(dict(node_id=root, source="jigsaw", level=1, parent_id="", name="Community rules",
                          node_text="Community rules: rules that moderators of online communities enforce on comments"))
        rules = {v: f"jigsaw_rule_{i + 1}" for i, v in enumerate(sorted(j.rule_text.unique()))}
        for v, nid in rules.items():
            name = j[j.rule_text == v].gold_rule.iloc[0]
            nodes.append(dict(node_id=nid, source="jigsaw", level=2, parent_id=root, name=name, node_text=v))
        gold.append(pd.DataFrame({"text_id": j.text_id, "gold_l1": root,
                                  "gold_l2": j.rule_text.map(rules), "gold_l3": ""}))

    if not nodes:
        raise SystemExit("정답 규칙이 있는 데이터(dna, jigsaw)가 texts_all.csv에 없습니다.")
    tree = pd.DataFrame(nodes)
    tree["is_leaf"] = ~tree.node_id.isin(set(tree.parent_id))
    g = pd.concat(gold, ignore_index=True)
    g["gold_leaf"] = g.gold_l3.where(g.gold_l3 != "", g.gold_l2)
    tree["n_gold"] = tree.node_id.map(g.gold_leaf.value_counts()).fillna(0).astype(int)

    tree.to_csv(PROCESSED / "rule_tree.csv", index=False)
    g.to_csv(PROCESSED / "gold_paths.csv", index=False)
    print(f"rule_tree.csv: 노드 {len(tree)}개 (잎 {int(tree.is_leaf.sum())}개)")
    print(tree.groupby(["source", "level"]).size().rename("노드 수").to_string())
    print(f"gold_paths.csv: {len(g):,}행")


if __name__ == "__main__":
    main()
