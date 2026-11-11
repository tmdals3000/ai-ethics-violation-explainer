"""
P1 규칙 찾기 (BFS) — make_assets.py, analyze.py, app.py 가 같이 씀

규칙 트리는 출처별로 따로 둔다 (가정 A-tree)
  - dna 트리   : 대분류 5 → 중분류 12 → 소분류 61 (챗봇 응답 정책). Do-Not-Answer 와 ProsocialDialog 가 사용
  - jigsaw 트리: Reddit 커뮤니티 규칙 1 → 규칙 2개
  이유: 플랫폼마다 규칙집이 따로 있다고 보는 게 자연스럽고, 한 트리에 섞으면 문장이 긴 Jigsaw 규칙이
       DNA 텍스트를 끌어가는 문제가 생김 (승민 님 보고서의 멘토링 질문 1번과 같은 현상)

BFS (τ = 중분류를 남기는 유사도 기준)
  - 대분류·중분류는 모두 비교
  - 중분류 중 유사도 ≥ τ 인 것만 남김 (모두 τ 미만이면 가장 높은 1개)
  - 남은 중분류의 자식(소분류)을 모두 비교. 자식이 없는 중분류(Jigsaw)는 그 자신이 후보
  - 후보를 유사도 순으로 정렬. 효율 = 비교한 노드 수 / 그 트리의 전체 노드 수
"""
import pandas as pd

TAU = 0.20
TOP_K = 5
TREE_OF_SOURCE = {"do_not_answer": "dna", "prosocial": "dna", "jigsaw": "jigsaw"}


class Tree:
    def __init__(self, rules):
        self.rules = rules
        self.name = dict(zip(rules["node_id"], rules["name"]))
        self.text = dict(zip(rules["node_id"], rules["text"]))
        self.parent = dict(zip(rules["node_id"], rules["parent"]))
        self.children = rules.groupby("parent")["node_id"].apply(list).to_dict()
        self.by_tree = {}
        for t, g in rules.groupby("tree"):
            self.by_tree[t] = {"L1": g[g.level == 1]["node_id"].tolist(),
                               "L2": g[g.level == 2]["node_id"].tolist(), "n": len(g)}

    def bfs(self, row_sim, source, tau=TAU):
        """텍스트 1개의 유사도(Series: node_id → 값) → (후보 목록, 남긴 중분류, 효율)"""
        t = self.by_tree[TREE_OF_SOURCE[source]]
        s2 = row_sim[t["L2"]]
        kept = s2[s2 >= tau].index.tolist() or [s2.idxmax()]
        leaves = []
        for n in kept:
            leaves += self.children.get(n, [n])
        ranked = row_sim[leaves].sort_values(ascending=False).index.tolist()
        n_cmp = len(t["L1"]) + len(t["L2"]) + sum(len(self.children.get(n, [])) for n in kept)
        return ranked, kept, n_cmp / t["n"]


def gold_leaf(row):
    """정답 잎 노드: DNA 는 소분류, Jigsaw 는 규칙(중분류). ProsocialDialog 는 없음"""
    g3, g2 = row.get("gold_l3", ""), row.get("gold_l2", "")
    g3 = "" if pd.isna(g3) else g3
    g2 = "" if pd.isna(g2) else g2
    return g3 or g2
