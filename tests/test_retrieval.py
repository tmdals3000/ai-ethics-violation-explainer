"""P1 BFS 테스트 (손으로 만든 작은 트리)."""
import pandas as pd

from src.retrieval import Tree, bfs, excluded_for, first_divergence

NODES = pd.DataFrame([
    ("A", 1, ""), ("B", 1, ""),
    ("A1", 2, "A"), ("A2", 2, "A"), ("B1", 2, "B"),
    ("A1a", 3, "A1"), ("A1b", 3, "A1"), ("A2a", 3, "A2"),
], columns=["node_id", "level", "parent_id"])
SIM = {"A": .5, "B": .1, "A1": .4, "A2": .1, "B1": .35, "A1a": .2, "A1b": .6, "A2a": .9}


def test_bfs_keeps_only_mid_nodes_above_tau():
    r = bfs(SIM, Tree(NODES), tau=0.3)
    assert r["kept"][2] == ["A1", "B1"]          # A2(0.1) 탈락
    assert r["ranking"] == ["A1b", "B1", "A1a"]  # 유사도순. B1 은 자식 없음 → 그 자체가 후보
    assert r["compared"] == 2 + 3 + 2            # 대 2 + 중 3 + 소 2


def test_bfs_fallback_when_all_below_tau():
    r = bfs(SIM, Tree(NODES), tau=0.99)
    assert r["kept"][2] == ["A1"]


def test_excluded_and_divergence():
    t = Tree(NODES)
    ex = excluded_for(t, "A>A1>A1a")
    assert ex == {"A1", "A1a", "A1b"}
    r = bfs(SIM, t, 0.3, excluded=ex)
    assert "A1a" not in r["ranking"] and r["ranking"] == ["B1"]
    assert first_divergence(["A", "A1", "A1b"], ["A", "A1", "A1a"]) == "소"
    assert first_divergence(["B", "B1"], ["A", "A1", "A1a"]) == "대"
    assert first_divergence(["A", "A1", "A1a"], ["A", "A1", "A1a"]) == "none"
