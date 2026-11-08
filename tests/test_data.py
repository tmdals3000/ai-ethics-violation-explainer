"""1단계 테스트: tests/fixtures 의 가짜 데이터만 사용한다."""
from pathlib import Path

from src import data

RAW = Path(__file__).parent / "fixtures/raw"


def test_dna_melt_one_row_per_response():
    d = data.load_dna(RAW)
    assert len(d) == 3 * 6  # 질문 3개 × 응답 LLM 6개
    assert d["text_id"].is_unique
    assert set(d["response_llm"]) == set(data.DNA_LLMS)
    assert d.loc[d.response_llm == "ChatGLM2", "human_label"].eq(1).all()


def test_jigsaw_examples_split_out():
    t, ex = data.load_jigsaw(RAW)
    assert len(t) == 6 and len(ex) == 6 * 4
    assert ex["label"].value_counts().to_dict() == {1: 12, 0: 12}


def test_tree_and_gold_path():
    d = data.load_dna(RAW)
    t, _ = data.load_jigsaw(RAW)
    nodes, _ = data.build_tree(d, t)
    assert nodes.groupby("level").size().to_dict() == {1: 3, 2: 4, 3: 3}  # 대: RA 2 + J
    assert d["gold_path"].str.count(">").eq(2).all()
    assert set(t["gold_rule_id"]) == {"J1", "J2"}
    # 모든 경로의 노드가 노드표에 있다
    ids = set(nodes.node_id)
    assert all(set(p.split(">")) <= ids for p in d["gold_path"])


def test_split_keeps_question_together_and_is_deterministic():
    d = data.load_dna(RAW)
    r = {"train": 0.34, "dev": 0.33, "test": 0.33}
    s1, s2 = data.temporary_split(d, r, 1), data.temporary_split(d, r, 1)
    assert (s1 == s2).all()
    assert d.assign(s=s1).groupby("group_id")["s"].nunique().eq(1).all()


def test_jigsaw_same_body_same_split():
    t, _ = data.load_jigsaw(RAW)
    s = data.temporary_split(t, {"train": 0.34, "dev": 0.33, "test": 0.33}, 1)
    u = data.unify_split_by_text(t, s)
    assert t.assign(s=u).groupby("text")["s"].nunique().eq(1).all()
    assert ((u != s) <= t["text"].duplicated(keep=False)).all()  # 중복 본문 행만 바뀜
