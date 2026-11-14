"""4가지 조건 판정 목록 만들기 → conditions_map.csv, conditions_jobs.csv

PDF: "dev·test에서 4가지 조건으로 판정해 judgments.csv 형식으로 저장합니다. 기본 조건은 상위 5개 후보마다 판정합니다."
대상: 정답 규칙이 있는 Do-Not-Answer와 Jigsaw (ProsocialDialog는 정답 규칙이 없어 제외)

    기본          BFS로 찾은 규칙 상위 5개 각각으로 판정 → 가장 높은 위반 확률 (어느 규칙이든 어기면 위반)
    규칙 제외     정답 규칙의 중분류(DNA 위해 유형, Jigsaw 규칙)를 트리에서 빼고 BFS로 찾은 상위 5개 → 같은 방식
                  = 처음 보는 규칙 (rule_search.csv의 found_leaves_excl)
    규칙 문장만   정답 규칙 문장 하나 (1주차 '규칙 문장만'과 같은 프롬프트 → 1주차 값이 재현되어야 정상)
    규칙 문장 + 예시  정답 규칙 문장 + 위반 1개·비위반 1개 예시

예시 고르는 법 (정답 유출 방지)
  Jigsaw : examples_jigsaw.csv에서 safe_for_eval = True, label_conflict = False인 것만 (dev/test 댓글과 겹치지 않음)
  DNA    : 예시 목록이 따로 없어서 dev 응답에서 고름. 같은 소분류 규칙의 다른 질문(qid가 다름)에서 위반·비위반 하나씩,
           위반이 없으면 같은 중분류에서. test를 판정할 때도 예시는 dev에서만 → test 라벨은 쓰지 않음

판정 횟수를 줄이려고 (텍스트, 규칙, 예시 여부)가 같은 판정은 한 번만 함 (conditions_jobs.csv가 중복 없는 목록).

    python scripts/build_conditions.py --ids-from data/processed/judgments_baseline_dev300s.csv data/processed/judgments_baseline_dev300_jigsaw-prosocial.csv
    python scripts/build_conditions.py --n 300 --stratified            # 기준 판정 파일 없이 새로 뽑기
    python scripts/build_conditions.py --split test --n 300 --stratified   # test 추가 (dev 목록에 이어 붙음)
"""
import argparse
import json

import pandas as pd

from analysis import Tree
from common import PROCESSED, hash_bucket, need

TOP = 5


def sample_ids(texts, sources, split, n, stratified, seed):
    out = []
    for src in sources:
        d = texts[(texts.source == src) & (texts.split == split)]
        if stratified:
            pos = d[d.human_label == 1]
            neg = d[d.human_label == 0].sample(n=max(min(n - len(pos), (d.human_label == 0).sum()), 0), random_state=seed)
            d = pd.concat([pos, neg])
        elif n < len(d):
            d = d.sample(n=n, random_state=seed)
        out.append(d.text_id)
    return pd.concat(out)


def qid_of(text_id):                       # dna_{qid}_{responder}
    return text_id.split("_")[1]


def dna_examples(target_id, leaf, tree, pool, n_each):
    """같은 소분류 → 없으면 같은 중분류에서, 다른 질문의 dev 응답을 위반·비위반 n_each개씩."""
    q = qid_of(target_id)
    picked = []
    for lab in (1, 0):
        for scope in ([leaf], tree.leaves_under(tree.unit_of(leaf))):
            c = pool[(pool.gold_leaf.isin(scope)) & (pool.human_label == lab) & (pool.qid != q)]
            if len(c):
                c = c.assign(_h=[hash_bucket(target_id + t, 10 ** 9) for t in c.text_id]).sort_values("_h")
                picked += [(r.text_id, f"Question: {r.context[:200]} Response: {r.text[:300]}", lab)
                           for r in c.head(n_each).itertuples()]
                break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-from", nargs="*", default=[], help="이 판정 파일들의 text_id를 그대로 사용 (기준 모델과 같은 표본)")
    ap.add_argument("--sources", nargs="+", default=["dna", "jigsaw"], choices=["dna", "jigsaw"])
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--n", type=int, default=300, help="--ids-from이 없을 때 출처마다 뽑을 수")
    ap.add_argument("--stratified", action="store_true", help="DNA: 위반 전부 + 정상 무작위")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top", type=int, default=TOP, help="기본·규칙 제외 조건에서 판정할 후보 수 (PDF 5)")
    ap.add_argument("--n-examples", type=int, default=1, help="위반·비위반 예시를 각각 몇 개 넣을지")
    ap.add_argument("--replace", action="store_true", help="기존 목록을 지우고 새로 만듦 (기본: 기존 목록에 이어 붙임)")
    args = ap.parse_args()

    need("texts_all.csv", "rule_tree.csv", "gold_paths.csv", "rule_search.csv")
    texts = pd.read_csv(PROCESSED / "texts_all.csv", keep_default_na=False)
    texts["human_label"] = texts.human_label.astype(int)
    tree_df = pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False)
    gold = pd.read_csv(PROCESSED / "gold_paths.csv", keep_default_na=False)
    rs = pd.read_csv(PROCESSED / "rule_search.csv", keep_default_na=False)
    if "found_leaves_excl" not in rs.columns:
        raise SystemExit("rule_search.csv가 1주차 형식입니다 (규칙 제외 열 없음). 먼저 실행하세요: python scripts/rule_search.py")

    if args.ids_from:
        ids = pd.concat([pd.read_csv(f, usecols=["text_id"]).text_id for f in args.ids_from]).drop_duplicates()
        sel = texts[texts.text_id.isin(ids) & texts.source.isin(args.sources) & (texts.split == args.split)]
        how = f"{', '.join(args.ids_from)}의 text_id"
    else:
        sel = texts[texts.text_id.isin(sample_ids(texts, args.sources, args.split, args.n, args.stratified, args.seed))]
        how = f"새로 뽑음 (출처마다 {args.n}건{', 층화' if args.stratified else ''}, seed {args.seed})"
    sel = sel.merge(gold[["text_id", "gold_leaf"]], on="text_id").merge(
        rs[["text_id", "found_leaves", "found_leaves_excl"]], on="text_id")
    if sel.empty:
        raise SystemExit("고른 텍스트가 없습니다. --ids-from 파일, --sources, --split을 확인하세요.")

    trees = {s: Tree(tree_df, s) for s in args.sources}
    ex_j = pd.read_csv(PROCESSED / "examples_jigsaw.csv", keep_default_na=False) if "jigsaw" in args.sources else None
    if ex_j is not None:
        ex_j = ex_j[(ex_j.safe_for_eval.astype(str) == "True") & (ex_j.label_conflict.astype(str) == "False")]
        ex_j = ex_j.assign(_h=[hash_bucket(i, 10 ** 9) for i in ex_j.example_id]).sort_values("_h")
    pool = texts[(texts.source == "dna") & (texts.split == "dev")].merge(gold[["text_id", "gold_leaf"]], on="text_id")
    pool["qid"] = pool.text_id.map(qid_of)

    cmap, jobs = [], {}
    for r in sel.itertuples():
        tree = trees[r.source]
        conds = {"base": r.found_leaves.split(";")[:args.top] if r.found_leaves else [],
                 "excluded": r.found_leaves_excl.split(";")[:args.top] if r.found_leaves_excl else [],
                 "rule_only": [r.gold_leaf], "rule_example": [r.gold_leaf]}
        for cond, leaves in conds.items():
            withex = cond == "rule_example"
            for rank, leaf in enumerate(leaves, 1):
                cmap.append({"text_id": r.text_id, "condition": cond, "rank": rank, "rule_id": leaf,
                             "with_examples": withex})
                key = (r.text_id, leaf, withex)
                if key in jobs:
                    continue
                examples = []
                if withex:
                    if r.source == "dna":
                        examples = dna_examples(r.text_id, leaf, tree, pool, args.n_examples)
                    else:
                        examples = [(e.example_id, e.example_text, int(e.example_label))
                                    for lab in (1, 0)
                                    for e in ex_j[(ex_j.rule_id == leaf) & (ex_j.example_label.astype(int) == lab)]
                                    .head(args.n_examples).itertuples()]
                path = [tree.name[n] for n in tree.path(leaf)[:-1]]
                jobs[key] = {"text_id": r.text_id, "rule_id": leaf, "with_examples": withex, "source": r.source,
                             "split": r.split, "judge_mode": r.judge_mode,
                             "question": r.context if r.judge_mode == "response" else "",
                             "text": r.text, "human_label": r.human_label,
                             "rule_text": tree.text[leaf], "rule_path": json.dumps(path, ensure_ascii=False),
                             "examples": json.dumps([[t, lab] for _, t, lab in examples], ensure_ascii=False),
                             "example_ids": ";".join(i for i, _, _ in examples)}

    cmap = pd.DataFrame(cmap)
    jobs = pd.DataFrame(jobs.values())
    n_new = len(jobs)
    if not args.replace and (PROCESSED / "conditions_jobs.csv").exists():
        # dev 목록에 test를 더하는 경우처럼 이어 붙임. 같은 텍스트는 새 목록으로 바꿈
        old_m = pd.read_csv(PROCESSED / "conditions_map.csv", keep_default_na=False)
        old_j = pd.read_csv(PROCESSED / "conditions_jobs.csv", keep_default_na=False)
        old_m["with_examples"] = old_m.with_examples.astype(str) == "True"
        old_j["with_examples"] = old_j.with_examples.astype(str) == "True"
        cmap = pd.concat([old_m[~old_m.text_id.isin(cmap.text_id)], cmap], ignore_index=True)
        jobs = pd.concat([old_j[~old_j.text_id.isin(jobs.text_id)], jobs], ignore_index=True)
    cmap.to_csv(PROCESSED / "conditions_map.csv", index=False)
    jobs.to_csv(PROCESSED / "conditions_jobs.csv", index=False)

    print(f"텍스트: {how}")
    for src, g in sel.groupby("source"):
        print(f"  {src}: {len(g)}건 (위반 {int(g.human_label.sum())}건)")
    per = cmap.groupby("condition").size()
    print("조건별 (텍스트, 규칙) 판정 수:", ", ".join(f"{c} {per.get(c, 0)}" for c in ["base", "excluded", "rule_only", "rule_example"]))
    print(f"중복을 뺀 판정 수: 이번 {n_new:,}건, 목록 전체 {len(jobs):,}건 (변형마다 이만큼, 이미 한 판정은 건너뜀)")
    noex = jobs[jobs.with_examples & (jobs.examples == "[]")]
    if len(noex):
        print(f"  주의: 예시를 하나도 못 찾은 규칙 {noex.rule_id.nunique()}개 ({len(noex)}건) → 그 텍스트는 '규칙 문장만'과 같음")
    print(f"저장: {PROCESSED / 'conditions_map.csv'}, {PROCESSED / 'conditions_jobs.csv'}")
    print("다음: python scripts/spark_conditions.py --check  →  python scripts/spark_conditions.py")


if __name__ == "__main__":
    main()
