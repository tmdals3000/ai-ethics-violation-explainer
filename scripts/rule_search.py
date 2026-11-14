"""BFS 규칙 찾기 → rule_search.csv (+ reports/rule_search_summary.md, tau_sweep.csv)

PDF 2주차: "저장된 유사도에서 임계값 τ 이상인 중분류만 남기고 소분류로 내려갑니다." (BFS 함수는 analysis.bfs)
  1) 중분류를 모두 비교 → 유사도 ≥ τ인 것만 남김 (하나도 없으면 가장 높은 하나만 남기고 fallback 표시)
     --start-level 1이면 1주차처럼 대분류부터 거름
  2) 남은 노드의 자식들을 비교 → 같은 방식으로 남김 … 잎(규칙)에 닿을 때까지
  3) 닿은 잎들을 유사도 순으로 정렬한 것이 '찾은 규칙'

τ는 출처마다 dev에서 정함. τ를 낮출수록(가지를 덜 자를수록) 정확도는 항상 오르고 효율은 나빠지므로
(τ = 0이면 모든 노드를 비교 = BFS를 안 쓰는 것), 기본 기준은
"top-3 정확도를 dev 최고치에서 1%p 이내로 지키면서 비교하는 노드가 가장 적은 τ" (--max-top3-loss로 조절).
DNA와 Jigsaw는 텍스트 길이와 규칙 문장 성격이 달라서 유사도 크기가 달라 τ를 따로 둠.

rule_search.csv 열 (텍스트 1행)
  found_leaves      찾은 규칙 (유사도 순, ';'로 구분)        top1 / top1_correct / top3_correct
  found_l1, found_l2, found_l3   1위 규칙의 경로
  l1_match, l2_match, l3_match   단계별로 정답 경로와 같은지   ← 분석용 변수 "찾은 경로와 정답 경로 일치 여부"
  first_divergence  1위 규칙의 경로가 정답과 처음 갈라진 단계 (L1/L2/L3, 같으면 '일치')
  gold_lost_at      BFS가 정답 경로를 잘라 낸 단계 (정답이 후보에 아예 없게 된 이유, 끝까지 남았으면 '남음')
  gold_rank         정답 규칙이 찾은 규칙 중 몇 위인지 (없으면 빈칸)
  l2_rank_of_gold   찾은 가지(중분류)들을 가장 높은 잎 유사도 순으로 줄 세웠을 때 정답 가지의 순위
                    ← 분석용 변수 "가지별 우선순위" (P4에서 우선순위를 바꾸는 실험의 기준점)
  n_compared, efficiency   비교한 노드 수, 그 출처 트리의 전체 노드 수 대비 비율 (PDF 규칙 찾기 효율)
  boundary          τ를 ±0.02 바꿨을 때 1위 규칙이 바뀌는지 (P1 경계 사례)
  fallback          어느 단계에서 τ를 넘는 노드가 없어 1위만 남겼는지
  excl_unit         '규칙 제외' 실험 단위 = 정답 규칙의 중분류 (DNA 위해 유형 12개, Jigsaw 규칙 2개)
  found_leaves_excl 그 단위를 트리에서 빼고 같은 τ로 찾은 규칙 → '규칙 제외' 조건의 후보

    python scripts/rule_search.py                  # 출처별 τ를 dev에서 자동 선택
    python scripts/rule_search.py --tau dna=0.35 jigsaw=0.30
"""
import argparse

import numpy as np
import pandas as pd

from analysis import Tree, bfs
from common import PROCESSED, need, REPORTS
from data_overview import md_table

DELTA = 0.02


def search_source(tree: Tree, sims: pd.DataFrame, gold: pd.DataFrame, tau: float, start: int = 2,
                  with_excl: bool = True) -> pd.DataFrame:
    rows = []
    node_cols = [c for c in sims.columns if c in tree.level]
    for rec, g in zip(sims[["text_id"] + node_cols].itertuples(index=False), gold.itertuples(index=False)):
        sim = dict(zip(node_cols, rec[1:]))
        leaves, n_cmp, kept, fb = bfs(tree, sim, tau, start)
        gold_path = [x for x in (g.gold_l1, g.gold_l2, g.gold_l3) if x]
        top1 = leaves[0]
        fpath = tree.path(top1)
        match = [fpath[i] == gold_path[i] if i < len(gold_path) and i < len(fpath) else None for i in range(3)]
        div = next((f"L{i + 1}" for i in range(len(gold_path)) if not match[i]), "일치")
        lost = next((f"L{lv}" for lv, node in zip(range(1, 4), gold_path)
                     if lv in kept and node not in kept[lv]), "남음")      # 시작 단계 위는 비교 안 함
        # 가지별 우선순위: 찾은 잎을 중분류(L2) 가지로 묶고 가지의 대표 유사도 = 그 가지 잎의 최댓값
        branch = {}
        for leaf in leaves:
            b = tree.path(leaf)[1] if len(tree.path(leaf)) > 2 else leaf
            branch.setdefault(b, sim[leaf])
        border = [b for b in branch]                     # leaves가 이미 유사도 순이라 처음 나온 순서 = 가지 순위
        gold_branch = gold_path[1] if len(gold_path) > 2 else gold_path[-1]
        lo, _, _, _ = bfs(tree, sim, tau - DELTA, start)
        hi, _, _, _ = bfs(tree, sim, tau + DELTA, start)
        unit = tree.unit_of(g.gold_leaf)                 # '규칙 제외' 실험 단위 (정답의 중분류)
        excl = bfs(tree, sim, tau, start, exclude={unit})[0] if with_excl else []
        rows.append({
            "text_id": rec[0], "tau": tau, "found_leaves": ";".join(leaves), "n_found": len(leaves),
            "top1": top1, "top1_sim": round(float(sim[top1]), 4),
            "gold_leaf": g.gold_leaf, "gold_sim": round(float(sim[g.gold_leaf]), 4),
            "top1_correct": top1 == g.gold_leaf, "top3_correct": g.gold_leaf in leaves[:3],
            "gold_rank": leaves.index(g.gold_leaf) + 1 if g.gold_leaf in leaves else np.nan,
            "found_l1": fpath[0], "found_l2": fpath[1] if len(fpath) > 1 else "",
            "found_l3": fpath[2] if len(fpath) > 2 else "",
            "l1_match": match[0], "l2_match": match[1], "l3_match": match[2],
            "first_divergence": div, "gold_lost_at": lost,
            "n_branches": len(border),
            "l2_rank_of_gold": border.index(gold_branch) + 1 if gold_branch in border else np.nan,
            "l2_top3_correct": gold_branch in border[:3],
            "n_compared": n_cmp, "efficiency": round(n_cmp / tree.n_nodes, 4),
            "boundary": (lo[0] != top1) or (hi[0] != top1), "fallback": fb,
            "excl_unit": unit, "found_leaves_excl": ";".join(excl),
        })
    return pd.DataFrame(rows)


def sweep(tree, sims, gold, taus, start):
    rows = []
    for tau in taus:
        r = search_source(tree, sims, gold, tau, start, with_excl=False)
        rows.append({"tau": round(float(tau), 4), "top1": r.top1_correct.mean(), "top3": r.top3_correct.mean(),
                     "l2_top1": (r.l2_rank_of_gold == 1).mean(), "efficiency": r.efficiency.mean(),
                     "fallback": r.fallback.mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", nargs="*", default=[], help="출처=값 (예: dna=0.35). 없으면 dev에서 자동 선택")
    ap.add_argument("--max-top3-loss", type=float, default=0.01,
                    help="자동 선택 때 허용하는 top-3 정확도 손실 (기본 0.01 = 1%p)")
    ap.add_argument("--start-level", type=int, default=2, choices=[1, 2],
                    help="2 = 중분류부터 거름 (PDF 2주차 정의, 기본), 1 = 대분류부터 (1주차 구현)")
    args = ap.parse_args()
    fixed = {k: float(v) for k, v in (x.split("=") for x in args.tau)}

    need("rule_tree.csv", "similarity.parquet", "gold_paths.csv", "texts_all.csv")
    tree_df = pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False)
    sims = pd.read_parquet(PROCESSED / "similarity.parquet")
    gold = pd.read_csv(PROCESSED / "gold_paths.csv", keep_default_na=False)
    texts = pd.read_csv(PROCESSED / "texts_all.csv", keep_default_na=False, usecols=["text_id", "source", "split"])
    df = sims.merge(gold, on="text_id").merge(texts, on="text_id")
    REPORTS.mkdir(parents=True, exist_ok=True)

    results, sweeps, out = [], [], ["# 규칙 찾기(BFS) 결과", ""]
    for src in sorted(df.source.unique()):
        tree = Tree(tree_df, src)
        d = df[df.source == src].reset_index(drop=True)
        cols = [c for c in d.columns if c in tree.level]
        dev = d[d.split == "dev"]
        vals = dev[cols].to_numpy().ravel()
        taus = np.unique(np.quantile(vals[~np.isnan(vals)], np.linspace(0.05, 0.95, 19)).round(3))
        sw = sweep(tree, dev[["text_id"] + cols], dev[["gold_l1", "gold_l2", "gold_l3", "gold_leaf"]], taus,
                   args.start_level)
        sw.insert(0, "source", src)
        sweeps.append(sw)
        if src in fixed:
            tau, how = fixed[src], "직접 지정"
        else:
            # 정확도만 보면 τ를 낮출수록(가지를 덜 자를수록) 항상 유리 → 효율과 균형을 잡아야 함
            ok = sw[sw.top3 >= sw.top3.max() - args.max_top3_loss]
            best = ok.sort_values(["efficiency", "top3"], ascending=[True, False]).iloc[0]
            tau = float(best.tau)
            how = (f"dev에서 top-3를 최고치({sw.top3.max():.3f})의 {args.max_top3_loss * 100:.0f}%p 이내로 "
                   f"지키면서 비교 노드가 가장 적은 값")
        r = search_source(tree, d[["text_id"] + cols], d[["gold_l1", "gold_l2", "gold_l3", "gold_leaf"]], tau,
                          args.start_level)
        r.insert(1, "source", src)
        r.insert(2, "split", d.split.values)
        results.append(r)

        out += [f"## {src} (τ = {tau:.3f}, {how}, 트리 노드 {tree.n_nodes}개, L{args.start_level}부터 비교)", ""]
        summ = r.groupby("split").agg(건수=("text_id", "count"), top1=("top1_correct", "mean"),
                                      top3=("top3_correct", "mean"), 효율=("efficiency", "mean"),
                                      경계사례=("boundary", "sum"), fallback=("fallback", "mean"))
        if src == "dna":
            summ["중분류 top1"] = r.groupby("split").l2_rank_of_gold.apply(lambda s: (s == 1).mean())
            summ["중분류 top3"] = r.groupby("split").l2_top3_correct.mean()
        out += [md_table(summ.round(3).reset_index()), ""]
        dv = r[r.split == "dev"]
        if len(dv):
            for col, title in [("first_divergence", "dev에서 1위 규칙이 정답과 처음 갈라진 단계"),
                               ("gold_lost_at", "dev에서 BFS가 정답 경로를 잘라 낸 단계")]:
                vc = dv[col].value_counts().rename_axis("단계").reset_index(name="건수")
                out += [f"**{title}**", "", md_table(vc), ""]
        out += ["τ 후보별 dev 결과 (`reports/tau_sweep.csv`)", "", md_table(sw.round(3)), ""]

    res = pd.concat(results, ignore_index=True)
    res.to_csv(PROCESSED / "rule_search.csv", index=False)
    pd.concat(sweeps).round(4).to_csv(REPORTS / "tau_sweep.csv", index=False)
    (REPORTS / "rule_search_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\n저장: {PROCESSED / 'rule_search.csv'} ({len(res):,}행), {REPORTS / 'rule_search_summary.md'}")


if __name__ == "__main__":
    main()
