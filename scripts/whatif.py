"""수정 효과 재계산 (P4): 임계값·τ·우선순위를 바꾸면 어떤 판정이 뒤집히고 지표가 얼마나 바뀌나

'기본' 조건 판정을 원래 설정과 바꾼 설정으로 계산해 나란히 놓고, 뒤집힌 판정을 맞게 바뀐 것·틀리게 바뀐 것으로 나눔.
τ·우선순위를 바꾸면 찾은 규칙 상위 5개가 달라지므로 판정 캐시(judgments_cache.csv)에서 확률을 다시 모음.
캐시에 없는 (텍스트, 규칙)은 conditions_jobs_whatif.csv로 저장 → 판정한 뒤 다시 실행.

    python scripts/whatif.py --model team_v3 --source dna --thr 0.30           # 임계값만
    python scripts/whatif.py --model team_v3 --source dna --tau 0.12           # τ
    python scripts/whatif.py --model team_v3 --source dna --boost dna_L2_3=0.05 # 우선순위: 그 가지를 앞으로
    python scripts/spark_conditions.py --jobs data/processed/conditions_jobs_whatif.csv   # 빠진 판정 채우기

원래 설정: τ = rule_search.csv의 값, 임계값 = dev κ 최대(--thr-by f1 가능), 우선순위 = 유사도 그대로
출력: reports/whatif_flips.csv, reports/whatif_summary.md
"""
import argparse
import json

import numpy as np
import pandas as pd

from analysis import Tree, bfs, flips, recompute_base, thresholds_on_dev
from common import PROCESSED, REPORTS, binary_metrics, need
from data_overview import md_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="team_v3", help="shieldgemma 또는 team_v1~v3")
    ap.add_argument("--source", default="dna", choices=["dna", "jigsaw"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--thr", type=float, default=None, help="바꿀 판정 임계값 (없으면 원래 값)")
    ap.add_argument("--tau", type=float, default=None, help="바꿀 τ (없으면 원래 값)")
    ap.add_argument("--start-level", type=int, default=2, choices=[1, 2])
    ap.add_argument("--boost", nargs="*", default=[], help="노드=더할 값 (예: dna_L2_3=0.05, 음수면 뒤로)")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--thr-by", default="kappa", choices=["kappa", "f1"], help="원래 임계값 기준")
    args = ap.parse_args()

    need("judgments_conditions.csv", "judgments_cache.csv", "rule_search.csv", "similarity.parquet", "rule_tree.csv")
    j = pd.read_csv(PROCESSED / "judgments_conditions.csv")
    j = j[(j.source == args.source) & (j.model == args.model) & (j.condition == "base")].dropna(subset=["p_violation"])
    if j.empty:
        raise SystemExit(f"{args.source}의 {args.model} '기본' 조건 판정이 없습니다.")
    thr0 = thresholds_on_dev(j, criterion=args.thr_by)[(args.source, args.model, "base")]
    thr1 = thr0 if args.thr is None else args.thr
    j = j[j.split == args.split].set_index("text_id")
    y = j.human_label.astype(int)

    rs = pd.read_csv(PROCESSED / "rule_search.csv", keep_default_na=False).set_index("text_id")
    tau0 = float(rs.loc[rs.source == args.source, "tau"].iloc[0])
    tau1 = tau0 if args.tau is None else args.tau
    boost = {k: float(v) for k, v in (x.split("=") for x in args.boost)}
    variant = "team_v0" if args.model == "shieldgemma" else args.model

    p_new, missing = j.p_violation.copy(), pd.DataFrame(columns=["text_id", "rule_id"])
    if args.tau is not None or boost:
        tree = Tree(pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False), args.source)
        sims = pd.read_parquet(PROCESSED / "similarity.parquet").set_index("text_id").loc[j.index]
        cols = [c for c in sims.columns if c in tree.level]
        found = pd.Series({tid: bfs(tree, dict(zip(cols, row)), tau1, args.start_level, boost=boost)[0]
                           for tid, row in zip(sims.index, sims[cols].to_numpy())})
        cache = pd.read_csv(PROCESSED / "judgments_cache.csv")
        cache["with_examples"] = cache.with_examples.astype(str) == "True"
        rec, missing = recompute_base(cache, found, variant, args.top)
        p_new = rec.set_index("text_id").p_violation.reindex(j.index)
        changed_rules = int(sum(found[t][:args.top] != rs.loc[t, "found_leaves"].split(";")[:args.top] for t in j.index))
    else:
        changed_rules = 0

    ok = p_new.notna()
    f = flips(j.p_violation[ok], p_new[ok], y[ok], thr0, thr1)
    m0 = binary_metrics(j.p_violation[ok], y[ok], thr0)
    m1 = binary_metrics(p_new[ok], y[ok], thr1)
    REPORTS.mkdir(parents=True, exist_ok=True)

    sub = j[ok].assign(p_after=p_new[ok], pred_before=(j.p_violation[ok] >= thr0).astype(int),
                       pred_after=(p_new[ok] >= thr1).astype(int))
    sub["flip"] = np.select([f["mask_fixed"], f["mask_broken"]], ["맞게 바뀜", "틀리게 바뀜"], "")
    sub = sub[sub.flip != ""].reset_index()[["text_id", "human_label", "p_violation", "p_after", "pred_before",
                                             "pred_after", "flip"]].rename(columns={"p_violation": "p_before"})
    sub.to_csv(REPORTS / "whatif_flips.csv", index=False)

    keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "kappa_human", "fn", "fp"]
    tab = pd.DataFrame({"지표": keys, "원래": [m0[k] for k in keys], "바꾼 설정": [m1[k] for k in keys]})
    tab["변화"] = tab["바꾼 설정"] - tab["원래"]
    setting = pd.DataFrame({"설정": ["판정 임계값", "τ", "우선순위(boost)", "상위 5개가 바뀐 텍스트"],
                            "원래": [round(thr0, 4), round(tau0, 4), "없음", "-"],
                            "바꾼 설정": [round(thr1, 4), round(tau1, 4), json.dumps(boost) if boost else "없음",
                                       changed_rules]})
    out = [f"# 수정 효과 ({args.source}, {args.model}, '기본' 조건, {args.split} {int(ok.sum())}건)", "",
           md_table(setting), "", md_table(tab.round(4)), "",
           f"뒤집힌 판정 {f['flipped']}건: 맞게 바뀜 {f['fixed']}건, 틀리게 바뀜 {f['broken']}건 → `reports/whatif_flips.csv`"]
    if len(missing):
        mp = missing.drop_duplicates()
        jobs = pd.read_csv(PROCESSED / "conditions_jobs.csv", keep_default_na=False)
        tree = Tree(pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False), args.source)
        tinfo = jobs.drop_duplicates("text_id").set_index("text_id")
        new = tinfo.loc[mp.text_id, ["source", "split", "judge_mode", "question", "text", "human_label"]].reset_index()
        new["rule_id"] = mp.rule_id.values
        new["with_examples"] = False
        new["rule_text"] = new.rule_id.map(tree.text)
        new["rule_path"] = [json.dumps([tree.name[n] for n in tree.path(r)[:-1]], ensure_ascii=False) for r in new.rule_id]
        new["examples"], new["example_ids"] = "[]", ""
        new.to_csv(PROCESSED / "conditions_jobs_whatif.csv", index=False)
        out += ["", f"캐시에 없는 판정 {len(mp)}건 때문에 텍스트 {int((~ok).sum())}건은 계산에서 뺐습니다. 채우려면:",
                "`python scripts/spark_conditions.py --jobs data/processed/conditions_jobs_whatif.csv` 후 다시 실행"]
    (REPORTS / "whatif_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
