"""
4단계: 분석 (P1 규칙 찾기 · P2 판정 비교 · P3 원인 진단 · P4 수정 효과) — 출처 3종

실행: python analyze.py
입력: texts.csv, rules.csv, similarity.csv, coords.csv          (1·2단계)
      judgments.csv                                             (1차 Colab: DNA × 기준 모델 3개)
      judgments_new.csv, base_candidates.csv, team_prompt_comparison.csv  (2차 Colab, 없으면 건너뜀)
출력: results/ 폴더 (화면 app.py 가 읽음), results/judgments_all.csv (판정 결과 전체)

※ 모든 분석은 dev 에서만 (PDF 원칙). test 는 보지 않음
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score
from retrieval import Tree, TAU, TOP_K, gold_leaf
from diagnose import diagnose, CAUSES, takes_rules, is_base

TAU_GRID = [0.10, 0.15, 0.20, 0.25, 0.30]
DELTA = 0.02
MIN_N = 10
SRC_NAME = {"do_not_answer": "Do-Not-Answer", "jigsaw": "Jigsaw", "prosocial": "ProsocialDialog"}
os.makedirs("results", exist_ok=True)

texts = pd.read_csv("texts.csv", keep_default_na=False)
df = texts[texts["sample"] == 1].reset_index(drop=True)
rules = pd.read_csv("rules.csv")
tree = Tree(rules)
sim = pd.read_csv("similarity.csv").set_index("text_id").loc[df["text_id"]]
df["gold"] = df.apply(gold_leaf, axis=1)
print("분석 대상:", df.groupby("source").size().to_dict())

# ---------------------------------------------------------------
# 1) P1 규칙 찾기 (정답 규칙이 있는 DNA·Jigsaw)
# ---------------------------------------------------------------
def run_p1(tau):
    out = []
    for i, r in df.iterrows():
        if not r["gold"]:
            continue
        ranked, kept, eff = tree.bfs(sim.iloc[i], r["source"], tau)
        top1 = ranked[0]
        if top1 == r["gold"]:
            diverge = "일치"
        elif r["source"] == "jigsaw":
            diverge = "규칙"
        elif r["gold_l2"] not in kept:
            diverge = "대" if tree.parent[tree.parent[top1]] != r["gold_l1"] else "중"
        else:
            diverge = "소"
        out.append({"text_id": r["text_id"], "source": r["source"], "tau": tau, "top1": top1,
                    "candidates": "|".join(ranked[:TOP_K]), "kept_l2": "|".join(kept),
                    "top1_correct": int(top1 == r["gold"]), "top3_correct": int(r["gold"] in ranked[:3]),
                    "top5_correct": int(r["gold"] in ranked[:TOP_K]), "efficiency": eff, "diverge_level": diverge})
    return pd.DataFrame(out)


print("P1: τ별 BFS 계산 중...")
sweep = {t: run_p1(t) for t in sorted(set(TAU_GRID + [TAU, TAU - DELTA, TAU + DELTA]))}
p1 = sweep[TAU]
p1.to_csv("results/p1_retrieval.csv", index=False)
pd.DataFrame([{"source": s, "tau": t, "n": len(g), "top1": g.top1_correct.mean(), "top3": g.top3_correct.mean(),
               "efficiency": g.efficiency.mean()}
              for t, d in sweep.items() if t in TAU_GRID for s, g in d.groupby("source")]) \
    .round(3).to_csv("results/p1_tau_sweep.csv", index=False)
lo, hi = sweep[TAU - DELTA], sweep[TAU + DELTA]
b = p1[(p1.top1.values != lo.top1.values) | (p1.top1.values != hi.top1.values)].copy()
b["top1_lower_tau"], b["top1_higher_tau"] = lo.loc[b.index, "top1"].values, hi.loc[b.index, "top1"].values
b.to_csv("results/p1_boundary_cases.csv", index=False)
for s, g in p1.groupby("source"):
    print(f"  {SRC_NAME[s]} (τ={TAU}): top-1 {g.top1_correct.mean():.3f}, top-3 {g.top3_correct.mean():.3f}, "
          f"효율 {g.efficiency.mean():.3f}, 경계 사례 {int(b.source.eq(s).sum())}개")

# ---------------------------------------------------------------
# 2) 판정 결과 모으기 (1차 + 2차)
# ---------------------------------------------------------------
parts = [pd.read_csv("judgments.csv")]
if os.path.exists("judgments_new.csv"):
    parts.append(pd.read_csv("judgments_new.csv"))
j = pd.concat(parts, ignore_index=True)
j = j[j["text_id"].isin(df["text_id"])].drop_duplicates(["text_id", "judge", "condition"], keep="last")
j["judge_cond"] = j["judge"] + " / " + j["condition"]
j.to_csv("results/judgments_all.csv", index=False)
wide = j.pivot_table(index="text_id", columns="judge_cond", values="p_violation")
print("\n판정 모델별 판정 수:", j.groupby("judge_cond").size().to_dict())

# 임계값: 기준 모델 0.5, 팀 판정기는 dev F1 최대 (가정 A-threshold)
#   단, 표본이 작은 출처(200개 미만: Jigsaw·ProsocialDialog 50개)는 F1 최대값이 '전부 위반'으로 쏠려서
#   믿기 어려움 → 0.5 그대로 둠
MIN_N_TUNE = 200
grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
th_rows = []
for jc in wide.columns:
    for s, g in df.groupby("source"):
        x = g[["text_id", "human_harmful"]].merge(wide[[jc]].dropna(), left_on="text_id", right_index=True)
        if len(x) == 0:
            continue
        th = 0.5
        if jc.startswith("Team") and x.human_harmful.nunique() > 1 and len(x) >= MIN_N_TUNE:
            f1s = [f1_score(x.human_harmful, (x[jc] >= t).astype(int), zero_division=0) for t in grid]
            th = float(grid[int(np.argmax(f1s))])
        th_rows.append({"judge": jc, "source": s, "threshold": th})
thr = pd.DataFrame(th_rows)
thr.to_csv("results/thresholds.csv", index=False)
TH = {(r.judge, r.source): r.threshold for r in thr.itertuples()}

# ---------------------------------------------------------------
# 3) P2 판정 비교 (출처별)
# ---------------------------------------------------------------
def metrics(y, p, th):
    pred = (p >= th).astype(int)
    two = len(set(y)) > 1
    return {"roc_auc": roc_auc_score(y, p) if two else np.nan, "pr_auc": average_precision_score(y, p) if two else np.nan,
            "f1": f1_score(y, pred, zero_division=0), "kappa_human": cohen_kappa_score(y, pred) if two else np.nan,
            "error_rate": (pred != y).mean(), "n_pred_pos": int(pred.sum())}


rows, pairs, long_err = [], [], []
for s, g in df.groupby("source"):
    m = g[["text_id", "human_harmful", "rule_l1", "rule_l2", "response_llm", "source"]].merge(
        wide, left_on="text_id", right_index=True)
    js = [c for c in wide.columns if m[c].notna().all()]
    y = m["human_harmful"].values
    for jc in js:
        rows.append({"source": s, "judge": jc, "n": len(m), "n_pos": int(y.sum()), "threshold": TH[(jc, s)],
                     **metrics(y, m[jc].values, TH[(jc, s)])})
        e = m[["text_id", "source", "rule_l1", "rule_l2", "response_llm", "human_harmful"]].copy()
        e["judge"], e["p_violation"] = jc, m[jc].values
        e["pred"] = (e.p_violation >= TH[(jc, s)]).astype(int)
        e["error"] = (e.pred != e.human_harmful).astype(int)
        long_err.append(e)
    for a_i, a in enumerate(js):
        for bj in js[a_i + 1:]:
            pairs.append({"source": s, "judge_a": a, "judge_b": bj,
                          "kappa": cohen_kappa_score((m[a] >= TH[(a, s)]).astype(int), (m[bj] >= TH[(bj, s)]).astype(int))})
p2 = pd.DataFrame(rows).round(4)
p2.to_csv("results/p2_metrics.csv", index=False)
pd.DataFrame(pairs).round(4).to_csv("results/p2_pairwise_kappa.csv", index=False)
long_err = pd.concat(long_err)
long_err.to_csv("results/p2_judgments_long.csv", index=False)
for gcol in ["rule_l1", "rule_l2", "response_llm", "source"]:
    t = long_err.groupby(["judge", gcol]).agg(n=("error", "size"), n_error=("error", "sum"),
                                              error_rate=("error", "mean")).reset_index()
    t["low_n"] = t.n < MIN_N
    t.round(4).to_csv(f"results/p2_errors_by_{gcol}.csv", index=False)
print("\nP2 판정 비교 (출처별):")
print(p2[["source", "judge", "n", "n_pos", "threshold", "roc_auc", "pr_auc", "f1", "kappa_human"]].to_string(index=False))

# ---------------------------------------------------------------
# 4) P3 원인 진단 (정답 규칙이 있는 DNA·Jigsaw)
# ---------------------------------------------------------------
base_argmax = {}
if os.path.exists("base_candidates.csv"):
    bc = pd.read_csv("base_candidates.csv")
    base_argmax = bc.loc[bc.groupby("text_id").p_violation.idxmax()].set_index("text_id")["cand"].to_dict()
gold_in_topk = dict(zip(p1.text_id, p1.top5_correct.astype(bool)))
gold_rule = dict(zip(df.text_id, df.gold))
coords = pd.read_csv("coords.csv")
diag = []
for s, g in df[df.gold != ""].groupby("source"):
    m = g[["text_id", "human_harmful"]].merge(wide, left_on="text_id", right_index=True)
    js = [c for c in wide.columns if m[c].notna().all()]
    d = diagnose(m, js, {jc: TH[(jc, s)] for jc in js}, gold_in_topk, base_argmax, gold_rule)
    diag.append(d.assign(source=s))
diag = pd.concat(diag).merge(coords, on="text_id", how="left")
diag.to_csv("results/p3_diagnosis.csv", index=False)
summ = diag.pivot_table(index=["source", "judge"], columns="cause", values="text_id", aggfunc="count", fill_value=0)
summ = summ.reindex(columns=CAUSES, fill_value=0)
summ["합계"] = summ.sum(axis=1)
summ.to_csv("results/p3_cause_summary.csv")
print("\nP3 원인별 잘못된 판정 수:")
print(summ.to_string())

# ---------------------------------------------------------------
# 5) P4 수정 효과 (데모 범위)
#    (a) 규칙 찾기 → 정답 규칙: '기본'(BFS 후보) 대비 '규칙 문장만'(정답 규칙)의 AUC 차이
#    (b) 팀 판정기 프롬프트(원칙 문장) 수정 전후 — Colab 의 비교표
#    (c) 임계값 0.5 → dev 최적으로 바꿨을 때 뒤집힌 판정 수
# ---------------------------------------------------------------
eff = []
for s in p2.source.unique():
    t = p2[p2.source == s].set_index("judge")
    for base_j in [x for x in t.index if is_base(x)]:
        gold_j = base_j.replace("/ base", "/ rule_only")
        if gold_j in t.index:
            eff.append({"source": s, "model": base_j.split(" / ")[0], "auc_base": t.loc[base_j, "roc_auc"],
                        "auc_rule_only": t.loc[gold_j, "roc_auc"],
                        "auc_change": t.loc[gold_j, "roc_auc"] - t.loc[base_j, "roc_auc"],
                        "prauc_base": t.loc[base_j, "pr_auc"], "prauc_rule_only": t.loc[gold_j, "pr_auc"]})
pd.DataFrame(eff).round(4).to_csv("results/p4_base_vs_rule_only.csv", index=False)
if os.path.exists("team_prompt_comparison.csv"):
    pd.read_csv("team_prompt_comparison.csv").to_csv("results/p4_team_prompt_comparison.csv", index=False)
flips = []
for (jc, s), th in TH.items():
    x = long_err[(long_err.judge == jc) & (long_err.source == s)]
    for new_th in [0.3, 0.5, 0.7, th]:
        a, b2 = (x.p_violation >= 0.5).astype(int), (x.p_violation >= new_th).astype(int)
        f = a != b2
        flips.append({"judge": jc, "source": s, "new_threshold": new_th, "flipped": int(f.sum()),
                      "to_correct": int((f & (b2 == x.human_harmful)).sum()),
                      "to_wrong": int((f & (b2 != x.human_harmful)).sum())})
pd.DataFrame(flips).drop_duplicates().to_csv("results/p4_threshold_flips.csv", index=False)
if eff:
    print("\nP4 '기본'(BFS 후보) → '규칙 문장만'(정답 규칙) AUC 변화:")
    print(pd.DataFrame(eff).round(3).to_string(index=False))
print("\n완료! results 폴더를 확인하세요.")
