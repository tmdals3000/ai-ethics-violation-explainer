"""
판정 모델이 Do-Not-Answer 에서 어떻게 판정하는지 (GPU 필요 없음)

질문: 판정 모델이 '거절문'을 위반으로 잘못 잡는가? 어떤 응답 유형·응답 LLM·길이에서 틀리는가?
실행: python dna_judge_analysis.py        (analyze.py 를 먼저 실행해 둘 것)
입력: texts.csv, data_en.csv, results/judgments_all.csv, results/thresholds.csv
출력: results/dna_judge/*.csv (원문 없음)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

os.makedirs("results/dna_judge", exist_ok=True)
ACTION = {0: "0 거절", 1: "1 전제 반박", 2: "2 양쪽 관점", 3: "3 면책 후 답변", 4: "4 모름·불확실", 5: "5 그대로 따름"}
REFUSAL_LIKE = {"0 거절", "1 전제 반박", "4 모름·불확실"}     # 내용을 제공하지 않는 응답

t = pd.read_csv("texts.csv", keep_default_na=False)
d = t[(t.source == "do_not_answer") & (t["sample"] == 1)].copy()
raw = pd.read_csv("data_en.csv")
llms = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]
amap = {f"dna-{i}-{m}": a for m in llms for i, a in zip(raw["id"], raw[f"{m}_action"])}
d["action"] = d.text_id.map(amap).map(ACTION)
d["n_words"] = d.text.astype(str).str.split().str.len()
d["length_bin"] = pd.qcut(d.n_words, 4, labels=["짧음(1/4)", "2/4", "3/4", "김(4/4)"])

j = pd.read_csv("results/judgments_all.csv")
j = j[j.text_id.isin(d.text_id)]
th = pd.read_csv("results/thresholds.csv")
th = th[th.source == "do_not_answer"].set_index("judge")["threshold"].to_dict()
wide = j.pivot_table(index="text_id", columns="judge_cond", values="p_violation")
judges = list(wide.columns)
d = d.merge(wide, left_on="text_id", right_index=True)
print(f"DNA dev {len(d)}개, 위반 {int(d.human_harmful.sum())}개 · 판정 모델 {len(judges)}개")

# ---------------------------------------------------------------
# 1) 응답 유형별: 잘못 잡은 위반(FP) 비율, 놓친 위반(FN)
# ---------------------------------------------------------------
rows = []
for jc in judges:
    pred = (d[jc] >= th.get(jc, 0.5)).astype(int)
    for a, g in d.assign(pred=pred).groupby("action"):
        neg, pos = g[g.human_harmful == 0], g[g.human_harmful == 1]
        rows.append({"judge": jc, "action": a, "n": len(g), "n_human_violation": len(pos),
                     "pred_violation_rate": g.pred.mean(),
                     "false_positive_rate": neg.pred.mean() if len(neg) else np.nan, "n_false_positive": int(neg.pred.sum()),
                     "n_false_negative": int((pos.pred == 0).sum()), "mean_p_violation": g[jc].mean()})
by_action = pd.DataFrame(rows).round(4)
by_action.to_csv("results/dna_judge/by_action.csv", index=False)

# 요약: 잘못 잡은 위반 중 '내용을 안 준 응답(거절·반박·모름)'의 비중
summ = []
for jc in judges:
    x = by_action[by_action.judge == jc]
    fp_all = x.n_false_positive.sum()
    fp_ref = x[x.action.isin(REFUSAL_LIKE)].n_false_positive.sum()
    share_ref = d.action.isin(REFUSAL_LIKE).mean()
    summ.append({"judge": jc, "false_positives": int(fp_all), "fp_from_refusal_like": int(fp_ref),
                 "fp_share_refusal_like": fp_ref / fp_all if fp_all else np.nan,
                 "refusal_like_share_of_data": share_ref,
                 "mean_p_refusal_like": d.loc[d.action.isin(REFUSAL_LIKE), jc].mean(),
                 "mean_p_followed(5)": d.loc[d.action == "5 그대로 따름", jc].mean()})
summ = pd.DataFrame(summ).round(3)
summ.to_csv("results/dna_judge/refusal_summary.csv", index=False)
print("\n① 잘못 잡은 위반(FP) 중 거절·반박·모름 응답의 비중")
print(summ.to_string(index=False))

print("\n② 응답 유형별 위반 판정 비율 (사람 라벨 위반 비율과 비교)")
human = d.groupby("action").human_harmful.mean().rename("사람 라벨")
pv = by_action.pivot(index="action", columns="judge", values="pred_violation_rate")
print(pd.concat([d.groupby("action").size().rename("n"), human, pv], axis=1).round(3).to_string())

# ---------------------------------------------------------------
# 2) 응답 유형 안에서도 구분이 되는가? (유형별 AUC, 위반이 있는 유형만)
# ---------------------------------------------------------------
rows = []
for a, g in d.groupby("action"):
    if g.human_harmful.nunique() < 2:
        continue
    for jc in judges:
        rows.append({"action": a, "judge": jc, "n": len(g), "n_pos": int(g.human_harmful.sum()),
                     "roc_auc": roc_auc_score(g.human_harmful, g[jc])})
auc_a = pd.DataFrame(rows).round(3)
auc_a.to_csv("results/dna_judge/auc_by_action.csv", index=False)

# ---------------------------------------------------------------
# 3) 응답 LLM별, 응답 길이별 오판
# ---------------------------------------------------------------
for col, name in [("response_llm", "by_llm"), ("length_bin", "by_length")]:
    rows = []
    for jc in judges:
        pred = (d[jc] >= th.get(jc, 0.5)).astype(int)
        for k, g in d.assign(pred=pred).groupby(col, observed=True):
            neg = g[g.human_harmful == 0]
            rows.append({"judge": jc, col: k, "n": len(g), "human_violation_rate": g.human_harmful.mean(),
                         "pred_violation_rate": g.pred.mean(),
                         "false_positive_rate": neg.pred.mean() if len(neg) else np.nan,
                         "error_rate": (g.pred != g.human_harmful).mean()})
    pd.DataFrame(rows).round(4).to_csv(f"results/dna_judge/{name}.csv", index=False)

# ---------------------------------------------------------------
# 4) 팀 판정기(거절 인정 원칙)가 ShieldGemma 대비 거절문에서 얼마나 덜 잡는가
# ---------------------------------------------------------------
sg = [c for c in judges if c.startswith("ShieldGemma") and c.endswith("rule_only")]
tm = [c for c in judges if c.startswith("Team") and c.endswith("rule_only")]
if sg and tm:
    a, b = sg[0], tm[0]
    cmp = d.groupby("action").agg(n=("text_id", "size"), sg_mean_p=(a, "mean"), team_mean_p=(b, "mean")).reset_index()
    cmp["change"] = cmp.team_mean_p - cmp.sg_mean_p
    cmp.round(4).to_csv("results/dna_judge/team_vs_shieldgemma_by_action.csv", index=False)
    print(f"\n③ 같은 '규칙 문장만' 조건에서 평균 위반 확률: {a} → {b}")
    print(cmp.round(3).to_string(index=False))

# ---------------------------------------------------------------
# 5) 직접 읽어 볼 사례 목록 (text_id 만 저장. 원문은 화면 P3 탭에서 '원문 보기'로)
#    - 거절인데 모든 모델이 위반으로 잡은 것 / 사람은 위반인데 모든 모델이 놓친 것
# ---------------------------------------------------------------
P = pd.DataFrame({jc: (d[jc] >= th.get(jc, 0.5)).astype(int) for jc in judges}, index=d.index)
all_pos, all_neg = P.all(axis=1), (P == 0).all(axis=1)
cases = pd.concat([
    d[d.action.isin(REFUSAL_LIKE) & (d.human_harmful == 0) & all_pos].assign(case="거절류인데 모든 모델이 위반"),
    d[(d.human_harmful == 1) & all_neg].assign(case="사람은 위반인데 모든 모델이 놓침"),
    d[(d.human_harmful == 1) & d.action.isin(REFUSAL_LIKE)].assign(case="사람 라벨 위반인데 응답 유형은 거절류"),
])[["case", "text_id", "response_llm", "action", "rule_l2", "human_harmful"] + judges]
cases.round(4).to_csv("results/dna_judge/cases_to_read.csv", index=False)
print("\n④ 직접 읽어 볼 사례:", cases.case.value_counts().to_dict())
print("\n완료! results/dna_judge 폴더를 확인하세요.")
