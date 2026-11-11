"""DNA 문제 확인 (dev 만). 결과는 results/dna_check/ 에 저장한다.

  python scripts/dna_check.py agg        # 저장된 판정으로 후보 집계 방식 비교 (새 판정 없음)
  python scripts/dna_check.py prompts    # 팀 프롬프트 v1·v2·v3 를 DNA dev rule_only 로 비교 (GPU)

응답 유형(action)은 DNA 원본의 *_action 열 (0 거절 · 1 반박 · 2 양쪽 관점 · 3 주의 문구 · 4 모름 · 5 지시 따름).
팀 판정기 임계값은 dev κ(사람 라벨) 최대로 고르고, F1 은 보고 지표로 함께 적는다 (ASSUMPTIONS A-threshold).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, roc_auc_score  # noqa: E402

from src import data, judges  # noqa: E402
from src.common import load_config  # noqa: E402
from src.retrieval import targets  # noqa: E402

LLMS = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]


def dna_dev(cfg):
    texts, nodes, ex, sample = data.load(cfg)
    tgt = targets(cfg, texts, sample)
    return tgt[tgt.source == "do_not_answer"], nodes, ex


def actions(cfg):
    raw = pd.read_csv(cfg["_raw"] / "dna/datasets/data_en.csv")
    return {f"dna-{r.id}-{l}": int(r[f"{l}_action"]) for _, r in raw.iterrows() for l in LLMS}


def metrics(y, p, th):
    pr = (p >= th).astype(int)
    return dict(n=len(y), n_pos=int(y.sum()), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                threshold=th, f1=f1_score(y, pr, zero_division=0), kappa=cohen_kappa_score(y, pr), pred_pos=int(pr.sum()))


def by_action(df, act):
    """df: text_id, y, p, pred → 응답 유형별 건수·위반 수·평균 p·위반 예측 수."""
    return (df.assign(action=df.text_id.map(act)).groupby("action")
            .agg(n=("y", "size"), n_harmful=("y", "sum"), mean_p=("p", "mean"), pred_pos=("pred", "sum")).reset_index())


def agg(cfg, out):
    d, _, _ = dna_dev(cfg)
    lab = dict(zip(d.text_id, d.human_label.astype(int)))
    J = pd.read_csv(cfg["_res"] / "judgments.csv", keep_default_na=False)
    sim = pd.read_csv(cfg["_data"] / "derived/similarity.csv").rename(columns={"node_id": "candidate_rule_id"})
    b = J[(J.condition == "base") & (J.candidate_rule_id != "") & ~J.candidate_rule_id.str.startswith("AGG:")
          & J.text_id.isin(lab) & J.model.isin(["shieldgemma", "team"])]
    b = b.merge(sim, on=["text_id", "candidate_rule_id"], how="left").sort_values(["text_id", "similarity"], ascending=[True, False])
    rows = []
    for mo, g in b.groupby("model"):
        w = g.similarity.clip(lower=0)
        wm = (w * g.p_violation).groupby(g.text_id).sum() / w.groupby(g.text_id).sum()
        n_fallback = int(wm.isna().sum())  # 후보 유사도가 모두 ≤ 0 → 단순 평균
        ways = {"max": g.groupby("text_id").p_violation.max(), "top1": g.groupby("text_id").p_violation.first(),
                "sim_weighted_mean": wm.fillna(g.groupby("text_id").p_violation.mean())}
        for name, p in ways.items():
            y = p.index.map(lab).values
            th = 0.5 if mo == "shieldgemma" else kappa_threshold(cfg, y, p.values)
            rows.append(dict(model=mo, aggregation=name, threshold_rule="고정 0.5" if mo == "shieldgemma" else "dev κ 최대",
                             note=f"유사도 합 0 → 단순 평균 {n_fallback}건" if name == "sim_weighted_mean" else "")
                        | metrics(y, p.values, th))
    r = pd.DataFrame(rows)
    r.to_csv(out / "aggregation_compare.csv", index=False)
    print(r.round(3).to_string(index=False))


def kappa_threshold(cfg, y, p):
    grid = judges.threshold_grid(cfg)
    k = [cohen_kappa_score(y, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(k))])  # 같은 κ 면 낮은 임계값


def prompts(cfg, out):
    d, nodes, ex = dna_dev(cfg)
    per = pd.read_csv(cfg["_res"] / "p1/retrieval_per_text.csv", keep_default_na=False)
    items, _ = judges.work_items(cfg, d, nodes, per, ex)
    items = [it for it in items if it["condition"] == "rule_only"]
    lab = dict(zip(d.text_id, d.human_label.astype(int)))
    act = actions(cfg)
    hf = cfg["judge"]["models"]["team"]["hf_id"]
    cache = judges.Cache(cfg["_data"] / "cache/judgments.sqlite")
    sg = judges.ShieldGemma(cfg, hf)
    rows, acts = [], []
    try:
        for pv, rid in [("team_v1", "T1"), ("team_v2", "T2"), ("team_v3", "T3")]:
            jd = judges.TeamJudge(cfg, hf, pv, shared=sg)
            p, n_new = judges.score_cached(jd, cache, f"{hf}|team|{pv}", [jd.build(it) for it in items])
            ids = [it["text_id"] for it in items]
            y = np.array([lab[t] for t in ids])
            th = kappa_threshold(cfg, y, p)
            rows.append(dict(run_id=rid, prompt_version=pv, condition="rule_only", threshold_rule="dev κ 최대",
                             n_new_forward=n_new) | metrics(y, p, th))
            a = by_action(pd.DataFrame({"text_id": ids, "y": y, "p": p, "pred": (p >= th).astype(int)}), act)
            acts.append(a.assign(run_id=rid, prompt_version=pv))
            print(f"[prompts] {pv}: 새로 계산 {n_new}")
    finally:
        sg.close()
    r = pd.DataFrame(rows)
    r.to_csv(out / "prompt_compare.csv", index=False)
    a = pd.concat(acts)
    a.to_csv(out / "prompt_by_action.csv", index=False)
    print(r.round(3).to_string(index=False))
    print(a.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["agg", "prompts"])
    a = ap.parse_args()
    cfg = load_config()
    out = cfg["_res"] / "dna_check"
    out.mkdir(parents=True, exist_ok=True)
    {"agg": agg, "prompts": prompts}[a.cmd](cfg, out)
