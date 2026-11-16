"""exp-v2: 팀원 프롬프트(T5~T7)를 우리 파이프라인 같은 조건에서 T2 와 비교 (dev 만). 결과는 runs/exp_v2/.

  python scripts/exp_v2.py plan    # 캐시에 없는 추론 수와 예상 시간만 (GPU 안 씀)
  python scripts/exp_v2.py run     # 판정 (GPU) → 지표
  python scripts/exp_v2.py diag    # base 에서 T2 가 ShieldGemma 보다 낮은 원인 (저장된 판정만)

같은 조건: DNA dev 전체 · Jigsaw dev 전체, base(상위 5후보 max) · rule_only, 우리 규칙 문장(R0)과 우리 ShieldGemma 틀.
임계값: 변형마다 출처별 base dev κ 최대 (judges.choose_thresholds, Baseline v2 와 같은 방식), 모든 조건에 같은 값.
원문은 저장하지 않는다 (text_id 와 확률만).
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, roc_auc_score  # noqa: E402

from src import data, judges  # noqa: E402
from src.common import load_config  # noqa: E402
from src.retrieval import targets  # noqa: E402

VARIANTS = [("official", "SG"), ("team_v2", "T2"), ("team_v5", "T5"), ("team_v6", "T6"), ("team_v7", "T7")]
SOURCES = ["do_not_answer", "jigsaw"]
CONDS = ["base", "rule_only"]
SEC_PER_FORWARD = 0.21  # exp-v1 B1 T4: 3,016회 623.5초 (RTX 3060)


def setup(cfg, conds=CONDS):
    texts, nodes, ex, sample = data.load(cfg)
    tgt = targets(cfg, texts, sample)
    tgt = tgt[tgt.source.isin(SOURCES) & (tgt.split == "dev")]
    assert (tgt.split == "dev").all()
    per = pd.read_csv(cfg["_res"] / "p1/retrieval_per_text.csv", keep_default_na=False)
    items, _ = judges.work_items(cfg, tgt, nodes, per, ex)
    return tgt, [it for it in items if it["condition"] in conds]


def builder(pv):
    cls = judges.ShieldGemma if pv == "official" else judges.TeamJudge
    b = cls.__new__(cls)
    b.prompt_version = pv
    return b


def key(cfg, pv):
    hf = cfg["judge"]["models"]["shieldgemma"]["hf_id"]
    return f"{hf}|shieldgemma|official" if pv == "official" else f"{hf}|team|{pv}"


def plan(cfg, out):
    tgt, items = setup(cfg)
    cache = judges.judgment_cache(cfg)
    print("dev 텍스트:", tgt.groupby("source").size().to_dict())
    rows = []
    for pv, rid in VARIANTS:
        b = builder(pv)
        keys = list({judges.Cache.key(key(cfg, pv), b.build(it)) for it in items})
        have = cache.get_many(keys)
        rows.append(dict(run_id=rid, prompt_version=pv, n_items=len(items), n_unique_prompts=len(keys),
                         n_cached=len(have), n_new=len(keys) - len(have)))
    r = pd.DataFrame(rows)
    r["est_minutes"] = (r.n_new * SEC_PER_FORWARD / 60).round(1)
    print(pd.Series([it["condition"] for it in items]).value_counts().to_dict())
    print(r.to_string(index=False))
    print(f"합계 새 추론 {r.n_new.sum()}회, 예상 {r.est_minutes.sum():.0f}분 (모델 로드 제외)")
    out.mkdir(parents=True, exist_ok=True)
    r.to_csv(out / "plan.csv", index=False)


def run(cfg, out, only=None, conds=CONDS):
    """only: 돌릴 run_id 목록. 캐시에 다 있으면 모델을 올리지 않는다. 이전 run_id 결과는 judgments_long.csv 에 남긴다."""
    tgt, items = setup(cfg, conds)
    hf = cfg["judge"]["models"]["shieldgemma"]["hf_id"]
    cache = judges.judgment_cache(cfg)
    sg, rows, times = None, [], []
    try:
        for pv, rid in VARIANTS:
            if only and rid not in only:
                continue
            t = time.time()
            prompts = [builder(pv).build(it) for it in items]
            keys = [judges.Cache.key(key(cfg, pv), x) for x in prompts]
            have = cache.get_many(list(set(keys)))
            if len(have) == len(set(keys)):
                p, n_new = np.array([have[k] for k in keys]), 0
            else:
                sg = sg or judges.ShieldGemma(cfg, hf)
                jd = sg if pv == "official" else judges.TeamJudge(cfg, hf, pv, shared=sg)
                p, n_new = judges.score_cached(jd, cache, key(cfg, pv), prompts)
            times.append(dict(run_id=rid, conditions="+".join(conds), n_prompts=len(items), n_new_forward=n_new, seconds=round(time.time() - t, 1)))
            print(f"[exp-v2] {rid}: 새로 계산 {n_new}, {time.time() - t:.0f}s", flush=True)
            rows.append(pd.DataFrame({"text_id": [it["text_id"] for it in items], "condition": [it["condition"] for it in items],
                                      "candidate_rule_id": [it["candidate_rule_id"] for it in items],
                                      "run_id": rid, "prompt_version": pv, "p_violation": p.round(6)}))
    finally:
        if sg is not None:
            sg.close()
    new, tm = pd.concat(rows), pd.DataFrame(times)
    for f, d in (("judgments_long.csv", new), ("times.csv", tm)):
        if (out / f).exists():
            old = pd.read_csv(out / f, keep_default_na=False)
            # 판정은 (run_id, 조건) 단위로 바꿔 쓰고, 실행 시간은 계속 쌓는다
            keep = (~old.run_id.isin(d.run_id) | ~old.condition.isin(conds)) if f == "judgments_long.csv" else old.index >= 0
            d = pd.concat([old[keep], d])
        d.to_csv(out / f, index=False)
    evaluate(cfg, out)


def per_text(cfg, out):
    """후보별 확률 → 텍스트별 하나 (base: 후보 max, rule_only: 그대로)."""
    j = pd.read_csv(out / "judgments_long.csv", keep_default_na=False)
    b = j[j.condition == "base"].groupby(["run_id", "prompt_version", "text_id"], as_index=False).p_violation.max()
    return pd.concat([b.assign(condition="base"), j[j.condition == "rule_only"].drop(columns="candidate_rule_id")])


BASE_K, RO_K, FIXED = "base dev κ 최대", "rule_only dev κ 최대 (임시)", "0.5 고정"


def thresholds(cfg, g, texts):
    """{기준: {출처: (임계값, dev κ)}}. base 판정이 있으면 base κ (Baseline v2 방식, 최종).
    base 가 아직 없는 변형끼리 비교하려고 rule_only κ 도 같이 계산 (임시). SG 는 기준 모델 기본값 0.5 도."""
    out = {}
    for name, cond in ((BASE_K, "base"), (RO_K, "rule_only")):
        h = g.loc[g.condition == cond, ["text_id", "p_violation"]]
        if len(h):
            out[name] = judges.choose_thresholds(cfg, h, texts, "kappa")["per_source"]
    if g.run_id.iloc[0] == "SG":
        out[FIXED] = {s: (0.5, None) for s in g.source.unique()}
    return out


def evaluate(cfg, out):
    texts, _, _, _ = data.load(cfg)
    t = per_text(cfg, out).merge(texts[["text_id", "source", "split", "human_label"]], on="text_id")
    assert (t.split == "dev").all()
    t["action"] = t.text_id.map(action_map(cfg))
    rows, ths, acts = [], [], []
    for rid, g in t.groupby("run_id", sort=False):
        for rule, per in thresholds(cfg, g, texts).items():
            for s, (v, sc) in per.items():
                ths.append(dict(run_id=rid, threshold_rule=rule, source=s, threshold=v, dev_kappa=sc))
            for (s, c), h in g.groupby(["source", "condition"]):
                y, p = h.human_label.astype(int).values, h.p_violation.values
                pr = (p >= per[s][0]).astype(int)
                rows.append(dict(run_id=rid, prompt_version=h.prompt_version.iloc[0], source=s, condition=c,
                                 threshold_rule=rule, threshold=per[s][0], n=len(y), n_pos=int(y.sum()),
                                 roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                                 f1=f1_score(y, pr, zero_division=0), kappa=cohen_kappa_score(y, pr), pred_pos=int(pr.sum())))
                if s == "do_not_answer":
                    a = h.assign(pred=pr).groupby("action").agg(n=("text_id", "size"), n_harmful=("human_label", "sum"),
                                                                mean_p=("p_violation", "mean"), pred_pos=("pred", "sum"))
                    acts.append(a.reset_index().assign(run_id=rid, condition=c, threshold_rule=rule, threshold=per[s][0]))
    m = pd.DataFrame(rows)
    m.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(ths).to_csv(out / "thresholds.csv", index=False)
    pd.concat(acts).to_csv(out / "by_action.csv", index=False)
    print(m.round(3).to_string(index=False))


def diag(cfg, out):
    """base 에서 T2 가 ShieldGemma 보다 낮은 원인. (1) max 집계가 정상 응답의 확률을 끌어올리는가 (2) 거절이 위반으로 잡히는가."""
    texts, _, _, _ = data.load(cfg)
    j = pd.read_csv(out / "judgments_long.csv", keep_default_na=False)
    j = j[j.run_id.isin(["SG", "T2"])].merge(texts[["text_id", "source", "human_label", "gold_rule_id"]], on="text_id")
    th = pd.read_csv(out / "thresholds.csv")
    th = th[th.threshold_rule == BASE_K].set_index(["run_id", "source"]).threshold
    rows, cases = [], []
    for (rid, s), g in j.groupby(["run_id", "source"]):
        b = g[g.condition == "base"]
        top = b.loc[b.groupby("text_id").p_violation.idxmax()].set_index("text_id")
        ro = g[g.condition == "rule_only"].set_index("text_id").p_violation
        gold_in = b[b.candidate_rule_id == b.gold_rule_id].set_index("text_id").p_violation
        d = top.assign(p_rule_only=ro, p_gold_in_base=gold_in, max_is_gold=top.candidate_rule_id == top.gold_rule_id)
        d["lift"] = d.p_violation - d.p_rule_only
        t = th[(rid, s)]
        for lab, h in d.groupby("human_label"):
            rows.append(dict(run_id=rid, source=s, human_label=int(lab), n=len(h), threshold=t,
                             mean_p_base=h.p_violation.mean(), mean_p_rule_only=h.p_rule_only.mean(),
                             mean_lift=h.lift.mean(), share_max_not_gold=1 - h.max_is_gold.mean(),
                             pred_pos_base=int((h.p_violation >= t).sum()), pred_pos_rule_only=int((h.p_rule_only >= t).sum()),
                             flipped_by_other_rule=int(((h.p_violation >= t) & (h.p_rule_only < t) & ~h.max_is_gold).sum())))
        if s == "do_not_answer":
            cases.append(d.reset_index().assign(run_id=rid, threshold=t)[
                ["run_id", "text_id", "human_label", "gold_rule_id", "candidate_rule_id", "max_is_gold",
                 "p_violation", "p_rule_only", "p_gold_in_base", "lift", "threshold"]])
    a = pd.DataFrame(rows)
    a.to_csv(out / "diag_aggregation.csv", index=False)
    print(a.round(3).to_string(index=False))
    c = pd.concat(cases)
    act = action_map(cfg)
    c["action"] = c.text_id.map(act)
    c["pred_base"] = (c.p_violation >= c.threshold).astype(int)
    c["pred_rule_only"] = (c.p_rule_only >= c.threshold).astype(int)
    c.to_csv(out / "diag_cases_dna.csv", index=False)
    g = c.groupby(["run_id", "action"]).agg(n=("text_id", "size"), n_harmful=("human_label", "sum"),
                                             mean_p_base=("p_violation", "mean"), mean_p_rule_only=("p_rule_only", "mean"),
                                             pred_pos_base=("pred_base", "sum"), pred_pos_rule_only=("pred_rule_only", "sum")
                                             ).reset_index()
    g.to_csv(out / "diag_by_action.csv", index=False)
    print(g.round(3).to_string(index=False))


def action_map(cfg):
    llms = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]
    names = {0: "0 거절", 1: "1 반박", 2: "2 양쪽 관점", 3: "3 주의 문구", 4: "4 모름·한계", 5: "5 지시 따름"}
    raw = pd.read_csv(cfg["_raw"] / "dna/datasets/data_en.csv")
    return {f"dna-{r.id}-{l}": names[int(r[f"{l}_action"])] for _, r in raw.iterrows() for l in llms}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "run", "eval", "diag"])
    ap.add_argument("--only", nargs="+", help="run 에서 돌릴 run_id (예: SG T2)")
    ap.add_argument("--conds", nargs="+", default=CONDS, choices=CONDS, help="run 에서 돌릴 조건")
    a = ap.parse_args()
    cfg = load_config()
    out = Path(__file__).resolve().parents[1] / "runs" / "exp_v2"
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "run":
        run(cfg, out, a.only, a.conds)
    else:
        {"plan": plan, "eval": evaluate, "diag": diag}[a.cmd](cfg, out)
