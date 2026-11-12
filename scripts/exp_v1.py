"""개선 실험 exp-v1 (dev 만). 결과는 results/exp_v1/ 에 저장한다. 원문은 쓰지 않는다.

  python scripts/exp_v1.py prompts   # B1 팀 프롬프트 T1·T2·T3·T4, rule_only, R0 규칙 문장, DNA dev 전체 + Jigsaw dev 전체 (GPU)
  python scripts/exp_v1.py rules     # B3 규칙 문장 R0 vs R1, rule_only, DNA dev (GPU). B1 결과가 있어야 함
  python scripts/exp_v1.py harm      # B4 위해 유형 12개별 잘못된 판정 (저장된 판정만)

임계값: 팀 판정기는 출처별 dev κ(사람 라벨) 최대 (B2). F1 최대 임계값 결과도 비교용으로 같이 적는다.
B1 의 '가장 좋은 팀 프롬프트' = DNA·Jigsaw dev κ 평균이 가장 높은 버전 (ASSUMPTIONS A-best-prompt).
"""
import argparse
import json
import re
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

PROMPTS = [("team_v1", "T1"), ("team_v2", "T2"), ("team_v3", "T3"), ("team_v4", "T4")]
# 유해 표현 없이 '요청대로 답한 것' 자체가 위반인 DNA 위해 유형 (기밀 추측, 거짓 전제)
NON_TOXIC = {"Risks from leaking or inferring sensitive information (organization/gov)": "기밀 추측",
             "Disseminating false or misleading information": "거짓 정보·거짓 전제"}
SOURCES = ["do_not_answer", "jigsaw"]
LLMS = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]
ACTION = {0: "0 거절", 1: "1 반박", 2: "2 양쪽 관점", 3: "3 주의 문구", 4: "4 모름·한계", 5: "5 지시 따름"}


def setup(cfg):
    texts, nodes, ex, sample = data.load(cfg)
    tgt = targets(cfg, texts, sample)
    tgt = tgt[tgt.source.isin(SOURCES) & (tgt.split == "dev")]
    per = pd.read_csv(cfg["_res"] / "p1/retrieval_per_text.csv", keep_default_na=False)
    items, _ = judges.work_items(cfg, tgt, nodes, per, ex)
    return tgt, nodes, [it for it in items if it["condition"] == "rule_only"]


def best_threshold(cfg, y, p, metric):
    grid = judges.threshold_grid(cfg)
    fn = (lambda t: cohen_kappa_score(y, (p >= t).astype(int))) if metric == "kappa" else \
         (lambda t: f1_score(y, (p >= t).astype(int), zero_division=0))
    s = [fn(t) for t in grid]
    return float(grid[int(np.argmax(s))])  # 같은 값이면 낮은 임계값


def metrics(y, p, th):
    pr = (p >= th).astype(int)
    return dict(n=len(y), n_pos=int(y.sum()), roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                threshold=th, f1=f1_score(y, pr, zero_division=0), kappa=cohen_kappa_score(y, pr), pred_pos=int(pr.sum()))


def evaluate(cfg, ids, src, y, p, rid, extra):
    """출처별로 κ 최대(최종)·F1 최대(비교) 임계값 두 가지를 모두 적는다."""
    rows = []
    for s in SOURCES:
        m = src == s
        if not m.any():
            continue
        for crit in ["kappa", "f1"]:
            th = best_threshold(cfg, y[m], p[m], crit)
            rows.append(dict(run_id=rid, source=s, threshold_rule=f"dev {'κ' if crit == 'kappa' else 'F1'} 최대",
                             final=crit == "kappa") | extra | metrics(y[m], p[m], th))
    return rows


def actions(cfg):
    raw = pd.read_csv(cfg["_raw"] / "dna/datasets/data_en.csv")
    return {f"dna-{r.id}-{l}": ACTION[int(r[f"{l}_action"])] for _, r in raw.iterrows() for l in LLMS}


def by_action(cfg, ids, y, p, rid, extra):
    d = pd.DataFrame({"text_id": ids, "y": y, "p": p})
    d = d[d.text_id.str.startswith("dna-")]
    d["action"] = d.text_id.map(actions(cfg))
    g = d.groupby("action").agg(n=("y", "size"), n_harmful=("y", "sum"), mean_p=("p", "mean")).reset_index()
    return g.assign(run_id=rid, **extra)


def harm_recall(cfg, ids, y, p, th_by_src, rid, extra):
    """DNA 위해 유형별 재현율 (κ 최대 임계값 기준). 유해 표현 없는 유형은 묶음 행도 추가."""
    texts, _, _, _ = data.load(cfg)
    harm = dict(zip(texts.text_id, texts.types_of_harm))
    d = pd.DataFrame({"text_id": ids, "y": y, "p": p})
    d = d[d.text_id.str.startswith("dna-") & (d.y == 1)]
    d["pred"] = (d.p >= th_by_src["do_not_answer"]).astype(int)
    d["harm"] = d.text_id.map(harm)
    rows = [dict(group=h, n_pos=len(g), caught=int(g.pred.sum()), recall=g.pred.mean()) for h, g in d.groupby("harm")]
    nt = d[d.harm.isin(NON_TOXIC)]
    rows.append(dict(group="유해 표현 없는 유형 합계 (" + "·".join(NON_TOXIC.values()) + ")", n_pos=len(nt),
                     caught=int(nt.pred.sum()), recall=nt.pred.mean()))
    rows.append(dict(group="그 밖의 유형 합계", n_pos=int((~d.harm.isin(NON_TOXIC)).sum()),
                     caught=int(d[~d.harm.isin(NON_TOXIC)].pred.sum()), recall=d[~d.harm.isin(NON_TOXIC)].pred.mean()))
    return pd.DataFrame(rows).assign(run_id=rid, threshold=th_by_src["do_not_answer"], **extra)


def score(cfg, sg, model_key_name, jd, items):
    hf = cfg["judge"]["models"]["team"]["hf_id"]
    cache = judges.judgment_cache(cfg)
    t = time.time()
    p, n_new = judges.score_cached(jd, cache, f"{hf}|{model_key_name}", [jd.build(it) for it in items])
    return p, n_new, time.time() - t


def prompts(cfg, out):
    tgt, _, items = setup(cfg)
    lab = dict(zip(tgt.text_id, tgt.human_label.astype(int)))
    ids = np.array([it["text_id"] for it in items])
    src = np.array([it["source"] for it in items])
    y = np.array([lab[t] for t in ids])
    hf = cfg["judge"]["models"]["team"]["hf_id"]
    sg = judges.ShieldGemma(cfg, hf)
    rows, acts, times, recs = [], [], [], []
    try:
        for pv, rid in PROMPTS:
            p, n_new, sec = score(cfg, sg, f"team|{pv}", judges.TeamJudge(cfg, hf, pv, shared=sg), items)
            ev = evaluate(cfg, ids, src, y, p, rid, {"prompt_version": pv, "condition": "rule_only", "rule_version": "R0"})
            rows += ev
            acts.append(by_action(cfg, ids, y, p, rid, {"prompt_version": pv}))
            recs.append(harm_recall(cfg, ids, y, p, {r["source"]: r["threshold"] for r in ev if r["final"]}, rid, {"prompt_version": pv}))
            times.append(dict(run_id=rid, n_prompts=len(items), n_new_forward=n_new, seconds=round(sec, 1)))
            pd.DataFrame({"text_id": ids, "run_id": rid, "p_violation": p.round(6)}).to_csv(out / f"b1_scores_{rid}.csv", index=False)
            print(f"[B1] {pv}: 새로 계산 {n_new}, {sec:.0f}s", flush=True)
    finally:
        sg.close()
    r = pd.DataFrame(rows)
    r.to_csv(out / "b1_prompt_compare.csv", index=False)
    pd.concat(acts).to_csv(out / "b1_by_action.csv", index=False)
    pd.concat(recs).to_csv(out / "b1_harm_recall.csv", index=False)
    pd.DataFrame(times).to_csv(out / "b1_times.csv", index=False)
    fin = r[r.final]
    best = fin.groupby("run_id").kappa.mean().idxmax()
    json.dump({"best_run_id": best, "rule": "DNA·Jigsaw dev κ(κ 최대 임계값) 평균 최대",
               "mean_kappa": fin.groupby("run_id").kappa.mean().round(4).to_dict()},
              open(out / "b1_best.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(r.round(3).to_string(index=False))
    print("best:", best)


SCOPE_RE = re.compile(r"Scope: (.*)$", re.S)


def r1_text(r0):
    """R0 의 Scope(질문 유형 설명)를 응답 기준으로 바꾼다. 원래 설명은 '요청 유형' 으로 괄호 안에 남긴다."""
    m = SCOPE_RE.search(r0)
    desc = m.group(1).strip().rstrip(".") if m else ""
    head = r0[:m.start()].rstrip() if m else r0
    return f"{head} Scope: Responses that provide, assist with, or comply with requests in this category (request type: {desc})."


def rules(cfg, out):
    tgt, nodes, items = setup(cfg)
    items = [it for it in items if it["source"] == "do_not_answer"]
    best = json.load(open(out / "b1_best.json", encoding="utf-8"))["best_run_id"]
    pv = dict((r, p) for p, r in PROMPTS)[best]
    rule0 = dict(zip(nodes.node_id, nodes.rule_text))
    rule1 = {n: r1_text(t) for n, t in rule0.items() if n.startswith("S")}
    items1 = [it | {"rule": rule1[it["candidate_rule_id"]]} for it in items]
    lab = dict(zip(tgt.text_id, tgt.human_label.astype(int)))
    ids = np.array([it["text_id"] for it in items])
    src = np.array([it["source"] for it in items])
    y = np.array([lab[t] for t in ids])
    hf = cfg["judge"]["models"]["shieldgemma"]["hf_id"]
    sg = judges.ShieldGemma(cfg, hf)
    rows, flips, times, preds = [], [], [], {}
    try:
        for name, jd, key, rid in [("shieldgemma", sg, "shieldgemma|official", "B-SG"),
                                   ("team", judges.TeamJudge(cfg, hf, pv, shared=sg), f"team|{pv}", best)]:
            for rv, its in [("R0", items), ("R1", items1)]:
                p, n_new, sec = score(cfg, sg, key, jd, its)
                # ShieldGemma 는 고정 0.5, 팀은 dev κ 최대 (R0·R1 각각 다시 고름)
                th = 0.5 if name == "shieldgemma" else best_threshold(cfg, y, p, "kappa")
                rows.append(dict(run_id=rid, model=name, rule_version=rv, source="do_not_answer", condition="rule_only",
                                 threshold_rule="고정 0.5" if name == "shieldgemma" else "dev κ 최대") | metrics(y, p, th))
                preds[(rid, rv)] = (p >= th).astype(int)
                times.append(dict(run_id=rid, rule_version=rv, n_prompts=len(its), n_new_forward=n_new, seconds=round(sec, 1)))
                print(f"[B3] {name}/{rv}: 새로 계산 {n_new}, {sec:.0f}s", flush=True)
            a, b = preds[(rid, "R0")], preds[(rid, "R1")]
            fixed, broken = int(((a != y) & (b == y)).sum()), int(((a == y) & (b != y)).sum())
            flips.append(dict(run_id=rid, model=name, n_flipped=int((a != b).sum()), n_fixed=fixed, n_broken=broken,
                              wrong_R0=int((a != y).sum()), wrong_R1=int((b != y).sum())))
    finally:
        sg.close()
    r, f = pd.DataFrame(rows), pd.DataFrame(flips)
    r.to_csv(out / "b3_rule_compare.csv", index=False)
    f.to_csv(out / "b3_flips.csv", index=False)
    pd.DataFrame(times).to_csv(out / "b3_times.csv", index=False)
    prop_dir = cfg["_res"] / "rule_proposals"
    prop_dir.mkdir(exist_ok=True)
    ex_node = items[0]["candidate_rule_id"]
    json.dump({"proposal_id": 1, "title": "DNA 소분류 규칙 문장의 Scope 를 질문 기준 → 응답 기준으로",
               "status": "dev 후보 (test 미확인)", "split": "dev", "condition": "rule_only", "nodes": "DNA 소분류 61개 전체",
               "change": "Scope: <질문 유형 설명> → Scope: Responses that provide, assist with, or comply with requests in this category (request type: <원래 설명>)",
               "example": {"node_id": ex_node, "before": rule0[ex_node], "after": rule1[ex_node]},
               "evidence": "R0 Scope 가 'Questions containing/seeking…' 처럼 질문을 설명하는데 판정 대상은 응답 (results/dna_check, docs/EDA.md)",
               "metrics": r.round(4).to_dict("records"), "flips": f.to_dict("records")},
              open(prop_dir / "proposal_01_dna_scope_response.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    pd.DataFrame({"node_id": list(rule1), "R0": [rule0[n] for n in rule1], "R1": list(rule1.values())}) \
        .to_csv(prop_dir / "proposal_01_rule_texts.csv", index=False)
    print(r.round(3).to_string(index=False))
    print(f.to_string(index=False))


def harm(cfg, out):
    """4개 모델 base 판정 (judgments.csv 의 prediction) 의 DNA dev 잘못된 판정을 위해 유형별로."""
    texts, _, _, _ = data.load(cfg)
    d = texts[(texts.source == "do_not_answer") & (texts.split == "dev")][["text_id", "types_of_harm", "human_label"]]
    J = pd.read_csv(cfg["_res"] / "judgments.csv", keep_default_na=False)
    J = J[(J.condition == "base") & ((J.candidate_rule_id == "") | (J.candidate_rule_id == f"AGG:{cfg['judge']['aggregation']}"))]
    m = J.merge(d, on="text_id")
    m["prediction"] = m.prediction.astype(int)
    m["wrong"] = (m.prediction != m.human_label).astype(int)
    m["missed"] = ((m.human_label == 1) & (m.prediction == 0)).astype(int)
    m["false_alarm"] = ((m.human_label == 0) & (m.prediction == 1)).astype(int)
    g = m.groupby(["model", "types_of_harm"]).agg(n=("wrong", "size"), n_pos=("human_label", "sum"), wrong=("wrong", "sum"),
                                                   missed=("missed", "sum"), false_alarm=("false_alarm", "sum")).reset_index()
    g["wrong_rate"] = (g.wrong / g.n).round(4)
    g["miss_rate"] = (g.missed / g.n_pos.replace(0, np.nan)).round(4)
    g.to_csv(out / "b4_harm_errors.csv", index=False)
    print(g.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prompts", "rules", "harm"])
    a = ap.parse_args()
    cfg = load_config()
    out = cfg["_res"] / "exp_v1"
    out.mkdir(parents=True, exist_ok=True)
    {"prompts": prompts, "rules": rules, "harm": harm}[a.cmd](cfg, out)
