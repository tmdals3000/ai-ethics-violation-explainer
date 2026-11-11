"""6단계(P4): 임계값·τ·우선순위 변경, 처음 보는 규칙(규칙 제외), 규칙 문장 수정의 전후 비교."""
import json
import time

import numpy as np
import pandas as pd

from src import adapters, data, judges, retrieval
from src.metrics import binary_metrics, load_joined, main_rows

RULE_MODELS = ["shieldgemma", "team"]


def flips(before, after, human):
    """before/after: text_id→prediction. 맞게 바뀐 것(fixed)과 틀리게 바뀐 것(broken)."""
    rows = []
    for t in before.index.intersection(after.index):
        b, a = int(before[t]), int(after[t])
        if a != b:
            rows.append({"text_id": t, "before": b, "after": a, "human_label": int(human[t]),
                         "kind": "fixed" if a == human[t] else "broken"})
    return pd.DataFrame(rows, columns=["text_id", "before", "after", "human_label", "kind"])


def per_source(df):
    return {s: binary_metrics(g) for s, g in df.groupby("source")}


def compare(name, change, before, after, meta):
    """before/after: text_id, p_violation, prediction (base 조건). 지표·뒤집힌 판정 반환."""
    b = before.merge(meta, on="text_id")
    a = after.merge(meta, on="text_id")
    f = flips(b.set_index("text_id").prediction, a.set_index("text_id").prediction, meta.set_index("text_id").human_label)
    f = f.merge(meta[["text_id", "source"]], on="text_id")
    return {"name": name, "change": change, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "split": "dev", "metrics_before": per_source(b), "metrics_after": per_source(a),
            "n_flipped": len(f), "n_fixed": int((f.kind == "fixed").sum()), "n_broken": int((f.kind == "broken").sum()),
            "flipped": f.to_dict("records")}


def save(cfg, prop):
    d = cfg["_res"] / "rule_proposals"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prop['name']}.json").write_text(json.dumps(prop, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[p4] {prop['name']}: 뒤집힘 {prop['n_flipped']} (맞게 {prop['n_fixed']} / 틀리게 {prop['n_broken']})")


def auc_drop_recovery(m):
    """정답 규칙이 있는 텍스트에서 조건별 AUC. 하락폭 = base − rule_excluded, 회복폭 = (rule_only|+examples) − rule_excluded."""
    g = m[(m.gold_rule_id.astype(str) != "") & m.model.isin(RULE_MODELS)]
    rows = []
    for by in ["source", "rule_group"]:
        for (mo, v), x in g.groupby(["model", by]):
            auc = {c: binary_metrics(y)["roc_auc"] for c, y in x.groupby("condition")}
            ex = auc.get("rule_excluded", np.nan)
            rows.append({"group_by": by, "group": v, "model": mo, "n_texts": x.text_id.nunique(),
                         "n_pos": int(x[x.condition == "base"].human_label.sum())}
                        | {f"auc_{c}": auc.get(c, np.nan) for c in ["base", "rule_excluded", "rule_only", "rule_plus_examples"]}
                        | {"auc_drop": auc.get("base", np.nan) - ex,
                           "recovery_rule_only": auc.get("rule_only", np.nan) - ex,
                           "recovery_rule_plus_examples": auc.get("rule_plus_examples", np.nan) - ex,
                           "note": "" if all(a == a for a in auc.values()) else "한 라벨뿐인 묶음은 AUC 계산 불가"})
    return pd.DataFrame(rows).round(4)


def base_items(cfg, tgt, per, rule):
    out = []
    for _, t in tgt.iterrows():
        cands = str(per.loc[t.text_id, "candidates"]).split("|")
        for c in cands:
            out.append({"text_id": t.text_id, "text": t.text, "context": t.context, "condition": "base",
                        "mode": cfg["judge"]["moderation_mode"][t.source], "candidate_rule_id": c, "rule": rule[c]})
    return out


def per_text_threshold(jj):
    """텍스트별 임계값 (팀 판정기는 출처별 임계값이라 텍스트마다 다를 수 있다)."""
    return jj.drop_duplicates("text_id").set_index("text_id").threshold.astype(float)


def rejudge_base(cfg, tgt, per, rule, pv, th, how):
    items = base_items(cfg, tgt, per, rule)
    p = judges.score_team(cfg, items, pv)
    j = pd.DataFrame(items)[["text_id", "condition", "candidate_rule_id"]].assign(model="team", prompt_version=pv, p_violation=p)
    agg = judges.aggregate(j, how)
    return agg.assign(prediction=(agg.p_violation >= agg.text_id.map(th)).astype(int))[["text_id", "p_violation", "prediction"]]


def run(cfg):
    m, _ = load_joined(cfg)
    texts, nodes, _, sample = data.load(cfg)
    tgt = retrieval.targets(cfg, texts, sample)
    tgt = tgt[tgt.split == "dev"]
    meta = m[["text_id", "source", "human_label"]].drop_duplicates("text_id")
    out = cfg["_res"] / "p4"
    out.mkdir(parents=True, exist_ok=True)
    ec = cfg["effects"]
    mo = ec["model"] if ec["model"] in set(m.model) else None
    status = []

    # (1) 처음 보는 규칙: AUC 하락폭·회복폭
    adr = auc_drop_recovery(m)
    adr.to_csv(out / "auc_drop_recovery.csv", index=False)
    if adr.empty:
        status.append("AUC 하락폭·회복폭: 규칙을 받는 모델(ShieldGemma/팀) 판정이 없어 계산 불가")

    # (2) 판정 임계값 변경 — 판정 결과가 있는 모든 모델 (4개)
    th_model = mo or (sorted(set(m.model))[0] if len(m) else None)
    for th_model in sorted(set(m.model)):
        cur = m[(m.model == th_model) & (m.condition == "base") & (m.split == "dev")]
        for th in ec["alt_thresholds"]:
            after = cur.assign(prediction=(cur.p_violation >= th).astype(int))
            prop = compare(f"threshold_{th_model}_{th}", {"model": th_model, "threshold": [cur.groupby("source").threshold.first().to_dict(), th],
                           "substitute_model": th_model != ec["model"]},
                           cur[["text_id", "p_violation", "prediction"]], after[["text_id", "p_violation", "prediction"]], meta)
            save(cfg, prop)
    if not mo:
        status.append(f"effects.model={ec['model']} 판정 결과 없음 → 임계값 실험만 {th_model} 로 대신, 우선순위·τ·규칙 문장 수정은 건너뜀")
        (out / "status.txt").write_text("\n".join(status), encoding="utf-8")
        print("\n".join(status))
        return

    j, _ = adapters.read(cfg, "judgments")
    pv = cfg["judge"]["team_prompt_versions"][-1]
    jj = j[(j.model == mo) & (j.prompt_version.astype(str) == (pv if mo == "team" else "official"))]
    th = per_text_threshold(jj)
    cur = main_rows(jj)
    cur = cur[(cur.condition == "base") & cur.text_id.isin(tgt.text_id)][["text_id", "p_violation", "prediction"]]

    # (3) 우선순위(후보 집계) 변경: max → top1 — 규칙을 받는 모델 (ShieldGemma, 팀)
    for rm in [x for x in RULE_MODELS if x in set(j.model)]:
        jr = j[(j.model == rm) & (j.prompt_version.astype(str) == (pv if rm == "team" else "official"))]
        thr = per_text_threshold(jr)
        cr = main_rows(jr)
        cr = cr[(cr.condition == "base") & cr.text_id.isin(tgt.text_id)][["text_id", "p_violation", "prediction"]]
        agg = judges.aggregate(jr[jr.text_id.isin(tgt.text_id)], ec["alt_aggregation"])
        agg = agg[agg.condition == "base"].assign(prediction=lambda x: (x.p_violation >= x.text_id.map(thr)).astype(int))
        save(cfg, compare(f"priority_{rm}_{ec['alt_aggregation']}", {"model": rm, "aggregation": [cfg["judge"]["aggregation"], ec["alt_aggregation"]]},
                          cr, agg[["text_id", "p_violation", "prediction"]], meta))

    # (4) τ 변경 → 후보가 바뀌므로 팀 판정기로 다시 판정 (캐시 재사용)
    tree = retrieval.Tree(nodes)
    sim, _ = adapters.read(cfg, "similarity")
    sw = retrieval.wide(sim)
    rule = dict(zip(nodes.node_id, nodes.rule_text))
    for tau in ec["alt_taus"] if mo == "team" else []:
        per = retrieval.retrieve_all(cfg, tgt, tree, sw, tau).set_index("text_id")
        after = rejudge_base(cfg, tgt, per, rule, pv, th, cfg["judge"]["aggregation"])
        save(cfg, compare(f"tau_{mo}_{tau}", {"model": mo, "tau": [cfg["retrieval"]["tau"], tau]}, cur, after, meta))

    # (5) 규칙 문장 수정
    re = ec.get("rule_edit") or {}
    if re.get("node_id") and mo == "team":
        save(cfg, rule_edit(cfg, re["node_id"], re["new_text"]))
    else:
        status.append("규칙 문장 수정: configs/baseline.yaml effects.rule_edit 미설정 → 건너뜀")
    (out / "status.txt").write_text("\n".join(status), encoding="utf-8")
    print(adr.to_string() if len(adr) else "", "\n".join(status))


def rule_edit(cfg, nid, new):
    """규칙 문장 하나를 고쳐 유사도 다시 계산 → BFS → 팀 판정기 base·rule_only 다시 판정 → 전후 비교.
    화면의 '다시 판정' 버튼도 이 함수를 부른다."""
    m, _ = load_joined(cfg)
    texts, nodes, _, sample = data.load(cfg)
    tgt = retrieval.targets(cfg, texts, sample)
    tgt = tgt[tgt.split == "dev"]
    meta = m[["text_id", "source", "human_label"]].drop_duplicates("text_id")
    mo, pv = "team", cfg["judge"]["team_prompt_versions"][-1]
    j, _ = adapters.read(cfg, "judgments")
    jj = j[(j.model == mo) & (j.prompt_version.astype(str) == pv)]
    th = per_text_threshold(jj)
    cur = main_rows(jj)
    cur = cur[(cur.condition == "base") & cur.text_id.isin(tgt.text_id)][["text_id", "p_violation", "prediction"]]
    tree = retrieval.Tree(nodes)
    sim, _ = adapters.read(cfg, "similarity")
    sw = retrieval.wide(sim)
    rule = dict(zip(nodes.node_id, nodes.rule_text))
    nodes2 = nodes.copy()
    nodes2.loc[nodes2.node_id == nid, "rule_text"] = new
    enc = retrieval.encoder(cfg)
    new_sim, _ = retrieval.compute_similarity(cfg, tgt, nodes2[nodes2.node_id == nid], enc)
    sw2 = {t: dict(d) for t, d in sw.items()}
    for r in new_sim.itertuples():
        sw2[r.text_id][nid] = r.similarity
    del enc
    per2 = retrieval.retrieve_all(cfg, tgt, tree, sw2).set_index("text_id")
    rule2 = rule | {nid: new}
    after = rejudge_base(cfg, tgt, per2, rule2, pv, th, cfg["judge"]["aggregation"])
    prop = compare(f"rule_edit_{nid}", {"node_id": nid, "old_text": rule[nid], "new_text": new, "model": mo}, cur, after, meta)
    # rule_only 도 전후 비교 (정답 규칙이 이 노드인 텍스트)
    gt = tgt[tgt.gold_rule_id == nid]
    if len(gt):
        items = [{"text_id": t.text_id, "text": t.text, "context": t.context, "mode": cfg["judge"]["moderation_mode"][t.source],
                  "rule": new} for t in gt.itertuples()]
        p = judges.score_team(cfg, items, pv)
        b = main_rows(jj)
        b = b[(b.condition == "rule_only") & b.text_id.isin(gt.text_id)][["text_id", "p_violation", "prediction"]]
        a = pd.DataFrame({"text_id": gt.text_id.values, "p_violation": p, "prediction": (p >= gt.text_id.map(th).values).astype(int)})
        prop["rule_only"] = compare("ro", {}, b, a, meta)
    return prop
