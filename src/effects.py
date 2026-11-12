"""6단계(P4): 임계값·τ·우선순위 변경, 처음 보는 규칙(규칙 제외), 규칙 문장 수정의 전후 비교."""
import json
import time
from datetime import datetime, timedelta, timezone

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
    if not before.text_id.is_unique or not after.text_id.is_unique:
        raise ValueError("전후 비교에는 텍스트마다 판정이 하나씩 있어야 합니다.")
    if set(before.text_id) != set(after.text_id):
        raise ValueError("전후 비교의 텍스트 범위가 다릅니다. 빠진 판정을 먼저 계산하세요.")
    meta = meta[meta.text_id.isin(before.text_id)].drop_duplicates("text_id")
    if set(meta.text_id) != set(before.text_id):
        raise ValueError("일부 판정의 사람 라벨 또는 출처가 없습니다.")
    splits = sorted(meta.split.unique()) if "split" in meta else ["dev"]
    if len(splits) > 1:
        raise ValueError("dev와 test는 분리해서 비교하세요.")
    b = before.merge(meta, on="text_id", validate="one_to_one")
    a = after.merge(meta, on="text_id", validate="one_to_one")
    f = flips(b.set_index("text_id").prediction, a.set_index("text_id").prediction, meta.set_index("text_id").human_label)
    f = f.merge(meta[["text_id", "source"]], on="text_id")
    return {"name": name, "change": change, "created": datetime.now(timezone(timedelta(hours=9))).isoformat(timespec="seconds"),
            "split": splits[0] if splits else "dev", "metrics_before": per_source(b), "metrics_after": per_source(a),
            "n_flipped": len(f), "n_fixed": int((f.kind == "fixed").sum()), "n_broken": int((f.kind == "broken").sum()),
            "flipped": f.to_dict("records")}


def save(cfg, prop):
    d = cfg["_res"] / "rule_proposals"
    d.mkdir(parents=True, exist_ok=True)
    prop.setdefault("prompt_version", cfg["judge"]["team_prompt_versions"][-1])
    name = f"{prop['name']}_{prop['prompt_version']}"
    p = d / f"{name}.json"
    if p.exists():
        p = d / f"{name}_{time.time_ns()}.json"
    p.write_text(json.dumps(prop, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"[p4] {prop['name']}: 뒤집힘 {prop['n_flipped']} (맞게 {prop['n_fixed']} / 틀리게 {prop['n_broken']})")
    return p


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
    p = np.round(judges.score_team(cfg, items, pv), 6)
    j = pd.DataFrame(items)[["text_id", "condition", "candidate_rule_id"]].assign(model="team", prompt_version=pv, p_violation=p)
    agg = judges.aggregate(j, how)
    return agg.assign(prediction=(agg.p_violation >= agg.text_id.map(th)).astype(int))[["text_id", "p_violation", "prediction"]]


def live_settings(cfg, tgt, nodes, sw, m, judgments, model, tau, aggregation, threshold=None):
    """같은 dev 텍스트·고정 임계값으로 검색 τ와 후보 우선순위를 함께 바꾼다."""
    if (tgt.split != "dev").any():
        raise ValueError("설정 탐색은 dev에서만 가능합니다.")
    pv = cfg["judge"]["team_prompt_versions"][-1] if model == "team" else "official"
    cur = m[(m.model == model) & (m.condition == "base") & m.text_id.isin(tgt.text_id)].copy()
    if cur.empty:
        raise ValueError("선택한 모델의 기본 판정이 없습니다.")
    if set(cur.text_id) != set(tgt.text_id):
        raise ValueError("일부 텍스트의 기본 판정이 없습니다. judge 단계를 먼저 실행하세요.")
    th = per_text_threshold(cur)
    changed_rules = tau != cfg["retrieval"]["tau"] or aggregation != cfg["judge"]["aggregation"]
    after = cur[["text_id", "p_violation", "prediction"]].copy()
    if changed_rules and model in RULE_MODELS:
        per = retrieval.retrieve_all(cfg, tgt, retrieval.Tree(nodes), sw, tau).set_index("text_id")
        items = base_items(cfg, tgt, per, dict(zip(nodes.node_id, nodes.rule_text)))
        raw = judgments[(judgments.model == model) & (judgments.prompt_version == pv)
                        & (judgments.condition == "base") & ~judgments.candidate_rule_id.str.startswith("AGG:")]
        known = {(r.text_id, r.candidate_rule_id): r.p_violation for r in raw.itertuples()}
        missing = [it for it in items if (it["text_id"], it["candidate_rule_id"]) not in known]
        if missing:
            p = judges.score_rule_model(cfg, missing, model, pv)
            known.update({(it["text_id"], it["candidate_rule_id"]): round(float(v), 6) for it, v in zip(missing, p)})
        jj = pd.DataFrame(items)[["text_id", "condition", "candidate_rule_id"]].assign(model=model, prompt_version=pv)
        jj["p_violation"] = [known[(r.text_id, r.candidate_rule_id)] for r in jj.itertuples()]
        after = judges.aggregate(jj, aggregation)[["text_id", "p_violation"]]
    thresholds = after.text_id.map(th) if threshold is None else float(threshold)
    after = after.assign(prediction=(after.p_violation >= thresholds).astype(int))
    change = dict(model=model, prompt_version=pv, tau=[cfg["retrieval"]["tau"], tau],
                  aggregation=[cfg["judge"]["aggregation"], aggregation],
                  threshold=[cur.groupby("source").threshold.first().to_dict(), threshold if threshold is not None else "출처별 원래 값 유지"])
    meta = tgt[["text_id", "source", "split", "human_label"]]
    return compare("settings", change, cur[["text_id", "p_violation", "prediction"]], after, meta)


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
    adr = auc_drop_recovery(m[m.split == "dev"])
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
    edits = list(ec.get("rule_edits") or [])
    single = ec.get("rule_edit") or {}
    if single.get("node_id"):
        edits.append(single)
    if edits and mo == "team":
        for edit in edits:
            scope = tgt[tgt.source == edit["source"]].text_id if edit.get("source") else None
            prop = rule_edit(cfg, edit["node_id"], edit["new_text"], text_ids=scope)
            prop["title"] = edit.get("title", edit["node_id"])
            prop["evidence"] = edit.get("evidence", "")
            save(cfg, prop)
    else:
        status.append("규칙 문장 수정: configs/baseline.yaml effects.rule_edit 미설정 → 건너뜀")
    (out / "status.txt").write_text("\n".join(status), encoding="utf-8")
    print(adr.to_string() if len(adr) else "", "\n".join(status))


def rule_edit(cfg, nid, new, examples=None, text_ids=None):
    """규칙 문장 하나를 고쳐 유사도 다시 계산 → BFS → 팀 판정기 base·rule_only 다시 판정 → 전후 비교.
    화면의 '다시 판정' 버튼도 이 함수를 부른다."""
    m, _ = load_joined(cfg)
    texts, nodes, _, sample = data.load(cfg)
    tgt = retrieval.targets(cfg, texts, sample)
    tgt = tgt[tgt.split == "dev"]
    if text_ids is not None:
        tgt = tgt[tgt.text_id.isin(text_ids)]
    if tgt.empty or not str(new).strip():
        raise ValueError("규칙 문장과 비교할 dev 텍스트가 필요합니다.")
    if nid not in set(nodes.node_id):
        raise ValueError(f"없는 규칙입니다: {nid}")
    meta = m[["text_id", "source", "human_label"]].drop_duplicates("text_id")
    mo, pv = "team", cfg["judge"]["team_prompt_versions"][-1]
    j, _ = adapters.read(cfg, "judgments")
    jj = j[(j.model == mo) & (j.prompt_version.astype(str) == pv)]
    th = per_text_threshold(jj)
    cur = main_rows(jj)
    cur = cur[(cur.condition == "base") & cur.text_id.isin(tgt.text_id)][["text_id", "p_violation", "prediction"]]
    if set(cur.text_id) != set(tgt.text_id):
        raise ValueError("일부 dev 텍스트의 팀 기본 판정이 없습니다. judge --models team을 먼저 실행하세요.")
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
    judge_rule = judges.rule_with_examples(new, examples)
    rule2 = rule | {nid: judge_rule}
    after = rejudge_base(cfg, tgt, per2, rule2, pv, th, cfg["judge"]["aggregation"])
    prop = compare(f"rule_edit_{nid}", {"node_id": nid, "old_text": rule[nid], "new_text": new, "model": mo}, cur, after, meta)
    prop["change"].update(tau=cfg["retrieval"]["tau"], aggregation=cfg["judge"]["aggregation"],
                           thresholds=jj[jj.text_id.isin(tgt.text_id)].assign(source=lambda x: x.text_id.map(tgt.set_index("text_id").source))
                           .groupby("source").threshold.first().to_dict())
    prop["prompt_version"] = pv
    prop["status"] = "dev 평가 완료 · test 미확인"
    prop["change"]["examples"] = examples.to_dict("records") if examples is not None else []
    # rule_only 도 전후 비교 (정답 규칙이 이 노드인 텍스트)
    gt = tgt[tgt.gold_rule_id == nid]
    if len(gt):
        items = [{"text_id": t.text_id, "text": t.text, "context": t.context, "mode": cfg["judge"]["moderation_mode"][t.source],
                  "rule": judge_rule} for t in gt.itertuples()]
        p = np.round(judges.score_team(cfg, items, pv), 6)
        b = main_rows(jj)
        b = b[(b.condition == "rule_only") & b.text_id.isin(gt.text_id)][["text_id", "p_violation", "prediction"]]
        a = pd.DataFrame({"text_id": gt.text_id.values, "p_violation": p, "prediction": (p >= gt.text_id.map(th).values).astype(int)})
        prop["rule_only"] = compare("ro", {}, b, a, meta)
    return prop
