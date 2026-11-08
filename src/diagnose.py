"""5단계(P3): 잘못된 판정(base 조건)을 원인 1개로 분류한다. 위에서부터 차례로 적용.

1 retrieval    규칙 찾기 문제: 정답 규칙이 상위 k 밖이거나, 정답 규칙만 넣으면(rule_only) 맞음
               = 함께 찾은 다른 규칙 때문에 판정이 바뀜. 규칙을 받는 모델에만 적용.
2 label_check  라벨 확인 필요: 4개 모델의 base 답이 모두 같고 사람 라벨만 다름.
3 rule_text    규칙 문장 문제: ShieldGemma 와 팀 판정기가 rule_only 에서 함께 틀림.
4 judge_model  판정 모델 문제: rule_only 에서 이 모델은 틀리고 다른 모델 중 맞은 것이 있음.
그 외 unclassified. 규칙을 받지 않는 모델의 rule_only 답은 base 답과 같다고 본다.
"""
import pandas as pd

from src.metrics import MODELS, load_joined

RULE_MODELS = {"shieldgemma", "team"}
CAUSES = ["retrieval", "label_check", "rule_text", "judge_model", "unclassified"]


def diagnose(m, per, top_k):
    gr_col = pd.to_numeric(per.get("gold_rank", pd.Series(dtype=float)), errors="coerce").fillna(-1).astype(int)
    gold_rank = dict(zip(per.text_id, gr_col))
    m = m[m.gold_rule_id.astype(str) != ""]
    pred = {(r.text_id, r.model, r.condition): r.prediction for r in m.itertuples()}
    human = dict(zip(m.text_id, m.human_label))
    models = sorted(set(m.model))

    def ro(t, mo):  # rule_only 답 (규칙 안 받는 모델은 base 답)
        return pred.get((t, mo, "rule_only" if mo in RULE_MODELS else "base"))

    rows = []
    for r in m[(m.condition == "base") & (m.correct == 0)].itertuples():
        t, mo, y = r.text_id, r.model, human[r.text_id]
        cause, by = "unclassified", "no rule matched"
        base_all = [pred.get((t, x, "base")) for x in MODELS]
        missing = [x for x, p in zip(MODELS, base_all) if p is None]
        ro_rule = [ro(t, x) for x in RULE_MODELS]
        gr = gold_rank.get(t, -1)
        if mo in RULE_MODELS and (gr < 1 or gr > top_k):
            cause, by = "retrieval", f"1: 정답 규칙 순위 {gr if gr > 0 else '없음'} > top-{top_k} (base 후보)"
        elif mo in RULE_MODELS and ro(t, mo) == y:
            cause, by = "retrieval", "1: base 틀림·rule_only 맞음 → 다른 후보 규칙이 판정을 바꿈"
        elif missing or None in ro_rule:
            # 규칙 2·3 을 확인할 수 없으면 규칙 4 로 건너뛰지 않는다 (차례대로 적용)
            by = f"판정 결과가 없는 모델({', '.join(missing) or 'rule_only'})이 있어 규칙 2–4 적용 불가"
        elif len(set(base_all)) == 1:
            cause, by = "label_check", "2: base 에서 4개 모델 모두 같은 답, 사람 라벨만 다름"
        elif all(p != y for p in ro_rule):
            cause, by = "rule_text", "3: rule_only 에서 ShieldGemma·팀 판정기 모두 틀림"
        elif ro(t, mo) != y and any(ro(t, x) == y for x in models if x != mo):
            cause, by = "judge_model", "4: rule_only 에서 일부 모델만 틀림"
        rows.append({"text_id": t, "model": mo, "source": r.source, "gold_rule_id": r.gold_rule_id,
                     "rule_group": r.rule_group, "human_label": y, "prediction": r.prediction,
                     "gold_rank": gr, "cause": cause, "decided_by": by,
                     "missing_models": "|".join(missing), "rule2_applicable": not missing})
    return pd.DataFrame(rows, columns=["text_id", "model", "source", "gold_rule_id", "rule_group", "human_label",
                                       "prediction", "gold_rank", "cause", "decided_by", "missing_models",
                                       "rule2_applicable"])


def summary(d):
    if d.empty:
        return pd.DataFrame(columns=["source", "model", "cause", "n", "ratio"])
    s = d.groupby(["source", "model", "cause"]).size().rename("n").reset_index()
    s["ratio"] = (s.n / s.groupby(["source", "model"]).n.transform("sum")).round(4)
    return s


def run(cfg):
    m, _ = load_joined(cfg)
    per = pd.read_csv(cfg["_res"] / "p1/retrieval_per_text.csv", keep_default_na=False)
    d = diagnose(m, per, cfg["diagnose"]["retrieval_top_k"])
    out = cfg["_res"] / "p3"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "diagnosis.csv", index=False)
    s = summary(d)
    s.to_csv(out / "cause_summary.csv", index=False)
    print(s.to_string() if len(s) else "잘못된 판정 없음 (정답 규칙이 있는 출처 기준)")
    print("unclassified:", int((d.cause == "unclassified").sum()), "건")
    if len(d) and (~d.rule2_applicable).any():
        miss = sorted({x for v in d.missing_models for x in v.split("|") if x})
        print(f"[p3] ⚠ 결과가 빠진 모델 {miss} → {int((~d.rule2_applicable).sum())}건은 '라벨 확인 필요' 적용 불가")
