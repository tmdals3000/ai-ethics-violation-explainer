"""잘못된 판정 집계: 원인 규칙 1–4로 dev의 잘못된 판정을 원인별로 나눔 (P3)

PDF 원인 규칙 (위에서부터 차례로 적용해 하나만 선택, 대상: Do-Not-Answer와 Jigsaw)
  1 규칙 찾기 문제  정답 규칙을 찾지 못했거나(상위 k 밖), 함께 찾은 다른 규칙 때문에 판정이 바뀜
  2 라벨 확인 필요  규칙 문장을 받지 않는 Llama Guard·Qwen3Guard까지 모든 판정 모델이 같은 답이고 사람 라벨만 다름
  3 규칙 문장 문제  규칙 문장을 받는 판정(ShieldGemma, 팀 판정기)이 정답 규칙 문장을 넣어도('규칙 문장만') 함께 틀림
  4 판정 모델 문제  정답 규칙 문장을 넣었을 때 일부 판정 모델만 틀림

구현에서 정한 것 (analysis.diagnose 설명 참고)
  - 잘못된 판정 = (텍스트, 모델)에서 '기본' 조건 판정이 사람 라벨과 다른 것.
    Llama Guard·Qwen3Guard는 조건이 없으므로 그 판정 그대로
  - 1번은 규칙을 받는 모델에만 적용 (규칙을 받지 않는 모델의 판정은 규칙 찾기와 상관없음)
  - '함께 찾은 다른 규칙 때문에 바뀜' = 그 모델이 '규칙 문장만'에서는 맞음
  - 2~4번에서 규칙을 받는 모델의 답은 '규칙 문장만' 조건 (정답 규칙을 넣었을 때)
  - 임계값은 출처·모델·조건마다 dev κ 최대 (--thr-by f1, --thr로 바꿀 수 있음).
    F1 최대는 판별력이 약한 조건에서 '전부 위반'이 되어 오판이 전부 FP가 되므로 진단에는 κ를 씀

    python scripts/diagnose.py                       # 팀 판정기 = team_v3, k = 5
    python scripts/diagnose.py --team team_v1 --k 3
    python scripts/diagnose.py --thr team_v3=0.4 shieldgemma=0.2

출력: reports/diagnosis.csv (잘못된 판정 1건 1행), reports/diagnosis_k.csv (k별 원인 건수), reports/diagnosis_summary.md
"""
import argparse

import numpy as np
import pandas as pd

from analysis import CAUSES, RULE_FREE, diagnose, thresholds_on_dev
from common import PROCESSED, REPORTS, need
from compare_conditions import load_baseline
from data_overview import md_table


def build_wide(j, base, rs, models, thr, team):
    """텍스트 1행: human_label, gold_rank, pred_{m}, pred_{m}_rule_only, p_{m}_{조건}"""
    rule_models = ["shieldgemma", team]
    w = j.drop_duplicates("text_id")[["text_id", "source", "split", "human_label"]].set_index("text_id")
    for m in models:
        if m in RULE_FREE:
            g = base[base.model == m].set_index("text_id")
            w[f"p_{m}"] = g.p_violation
            w[f"pred_{m}"] = (g.p_violation >= g.source.map(lambda s: thr[(s, m, "none")])).astype(float)
            w.loc[w[f"p_{m}"].isna(), f"pred_{m}"] = np.nan
        else:
            for c in ["base", "excluded", "rule_only", "rule_example"]:
                g = j[(j.model == m) & (j.condition == c)].set_index("text_id")
                w[f"p_{m}_{c}"] = g.p_violation
                if c in ("base", "rule_only"):
                    pred = (g.p_violation >= g.source.map(lambda s: thr.get((s, m, c), 0.5))).astype(float)
                    w[f"pred_{m}" if c == "base" else f"pred_{m}_rule_only"] = pred
    w = w.join(rs.set_index("text_id")[["gold_rank", "gold_leaf", "top1", "found_leaves"]])
    need_cols = [f"pred_{m}" for m in models] + [f"pred_{m}_rule_only" for m in rule_models]
    ok = w[need_cols].notna().all(axis=1)
    return w[ok].reset_index(), int((~ok).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", default="team_v3", help="진단에 넣을 팀 판정기 변형")
    ap.add_argument("--k", type=int, default=5, help="정답 규칙이 상위 k 밖이면 '규칙 찾기 문제' (기본 5 = 기본 조건 후보 수)")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--baseline", nargs="*", default=[])
    ap.add_argument("--thr-by", default="kappa", choices=["kappa", "f1"], help="임계값 기준 (기본 kappa)")
    ap.add_argument("--thr", nargs="*", default=[], help="모델=임계값 (기본 조건에 적용, 예: team_v3=0.4)")
    args = ap.parse_args()

    need("judgments_conditions.csv", "rule_search.csv", "rule_tree.csv")
    j = pd.read_csv(PROCESSED / "judgments_conditions.csv").dropna(subset=["p_violation"])
    if args.team not in set(j.model):
        raise SystemExit(f"judgments_conditions.csv에 {args.team}이 없습니다. 있는 모델: {sorted(set(j.model))}")
    rs = pd.read_csv(PROCESSED / "rule_search.csv", keep_default_na=False,
                     usecols=["text_id", "gold_rank", "gold_leaf", "top1", "found_leaves"])
    rs["gold_rank"] = pd.to_numeric(rs.gold_rank, errors="coerce")
    names = pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False).set_index("node_id").name
    info = j.drop_duplicates("text_id")[["text_id", "source", "split"]]
    base = load_baseline(args.baseline, set(info.text_id))
    if base.empty or set(base.model) != set(RULE_FREE):
        raise SystemExit("Llama Guard·Qwen3Guard 판정이 필요합니다 (원인 2 '모든 모델이 같은 답'). "
                         "--baseline으로 judgments_baseline_*.csv를 지정하세요.")
    base = base.drop(columns=[c for c in ("source", "split") if c in base.columns]).merge(info, on="text_id")
    base["condition"] = "none"

    models = list(RULE_FREE) + ["shieldgemma", args.team]
    thr = thresholds_on_dev(pd.concat([j[j.model.isin(models)], base], ignore_index=True), criterion=args.thr_by)
    for kv in args.thr:
        m, v = kv.split("=")
        for key in [k for k in thr if k[1] == m and k[2] in ("base", "none")]:
            thr[key] = float(v)

    j = j[j.split == args.split]
    wide, dropped = build_wide(j, base[base.split == args.split], rs, models, thr, args.team)
    REPORTS.mkdir(parents=True, exist_ok=True)

    res = []
    for src, w in wide.groupby("source"):
        d = diagnose(w, ["shieldgemma", args.team], k=args.k)
        d.insert(1, "source", src)
        res.append(d)
    diag = pd.concat(res, ignore_index=True)
    diag = diag.merge(wide[["text_id", "human_label", "gold_rank", "gold_leaf", "top1", "found_leaves"]], on="text_id")
    diag["found_leaves"] = diag.found_leaves.str.split(";").str[:args.k].str.join(";")   # 상위 k만
    diag["gold_rule"] = diag.gold_leaf.map(names)
    diag["top1_rule"] = diag.top1.map(names)
    for c in ["base", "excluded", "rule_only", "rule_example"]:          # 진단 팝업용: 그 모델의 조건별 확률
        diag[f"p_{c}"] = [wide.set_index("text_id").get(f"p_{m}_{c}", pd.Series(dtype=float)).get(t, np.nan)
                          if m not in RULE_FREE else np.nan for t, m in zip(diag.text_id, diag.model)]
    diag["p_native"] = [wide.set_index("text_id")[f"p_{m}"].get(t) if m in RULE_FREE else np.nan
                        for t, m in zip(diag.text_id, diag.model)]
    diag.to_csv(REPORTS / "diagnosis.csv", index=False)

    # k를 바꾸면 원인별 건수가 어떻게 바뀌는지 (뷰의 '기준값 바꾸기'를 미리 계산)
    ks = []
    for k in [1, 3, 5]:
        for src, w in wide.groupby("source"):
            d = diagnose(w, ["shieldgemma", args.team], k=k)
            ks.append(d.groupby("cause_name").size().rename("건수").reset_index().assign(source=src, k=k))
    ks = pd.concat(ks).pivot_table(index=["source", "k"], columns="cause_name", values="건수", fill_value=0)
    ks = ks.reindex(columns=[c for c in CAUSES.values() if c in ks.columns]).astype(int)
    ks.to_csv(REPORTS / "diagnosis_k.csv")

    out = ["# 잘못된 판정 원인 (P3)", "",
           f"분할 {args.split}, 팀 판정기 {args.team}, '규칙 찾기 문제' 기준 상위 k = {args.k}. "
           f"원인 규칙은 위에서부터 차례로 적용해 하나만 고름.", ""]
    if dropped:
        out += [f"모델 판정이 다 있지 않은 텍스트 {dropped}건은 뺌.", ""]
    thr_rows = [{"source": s, "model": m, "condition": c, "threshold": round(v, 4)} for (s, m, c), v in thr.items()
                if m in models]
    for src, d in diag.groupby("source"):
        n_text = int((wide.source == src).sum())
        out += [f"## {src} (텍스트 {n_text}건)", ""]
        tab = d.pivot_table(index="cause_name", columns="model", values="text_id", aggfunc="count", fill_value=0)
        tab = tab.reindex([c for c in CAUSES.values() if c in tab.index]).reindex(columns=models).fillna(0).astype(int)
        tab.loc["잘못된 판정 합계"] = tab.sum()
        share = (tab.iloc[:-1] / tab.iloc[-1].replace(0, np.nan) * 100).round(1)
        show = tab.astype(str)
        show.iloc[:-1] = tab.iloc[:-1].astype(str) + " (" + share.astype(str) + "%)"
        out += ["원인별 잘못된 판정 건수 (비율 = 그 모델의 잘못된 판정 중)", "",
                md_table(show.reset_index().rename(columns={"cause_name": "원인"})), ""]
        why = d.groupby(["cause_name", "why"]).size().rename("건수").reset_index()
        out += ["원인이 정해진 이유", "", md_table(why.rename(columns={"cause_name": "원인", "why": "이유"})), ""]
        out += ["k에 따른 원인별 건수 (모델 합계)", "", md_table(ks.loc[src].reset_index()), ""]
        lc = d[d.all_models_wrong].text_id.nunique()
        out += [f"모든 모델이 정답 규칙으로도 틀린 텍스트(라벨 확인 후보): {lc}건. "
                "규칙을 받는 모델은 순서상 1번(규칙 찾기)이 먼저 정해질 수 있어 `all_models_wrong` 열로 따로 표시", ""]
    out += [f"## 사용한 임계값 (dev {args.thr_by} 최대)", "", md_table(pd.DataFrame(thr_rows)), ""]
    (REPORTS / "diagnosis_summary.md").write_text("\n".join(out), encoding="utf-8")
    pd.DataFrame(thr_rows).to_csv(REPORTS / "thresholds.csv", index=False)
    print("\n".join(out))
    print(f"\n저장: {REPORTS / 'diagnosis.csv'} ({len(diag)}행), {REPORTS / 'diagnosis_k.csv'}, "
          f"{REPORTS / 'diagnosis_summary.md'}, {REPORTS / 'thresholds.csv'}")


if __name__ == "__main__":
    main()
