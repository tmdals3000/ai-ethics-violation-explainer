"""실험 조건별 비교 + 처음 보는 규칙의 AUC 하락폭·회복폭 + 팀 판정기 버전별 dev 지표 기록

PDF 2주차: "실험 조건별 비교: 규칙 제외 → 규칙 문장만 → 규칙 문장 + 예시 순서로 봅니다."
PDF P4 지표
  처음 보는 규칙의 AUC 하락폭 = 정답 규칙을 뺀 텍스트에서 '기본' AUC − '규칙 제외' AUC
  회복폭 = '규칙 문장만' AUC − '규칙 제외' AUC,  '규칙 문장 + 예시' AUC − '규칙 제외' AUC
  실험 규칙: Jigsaw 규칙 2개, Do-Not-Answer 위해 유형(중분류) 12개
  → 규칙마다 '그 규칙이 정답인 텍스트'만 모아 네 조건의 AUC를 계산 (그 텍스트들에서는 그 규칙이 트리에서 빠짐)

출력
  reports/conditions_metrics.csv   출처·모델·조건별 ROC-AUC, PR-AUC, F1, κ (임계값은 조건마다 dev κ 최대, --thr-by f1 가능)
  reports/unseen_rules.csv         실험 규칙 × 모델별 네 조건 AUC, 하락폭, 회복폭 (건수 포함)
  reports/team_versions.csv        팀 판정기 버전별 dev 지표 기록 (실행할 때마다 쌓임, 같은 결과는 한 번만)
  reports/conditions_summary.md

    python scripts/compare_conditions.py
    python scripts/compare_conditions.py --baseline data/processed/judgments_baseline_dev300s.csv
"""
import argparse
import glob
from datetime import datetime

import pandas as pd

from analysis import auc_or_nan, degenerate, source_metrics, thresholds_on_dev
from common import PROCESSED, REPORTS, need
from data_overview import md_table

ORDER = ["base", "excluded", "rule_only", "rule_example"]
KO = {"base": "기본", "excluded": "규칙 제외", "rule_only": "규칙 문장만", "rule_example": "규칙 문장+예시",
      "none": "(규칙 없음)"}
MIN_POS = 5          # 위반이나 정상이 이보다 적은 칸은 '참고'로 표시 (PDF: 건수가 적은 칸은 흐리게)


def load_baseline(files, ids):
    files = files or sorted(glob.glob(str(PROCESSED / "judgments_baseline_*.csv")))
    if not files:
        return pd.DataFrame()
    b = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    b = b[b.model.isin(["llamaguard", "qwen3guard"]) & b.text_id.isin(ids)]
    return b.drop_duplicates(["text_id", "model"], keep="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="*", default=[], help="Llama Guard·Qwen3Guard 판정 파일 (기본: judgments_baseline_*.csv 전부)")
    ap.add_argument("--thr-by", default="kappa", choices=["kappa", "f1"],
                    help="임계값 기준 (기본 kappa: F1 최대가 '전부 위반'으로 무너지는 것을 막음)")
    ap.add_argument("--note", default="", help="버전 기록에 남길 메모 (예: '예시 2개씩')")
    args = ap.parse_args()

    need("judgments_conditions.csv", "rule_search.csv", "rule_tree.csv")
    j = pd.read_csv(PROCESSED / "judgments_conditions.csv").dropna(subset=["p_violation"])
    rs = pd.read_csv(PROCESSED / "rule_search.csv", usecols=["text_id", "excl_unit"])
    names = pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False).set_index("node_id").name
    info = j.drop_duplicates("text_id")[["text_id", "source", "split", "human_label"]]
    base = load_baseline(args.baseline, set(info.text_id))
    if len(base):
        base = base.drop(columns=[c for c in ("source", "split") if c in base.columns]).merge(
            info.drop(columns="human_label"), on="text_id")
        base["condition"] = "none"
    allj = pd.concat([j, base[j.columns.intersection(base.columns)]], ignore_index=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # 1) 조건별 지표 (조건마다 dev 임계값)
    thr = thresholds_on_dev(allj, criterion=args.thr_by)
    degen = degenerate(allj, thr)
    met = source_metrics(allj, thr)
    met["condition"] = pd.Categorical(met.condition, ORDER + ["none"], ordered=True)
    met = met.sort_values(["source", "split", "condition", "model"])
    met.round(4).to_csv(REPORTS / "conditions_metrics.csv", index=False)

    # 2) 처음 보는 규칙: 실험 규칙(정답 중분류)마다 네 조건 AUC
    jj = j.merge(rs, on="text_id")
    rows = []
    for (src, split, unit, model), g in jj.groupby(["source", "split", "excl_unit", "model"]):
        w = g.pivot_table(index="text_id", columns="condition", values="p_violation")
        y = g.drop_duplicates("text_id").set_index("text_id").human_label.reindex(w.index)
        r = {"source": src, "split": split, "rule": unit, "rule_name": names.get(unit, unit), "model": model,
             "n": len(w), "violations": int(y.sum())}
        for c in ORDER:
            r[f"auc_{c}"] = auc_or_nan(w[c], y) if c in w else float("nan")
        rows.append(r)
    # 전체 (텍스트마다 자기 정답 중분류를 뺀 것을 모두 모음)
    for (src, split, model), g in jj.groupby(["source", "split", "model"]):
        w = g.pivot_table(index="text_id", columns="condition", values="p_violation")
        y = g.drop_duplicates("text_id").set_index("text_id").human_label.reindex(w.index)
        r = {"source": src, "split": split, "rule": "(전체)", "rule_name": "(전체)", "model": model,
             "n": len(w), "violations": int(y.sum())}
        for c in ORDER:
            r[f"auc_{c}"] = auc_or_nan(w[c], y) if c in w else float("nan")
        rows.append(r)
    un = pd.DataFrame(rows)
    un["drop"] = un.auc_base - un.auc_excluded                        # 하락폭
    un["recover_rule"] = un.auc_rule_only - un.auc_excluded           # 회복폭 (규칙 문장만)
    un["recover_example"] = un.auc_rule_example - un.auc_excluded     # 회복폭 (규칙 문장 + 예시)
    un["small"] = (un.violations < MIN_POS) | (un.n - un.violations < MIN_POS)
    un.round(4).to_csv(REPORTS / "unseen_rules.csv", index=False)

    # 3) 팀 판정기 버전 기록 (dev)
    t = met[(met.split == "dev") & met.model.astype(str).str.startswith(("team_", "shieldgemma"))].copy()
    t.insert(0, "run_at", datetime.now().strftime("%Y-%m-%d %H:%M"))
    t["variant"] = t.model.astype(str).replace({"shieldgemma": "team_v0"})
    t["condition"] = t.condition.astype(str)
    t["note"] = (args.note + " " if args.note else "") + f"thr={args.thr_by}"
    keep = ["run_at", "variant", "source", "condition", "split", "n", "violations", "roc_auc", "pr_auc",
            "threshold", "f1", "precision", "recall", "kappa_human", "note"]
    t = t[keep].round(4)
    logf = REPORTS / "team_versions.csv"
    if logf.exists():
        old = pd.read_csv(logf, keep_default_na=False)
        sig = ["variant", "source", "condition", "split", "n", "roc_auc", "f1", "note"]
        key = lambda d: d[sig].apply(lambda col: col.map(lambda v: "" if pd.isna(v) else str(v))).agg("|".join, axis=1)
        t = t[~key(t).isin(key(old))]
        t = pd.concat([old, t], ignore_index=True)
    t.to_csv(logf, index=False)

    # 요약
    out = ["# 실험 조건별 비교 (dev)", "",
           f"임계값은 출처·모델·조건마다 dev에서 {'κ' if args.thr_by == 'kappa' else 'F1'}가 최대인 값. "
           "Llama Guard·Qwen3Guard는 규칙 문장을 받지 않아 조건이 없음.", ""]
    if degen:
        out += ["**주의: 임계값이 dev의 95% 이상을 한쪽으로 판정하는 묶음** (F1·κ 해석 불가)", "",
                md_table(pd.DataFrame([{"source": k[0], "model": k[1], "condition": k[2], "위반 판정 비율": round(r, 3)}
                                       for k, r in degen])), ""]
    for src, g in met[met.split == "dev"].groupby("source"):
        tab = g.pivot_table(index="model", columns="condition", values="roc_auc", observed=True)
        tab.columns = [KO[str(c)] for c in tab.columns]
        out += [f"## {src}: 조건별 ROC-AUC (텍스트 {int(g.n.max())}건)", "", md_table(tab.round(3).reset_index()), ""]
        tab = g.pivot_table(index="model", columns="condition", values="f1", observed=True)
        tab.columns = [KO[str(c)] for c in tab.columns]
        out += ["F1", "", md_table(tab.round(3).reset_index()), ""]
    show = un[un.split == "dev"].copy()
    for src, g in show.groupby("source"):
        g = g.assign(규칙=g.rule_name.str.slice(0, 40) + g.small.map({True: " (건수 적음)", False: ""}))
        cols = ["규칙", "model", "n", "violations", "auc_base", "auc_excluded", "auc_rule_only", "auc_rule_example",
                "drop", "recover_rule", "recover_example"]
        out += [f"## {src}: 처음 보는 규칙 (하락폭 = 기본 − 규칙 제외, 회복폭 = 각 조건 − 규칙 제외)", "",
                md_table(g[cols].round(3)), ""]
    out += [f"버전 기록: `reports/team_versions.csv` ({len(t)}행)"]
    (REPORTS / "conditions_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:40]))
    print(f"\n저장: {REPORTS / 'conditions_metrics.csv'}, {REPORTS / 'unseen_rules.csv'}, {logf}, "
          f"{REPORTS / 'conditions_summary.md'}")


if __name__ == "__main__":
    main()
