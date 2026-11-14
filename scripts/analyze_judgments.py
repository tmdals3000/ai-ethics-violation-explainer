"""판정 결과 → 분석용 변수 + 집계표 (PDF 1주차)

    python scripts/analyze_judgments.py --judgments data/processed/judgments_baseline_dev300s.csv data/processed/judgments_team_dev300s.csv
    python scripts/analyze_judgments.py --judgments ... --models llamaguard qwen3guard shieldgemma team_v3

만드는 것 (reports/ 폴더):
  analysis_vars.csv          텍스트 1행. 모델별 위반 확률·판정·정답 여부 + 아래 변수
      n_wrong               틀린 모델 수
      models_disagree       모델 간 불일치 여부 (판정이 하나라도 다르면 True)
      label_check_candidate 모든 모델이 같은 판정인데 사람 라벨만 다름 (P3 '라벨 확인 필요'의 재료)
  table_models.csv           출처·분할·모델별 성능 (ROC-AUC, PR-AUC, F1, 정밀도, 재현율, κ)
  table_rule_by_model.csv    규칙(범주) × 모델: 건수, 위반 수, 놓친 위반(FN), 잘못 잡은 정상(FP), 오판율
  table_responder_by_model.csv   응답 LLM × 모델 (Do-Not-Answer만)
  table_kappa_pairs.csv      모델 쌍별 Cohen's κ (모델 4개면 6쌍)
  judgment_summary.md        위 표를 읽기 좋게 모은 보고서

규칙:
  - 임계값은 모델·출처마다 dev에서 F1이 최대인 값 (test에도 그 값을 씀)
  - 공정한 비교를 위해 선택한 모델이 모두 판정한 텍스트만 사용
  - test는 --include-test 를 줘야 계산 (PDF: test 지표는 5단계에서 처음 확인)
  - 'rule_search.csv'(rule_search.py 결과)가 있으면 찾은 규칙 경로와 정답 경로의 일치 여부,
    처음 갈라진 단계, 가지별 우선순위(l2_rank_of_gold), 경계 사례 여부도 붙임
"""
import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from common import PROCESSED, REPORTS, best_f1_threshold, binary_metrics, confusion
from data_overview import SRC, md_table

LOW_N = 20          # 이보다 건수가 적은 묶음은 표에 표시 (PDF: 건수가 적은 묶음은 건수와 함께 볼 것)


def load_texts() -> pd.DataFrame:
    p = PROCESSED / "texts_all.csv"
    if not p.exists():
        raise SystemExit("texts_all.csv가 없습니다. python scripts/combine_texts.py 를 먼저 실행하세요.")
    t = pd.read_csv(p, keep_default_na=False)
    t["human_label"] = t["human_label"].astype(int)
    return t


def group_table(d: pd.DataFrame, models, key: str) -> pd.DataFrame:
    rows = []
    for (src, split, grp), g in d.groupby(["source", "split", key]):
        if grp == "":
            continue
        for m in models:
            tp, fp, fn, tn = confusion(g[f"pred_{m}"], g["human_label"])
            n = len(g)
            rows.append({"source": src, "split": split, key: grp, "model": m, "n": n,
                         "violations": int(g.human_label.sum()), "fn": fn, "fp": fp,
                         "wrong": fn + fp, "error_rate": round((fn + fp) / n, 4), "low_n": n < LOW_N})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=None, help="비교할 모델 (기본: 파일에 있는 전부)")
    ap.add_argument("--include-test", action="store_true", help="최종 평가 단계에서만")
    args = ap.parse_args()

    texts = load_texts()
    j = pd.concat([pd.read_csv(f)[["text_id", "model", "p_violation"]] for f in args.judgments],
                  ignore_index=True).drop_duplicates(["text_id", "model"], keep="last")
    models = args.models or sorted(j["model"].unique())
    missing = [m for m in models if m not in set(j["model"])]
    if missing:
        raise SystemExit(f"판정 파일에 없는 모델: {missing} (있는 모델: {sorted(j['model'].unique())})")

    wide = j[j.model.isin(models)].pivot(index="text_id", columns="model", values="p_violation")
    n_any = len(wide)
    wide = wide.dropna(subset=models)                      # 모든 모델이 판정한 텍스트만
    d = texts.merge(wide.add_prefix("p_"), left_on="text_id", right_index=True, how="inner")
    print(f"판정이 하나라도 있는 텍스트 {n_any:,}건 → 모델 {len(models)}개가 모두 판정한 {len(d):,}건 사용")
    if not args.include_test and (d.split == "test").any():
        print(f"※ test {int((d.split == 'test').sum())}건은 제외 (--include-test 를 주면 포함)")
        d = d[d.split != "test"]
    d = d[d.split.isin(["dev", "test"])].copy()
    if d.empty:
        raise SystemExit("dev/test에 해당하는 판정이 없습니다.")

    # ---- 임계값: 모델·출처마다 dev에서 F1 최대 ----
    thr = {}
    for src in d.source.unique():
        dev = d[(d.source == src) & (d.split == "dev")]
        for m in models:
            thr[(src, m)] = best_f1_threshold(dev[f"p_{m}"], dev.human_label) if len(dev) else 0.5
            if not len(dev):
                print(f"※ {src}에 dev 판정이 없어 {m} 임계값을 0.5로 둠")

    # ---- 분석용 변수 ----
    for m in models:
        t_m = d.source.map(lambda s: thr[(s, m)])
        d[f"pred_{m}"] = (d[f"p_{m}"] >= t_m).astype(int)
        d[f"correct_{m}"] = d[f"pred_{m}"] == d.human_label
    preds = d[[f"pred_{m}" for m in models]].to_numpy()
    corr = d[[f"correct_{m}" for m in models]].to_numpy()
    d["n_models"] = len(models)
    d["n_wrong"] = (~corr).sum(axis=1)
    d["any_wrong"] = d.n_wrong > 0
    d["all_wrong"] = d.n_wrong == len(models)
    d["models_disagree"] = preds.min(axis=1) != preds.max(axis=1)
    d["label_check_candidate"] = (~d.models_disagree) & d.all_wrong if len(models) >= 2 else False
    d["p_spread"] = d[[f"p_{m}" for m in models]].max(axis=1) - d[[f"p_{m}" for m in models]].min(axis=1)

    rs_cols = []
    rs_path = PROCESSED / "rule_search.csv"
    if rs_path.exists():                                   # 규칙 찾기 결과가 있으면 경로 변수 추가
        rs = pd.read_csv(rs_path, keep_default_na=False)
        rs_cols = ["top1", "top1_correct", "top3_correct", "l1_match", "l2_match", "l3_match",
                   "first_divergence", "gold_lost_at", "l2_rank_of_gold", "boundary"]
        d = d.merge(rs[["text_id"] + rs_cols].rename(columns={"top1": "found_rule"}), on="text_id", how="left")
        rs_cols = ["found_rule"] + rs_cols[1:]
    else:
        print("※ rule_search.csv 없음 → 경로 일치 여부·가지별 우선순위 변수는 빠짐 (rule_search.py 먼저 실행)")

    REPORTS.mkdir(parents=True, exist_ok=True)
    keep = (["text_id", "source", "split", "human_label", "rule_l1", "rule_l2", "gold_rule", "responder",
             "label_raw"] + [f"{k}_{m}" for m in models for k in ("p", "pred", "correct")]
            + ["n_models", "n_wrong", "any_wrong", "all_wrong", "models_disagree",
               "label_check_candidate", "p_spread"] + rs_cols)
    d[keep].to_csv(REPORTS / "analysis_vars.csv", index=False)

    # ---- 집계표 ----
    rows = []
    for (src, split), g in d.groupby(["source", "split"]):
        for m in models:
            rows.append({"source": src, "split": split, "model": m,
                         **binary_metrics(g[f"p_{m}"], g.human_label, thr[(src, m)])})
    t_models = pd.DataFrame(rows)
    t_rule = group_table(d, models, "rule_l2")
    t_resp = group_table(d[d.source == "dna"], models, "responder")
    rows = []
    for (src, split), g in d.groupby(["source", "split"]):
        for a, b in combinations(models, 2):
            pa, pb = g[f"pred_{a}"], g[f"pred_{b}"]
            k = cohen_kappa_score(pa, pb) if pa.nunique() == 2 and pb.nunique() == 2 else np.nan
            rows.append({"source": src, "split": split, "model_a": a, "model_b": b, "kappa": k,
                         "agree_rate": float((pa == pb).mean()), "n": len(g)})
    t_kappa = pd.DataFrame(rows)

    t_models.round(4).to_csv(REPORTS / "table_models.csv", index=False)
    t_rule.to_csv(REPORTS / "table_rule_by_model.csv", index=False)
    t_resp.to_csv(REPORTS / "table_responder_by_model.csv", index=False)
    t_kappa.round(4).to_csv(REPORTS / "table_kappa_pairs.csv", index=False)

    # ---- 읽기 좋은 보고서 ----
    f3 = lambda v: "-" if pd.isna(v) else f"{v:.3f}"
    out = ["# 판정 집계 보고서", "",
           "판정 파일: " + ", ".join(Path(f).name for f in args.judgments), "",
           f"모델 {len(models)}개가 모두 판정한 텍스트만 사용했습니다. 임계값은 모델·출처마다 dev에서 F1이 최대인 값입니다.", ""]
    for (src, split), g in d.groupby(["source", "split"]):
        pos = int(g.human_label.sum())
        out += [f"## {SRC.get(src, src)} · {split} ({len(g):,}건, 위반 {pos}건, {pos / len(g):.1%})", ""]
        tm = t_models[(t_models.source == src) & (t_models.split == split)]
        out += ["### 모델별 성능", "", md_table(pd.DataFrame({
            "모델": tm.model, "ROC-AUC": tm.roc_auc.map(f3), "PR-AUC": tm.pr_auc.map(f3),
            "임계값": tm.threshold.map(f3), "F1": tm.f1.map(f3), "정밀도": tm.precision.map(f3),
            "재현율": tm.recall.map(f3), "사람 라벨과 κ": tm.kappa_human.map(f3),
            "놓친 위반": tm.fn, "잘못 잡은 정상": tm.fp})), ""]
        tk = t_kappa[(t_kappa.source == src) & (t_kappa.split == split)]
        if len(tk):
            out += [f"### 모델 간 일치도 ({len(tk)}쌍)", "", md_table(pd.DataFrame({
                "모델 A": tk.model_a, "모델 B": tk.model_b, "κ": tk.kappa.map(f3),
                "같은 판정 비율": tk.agree_rate.map("{:.1%}".format)})), ""]
        for tbl, key, title in [(t_rule, "rule_l2", "규칙(범주) × 모델 오판 수"),
                                (t_resp, "responder", "응답 LLM × 모델 오판 수")]:
            x = tbl[(tbl.source == src) & (tbl.split == split)] if len(tbl) else tbl
            if len(x):
                x = x.assign(cell=[f"{w} ({e:.0%})" for w, e in zip(x.wrong, x.error_rate)])
                pv = x.pivot(index=key, columns="model", values="cell")[models]
                info = x.drop_duplicates(key).set_index(key)[["n", "violations"]]
                pv.insert(0, "위반/건수", [f"{info.loc[i, 'violations']} / {info.loc[i, 'n']}"
                                        + (" ⚠" if info.loc[i, "n"] < LOW_N else "") for i in pv.index])
                out += [f"### {title}", "", md_table(pv.reset_index().rename(columns={key: "묶음"})), "",
                        f"칸 = 오판 수 (오판율). ⚠ = {LOW_N}건 미만이라 해석 주의.", ""]
        out += ["### 분석용 변수 요약", "",
                f"- 하나라도 틀린 텍스트: {int(g.any_wrong.sum())}건",
                f"- 모델 간 판정이 엇갈린 텍스트: {int(g.models_disagree.sum())}건",
                f"- 모든 모델이 같은 판정인데 사람 라벨만 다른 텍스트(라벨 확인 후보): "
                f"{int(g.label_check_candidate.sum())}건", ""]
    (REPORTS / "judgment_summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    print(f"\n저장: {REPORTS}  (analysis_vars.csv, table_*.csv, judgment_summary.md)")


if __name__ == "__main__":
    main()
