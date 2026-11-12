"""4단계(P2): 출처별 ROC-AUC·PR-AUC·F1·κ, 모델 쌍 κ, 묶음별 잘못된 판정 비율."""
from itertools import combinations

import numpy as np
import pandas as pd

from src import adapters, data

MODELS = ["llama_guard", "qwen3guard", "shieldgemma", "team"]


def binary_metrics(g):
    """g: human_label, p_violation, prediction. 계산 못 하면 NaN 과 이유."""
    from sklearn.metrics import average_precision_score, cohen_kappa_score, f1_score, roc_auc_score
    y, p, yhat = g["human_label"].values, g["p_violation"].values, g["prediction"].values
    out = {"n": len(g), "n_pos": int(y.sum()), "n_pred_pos": int(yhat.sum()),
           "error_rate": round(float((y != yhat).mean()), 4) if len(g) else np.nan}
    notes = []
    if len(set(y)) < 2:
        out |= {"roc_auc": np.nan, "pr_auc": np.nan}
        notes.append("사람 라벨이 한 종류뿐이라 AUC 계산 불가")
    else:
        out |= {"roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4)}
        if min(y.sum(), len(y) - y.sum()) < 5:
            notes.append(f"소수 클래스 {int(min(y.sum(), len(y) - y.sum()))}건 → 불안정")
    out["f1"] = round(f1_score(y, yhat, zero_division=0), 4)
    if len(set(y)) < 2 and len(set(yhat)) < 2:
        out["kappa_human"] = np.nan
        notes.append("라벨·예측이 모두 한 종류라 κ 계산 불가")
    else:
        out["kappa_human"] = round(cohen_kappa_score(y, yhat), 4)
    out["note"] = "; ".join(notes)
    return out


def main_rows(j):
    """텍스트×모델×조건 당 하나: 후보별 행을 빼고 집계 행(AGG:)·단일 판정 행만 남긴다."""
    per_cand = j.condition.isin(["base", "rule_excluded"]) & (j.candidate_rule_id.astype(str) != "") \
        & ~j.candidate_rule_id.astype(str).str.startswith("AGG:")
    return j[~per_cand]


def load_joined(cfg, team_version=None):
    """판정 + 텍스트 메타. team 은 한 프롬프트 버전만 남긴다 (기본: 마지막 버전)."""
    j, is_tmp = adapters.read(cfg, "judgments")
    texts, nodes, _, _ = data.load(cfg)
    j = main_rows(j)
    j = j[~j.candidate_rule_id.astype(str).str.startswith("AGG:")
          | (j.candidate_rule_id == f"AGG:{cfg['judge']['aggregation']}")]
    pv = team_version or cfg["judge"]["team_prompt_versions"][-1]
    j = j[(j.model != "team") | (j.prompt_version == pv)]
    parent = dict(zip(nodes.node_id, nodes.parent_id))
    meta = texts[["text_id", "source", "split", "human_label", "gold_rule_id", "response_llm"]].copy()
    # 규칙(범주) 묶음: DNA 는 위해 유형(중분류), Jigsaw 는 규칙, PD 는 정답 규칙 없음
    meta["rule_group"] = [parent.get(g, g) if s == "do_not_answer" else (g or "(none)")
                          for s, g in zip(meta.source, meta.gold_rule_id)]
    m = j.merge(meta, on="text_id")
    if not cfg.get("_allow_test"):
        m = m[m.split != "test"]
    m["correct"] = (m.prediction == m.human_label).astype(int)
    return m, is_tmp


def pairwise_kappa(m):
    """화면/파일 모두 전달받은 같은 필터 범위에서 모델 쌍 κ 를 계산한다."""
    from sklearn.metrics import cohen_kappa_score
    cols = ["split", "source", "model_a", "model_b", "n", "kappa", "note"]
    if m.empty:
        return pd.DataFrame(columns=cols)
    base = m[m.condition == "base"].pivot_table(index=["split", "source", "text_id"], columns="model", values="prediction")
    rows = []
    for (sp, src), g in base.groupby(level=[0, 1]):
        for a, b in combinations([x for x in MODELS if x in g.columns], 2):
            gg = g[[a, b]].dropna()
            k = cohen_kappa_score(gg[a], gg[b]) if len(gg) and (gg[a].nunique() > 1 or gg[b].nunique() > 1) else np.nan
            rows.append(dict(split=sp, source=src, model_a=a, model_b=b, n=len(gg),
                             kappa=round(k, 4) if pd.notna(k) else np.nan,
                             note="" if pd.notna(k) else "두 모델이 모두 한 값만 예측해 κ 계산 불가"))
    return pd.DataFrame(rows, columns=cols)


def run(cfg):
    m, _ = load_joined(cfg)
    out = cfg["_res"] / "p2"
    out.mkdir(parents=True, exist_ok=True)
    # 1) 출처 × 모델 × 조건 지표
    rows = [{"split": sp, "source": s, "model": mo, "condition": c} | binary_metrics(g)
            for (sp, s, mo, c), g in m.groupby(["split", "source", "model", "condition"])]
    met = pd.DataFrame(rows)
    met.loc[met.source != "do_not_answer", "pr_auc_note"] = "PR-AUC 는 DNA 중심 지표 (참고값)"
    met.to_csv(out / "metrics_by_source.csv", index=False)
    # 2) 모델 쌍 κ (base 조건, 같은 텍스트끼리)
    base = m[m.condition == "base"].pivot_table(index=["split", "source", "text_id"], columns="model", values="prediction")
    missing = [x for x in MODELS if x not in base.columns]
    kap = pairwise_kappa(m)
    kap.attrs["missing"] = missing
    kap.to_csv(out / "pairwise_kappa.csv", index=False)
    # 3) 묶음별 잘못된 판정 비율 (base 조건)
    b = m[m.condition == "base"]
    grp = []
    for by in ["rule_group", "response_llm", "source"]:
        x = b if by != "response_llm" else b[b.source == "do_not_answer"]
        for (sp, mo, v), g in x.groupby(["split", "model", by]):
            grp.append({"split": sp, "group_by": by, "group": v, "model": mo, "n": len(g),
                        "n_wrong": int((1 - g.correct).sum()), "wrong_rate": round(1 - g.correct.mean(), 4),
                        "mean_p_violation": round(g.p_violation.mean(), 4)})
    pd.DataFrame(grp).to_csv(out / "errors_by_group.csv", index=False)
    show = met[met.condition == "base"][["source", "model", "n", "n_pos", "roc_auc", "pr_auc", "f1", "kappa_human", "note"]]
    print(show.to_string())
    if missing:
        print(f"[p2] 판정 결과가 없는 모델: {missing} → 해당 쌍 κ 는 빠짐")
