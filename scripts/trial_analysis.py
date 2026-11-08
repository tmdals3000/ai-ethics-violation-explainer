"""시험 실행(results/trial) + 규칙 없는 모델 본 결과를 같은 텍스트로 묶어 P2·κ·P3 를 계산한다 (출처별 50건)."""
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import load_config  # noqa: E402

cfg = load_config()
import pandas as pd  # noqa: E402
from sklearn.metrics import cohen_kappa_score  # noqa: E402

from src import adapters, data, diagnose, judges  # noqa: E402
from src.metrics import MODELS, binary_metrics, main_rows  # noqa: E402

texts, nodes, _, _ = data.load(cfg)
tr = pd.read_csv(cfg["_res"] / "trial/judgments.csv", keep_default_na=False)
ids = set(tr.text_id)
tr = pd.concat([tr, judges.aggregate(tr, cfg["judge"]["aggregation"])], ignore_index=True)
tr["threshold"] = cfg["judge"]["default_threshold"]
team = (tr.model == "team")
th = judges.choose_threshold(cfg, tr[team & (tr.condition == "base") & tr.candidate_rule_id.str.startswith("AGG:")], texts)
tr.loc[team, "threshold"] = th
main, _ = adapters.read(cfg, "judgments")
main = main[main.model.isin(["llama_guard", "qwen3guard"]) & main.text_id.isin(ids)]
j = main_rows(pd.concat([main, tr], ignore_index=True))
j["prediction"] = (j.p_violation >= j.threshold).astype(int)
parent = dict(zip(nodes.node_id, nodes.parent_id))
meta = texts[["text_id", "source", "split", "human_label", "gold_rule_id", "response_llm"]].copy()
meta["rule_group"] = [parent.get(g, g) if s == "do_not_answer" else (g or "(none)") for s, g in zip(meta.source, meta.gold_rule_id)]
m = j.merge(meta, on="text_id")
m["correct"] = (m.prediction == m.human_label).astype(int)
out = cfg["_res"] / "trial"
met = pd.DataFrame([{"source": s, "model": mo, "condition": c} | binary_metrics(g) for (s, mo, c), g in m.groupby(["source", "model", "condition"])])
met.to_csv(out / "metrics.csv", index=False)
b = m[m.condition == "base"].pivot_table(index=["source", "text_id"], columns="model", values="prediction")
kap = [{"source": s, "model_a": a, "model_b": c, "n": len(g), "kappa": round(cohen_kappa_score(g[a], g[c]), 4)}
       for s, g in b.groupby(level=0) for a, c in combinations(MODELS, 2)]
pd.DataFrame(kap).to_csv(out / "pairwise_kappa.csv", index=False)
per = pd.read_csv(cfg["_res"] / "p1/retrieval_per_text.csv", keep_default_na=False)
d = diagnose.diagnose(m, per, cfg["diagnose"]["retrieval_top_k"])
d.to_csv(out / "diagnosis.csv", index=False)
diagnose.summary(d).to_csv(out / "cause_summary.csv", index=False)
print(f"팀 판정기 임계값 (시험 표본 dev 에서 선택): {th}")
print(met[met.condition == "base"][["source", "model", "n", "n_pos", "n_pred_pos", "roc_auc", "pr_auc", "f1", "kappa_human", "note"]].to_string(index=False))
print(met[met.model.isin(["shieldgemma", "team"])].pivot_table(index=["source", "model"], columns="condition", values="roc_auc").round(3).to_string())
print(pd.DataFrame(kap).pivot_table(index=["model_a", "model_b"], columns="source", values="kappa").to_string())
print(pd.crosstab([d.source, d.model], d.cause, margins=True).to_string())
