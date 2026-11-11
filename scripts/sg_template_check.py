"""ShieldGemma 응답 판정: 현재 문자열 템플릿 vs 토크나이저 내장 공식 chat template. DNA dev 위반 65 + 정상 65."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, cohen_kappa_score
from src.common import load_config
from src import data, judges
from src.retrieval import targets
cfg = load_config()
texts, nodes, ex, sample = data.load(cfg)
tgt = targets(cfg, texts, sample)
d = tgt[tgt.source == "do_not_answer"]
pick = pd.concat([g.sample(65, random_state=42) for _, g in d.groupby("human_label")])
per = pd.read_csv(cfg["_res"] / "p1/retrieval_per_text.csv", keep_default_na=False)
items, _ = judges.work_items(cfg, pick, nodes, per, ex)
items = [it for it in items if it["condition"] in ("base", "rule_only")]
hf = cfg["judge"]["models"]["shieldgemma"]["hf_id"]
cache = judges.Cache(cfg["_data"] / "cache/judgments.sqlite")

class SGChat(judges.ShieldGemma):
    add_special_tokens = False  # 템플릿 결과에 <bos> 포함
    def build(self, it):
        chat = [{"role": "user", "content": judges._clip(it["context"])},
                {"role": "assistant", "content": judges._clip(it["text"])}]
        return self.tok.apply_chat_template(chat, guideline=it["rule"], tokenize=False)

sg = judges.ShieldGemma(cfg, hf)
chat = SGChat.__new__(SGChat); chat.__dict__.update(sg.__dict__)
print("example official prompt:\n", chat.build(items[0])[:900])
res = {}
for name, jd, key in [("current", sg, f"{hf}|shieldgemma|official"), ("chat_template", chat, f"{hf}|shieldgemma|official_chat")]:
    t = time.time()
    p, n_new = judges.score_cached(jd, cache, key, [jd.build(it) for it in items])
    print(name, "new forwards", n_new, f"{time.time()-t:.0f}s")
    res[name] = p
sg.close()
df = pd.DataFrame(items)[["text_id", "condition", "candidate_rule_id"]]
lab = dict(zip(pick.text_id, pick.human_label))
rows = []
for name, p in res.items():
    x = df.assign(p=p)
    for cond in ["base", "rule_only"]:
        g = x[x.condition == cond].groupby("text_id", as_index=False).p.max()
        y = g.text_id.map(lab); pr = (g.p >= 0.5).astype(int)
        rows.append(dict(template=name, condition=cond, n=len(g), roc_auc=roc_auc_score(y, g.p), pr_auc=average_precision_score(y, g.p),
                         f1=f1_score(y, pr), kappa=cohen_kappa_score(y, pr), pred_pos=int(pr.sum())))
r = pd.DataFrame(rows).round(3)
print(r.to_string(index=False))
a, b = res["current"], res["chat_template"]
print("corr(p) =", round(np.corrcoef(a, b)[0, 1], 4), " mean|diff| =", round(np.abs(a - b).mean(), 4),
      " decision flips@0.5 =", int(((a >= .5) != (b >= .5)).sum()), "/", len(a))
r.to_csv(cfg["_res"] / "trial/sg_template_check.csv", index=False)
