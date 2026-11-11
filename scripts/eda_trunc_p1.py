"""DNA dev: 임베딩 입력(질문+응답)이 256 토큰에서 잘린 텍스트 vs 안 잘린 텍스트의 P1 top-1·top-3 (저장된 P1 결과만 사용).

  python scripts/eda_trunc_p1.py   → results/eda/dna_trunc_p1.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from src import data  # noqa: E402
from src.common import load_config  # noqa: E402
from src.retrieval import embed_input  # noqa: E402

MAX_SEQ = 256  # all-roberta-large-v1 max_seq_length
cfg = load_config()
texts, _, _, _ = data.load(cfg)
d = texts[(texts.source == "do_not_answer") & (texts.split == "dev")]
tok = AutoTokenizer.from_pretrained(cfg["retrieval"]["encoder"])
n_tok = [len(x) for x in tok([embed_input(cfg, r) for _, r in d.iterrows()]).input_ids]
d = d.assign(truncated=[n > MAX_SEQ for n in n_tok])
per = pd.read_csv(cfg["_res"] / "p1/retrieval_per_text.csv", keep_default_na=False)
m = d[["text_id", "truncated", "human_label"]].merge(per, on="text_id")
for c in ["hit1", "hit3", "hit1_mid", "hit3_mid"]:
    m[c] = m[c].astype(str).eq("True")
g = m.groupby("truncated").agg(n=("text_id", "size"), n_violation=("human_label", "sum"), top1_acc=("hit1", "mean"),
                               top3_acc=("hit3", "mean"), top1_acc_mid=("hit1_mid", "mean"), top3_acc_mid=("hit3_mid", "mean"),
                               top1_is_jigsaw=("top1", lambda s: s.str.startswith("J").mean())).round(4).reset_index()
g["truncated"] = g.truncated.map({True: f"잘림 (> {MAX_SEQ} 토큰)", False: f"안 잘림 (≤ {MAX_SEQ} 토큰)"})
g.to_csv(cfg["_res"] / "eda/dna_trunc_p1.csv", index=False)
print(g.to_string(index=False))
