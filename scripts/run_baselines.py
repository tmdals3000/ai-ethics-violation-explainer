"""기준 판정 모델 3종을 Spark 없이 단일 머신에서 실행 -> judgments_baseline.csv

판정 로직은 judges.py (spark_judge.py와 공통).

    python scripts/run_baselines.py --check          # 모델별 3건 원문 출력
    python scripts/run_baselines.py --limit 200
    python scripts/run_baselines.py                  # 전체
"""
import argparse
import gc
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from judges import MODEL_IDS, make_judge, policy_for

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=ROOT / "data/processed/texts_all.csv")
    ap.add_argument("--sources", nargs="+", default=["dna"], choices=["dna", "jigsaw", "prosocial"])
    ap.add_argument("--out", default=ROOT / "data/processed/judgments_baseline.csv")
    ap.add_argument("--models", nargs="+", default=list(MODEL_IDS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--controversial-weight", type=float, default=0.5)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    texts = pd.read_csv(args.texts, keep_default_na=False)
    texts = texts[texts.source.isin(args.sources)]
    texts = texts.sample(3, random_state=0) if args.check else texts.head(args.limit)

    rows = []
    for m in args.models:
        judge = make_judge(m, controversial_weight=args.controversial_weight)
        for _, row in tqdm(texts.iterrows(), total=len(texts), desc=m):
            policy, cond = policy_for(row["rule_text"], row["judge_mode"])
            p, raw = judge(row["context"], row["text"], policy, mode=row["judge_mode"])
            rows.append({"text_id": row["text_id"], "model": m,
                         "condition": cond if m == "shieldgemma" else "none", "p_violation": p})
            if args.check:
                print(f"[{m}] {row['text_id']} human={row['human_label']} p={p:.3f} raw={raw!r}")
        del judge
        gc.collect()
        torch.cuda.empty_cache()

    if not args.check:
        out = pd.DataFrame(rows)
        out.to_csv(args.out, index=False)
        print(f"저장: {args.out} ({len(out):,}행), 판정 실패 {out.p_violation.isna().sum()}건")


if __name__ == "__main__":
    main()
