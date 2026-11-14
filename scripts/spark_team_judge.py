"""Spark로 팀 판정기 프롬프트 변형을 같은 표본에 실행 → judgments_team_{split}{n}.csv

    python scripts/spark_team_judge.py --check                    # 3건, 변형별 확률 출력
    python scripts/spark_team_judge.py --n 300 --stratified       # dev: 위반 전부 + 정상 무작위
    python scripts/spark_team_judge.py --n 2808                   # dev 전체 (최종 비교용)

- ShieldGemma는 한 번만 로드하고, 텍스트마다 변형 수만큼 판정 (변형 4개 = 판정 4회)
- --stratified: 위반이 3.4%뿐이라 무작위 100건이면 위반이 3건 남짓 → 비교가 불가능.
  dev 위반을 전부 넣고 나머지를 정상으로 채움. ROC-AUC는 비율 영향을 받지 않지만
  PR-AUC·F1은 위반 비율이 높아져 실제보다 좋게 나오므로 '변형끼리 순위 비교'용으로만 쓸 것
- 같은 --seed면 기준 모델(spark_judge.py)과 같은 표본이 뽑히지 않을 수 있으니,
  기준 모델 비교는 판별 페이지가 공통 text_id만 골라 계산함
"""
import argparse
import os
import sys
import time

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import FloatType, StructField, StructType

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)
from team_judge import VARIANTS  # noqa: E402  (프롬프트 정의만 가져옴, 모델은 워커에서 로드)


def make_udf(variants, batch_size, check):
    schema = StructType([StructField(v, FloatType()) for v in variants])

    def make_predict_fn():
        sys.path.insert(0, SCRIPTS_DIR)
        import numpy as np
        from team_judge import make_team_judge
        judge = make_team_judge(variants)

        def predict(q, r, ra, th, sh):
            cols = {v: [] for v in variants}
            for a, b, c, d, e in zip(q, r, ra, th, sh):
                probs = judge(a, b, {"risk_area": c, "types_of_harm": d, "specific_harms": e})
                if check:
                    print("  " + "  ".join(f"{v}={probs[v]:.3f}" for v in variants), flush=True)
                for v in variants:
                    cols[v].append(probs[v])
            return {v: np.asarray(x, dtype=np.float32) for v, x in cols.items()}
        return predict

    return predict_batch_udf(make_predict_fn, return_type=schema, batch_size=batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--stratified", action="store_true", help="위반 전부 + 정상 무작위로 n건")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.check:
        args.n, args.stratified = 3, False

    spark = (SparkSession.builder.master("local[1]").appName("team-judge")
             .config("spark.python.worker.reuse", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
          .csv(os.path.join(ROOT, "data/processed/texts_dna.csv"))
          .filter(F.col("split") == args.split))
    if args.stratified:
        pos = df.filter(F.col("human_label") == "1")
        n_pos = pos.count()
        neg = df.filter(F.col("human_label") == "0").orderBy(F.rand(args.seed)).limit(max(args.n - n_pos, 0))
        sample = pos.unionByName(neg)
    else:
        sample = df.orderBy(F.rand(args.seed)).limit(args.n)
    sample = sample.coalesce(1).cache()
    n = sample.count()
    n_pos = sample.filter(F.col("human_label") == "1").count()
    print(f"표본: {args.split} {n}건 (위반 {n_pos}건, {n_pos / max(n, 1):.1%})"
          f"{' · 층화 표본: PR-AUC·F1은 변형끼리 비교용' if args.stratified else ''}")

    t0 = time.time()
    udf = make_udf(args.variants, args.batch_size, args.check)
    res = (sample.withColumn("p", udf("question", "response", "risk_area", "types_of_harm", "specific_harms"))
           .select("text_id", "human_label", *[F.col(f"p.{v}").alias(v) for v in args.variants])
           .toPandas())
    spark.stop()
    dt = time.time() - t0
    print(f"{n}건 × 변형 {len(args.variants)}개 = 판정 {n * len(args.variants)}회, {dt:.0f}초 "
          f"(판정 1회당 {dt / max(n * len(args.variants), 1):.2f}초, 모델 로드 포함)")
    if args.check:
        return

    long = res.melt(id_vars=["text_id", "human_label"], var_name="model", value_name="p_violation")
    long.insert(2, "condition", "rule_only")
    long["human_label"] = long["human_label"].astype(int)
    tag = f"{args.split}{n}{'s' if args.stratified else ''}"
    path = os.path.join(ROOT, f"data/processed/judgments_team_{tag}.csv")
    long[["text_id", "model", "condition", "p_violation", "human_label"]].to_csv(path, index=False)
    print(f"저장: {path}\n→ team_eval.html의 'CSV 불러오기'로 열어서 비교")


if __name__ == "__main__":
    main()
