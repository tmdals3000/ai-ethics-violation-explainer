"""Spark로 4가지 조건 판정 → judgments_cache.csv (후보별) → judgments_conditions.csv (조건별, judgments.csv 형식)

    python scripts/spark_conditions.py --check                     # 3건, 프롬프트 1개와 확률 출력
    python scripts/spark_conditions.py                             # conditions_jobs.csv 전부 (이미 한 것은 건너뜀)
    python scripts/spark_conditions.py --variants team_v0 team_v3  # 변형을 줄이면 시간이 줄어듦
    python scripts/spark_conditions.py --assemble-only             # 판정 없이 캐시로 조건별 파일만 다시 만들기

- 판정기: 팀 판정기(ShieldGemma-2B). team_v0 = 기준 ShieldGemma와 같은 프롬프트라서 결과 파일에서는
  model = "shieldgemma"로 저장함. Llama Guard·Qwen3Guard는 규칙 문장을 받지 않아 조건과 상관없이 값이 같으므로
  1주차 기준 판정(judgments_baseline_*.csv)을 그대로 씀
- --chunk건마다 캐시에 저장 → 중간에 꺼도 다시 실행하면 이어서 함
- FAKE_JUDGE=1이면 모델 없이 가짜 확률로 흐름만 확인
"""
import argparse
import json
import os
import sys
import time

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import pandas as pd

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPTS_DIR)
from analysis import assemble_conditions  # noqa: E402
from common import PROCESSED, need  # noqa: E402
from team_judge import VARIANTS  # noqa: E402

KEYS = ["text_id", "rule_id", "with_examples"]
CACHE = PROCESSED / "judgments_cache.csv"


def make_udf(variants, batch_size, check):
    from pyspark.ml.functions import predict_batch_udf
    from pyspark.sql.types import FloatType, StructField, StructType
    schema = StructType([StructField(v, FloatType()) for v in variants])

    def make_predict_fn():
        sys.path.insert(0, SCRIPTS_DIR)
        import numpy as np
        from team_judge import make_fake_team_judge, make_team_judge
        judge = (make_fake_team_judge if os.environ.get("FAKE_JUDGE") else make_team_judge)(variants)
        shown = [False]

        def predict(q, t, mode, rule_text, rule_path, examples):
            cols = {v: [] for v in variants}
            for a, b, md, rt, rp, ex in zip(q, t, mode, rule_text, rule_path, examples):
                rule = {"rule_text": rt, "path": json.loads(rp), "examples": json.loads(ex)}
                probs = judge(a, b, rule, md)
                if check:
                    if not shown[0] and hasattr(judge, "prompt_text"):
                        print("---- 프롬프트 예 (" + variants[-1] + ") ----\n"
                              + judge.prompt_text(variants[-1], a, b, rule, md) + "\n----", flush=True)
                        shown[0] = True
                    print("  " + "  ".join(f"{v}={probs[v]:.3f}" for v in variants), flush=True)
                for v in variants:
                    cols[v].append(probs[v])
            return {v: np.asarray(x, dtype=np.float32) for v, x in cols.items()}
        return predict

    return predict_batch_udf(make_predict_fn, return_type=schema, batch_size=batch_size)


def load_cache():
    if CACHE.exists():
        c = pd.read_csv(CACHE)
        c["with_examples"] = c.with_examples.astype(str) == "True"
        return c
    return pd.DataFrame(columns=KEYS + ["variant", "p"])


def assemble():
    need("conditions_map.csv", "conditions_jobs.csv")
    cmap = pd.read_csv(PROCESSED / "conditions_map.csv")
    cmap["with_examples"] = cmap.with_examples.astype(str) == "True"
    jobs = pd.read_csv(PROCESSED / "conditions_jobs.csv", usecols=["text_id", "source", "split", "human_label"])
    cache = load_cache()
    if cache.empty:
        raise SystemExit("판정 캐시가 비어 있습니다. 먼저 python scripts/spark_conditions.py 를 실행하세요.")
    res = assemble_conditions(cache, cmap.merge(jobs[["text_id"]].drop_duplicates(), on="text_id"))
    info = jobs.drop_duplicates("text_id")
    res = res.merge(info, on="text_id")
    cols = ["text_id", "source", "split", "model", "condition", "p_violation", "human_label", "n_rules", "top_rule"]
    res = res[cols].sort_values(["source", "model", "condition", "text_id"])
    res.to_csv(PROCESSED / "judgments_conditions.csv", index=False)
    done = res.p_violation.notna()
    print(f"저장: {PROCESSED / 'judgments_conditions.csv'} ({done.sum():,}행 완료"
          f"{f', {(~done).sum():,}행은 후보 판정이 덜 끝나 비어 있음' if (~done).any() else ''})")
    print(res[done].groupby(["source", "condition", "model"]).size().unstack("model").to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--jobs", default=str(PROCESSED / "conditions_jobs.csv"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=1000, help="이만큼 판정할 때마다 캐시에 저장")
    ap.add_argument("--limit", type=int, default=0, help="이번에 판정할 최대 건수 (0 = 전부)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    if args.assemble_only:
        return assemble()

    need("conditions_jobs.csv")
    jobs = pd.read_csv(args.jobs, keep_default_na=False)
    jobs["with_examples"] = jobs.with_examples.astype(str) == "True"
    cache = load_cache()
    have = cache.groupby(KEYS).variant.agg(set) if len(cache) else pd.Series(dtype=object)
    need_v = [all(v in have.get(tuple(k), set()) for v in args.variants)
              for k in jobs[KEYS].itertuples(index=False)]
    todo = jobs[[not x for x in need_v]].reset_index(drop=True)
    if args.check:
        todo = jobs.groupby("source").head(2).head(3).reset_index(drop=True)
    elif args.limit:
        todo = todo.head(args.limit)
    print(f"판정 목록 {len(jobs):,}건 중 남은 것 {len(todo):,}건 × 변형 {len(args.variants)}개")
    if todo.empty:
        return assemble()

    from pyspark.sql import SparkSession
    spark = (SparkSession.builder.master("local[1]").appName("conditions")
             .config("spark.python.worker.reuse", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    udf = make_udf(args.variants, args.batch_size, args.check)
    cols = ["text_id", "rule_id", "with_examples", "question", "text", "judge_mode", "rule_text", "rule_path", "examples"]

    t0, done = time.time(), 0
    for start in range(0, len(todo), args.chunk):
        part = todo.iloc[start:start + args.chunk][cols]
        sdf = spark.createDataFrame(part.astype({"with_examples": bool})).coalesce(1)
        out = (sdf.withColumn("p", udf("question", "text", "judge_mode", "rule_text", "rule_path", "examples"))
                  .select(*KEYS, *[f"p.{v}" for v in args.variants]).toPandas())
        long = out.melt(id_vars=KEYS, var_name="variant", value_name="p")
        if args.check:
            break
        cache = load_cache()
        cache = pd.concat([cache, long]).drop_duplicates(KEYS + ["variant"], keep="last")
        cache.to_csv(CACHE, index=False)
        done += len(part)
        dt = time.time() - t0
        print(f"  {done:,}/{len(todo):,}건 저장 ({dt:.0f}초, 1건(변형 {len(args.variants)}개)당 {dt / done:.2f}초, "
              f"남은 시간 약 {dt / done * (len(todo) - done) / 60:.0f}분)", flush=True)
    spark.stop()
    if args.check:
        print(f"{len(todo)}건 확인 끝 ({time.time() - t0:.0f}초, 모델 로드 포함). 캐시에는 저장하지 않음")
        return
    assemble()
    print("다음: python scripts/compare_conditions.py  →  python scripts/diagnose.py")


if __name__ == "__main__":
    main()
