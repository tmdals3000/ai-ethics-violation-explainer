"""Spark로 기준 판정 모델 3종을 같은 표본에 돌림 → judgments_baseline_{tag}.csv

texts_all.csv(세 데이터 공통 형식)를 읽고, 텍스트마다 맞는 방식으로 판정
  - dna       : 질문 + 응답 → 응답 판정 / ShieldGemma 정책 = 정답 규칙 문장 (조건 rule_only)
  - jigsaw    : 댓글 하나 판정 / ShieldGemma 정책 = 정답 규칙 문장 (조건 rule_only)
  - prosocial : 발화 하나 판정 / 정답 규칙이 없어 ShieldGemma는 일반 정책 (조건 generic, 임시)

    python scripts/spark_judge.py                                   # dna dev 100건, 3개 모델
    python scripts/spark_judge.py --sources jigsaw --split dev --n 100000   # jigsaw dev 전체
    python scripts/spark_judge.py --sources prosocial --split dev --n 500
    python scripts/spark_judge.py --models shieldgemma qwen3guard   # 일부 모델만
    python scripts/spark_judge.py --n 300 --split dev
    python scripts/spark_judge.py --check                           # 모델별 3건 원문 출력(라벨 파싱 확인)
    python scripts/spark_judge.py --ids-from data/processed/judgments_team_dev300s.csv   # 팀 판정기와 같은 텍스트

- 표본은 한 번 뽑아 캐시 → 세 모델이 정확히 같은 텍스트를 판정
- 모델은 하나씩 순서대로 실행. 워커 재사용을 꺼서 모델이 끝나면 GPU 메모리가 비워짐
- 한 모델이 실패해도(예: Llama Guard 승인 대기) 나머지는 계속 진행
"""
import argparse
import os
import sys
import time

os.environ["PYSPARK_PYTHON"] = sys.executable           # Windows: 워커도 .venv 파이썬
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

import numpy as np
import pandas as pd
from pyspark.ml.functions import predict_batch_udf
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import FloatType

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS_DIR)
ALL_MODELS = ["shieldgemma", "qwen3guard", "llamaguard"]


def make_udf(model_key, contro_w, batch_size, check):
    def make_predict_fn():
        # executor(워커) 안에서 실행됨: 판정 모듈을 불러와 모델을 1회 로드
        sys.path.insert(0, SCRIPTS_DIR)
        from judges import make_judge, policy_for
        judge = make_judge(model_key, controversial_weight=contro_w)

        def predict(ctx, txt, rule, mode):
            out = []
            for a, b, c, md in zip(ctx, txt, rule, mode):
                p, raw = judge(a, b, policy_for(c, md)[0], mode=md)
                if check:
                    print(f"[{model_key}] ({md}) p={p:.3f} raw={raw!r}", flush=True)
                out.append(p)
            return np.asarray(out, dtype=np.float32)
        return predict

    return predict_batch_udf(make_predict_fn, return_type=FloatType(), batch_size=batch_size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=ALL_MODELS, choices=ALL_MODELS)
    ap.add_argument("--sources", nargs="+", default=["dna"], choices=["dna", "jigsaw", "prosocial"])
    ap.add_argument("--split", default="dev")
    ap.add_argument("--n", type=int, default=100, help="출처마다 뽑을 건수")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--controversial-weight", type=float, default=0.5)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--ids-from", default=None,
                    help="이 CSV의 text_id와 같은 텍스트만 판정 (예: 팀 판정기 결과 CSV)")
    args = ap.parse_args()
    if args.check:
        args.n = 3

    spark = (SparkSession.builder.master("local[1]").appName("baseline-judges")
             .config("spark.python.worker.reuse", "false")      # 모델 끝나면 워커 종료 → GPU 메모리 반환
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    df = (spark.read.option("header", True).option("multiLine", True).option("escape", '"')
          .csv(os.path.join(ROOT, "data/processed/texts_all.csv"))
          .fillna("", subset=["context", "rule_text"]))
    cols = ["text_id", "source", "context", "text", "rule_text", "judge_mode", "human_label"]
    if args.ids_from:                                            # 팀 판정기와 같은 텍스트
        ids = pd.read_csv(args.ids_from)["text_id"].drop_duplicates().to_frame()
        sample = df.join(spark.createDataFrame(ids), "text_id").select(*cols)
        tag = os.path.splitext(os.path.basename(args.ids_from))[0].replace("judgments_team_", "")
    else:
        sample = None                                            # 출처마다 n건씩 (큰 데이터가 표본을 독차지하지 않게)
        for src in args.sources:
            part = (df.filter((F.col("source") == src) & (F.col("split") == args.split))
                    .orderBy(F.rand(args.seed)).limit(args.n).select(*cols))
            sample = part if sample is None else sample.unionByName(part)
        tag = f"{args.split}{args.n}" + ("" if args.sources == ["dna"] else "_" + "-".join(args.sources))
    sample = sample.coalesce(1).cache()
    cnt = sample.groupBy("source").count().toPandas()
    print(f"표본: {int(cnt['count'].sum())}건 ({', '.join(f'{a} {b}' for a, b in zip(cnt.source, cnt['count']))})"
          f"{' · --ids-from ' + os.path.basename(args.ids_from) if args.ids_from else ''}")

    results = []
    for m in args.models:
        t0 = time.time()
        try:
            udf = make_udf(m, args.controversial_weight, args.batch_size, args.check)
            res = (sample.withColumn("p_violation", udf("context", "text", "rule_text", "judge_mode"))
                   .select("text_id", "rule_text", "judge_mode", "p_violation", "human_label").toPandas())
        except Exception as e:                                   # 승인 대기·메모리 부족 등
            lines = [l.strip() for l in str(e).splitlines() if "Error" in l or "error" in l]
            print(f"[{m}] 실패 → 건너뜀: {(lines[-1] if lines else str(e).splitlines()[0])[:300]}")
            if "gated" in str(e) or "403" in str(e) or "401" in str(e):
                print(f"    → Hugging Face 모델 페이지에서 라이선스 승인 여부와 hf auth login 확인")
            elif "out of memory" in str(e).lower():
                print(f"    → --batch-size를 줄이거나 다른 프로그램의 GPU 사용을 확인")
            continue
        from judges import policy_for
        res.insert(1, "model", m)
        res.insert(2, "condition", [policy_for(r, md)[1] if m == "shieldgemma" else "none"
                                    for r, md in zip(res.rule_text, res.judge_mode)])
        res = res.drop(columns=["rule_text", "judge_mode"])
        results.append(res)
        dt = time.time() - t0
        print(f"[{m}] {len(res)}건, {dt:.0f}초 (건당 {dt / max(len(res), 1):.2f}초, 모델 로드 포함), "
              f"판정 실패 {res.p_violation.isna().sum()}건")

    spark.stop()
    if not results or args.check:
        return

    out = pd.concat(results, ignore_index=True)
    out["human_label"] = out["human_label"].astype(int)
    path = os.path.join(ROOT, f"data/processed/judgments_baseline_{tag}.csv")
    out.to_csv(path, index=False)
    print(f"\n저장: {path}")

    # 간단 지표 (표본이 작으면 위반 건수가 적어 지표가 불안정함)
    try:
        from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
        for m, g in out.dropna(subset=["p_violation"]).groupby("model"):
            y, p = g.human_label, g.p_violation
            pred = (p >= 0.5).astype(int)
            auc = roc_auc_score(y, p) if y.nunique() == 2 else float("nan")
            ap_ = average_precision_score(y, p) if y.sum() else float("nan")
            print(f"  {m:<12} 위반 {y.sum()}/{len(y)}건  ROC-AUC {auc:.3f}  PR-AUC {ap_:.3f}  "
                  f"F1 {f1_score(y, pred, zero_division=0):.3f}")
    except ImportError:
        print("지표를 보려면 scikit-learn 설치: python -m pip install scikit-learn")


if __name__ == "__main__":
    main()
