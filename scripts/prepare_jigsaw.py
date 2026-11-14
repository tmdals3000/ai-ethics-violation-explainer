"""Jigsaw ACRC 전처리 → texts_jigsaw.csv, examples_jigsaw.csv

원본: Kaggle 'jigsaw-agile-community-rules' 대회의 train.csv (로그인 + 대회 규칙 동의 후 다운로드)
      → data/raw/jigsaw/train.csv 에 둘 것
원본 열: row_id, body, rule, subreddit, positive_example_1, positive_example_2,
         negative_example_1, negative_example_2, rule_violation

하는 일 (PDF 1주차: "행마다 딸린 위반·비위반 예시를 떼어 예시 후보로 정리"):
  1) 댓글(body)을 판정 대상 텍스트로 정리 → texts_jigsaw.csv
  2) 행마다 붙은 예시 4개를 떼어 규칙별 예시 후보 표로 정리 → examples_jigsaw.csv
     - 같은 예시가 여러 행에 반복되므로 (규칙, 예시, 라벨) 기준으로 중복 제거
     - judged_splits: 예시와 같은 문장이 판정 대상 댓글로 들어 있는 분할
     - safe_for_eval: dev/test의 판정 대상 댓글과 같은 문장이 아님
       (같은 문장을 라벨과 함께 예시로 넣으면 정답을 알려 주는 셈이라 지표가 부풀려짐)
     - from_splits: 그 예시가 딸려 있던 행들의 분할 (참고용)

분할(임시): 댓글 내용의 해시로 train 60 / dev 20 / test 20.
  같은 댓글이 여러 행에 있어도 같은 분할에 들어가도록 row_id가 아니라 내용 기준.
  팀에서 분할을 따로 정하면 --splits 로 교체.

    python scripts/prepare_jigsaw.py
    python scripts/prepare_jigsaw.py --splits data/raw/splits.csv
"""
import argparse

import pandas as pd

from common import PROCESSED, RAW, norm_text, require_columns, temp_split

EX_COLS = [("positive_example_1", 1), ("positive_example_2", 1),
           ("negative_example_1", 0), ("negative_example_2", 0)]


def rule_name(rule: str) -> str:
    """'No Advertising: Spam, referral links...' → 'No Advertising' (콜론이 없으면 앞 6단어)."""
    head, sep, _ = str(rule).partition(":")
    return head.strip() if sep else " ".join(str(rule).split()[:6])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW / "jigsaw" / "train.csv")
    ap.add_argument("--splits", default=None, help="팀이 확정한 분할 파일 (text_id, split 열)")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    require_columns(df, ["row_id", "body", "rule", "subreddit", "rule_violation"] + [c for c, _ in EX_COLS],
                    "Jigsaw train.csv")
    n0 = len(df)
    df = df.dropna(subset=["body", "rule", "rule_violation"]).copy()
    print(f"원본 {n0:,}행, 본문·규칙·라벨 결측 제거 후 {len(df):,}행")

    df["text_id"] = "jigsaw_" + df["row_id"].astype(str)
    df["_key"] = df["body"].map(norm_text)
    if args.splits:
        sp = pd.read_csv(args.splits)[["text_id", "split"]]
        df = df.merge(sp, on="text_id", how="left")
        print("split 결측:", int(df["split"].isna().sum()))
    else:
        df["split"] = df["_key"].map(lambda k: temp_split(k, (("train", 60), ("dev", 20), ("test", 20))))
        print("※ splits.csv 없음: 댓글 내용 해시 기반 임시 분할 (train 60 / dev 20 / test 20)")

    rules = sorted(df["rule"].unique())
    rule_id = {r: f"jigsaw_rule_{i + 1}" for i, r in enumerate(rules)}

    texts = pd.DataFrame({
        "text_id": df["text_id"], "source": "jigsaw", "split": df["split"],
        "text": df["body"].astype(str), "context": "r/" + df["subreddit"].astype(str),
        "judge_mode": "text", "human_label": df["rule_violation"].astype(int),
        "gold_rule": df["rule"].map(rule_name), "rule_text": df["rule"].astype(str),
        "rule_l1": "Community rules", "rule_l2": df["rule"].map(rule_name),
        "responder": "", "label_raw": df["rule_violation"].astype(int).astype(str),
        "rule_id": df["rule"].map(rule_id), "subreddit": df["subreddit"].astype(str),
    })

    # ---- 예시 후보: 행마다 붙은 예시 4개를 세로로 펼친 뒤 중복 제거 ----
    parts = []
    for col, lab in EX_COLS:
        p = df[["row_id", "rule", "split", col]].rename(columns={col: "example_text"})
        p["example_label"], p["slot"] = lab, col
        parts.append(p)
    ex = pd.concat(parts, ignore_index=True).dropna(subset=["example_text"])
    ex["_key"] = ex["example_text"].map(norm_text)
    ex = ex[ex["_key"] != ""]
    judged = {}                                  # (규칙, 문장) → 그 문장이 판정 대상인 분할들
    for r, k, sp in zip(df["rule"], df["_key"], df["split"]):
        judged.setdefault((r, k), set()).add(sp)
    examples = (ex.groupby(["rule", "_key", "example_label"], sort=False)
                .agg(example_text=("example_text", "first"), n_rows=("row_id", "nunique"),
                     from_splits=("split", lambda s: ",".join(sorted(set(s.dropna())))),
                     first_row_id=("row_id", "min")).reset_index())
    examples["rule_id"] = examples["rule"].map(rule_id)
    examples["rule_name"] = examples["rule"].map(rule_name)
    js = [judged.get((r, k), set()) for r, k in zip(examples["rule"], examples["_key"])]
    examples["judged_splits"] = [",".join(sorted(x)) for x in js]
    examples["safe_for_eval"] = [not (x & {"dev", "test"}) for x in js]
    conflict = examples.groupby(["rule", "_key"])["example_label"].nunique()
    examples["label_conflict"] = [conflict[(r, k)] > 1 for r, k in zip(examples["rule"], examples["_key"])]
    examples.insert(0, "example_id", ["jigsaw_ex_" + str(i) for i in range(len(examples))])
    examples = examples[["example_id", "rule_id", "rule_name", "example_label", "example_text",
                         "n_rows", "from_splits", "judged_splits", "safe_for_eval", "label_conflict", "first_row_id"]]

    PROCESSED.mkdir(parents=True, exist_ok=True)
    texts.to_csv(PROCESSED / "texts_jigsaw.csv", index=False)
    examples.to_csv(PROCESSED / "examples_jigsaw.csv", index=False)

    dup = int(df.duplicated(["_key", "rule"]).sum())
    print(f"\ntexts_jigsaw.csv: {len(texts):,}행, 위반 {int(texts.human_label.sum()):,}건 "
          f"({texts.human_label.mean():.1%}), (댓글, 규칙) 중복 {dup}행")
    print(texts.groupby(["rule_l2", "split"]).human_label.agg(건수="count", 위반="sum").to_string())
    print(f"\nexamples_jigsaw.csv: 예시 {len(ex):,}개 → 중복 제거 후 {len(examples):,}개")
    print(examples.groupby(["rule_name", "example_label"]).size().rename("예시 수").to_string())
    print(f"  판정 대상 댓글과 같은 문장인 예시: {int((examples.judged_splits != '').sum())}개")
    print(f"  그중 dev/test 댓글과 같은 문장: {int((~examples.safe_for_eval).sum())}개 "
          f"→ 예시로 넣으면 정답 유출. safe_for_eval = True인 {int(examples.safe_for_eval.sum())}개만 쓸 것")
    print(f"  위반·비위반 양쪽에 모두 나온 예시: {int(examples.label_conflict.sum())}개")

if __name__ == "__main__":
    main()
