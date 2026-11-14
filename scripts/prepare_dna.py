"""Do-Not-Answer 전처리 (P2 데모용)

- 한 행에 붙은 응답 LLM 6개를 펼쳐 응답 하나가 한 행이 되게 함 (939 -> 5,634행)
- text_id 생성: dna_{질문id}_{응답LLM}   ※ 팀에서 규칙을 바꾸면 여기만 고치면 됨
- split: --splits 파일이 있으면 그걸 쓰고, 없으면 질문 단위 해시로 임시 분할(dev/test 50:50)
         같은 질문의 응답 6개가 dev/test에 나뉘어 들어가지 않도록 질문 id 기준으로 나눔
- --mock: 화면 시연용 가상 위반 확률 생성 (실제 판정값은 run_baselines.py로 만듦)

사용:
    python scripts/prepare_dna.py --mock
    python scripts/prepare_dna.py --splits data/raw/splits.csv
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESPONDERS = ["GPT4", "ChatGPT", "Claude", "ChatGLM2", "llama2-7b-chat", "vicuna-7b"]


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in RESPONDERS:
        part = df[["id", "risk_area", "types_of_harm", "specific_harms", "question"]].copy()
        part["responder"] = m
        part["response"] = df[f"{m}_response"]
        part["human_label"] = df[f"{m}_harmful"].astype(int)   # 1 = 위반(harmful)
        part["action"] = df[f"{m}_action"].astype(int)
        rows.append(part)
    long = pd.concat(rows, ignore_index=True).rename(columns={"id": "question_id"})
    long["source"] = "dna"
    long["text_id"] = "dna_" + long["question_id"].astype(str) + "_" + long["responder"]
    return long


def temp_split(qid: int) -> str:
    h = int(hashlib.md5(str(qid).encode()).hexdigest(), 16)
    return "dev" if h % 2 == 0 else "test"


def mock_judgments(long: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """시연용 가상 확률. 모델별 성향을 일부러 다르게 넣어 행렬에 차이가 보이게 함."""
    rng = np.random.default_rng(seed)
    y = long["human_label"].to_numpy()
    base = np.where(y == 1, 0.1, -2.0)
    base += np.where(long["action"] == 5, 1.0, 0.0) - np.where(long["action"] == 0, 0.7, 0.0)
    base += long["responder"].isin(["ChatGLM2", "vicuna-7b"]).to_numpy() * 0.3

    ra = long["risk_area"]
    th = long["types_of_harm"]
    out = []
    specs = {
        # Llama Guard: 오정보·사람-챗봇 상호작용은 자체 범주에 거의 없음 -> 놓치기 쉬움
        "llamaguard": (ra.isin(["Misinformation Harms", "Human-Chatbot Interaction Harms"]) * -1.6, 1.1),
        # Qwen3Guard: 차별·혐오 범주에서 과잉 탐지 경향
        "qwen3guard": (ra.str.startswith("Discrimination") * 1.1, 1.2),
        # ShieldGemma: 학습한 4개 유형 밖의 규칙에서 흔들림
        "shieldgemma": ((~th.isin(["Adult Content", "Toxic language (hate speech)",
                                   "Social stereotypes and unfair discrimination"])) * -0.6, 1.4),
    }
    for model, (bias, sd) in specs.items():
        logit = base + bias.to_numpy() + rng.normal(0, sd, len(long))
        p = 1 / (1 + np.exp(-logit))
        out.append(pd.DataFrame({"text_id": long["text_id"], "model": model,
                                 "condition": "rule_only" if model == "shieldgemma" else "none",
                                 "p_violation": p.round(4)}))
    return pd.concat(out, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=ROOT / "data/raw/dna_data_en.csv")
    ap.add_argument("--splits", default=None, help="팀이 확정한 분할 파일 (text_id, split 열)")
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.raw)
    long = to_long(df)

    if args.splits:
        sp = pd.read_csv(args.splits)[["text_id", "split"]]
        long = long.merge(sp, on="text_id", how="left")
        print("split 결측:", long["split"].isna().sum())
    else:
        long["split"] = long["question_id"].map(temp_split)
        print("※ splits.csv 없음: 질문 id 해시 기반 임시 분할 사용")

    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "texts_dna.csv", index=False)
    print(f"texts_dna.csv: {len(long):,}행, 위반 {long.human_label.sum()}건 "
          f"({long.human_label.mean():.1%})")
    print(long.groupby("split").human_label.agg(["count", "sum"]))

    if args.mock:
        j = mock_judgments(long)
        j.to_csv(out / "judgments_mock.csv", index=False)
        print(f"judgments_mock.csv: {len(j):,}행 (가상값)")


if __name__ == "__main__":
    main()
