"""ProsocialDialog 전처리 → texts_prosocial.csv

원본: Hugging Face 'allenai/prosocial-dialog' (train / validation / test)
      → data/raw/prosocial/ 아래에 둘 것 (json, jsonl, parquet, csv 모두 읽음)
      받는 법:  hf download allenai/prosocial-dialog --repo-type dataset --local-dir data/raw/prosocial
원본 열: context(사용자 발화), response(친사회적 답변), rots(경험칙 목록), safety_label,
         dialogue_id, response_id(대화 안에서 몇 번째 턴인지), ...

하는 일 (PDF 1주차: "영어만 남기고, 안전 라벨을 위반/정상으로 바꿉니다"):
  1) 판정 대상 = context(사용자 발화). safety_label이 이 발화에 대한 평가이기 때문.
     같은 대화의 앞선 턴들은 맥락(context 열)으로 붙임
  2) 영어가 아닌 발화 제거 (글자 중 영문자 비율로 판단하는 간단한 기준)
  3) safety_label → human_label (※ 임시 기준. 팀에서 확정할 것)
        __casual__                                   → 0 (정상)
        __needs_caution__, __needs_intervention__    → 1 (위반)
        __possibly_needs_caution__, __probably_needs_caution__  → --ambiguous 옵션 (기본: 위반)
     원본 값은 label_raw 열에 남겨 두므로 기준이 바뀌어도 다시 계산하기 쉬움
  4) 정답 규칙이 없는 데이터 → gold_rule은 비워 둠 (P1·P3에서는 빠지고 P2에서만 씀)

분할: 원본 validation → dev, test → test, train → train. 기본은 dev·test만 저장
     (이 프로젝트는 ProsocialDialog로 학습하지 않음).

    python scripts/prepare_prosocial.py
    python scripts/prepare_prosocial.py --ambiguous drop --max-per-split 3000
"""
import argparse
from pathlib import Path

import pandas as pd

from common import PROCESSED, RAW, hash_bucket, norm_text, require_columns

SPLIT_NAMES = {"train": "train", "validation": "dev", "valid": "dev", "dev": "dev", "test": "test"}
POSITIVE = {"__needs_caution__", "__needs_intervention__"}
AMBIGUOUS = {"__possibly_needs_caution__", "__probably_needs_caution__"}
NEGATIVE = {"__casual__"}


def load_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf == ".csv":
        return pd.read_csv(path)
    with open(path, encoding="utf-8") as f:   # json: 첫 글자로 형식 판단
        head = f.read(64).lstrip()
    # '[' 로 시작하면 배열 하나, '{' 로 시작하면 한 줄에 한 건(jsonl)
    return pd.read_json(path, lines=not head.startswith("["))


def find_split_files(folder: Path) -> dict:
    found = {}
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() not in (".json", ".jsonl", ".parquet", ".csv"):
            continue
        stem = p.stem.lower()
        for key, name in SPLIT_NAMES.items():
            if stem == key or stem.startswith(key + "-") or stem.startswith(key + "_"):
                found.setdefault(name, p)
    return found


def looks_english(s: str, min_ratio=0.9) -> bool:
    letters = [c for c in str(s) if c.isalpha()]
    if not letters:
        return False
    return sum(c.isascii() for c in letters) / len(letters) >= min_ratio


def add_history(df: pd.DataFrame, max_turns=3, max_chars=1500) -> pd.Series:
    """같은 대화의 앞선 턴들을 'User: ...\\nBot: ...' 형태로 이어 붙임."""
    hist, out = {}, []
    for did, ctx, resp in zip(df["dialogue_id"], df["context"], df["response"]):
        prev = hist.setdefault(did, [])
        out.append("\n".join(prev[-max_turns:])[-max_chars:])
        prev.append(f"User: {ctx}\nBot: {resp}")
    return pd.Series(out, index=df.index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=RAW / "prosocial")
    ap.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"])
    ap.add_argument("--ambiguous", choices=["violation", "normal", "drop"], default="violation",
                    help="possibly/probably_needs_caution 처리")
    ap.add_argument("--max-per-split", type=int, default=None,
                    help="분할마다 최대 발화 수 (대화 단위로 뽑음). 판정 시간을 줄일 때")
    args = ap.parse_args()

    files = find_split_files(Path(args.raw_dir))
    if not files:
        raise SystemExit(f"{args.raw_dir} 에서 train/validation/test 파일을 찾지 못했습니다.")
    frames = []
    for split in args.splits:
        if split not in files:
            print(f"※ {split} 파일 없음 → 건너뜀")
            continue
        d = load_any(files[split])
        require_columns(d, ["context", "response", "safety_label", "dialogue_id", "response_id"],
                        f"ProsocialDialog {files[split].name}")
        d = d.sort_values(["dialogue_id", "response_id"]).reset_index(drop=True)
        d["split"] = split
        d["history"] = add_history(d)
        frames.append(d)
        print(f"{split}: {files[split].name} {len(d):,}행, 대화 {d.dialogue_id.nunique():,}개")
    df = pd.concat(frames, ignore_index=True)

    n0 = len(df)
    df = df.dropna(subset=["context", "safety_label"])
    df = df[df["context"].astype(str).str.strip() != ""]
    eng = df["context"].map(looks_english)
    print(f"\n결측·빈 발화 제거 {n0 - len(df)}행, 영어가 아닌 발화 제거 {int((~eng).sum())}행")
    df = df[eng].copy()

    known = POSITIVE | AMBIGUOUS | NEGATIVE
    unknown = sorted(set(df["safety_label"]) - known)
    if unknown:
        print(f"※ 모르는 safety_label 값 {unknown} → 해당 행 제거")
        df = df[df["safety_label"].isin(known)].copy()
    if args.ambiguous == "drop":
        df = df[~df["safety_label"].isin(AMBIGUOUS)].copy()
    pos = POSITIVE | (AMBIGUOUS if args.ambiguous == "violation" else set())
    df["human_label"] = df["safety_label"].isin(pos).astype(int)
    print(f"※ 임시 라벨 기준: 위반 = {sorted(pos)}"
          f"{' (possibly/probably 행은 제거)' if args.ambiguous == 'drop' else ''}")

    if args.max_per_split:                       # 대화 단위로 뽑아서 맥락이 끊기지 않게
        keep = []
        for split, g in df.groupby("split"):
            order = sorted(g["dialogue_id"].unique(), key=lambda x: hash_bucket(f"{split}-{x}", 10**9))
            sizes = g.groupby("dialogue_id").size()
            total, chosen = 0, set()
            for did in order:
                if total >= args.max_per_split:
                    break
                chosen.add(did); total += int(sizes[did])
            keep.append(g[g["dialogue_id"].isin(chosen)])
        df = pd.concat(keep, ignore_index=True)

    df["text_id"] = ("prosocial_" + df["split"] + "_" + df["dialogue_id"].astype(str)
                     + "_" + df["response_id"].astype(str))
    rots = df["rots"].map(lambda x: " | ".join(map(str, x)) if isinstance(x, (list, tuple)) or hasattr(x, "tolist")
                          else ("" if pd.isna(x) else str(x))) if "rots" in df else ""
    texts = pd.DataFrame({
        "text_id": df["text_id"], "source": "prosocial", "split": df["split"],
        "text": df["context"].astype(str), "context": df["history"], "judge_mode": "text",
        "human_label": df["human_label"], "gold_rule": "", "rule_text": "", "rule_l1": "", "rule_l2": "",
        "responder": "", "label_raw": df["safety_label"],
        "dialogue_id": df["dialogue_id"], "turn": df["response_id"], "rots": rots,
        "prosocial_response": df["response"].astype(str),
    })
    assert texts["text_id"].is_unique, "text_id가 겹칩니다"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    texts.to_csv(PROCESSED / "texts_prosocial.csv", index=False)
    dup = int(texts["text"].map(norm_text).duplicated().sum())
    print(f"\ntexts_prosocial.csv: {len(texts):,}행, 위반 {int(texts.human_label.sum()):,}건 "
          f"({texts.human_label.mean():.1%}), 같은 발화 중복 {dup}행")
    print(pd.crosstab(texts["label_raw"], texts["split"], margins=True).to_string())


if __name__ == "__main__":
    main()
