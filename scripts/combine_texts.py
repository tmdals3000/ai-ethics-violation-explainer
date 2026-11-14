"""세 데이터의 전처리 결과를 공통 형식 하나로 합침 → texts_all.csv

있는 파일만 합침 (Jigsaw나 ProsocialDialog를 아직 못 받았어도 실행됨).
texts_dna.csv는 기존 스크립트(판정, 화면)가 그대로 쓰므로 건드리지 않고, 여기서 공통 열로 옮겨 담기만 함.

    python scripts/combine_texts.py
"""
import pandas as pd

from common import COMMON_COLS, PROCESSED


def dna_to_common(d: pd.DataFrame) -> pd.DataFrame:
    name = d["specific_harms"].astype(str).str.partition(":")[0].str.strip()
    return pd.DataFrame({
        "text_id": d["text_id"], "source": "dna", "split": d["split"],
        "text": d["response"].astype(str), "context": d["question"].astype(str), "judge_mode": "response",
        "human_label": d["human_label"].astype(int), "gold_rule": name,
        "rule_text": d["specific_harms"].astype(str), "rule_l1": d["risk_area"], "rule_l2": d["types_of_harm"],
        "responder": d["responder"], "label_raw": d["action"].astype(str),
    })


def main():
    frames = []
    p = PROCESSED / "texts_dna.csv"
    if p.exists():
        frames.append(dna_to_common(pd.read_csv(p)))
    for name in ["texts_jigsaw.csv", "texts_prosocial.csv"]:
        p = PROCESSED / name
        if p.exists():
            frames.append(pd.read_csv(p, keep_default_na=False)[COMMON_COLS])
        else:
            print(f"※ {name} 없음 → 건너뜀")
    if not frames:
        raise SystemExit("합칠 전처리 결과가 없습니다. prepare_*.py를 먼저 실행하세요.")

    allt = pd.concat(frames, ignore_index=True)[COMMON_COLS]
    allt["human_label"] = allt["human_label"].astype(int)
    assert allt["text_id"].is_unique, "text_id가 출처 사이에서 겹칩니다"
    allt.to_csv(PROCESSED / "texts_all.csv", index=False)

    print(f"texts_all.csv: {len(allt):,}행")
    t = allt.groupby(["source", "split"]).human_label.agg(건수="count", 위반="sum")
    t["위반 비율"] = (t["위반"] / t["건수"]).map("{:.1%}".format)
    print(t.to_string())


if __name__ == "__main__":
    main()
