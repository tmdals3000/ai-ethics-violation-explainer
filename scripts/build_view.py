"""텍스트 + 판정값을 HTML 뷰에 넣어 단일 파일로 만듦 (file://로 바로 열리게).

    python scripts/build_view.py                                   # 가상값
    python scripts/build_view.py --judgments data/processed/judgments_baseline.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["llamaguard", "qwen3guard", "shieldgemma"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=ROOT / "data/processed/texts_dna.csv")
    ap.add_argument("--judgments", default=ROOT / "data/processed/judgments_mock.csv")
    ap.add_argument("--template", default=ROOT / "app/p2_template.html")
    ap.add_argument("--out", default=ROOT / "app/p2_view.html")
    ap.add_argument("--max-chars", type=int, default=1200)
    args = ap.parse_args()

    t = pd.read_csv(args.texts)
    j = pd.read_csv(args.judgments)
    is_mock = "mock" in Path(args.judgments).name

    def codes(col):
        cats = sorted(t[col].astype(str).unique())
        idx = {c: i for i, c in enumerate(cats)}
        return cats, t[col].astype(str).map(idx).tolist()

    data = {"mock": is_mock, "judgmentsFile": Path(args.judgments).name, "n": len(t)}
    for col in ["source", "split", "risk_area", "types_of_harm", "specific_harms", "responder", "question"]:
        data[col + "_dict"], data[col] = codes(col)
    data["text_id"] = t["text_id"].tolist()
    data["response"] = [r if len(r) <= args.max_chars else r[: args.max_chars] + " …(이하 생략)"
                        for r in t["response"].astype(str)]
    data["y"] = t["human_label"].astype(int).tolist()
    data["p"] = {}
    for m in MODELS:
        s = j[j.model == m].drop_duplicates("text_id", keep="last").set_index("text_id")["p_violation"]
        data["p"][m] = [None if pd.isna(v) else round(float(v), 4) for v in t["text_id"].map(s)]

    html = Path(args.template).read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{args.out}  ({Path(args.out).stat().st_size / 1e6:.1f} MB, mock={is_mock})")


if __name__ == "__main__":
    main()
