"""팀 판정기 성능 판별 페이지 빌드 → app/team_eval.html

    python scripts/build_team_eval.py --mock                     # 시연용 가상값
    python scripts/build_team_eval.py --judgments data/processed/judgments_baseline_dev100.csv \
                                                 data/processed/judgments_team_dev300s.csv
페이지에서 'CSV 불러오기'로 판정 파일을 더 넣을 수도 있음.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def mock_team(t: pd.DataFrame, base: pd.Series, seed=7) -> pd.DataFrame:
    """가상 팀 판정값: v0 = 기준 ShieldGemma와 동일, v1~v3은 의도한 방향으로 조금씩 바뀌게."""
    rng = np.random.default_rng(seed)
    p0 = base.clip(1e-4, 1 - 1e-4).to_numpy()
    z = np.log(p0 / (1 - p0))
    y = t["human_label"].to_numpy()
    soft = t["action"].isin([0, 1, 2, 3]).to_numpy()            # 거절·반박·양쪽 관점·조심스러운 답
    out = {"team_v0": p0}
    z1 = z - 1.2 * ((y == 0) & soft) - 0.3 * (y == 1) + rng.normal(0, .35, len(z))
    z2 = z + 0.45 * (y == 1) - 0.15 * (y == 0) + rng.normal(0, .5, len(z))
    z3 = z - 1.0 * ((y == 0) & soft) + 0.2 * (y == 1) + rng.normal(0, .45, len(z))
    for k, zz in [("team_v1", z1), ("team_v2", z2), ("team_v3", z3)]:
        out[k] = 1 / (1 + np.exp(-zz))
    rows = [pd.DataFrame({"text_id": t["text_id"], "model": k, "p_violation": np.round(v, 4)})
            for k, v in out.items()]
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", default=ROOT / "data/processed/texts_dna.csv")
    ap.add_argument("--judgments", nargs="*", default=[])
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--template", default=ROOT / "app/team_eval_template.html")
    ap.add_argument("--out", default=ROOT / "app/team_eval.html")
    ap.add_argument("--max-chars", type=int, default=1200)
    args = ap.parse_args()

    t = pd.read_csv(args.texts)
    frames, files = [], []
    if args.mock:
        base = pd.read_csv(ROOT / "data/processed/judgments_mock.csv")
        frames.append(base[["text_id", "model", "p_violation"]])
        sg = base[base.model == "shieldgemma"].set_index("text_id")["p_violation"].reindex(t["text_id"])
        frames.append(mock_team(t, sg.reset_index(drop=True)))
        files.append("가상값 (judgments_mock.csv + 가상 팀 변형)")
    for f in args.judgments:
        frames.append(pd.read_csv(f)[["text_id", "model", "p_violation"]])
        files.append(Path(f).name)
    j = pd.concat(frames, ignore_index=True).drop_duplicates(["text_id", "model"], keep="last")

    def codes(col):
        cats = sorted(t[col].astype(str).unique())
        idx = {c: i for i, c in enumerate(cats)}
        return cats, t[col].astype(str).map(idx).tolist()

    data = {"mock": args.mock and not args.judgments, "files": files, "n": len(t)}
    for col in ["split", "risk_area", "types_of_harm", "specific_harms", "responder", "question"]:
        data[col + "_dict"], data[col] = codes(col)
    data["text_id"] = t["text_id"].tolist()
    data["response"] = [r if len(r) <= args.max_chars else r[: args.max_chars] + " …(이하 생략)"
                        for r in t["response"].astype(str)]
    data["y"] = t["human_label"].astype(int).tolist()
    data["p"] = {}
    for m, g in j.groupby("model"):
        s = g.set_index("text_id")["p_violation"]
        data["p"][m] = [None if pd.isna(v) else round(float(v), 4) for v in t["text_id"].map(s)]

    html = Path(args.template).read_text(encoding="utf-8")
    html = html.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    Path(args.out).write_text(html, encoding="utf-8")
    print(f"{args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB), 모델: {sorted(data['p'])}")


if __name__ == "__main__":
    main()
