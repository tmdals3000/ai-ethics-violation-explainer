"""3주차 시스템 화면 만들기 → app/system.html (필수 뷰 P1~P4 + 공통 필터·선택 연동·원문 보기)

2주차 결과 파일을 한 페이지에 묶고, analysis.py의 계산(BFS, 지표, 원인 규칙, 뒤집힌 판정)을 화면 안에서
그대로 다시 하도록 데이터를 넣음 → τ·임계값·k를 바꾸면 서버 없이 바로 다시 계산.
'다시 판정'(규칙 문장 수정)만 GPU가 필요해서 serve.py를 켜 둔 경우에 동작.

    python scripts/build_system.py                # dev만
    python scripts/build_system.py --include-test # test도 넣음 (화면에서 'test 열기'를 체크해야 보임, PDF 5단계용)
    start app/system.html                         # 또는 python scripts/serve.py 후 http://127.0.0.1:8765

넣는 것
  규칙 트리, dev 텍스트(P1은 DNA·Jigsaw dev 전체, P2~P4는 판정한 표본), 텍스트 × 규칙 노드 유사도,
  기준 판정(Llama Guard·Qwen3Guard·ShieldGemma), 4가지 조건 판정(judgments_conditions.csv),
  후보별 판정 캐시(τ·우선순위를 바꿀 때 '기본' 조건을 다시 모음), t-SNE 좌표
"""
import argparse
import base64
import glob
import json
from datetime import datetime

import numpy as np
import pandas as pd

from common import PROCESSED, ROOT, need

TEMPLATE = ROOT / "app" / "system_template.html"
OUT = ROOT / "app" / "system.html"
MAX_TEXT, MAX_CTX = 3000, 1500


def rnd(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-test", action="store_true")
    ap.add_argument("--baseline", nargs="*", default=[], help="기준 판정 파일 (기본: judgments_baseline_*.csv 전부)")
    ap.add_argument("--start-level", type=int, default=2, help="rule_search.py를 돌린 시작 단계")
    args = ap.parse_args()

    need("texts_all.csv", "rule_tree.csv", "gold_paths.csv", "similarity.parquet", "rule_search.csv",
         "judgments_conditions.csv", "judgments_cache.csv")
    if not TEMPLATE.exists():
        raise SystemExit(f"화면 틀이 없습니다: {TEMPLATE}")
    splits = ["dev", "test"] if args.include_test else ["dev"]

    texts = pd.read_csv(PROCESSED / "texts_all.csv", keep_default_na=False)
    tree = pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False)
    gold = pd.read_csv(PROCESSED / "gold_paths.csv", keep_default_na=False)
    rs = pd.read_csv(PROCESSED / "rule_search.csv", keep_default_na=False)
    cond = pd.read_csv(PROCESSED / "judgments_conditions.csv")
    cache = pd.read_csv(PROCESSED / "judgments_cache.csv")
    files = args.baseline or sorted(glob.glob(str(PROCESSED / "judgments_baseline_*.csv")))
    base = pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame(
        columns=["text_id", "model", "condition", "p_violation"])
    base = base.drop_duplicates(["text_id", "model"], keep="last")

    judged = set(cond.text_id) | set(base.text_id)
    p1_ids = set(rs[rs.split.isin(splits)].text_id)                  # P1: 정답이 있는 dev 전체
    keep = texts[texts.split.isin(splits) & (texts.text_id.isin(p1_ids) | texts.text_id.isin(judged))].copy()
    keep = keep.merge(gold[["text_id", "gold_leaf"]], on="text_id", how="left").fillna({"gold_leaf": ""})
    parent, level = dict(zip(tree.node_id, tree.parent_id)), dict(zip(tree.node_id, tree.level))
    keep["unit"] = [parent[g] if g and level.get(g) == 3 else g for g in keep.gold_leaf]   # 정답의 중분류
    coords_f = PROCESSED / "coords.csv"
    if coords_f.exists():
        co = pd.read_csv(coords_f)[["text_id", "x", "y"]]
        keep = keep.merge(co, on="text_id", how="left")
    else:
        keep["x"], keep["y"] = np.nan, np.nan
    keep = keep.sort_values(["source", "text_id"]).reset_index(drop=True)
    index = {t: i for i, t in enumerate(keep.text_id)}

    data_texts = {
        "id": keep.text_id.tolist(), "src": keep.source.tolist(), "split": keep.split.tolist(),
        "y": keep.human_label.astype(int).tolist(), "gold": keep.gold_leaf.tolist(), "unit": keep.unit.tolist(),
        "resp": keep.responder.tolist(), "mode": keep.judge_mode.tolist(),
        "cx": [rnd(v, 3) for v in keep.x], "cy": [rnd(v, 3) for v in keep.y],
        "text": [str(s)[:MAX_TEXT] for s in keep.text], "ctx": [str(s)[:MAX_CTX] for s in keep.context],
        "judged": [t in judged for t in keep.text_id],
    }

    sims = pd.read_parquet(PROCESSED / "similarity.parquet").set_index("text_id")
    data_sims = {}
    for src in ["dna", "jigsaw"]:
        cols = tree[tree.source == src].node_id.tolist()
        rows = keep.index[(keep.source == src) & keep.text_id.isin(sims.index)].tolist()
        if not rows:
            continue
        m = sims.loc[keep.text_id[rows], cols].to_numpy()
        # float32 그대로 base64로 넣음 → 화면의 BFS가 rule_search.py와 동점 처리까지 똑같음
        data_sims[src] = {"cols": cols, "rows": rows, "b64": base64.b64encode(m.astype("<f4").tobytes()).decode()}

    # 판정: model → condition → [p per text]
    judg = {}

    def put(model, c, tids, ps):
        arr = judg.setdefault(model, {}).setdefault(c, [None] * len(keep))
        for t, p in zip(tids, ps):
            if t in index:
                arr[index[t]] = rnd(p, 6)          # 임계값이 0.0005처럼 작을 수 있어 6자리

    for m in ["llamaguard", "qwen3guard"]:
        b = base[base.model == m]
        put(m, "none", b.text_id, b.p_violation)
    sg = base[(base.model == "shieldgemma") & (base.condition == "generic")]      # Prosocial: 일반 정책
    put("shieldgemma", "base", sg.text_id, sg.p_violation)
    for (m, c), g in cond.dropna(subset=["p_violation"]).groupby(["model", "condition"]):
        put(m, c, g.text_id, g.p_violation)

    cache = cache[cache.with_examples.astype(str) == "False"]
    data_cache = {v: {f"{t}|{r}": rnd(p, 6) for t, r, p in zip(g.text_id, g.rule_id, g.p)}
                  for v, g in cache.groupby("variant")}

    tau = rs.groupby("source").tau.first().to_dict()
    data = {
        "meta": {"built": datetime.now().strftime("%Y-%m-%d %H:%M"), "splits": splits, "tau": tau,
                 "start_level": args.start_level, "top": 5,
                 "variants": sorted(m for m in judg if m.startswith("team_")),
                 "coords": "t-SNE" if coords_f.exists() else ""},
        "nodes": [{"id": r.node_id, "src": r.source, "lv": int(r.level), "par": r.parent_id, "name": r.name,
                   "text": r.node_text} for r in tree.itertuples()],
        "texts": data_texts, "sims": data_sims, "judg": judg, "cache": data_cache,
    }
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/{}", payload)
    OUT.write_text(html, encoding="utf-8")
    n_j = sum(data_texts["judged"])
    print(f"저장: {OUT} ({OUT.stat().st_size / 1e6:.1f}MB) · 텍스트 {len(keep):,}건 (판정한 표본 {n_j:,}건), "
          f"분할 {', '.join(splits)}")
    print("열기: start app\\system.html   ('다시 판정'까지 쓰려면 python scripts\\serve.py 후 http://127.0.0.1:8765)")


if __name__ == "__main__":
    main()
