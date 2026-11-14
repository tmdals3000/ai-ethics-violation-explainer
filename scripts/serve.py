"""시스템 화면용 로컬 서버: P4 '다시 판정'과 수정안 저장 (내 PC에서만, 127.0.0.1)

    python scripts/serve.py            → 브라우저에서 http://127.0.0.1:8765
    python scripts/serve.py --port 9000

하는 일
  GET  /                  app/system.html (먼저 build_system.py로 만들기)
  GET  /api/health        서버가 켜져 있는지 (화면 상단에 '로컬 서버 연결됨' 표시)
  POST /api/rejudge       규칙 문장 수정 → 그 규칙만 다시 인코딩해 유사도 재계산 → BFS 상위 5개를 다시 구함 →
                          (1) 고친 규칙이 상위 5개에 들거나 정답인 텍스트는 새 문장으로 판정
                          (2) 정답 텍스트에 예시가 있으면 '새 문장 + 예시'로도 판정
                          (3) 상위 5개에 새로 든 다른 규칙 중 캐시에 없는 것은 원래 문장으로 판정 (캐시에도 저장)
  POST /api/judge_pairs   (텍스트, 규칙) 판정을 원래 문장으로 채움 → judgments_cache.csv에 추가 (P4 τ·우선순위 변경용)
  POST /api/proposals     수정안을 reports/rule_proposals.jsonl에 한 줄씩 저장

모델은 처음 요청 때 한 번만 올림 (RoBERTa 약 1.4GB, ShieldGemma 약 5GB). GPU 판정은 한 번에 하나씩 (잠금).
FAKE_ENCODER=1, FAKE_JUDGE=1이면 모델 없이 흐름만 확인.
"""
import argparse
import json
import os
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pandas as pd

from analysis import Tree, bfs
from common import PROCESSED, REPORTS, ROOT, need

APP = ROOT / "app"
LOCK = threading.Lock()
STATE = {}


def load():
    """데이터는 서버를 켤 때 한 번 읽음."""
    if "tree_df" in STATE:
        return STATE
    need("rule_tree.csv", "similarity.parquet", "embeddings.npz", "texts_all.csv", "gold_paths.csv")
    tree_df = pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False)
    STATE["tree_df"] = tree_df
    STATE["trees"] = {s: Tree(tree_df, s) for s in tree_df.source.unique()}
    STATE["sims"] = pd.read_parquet(PROCESSED / "similarity.parquet").set_index("text_id")
    z = np.load(PROCESSED / "embeddings.npz", allow_pickle=True)
    STATE["emb_index"] = {t: i for i, t in enumerate(z["text_ids"])}
    STATE["text_emb"] = z["text_emb"]
    STATE["dna_input"] = str(z["dna_input"]) if "dna_input" in z.files else "both"
    t = pd.read_csv(PROCESSED / "texts_all.csv", keep_default_na=False).set_index("text_id")
    STATE["texts"] = t
    STATE["gold"] = pd.read_csv(PROCESSED / "gold_paths.csv", keep_default_na=False).set_index("text_id").gold_leaf
    cf = PROCESSED / "judgments_cache.csv"
    c = pd.read_csv(cf) if cf.exists() else pd.DataFrame(columns=["text_id", "rule_id", "with_examples", "variant", "p"])
    c = c[c.with_examples.astype(str) == "False"]
    STATE["cache"] = {(r.variant, r.text_id, r.rule_id): float(r.p) for r in c.itertuples()}
    return STATE


def encoder():
    if "enc" not in STATE:
        from compute_embeddings import get_encoder
        STATE["enc"] = get_encoder(16)[0]
    return STATE["enc"]


def judge_fn(variant):
    key = "judge_" + variant
    if key not in STATE:
        from team_judge import make_fake_team_judge, make_team_judge
        STATE[key] = (make_fake_team_judge if os.environ.get("FAKE_JUDGE") else make_team_judge)([variant])
    return STATE[key]


def rule_dict(tree, leaf, text=None, examples=()):
    return {"rule_text": text if text is not None else tree.text[leaf],
            "path": [tree.name[n] for n in tree.path(leaf)[:-1]], "examples": [list(e) for e in examples]}


def judge_one(variant, tid, rule):
    t = STATE["texts"].loc[tid]
    mode = t.judge_mode
    q = t.context if mode == "response" else ""
    return float(judge_fn(variant)(q, t.text, rule, mode)[variant])


def append_cache(rows):
    if not rows:
        return
    f = PROCESSED / "judgments_cache.csv"
    df = pd.DataFrame(rows, columns=["text_id", "rule_id", "with_examples", "variant", "p"])
    df.to_csv(f, mode="a", header=not f.exists(), index=False)


def rejudge(body):
    st = load()
    src, leaf, text = body["source"], body["rule_id"], str(body["rule_text"]).strip()
    variant, tau, start, top = body["variant"], float(body["tau"]), int(body.get("start_level", 2)), int(body.get("top", 5))
    examples = [(str(t), int(l)) for t, l in body.get("examples", [])]
    tree = st["trees"][src]
    if leaf not in tree.leaves:
        raise ValueError(f"{src} 트리에 없는 규칙: {leaf}")
    ids = [t for t in body["text_ids"] if t in st["sims"].index]
    # 1) 고친 문장만 다시 인코딩 → 유사도
    v = encoder()([text])[0]
    rows = [st["emb_index"][t] for t in ids if t in st["emb_index"]]
    have = [t for t in ids if t in st["emb_index"]]
    new_sim = dict(zip(have, (st["text_emb"][rows] @ v).astype(float)))
    cols = [c for c in st["sims"].columns if c in tree.level]
    t0, n, extra, p_new, p_ex, cache_rows = time.time(), 0, {}, {}, {}, []
    rule_new = rule_dict(tree, leaf, text)
    for tid in have:
        sim = st["sims"].loc[tid, cols].to_dict()
        sim[leaf] = new_sim[tid]
        leaves = bfs(tree, sim, tau, start)[0][:top]
        gold = st["gold"].get(tid, "")
        if leaf in leaves or gold == leaf:
            p_new[tid] = judge_one(variant, tid, rule_new); n += 1
        if gold == leaf and examples:
            p_ex[tid] = judge_one(variant, tid, rule_dict(tree, leaf, text, examples)); n += 1
        for other in leaves:
            if other != leaf and (variant, tid, other) not in st["cache"]:
                p = judge_one(variant, tid, rule_dict(tree, other)); n += 1
                st["cache"][(variant, tid, other)] = p
                extra[f"{tid}|{other}"] = p
                cache_rows.append([tid, other, False, variant, p])
    append_cache(cache_rows)
    print(f"다시 판정: {src} {leaf} ({variant}) {n}건, {time.time() - t0:.0f}초", flush=True)
    return {"sims": {t: round(s, 4) for t, s in new_sim.items()}, "p": p_new, "p_example": p_ex,
            "extra": extra, "n_judged": n}


def judge_pairs(body):
    st = load()
    variant, out, rows = body["variant"], {}, []
    for tid, leaf in body["pairs"]:
        src = st["texts"].loc[tid].source
        if (variant, tid, leaf) in st["cache"]:
            out[f"{tid}|{leaf}"] = st["cache"][(variant, tid, leaf)]
            continue
        p = judge_one(variant, tid, rule_dict(st["trees"][src], leaf))
        st["cache"][(variant, tid, leaf)] = p
        out[f"{tid}|{leaf}"] = p
        rows.append([tid, leaf, False, variant, p])
    append_cache(rows)
    print(f"빠진 판정 채움: {len(rows)}건 ({variant})", flush=True)
    return {"p": out}


def save_proposal(body):
    REPORTS.mkdir(parents=True, exist_ok=True)
    with open(REPORTS / "rule_proposals.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(body, ensure_ascii=False) + "\n")
    return {"ok": True}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(APP), **k)

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")       # file://로 연 화면에서도 호출 가능
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/health"):
            return self._json(200, {"ok": True})
        if self.path in ("/", "/index.html"):
            self.path = "/system.html"
        return super().do_GET()

    def do_POST(self):
        routes = {"/api/rejudge": rejudge, "/api/judge_pairs": judge_pairs, "/api/proposals": save_proposal}
        fn = routes.get(self.path)
        if not fn:
            return self._json(404, {"error": "없는 주소"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if fn is save_proposal:
                return self._json(200, fn(body))
            with LOCK:                                   # GPU는 한 번에 하나씩
                return self._json(200, fn(body))
        except Exception as e:                           # 화면에 이유를 보여 줌
            import traceback
            traceback.print_exc()
            return self._json(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):
        if "/api/" in (args[0] if args else ""):
            super().log_message(fmt, *args)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    if not (APP / "system.html").exists():
        raise SystemExit("app/system.html이 없습니다. 먼저 실행하세요: python scripts/build_system.py")
    load()
    print(f"열기: http://127.0.0.1:{args.port}   (끄기: Ctrl + C)")
    print("모델은 처음 '다시 판정'을 누를 때 올라갑니다 (수십 초).")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
