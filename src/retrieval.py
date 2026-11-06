"""2단계(P1): 유사도 계산·저장 → BFS 규칙 찾기 → top-k·효율·처음 갈라진 단계·경계 사례."""
import numpy as np
import pandas as pd

from src import adapters, data

LEVEL_NAME = {1: "대", 2: "중", 3: "소"}


# ---------- 임베딩·유사도 ----------
def embed_input(cfg, row):
    if cfg["retrieval"]["embed_with_context"].get(row["source"]) and row["context"]:
        return f"{row['context']}\n{row['text']}"
    return row["text"]


def encoder(cfg):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(cfg["retrieval"]["encoder"], device="cuda")


def compute_similarity(cfg, texts, nodes, enc=None):
    """텍스트 × 전체 노드 코사인 유사도 (long format) 와 텍스트 임베딩."""
    enc = enc or encoder(cfg)
    t = enc.encode([embed_input(cfg, r) for _, r in texts.iterrows()], batch_size=32,
                   normalize_embeddings=True, show_progress_bar=False)
    n = enc.encode(nodes["rule_text"].tolist(), normalize_embeddings=True, show_progress_bar=False)
    s = t @ n.T
    sim = pd.DataFrame({"text_id": np.repeat(texts["text_id"].values, len(nodes)),
                        "node_id": np.tile(nodes["node_id"].values, len(texts)),
                        "similarity": s.ravel().round(5)})
    return sim, t


def coords_from_embeddings(text_ids, emb):
    """임시 coords: 임베딩을 PCA 2차원으로 줄인다 (조교 coords 가 오면 교체)."""
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=0).fit_transform(emb)
    return pd.DataFrame({"text_id": text_ids, "x": xy[:, 0].round(4), "y": xy[:, 1].round(4)})


# ---------- BFS ----------
class Tree:
    def __init__(self, nodes):
        self.nodes = nodes
        self.level = dict(zip(nodes.node_id, nodes.level))
        self.parent = dict(zip(nodes.node_id, nodes.parent_id))
        self.children = nodes.groupby("parent_id")["node_id"].apply(list).to_dict()
        self.roots = self.children.get("", [])
        self.total = len(nodes)

    def path(self, nid):
        p = [nid]
        while self.parent.get(p[0]):
            p.insert(0, self.parent[p[0]])
        return p

    def subtree(self, nid):
        out, stack = set(), [nid]
        while stack:
            x = stack.pop()
            out.add(x)
            stack += self.children.get(x, [])
        return out


def bfs(sim, tree, tau, tau_levels=(2,), excluded=frozenset()):
    """sim: {node_id: 유사도}. 한 단계씩 내려가며 τ 미만 노드를 버린다 (모두 τ 미만이면 최고 1개 유지).
    자식이 없는 노드에 도달하면 그 노드가 후보(잎)가 된다. 반환: 후보 순위, 비교한 노드 수, 단계별 유지 노드."""
    frontier = [n for n in tree.roots if n not in excluded]
    compared, leaves, kept = 0, [], {}
    while frontier:
        compared += len(frontier)
        lvl = tree.level[frontier[0]]
        if lvl in tau_levels:
            keep = [n for n in frontier if sim[n] >= tau] or [max(frontier, key=sim.get)]
        else:
            keep = frontier
        kept[lvl] = keep
        nxt = []
        for n in keep:
            ch = [c for c in tree.children.get(n, []) if c not in excluded]
            if ch:
                nxt += ch
            else:
                leaves.append(n)
        frontier = nxt
    ranking = sorted(leaves, key=lambda n: -sim[n])
    return {"ranking": ranking, "compared": compared, "kept": kept}


def first_divergence(pred_path, gold_path):
    for i, (p, g) in enumerate(zip(pred_path, gold_path)):
        if p != g:
            return LEVEL_NAME[i + 1]
    return "none" if len(pred_path) == len(gold_path) else LEVEL_NAME[min(len(pred_path), len(gold_path)) + 1]


def excluded_for(tree, gold_path):
    """'규칙 제외' 조건에서 숨기는 노드: DNA는 정답 위해 유형(중분류) 전체, Jigsaw는 정답 규칙 노드."""
    return frozenset(tree.subtree(gold_path.split(">")[1])) if gold_path else frozenset()


def retrieve_all(cfg, texts, tree, sim_wide, tau=None):
    rc = cfg["retrieval"]
    tau = rc["tau"] if tau is None else tau
    k = cfg["judge"]["top_k_candidates"]
    rows = []
    for _, r in texts.iterrows():
        sim = sim_wide[r["text_id"]]
        res = bfs(sim, tree, tau, tuple(rc["tau_levels"]))
        rank = res["ranking"]
        row = {"text_id": r["text_id"], "source": r["source"], "tau": tau,
               "candidates": "|".join(rank[:k]), "top1": rank[0] if rank else "",
               "n_compared": res["compared"], "efficiency": round(res["compared"] / tree.total, 4),
               "kept_lvl2": "|".join(res["kept"].get(2, []))}
        # 경계 사례: τ±δ 에서 top-1 이 바뀌면
        d = rc["boundary_delta"]
        alt = {bfs(sim, tree, t, tuple(rc["tau_levels"]))["ranking"][0] for t in (tau - d, tau + d)}
        row["boundary"] = bool(alt - {row["top1"]})
        lvl2 = [n for n in tree.nodes.node_id if tree.level[n] == 2]
        row["min_gap_to_tau"] = round(min(abs(sim[n] - tau) for n in lvl2), 4)
        gp = r["gold_path"]
        if gp:
            gold = gp.split(">")
            row["gold_rule_id"] = r["gold_rule_id"]
            row["hit1"] = r["gold_rule_id"] in rank[:1]
            row["hit3"] = r["gold_rule_id"] in rank[:3]
            row["gold_rank"] = rank.index(r["gold_rule_id"]) + 1 if r["gold_rule_id"] in rank else -1
            if len(gold) == 3:  # DNA 중분류 정확도
                mids = list(dict.fromkeys(tree.parent[n] for n in rank))
                row["hit1_mid"], row["hit3_mid"] = gold[1] in mids[:1], gold[1] in mids[:3]
            row["first_divergence"] = first_divergence(tree.path(rank[0]), gold) if rank else LEVEL_NAME[1]
            ex = bfs(sim, tree, tau, tuple(rc["tau_levels"]), excluded_for(tree, gp))
            row["candidates_excluded"] = "|".join(ex["ranking"][:k])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(per):
    g = per[per.get("gold_rule_id", pd.Series(dtype=str)).fillna("") != ""] if "gold_rule_id" in per else per.iloc[:0]
    out = []
    for src, x in per.groupby("source"):
        row = {"source": src, "n": len(x), "efficiency_mean": round(x["efficiency"].mean(), 4),
               "n_boundary": int(x["boundary"].sum())}
        gx = g[g.source == src]
        if len(gx):
            row |= {"top1_acc": round(gx["hit1"].mean(), 4), "top3_acc": round(gx["hit3"].mean(), 4)}
            if gx["hit1_mid"].notna().any():
                row |= {"top1_acc_mid": round(gx["hit1_mid"].astype(float).mean(), 4),
                        "top3_acc_mid": round(gx["hit3_mid"].astype(float).mean(), 4)}
            row["first_divergence"] = gx["first_divergence"].value_counts().to_dict()
        else:
            row["note"] = "정답 규칙 없음 → top-k 계산 불가"
        out.append(row)
    return pd.DataFrame(out)


def wide(sim):
    return {t: dict(zip(g.node_id, g.similarity)) for t, g in sim.groupby("text_id")}


def targets(cfg, texts, sample):
    t = texts[texts.text_id.isin(sample.text_id)]
    if cfg.get("_allow_test"):
        test = data.dev_sample(texts, cfg["data"]["dev_sample_per_source"], cfg["data"]["stratify_sample"], cfg["seed"], split="test")
        t = pd.concat([t, test])
    return t.reset_index(drop=True)


def run(cfg):
    texts, nodes, _, sample = data.load(cfg)
    tgt = targets(cfg, texts, sample)
    sim, is_tmp = adapters.read(cfg, "similarity")
    need = set(tgt.text_id) - (set(sim.text_id) if sim is not None else set())
    if need or (sim is not None and set(nodes.node_id) - set(sim.node_id)):
        if not is_tmp:
            raise SystemExit(f"조교 similarity 에 없는 텍스트 {len(need)}건. 조교에게 확인 필요.")
        print(f"[p1] 유사도 계산: 텍스트 {len(tgt)} × 노드 {len(nodes)}")
        sim, emb = compute_similarity(cfg, tgt, nodes)
        adapters.write_ours(cfg, "similarity", sim)
        np.save(cfg["_data"] / "derived/text_emb.npy", emb)
        if not adapters.has_ta(cfg, "coords"):
            adapters.write_ours(cfg, "coords", coords_from_embeddings(tgt.text_id.values, emb))
    tree = Tree(nodes)
    sw = wide(sim)
    out = cfg["_res"] / "p1"
    out.mkdir(parents=True, exist_ok=True)
    per = retrieve_all(cfg, tgt, tree, sw)
    per.to_csv(out / "retrieval_per_text.csv", index=False)
    s = summarize(per)
    s.to_csv(out / "metrics.csv", index=False)
    dev = tgt[tgt.split == "dev"]  # τ 는 dev 에서만 정한다
    sweep = pd.concat([summarize(retrieve_all(cfg, dev, tree, sw, t)).assign(tau=t) for t in cfg["retrieval"]["tau_grid"]])
    sweep.to_csv(out / "tau_sweep.csv", index=False)
    comp, best = tau_comparison(sweep, cfg["retrieval"]["tau"])
    comp.to_csv(out / "tau_comparison.csv", index=False)
    per[per.boundary].to_csv(out / "boundary_cases.csv", index=False)
    g = per[per.get("gold_rule_id", "").fillna("") != ""] if "gold_rule_id" in per else per.iloc[:0]
    if len(g):
        g.groupby("gold_rule_id").agg(n=("text_id", "size"), top1_acc=("hit1", "mean"), top3_acc=("hit3", "mean")) \
            .round(4).reset_index().to_csv(out / "node_accuracy.csv", index=False)
    print(s.to_string())
    print("τ 비교 (dev):\n", comp.to_string(index=False))
    if best is not None and abs(best - cfg["retrieval"]["tau"]) > 1e-9:
        print(f"[p1] ⚠ dev 선택 규칙의 τ = {best} 가 설정값 {cfg['retrieval']['tau']} 과 다릅니다. configs 를 확인하세요.")


def tau_comparison(sweep, tau_cfg):
    """τ별 dev top-1·top-3·효율 비교표. 선택 규칙: 정답 규칙이 있는 출처의 top-1 평균 최대,
    같으면 효율 값(비교한 노드 비율)이 작은 τ."""
    cols = [c for c in ["source", "tau", "n", "top1_acc", "top3_acc", "top1_acc_mid", "top3_acc_mid",
                        "efficiency_mean", "n_boundary"] if c in sweep]
    comp = sweep[cols].sort_values(["source", "tau"]).reset_index(drop=True)
    g = comp.dropna(subset=["top1_acc"]) if "top1_acc" in comp else comp.iloc[:0]
    best = None
    if len(g):
        score = g.groupby("tau").agg(top1=("top1_acc", "mean"), eff=("efficiency_mean", "mean")).reset_index()
        best = float(score.sort_values(["top1", "eff"], ascending=[False, True]).iloc[0].tau)
    comp["dev_selected"] = comp.tau == best
    comp["config_tau"] = comp.tau == tau_cfg
    return comp, best
