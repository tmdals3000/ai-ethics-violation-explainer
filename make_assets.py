"""
2단계: 규칙 트리 · 유사도 · 좌표 · 판정 입력 파일 만들기

실행: python make_assets.py          (빠른 시험: python make_assets.py --fast)
입력: texts.csv (1단계 결과)
      text_emb.npy + dna_long.csv (있으면) → 지난번에 계산한 DNA 임베딩을 다시 써서 시간을 아낌
출력:
  rules.csv        규칙 트리 (출처별 트리, 노드마다 임베딩·판정에 쓰는 규칙 문장 text)
  similarity.csv   판정 표본 텍스트 × 규칙 노드 유사도
  coords.csv       텍스트 지도용 2차원 좌표 (UMAP)
  judge_input.csv  Colab 판정 입력 (원문 포함 → GitHub 에 올리지 않음)
  text_emb.npy, text_emb_ids.csv   임베딩 캐시
"""
import os
import sys
import numpy as np
import pandas as pd
from retrieval import Tree, TAU, TOP_K, gold_leaf

FAST = "--fast" in sys.argv
MODEL = ("sentence-transformers/all-MiniLM-L6-v2" if FAST else "sentence-transformers/all-roberta-large-v1")

texts = pd.read_csv("texts.csv", keep_default_na=False)
target = texts[texts["sample"] == 1].reset_index(drop=True)      # 판정·분석 대상 (DNA dev 전체 + 표본)
print("판정·분석 대상 텍스트:", target.groupby("source").size().to_dict())

# ---------------------------------------------------------------
# 1) 규칙 트리 (가정 A-tree: 출처별 트리, A-ruletext: 노드는 문장형 규칙)
#    text      : 유사도 계산(임베딩)에 쓰는 문장. 이름만 있으면 짧아서 유사도가 불안정 → "~하면 안 된다" 형태로 씀
#    judge_text: 판정 모델(ShieldGemma·팀)에 넣는 규칙 문장 = "소분류 이름: 설명" (지난번 판정과 같은 형식)
# ---------------------------------------------------------------
dna = texts[texts["source"] == "do_not_answer"]
rows = []
for l1 in sorted(dna["rule_l1"].unique()):
    rows.append({"node_id": "L1:" + l1, "tree": "dna", "level": 1, "parent": "ROOT", "name": l1,
                 "text": f"Chatbot safety policy area: {l1}."})
for (l1, l2), _ in dna.groupby(["rule_l1", "rule_l2"]):
    rows.append({"node_id": "L2:" + l2, "tree": "dna", "level": 2, "parent": "L1:" + l1, "name": l2,
                 "text": f"The chatbot must not produce responses involving {l2} ({l1})."})
for (l2, l3, desc), _ in dna.groupby(["rule_l2", "rule_l3", "rule_l3_desc"]):
    scope = desc if desc != l3 else l3
    rows.append({"node_id": "L3:" + l3, "tree": "dna", "level": 3, "parent": "L2:" + l2, "name": l3,
                 "text": f"The chatbot must not comply with or assist requests about {l3} ({l2}). "
                         f"This covers: {scope}",
                 # 판정 모델에 넣는 규칙 문장은 지난번 ShieldGemma 판정과 같은 형식으로 둠 (결과를 섞어 쓰려고)
                 "judge_text": l3 if desc == l3 else f"{l3}: {desc}"})
jig = texts[texts["source"] == "jigsaw"]
if len(jig):
    rows.append({"node_id": "J1:Reddit community rules", "tree": "jigsaw", "level": 1, "parent": "ROOT",
                 "name": "Reddit community rules", "text": "Reddit community rules for comments."})
    for (l2, full), _ in jig.groupby(["rule_l2", "rule_l3_desc"]):
        rows.append({"node_id": "J2:" + l2, "tree": "jigsaw", "level": 2, "parent": "J1:Reddit community rules",
                     "name": l2, "text": full, "judge_text": full})
rules = pd.DataFrame(rows).drop_duplicates("node_id")
rules["judge_text"] = rules["judge_text"].fillna(rules["text"])
rules.to_csv("rules.csv", index=False)
print("규칙 노드 수:", rules.groupby(["tree", "level"]).size().to_dict())

# ---------------------------------------------------------------
# 2) 임베딩 (판정 대상만). 지난번 DNA 임베딩이 있으면 다시 씀
# ---------------------------------------------------------------
from sentence_transformers import SentenceTransformer
print("모델 불러오는 중:", MODEL)
model = SentenceTransformer(MODEL)

cache = {}
if os.path.exists("text_emb.npy"):
    emb = np.load("text_emb.npy")
    if os.path.exists("text_emb_ids.csv"):
        ids = pd.read_csv("text_emb_ids.csv")["text_id"].tolist()
    elif os.path.exists("dna_long.csv"):            # 지난번 결과: dna_long.csv 와 같은 순서
        ids = pd.read_csv("dna_long.csv", usecols=["text_id"])["text_id"].tolist()
    else:
        ids = []
    if len(ids) == len(emb) and emb.shape[1] == model.get_sentence_embedding_dimension():
        cache = dict(zip(ids, emb))
        print(f"지난번 임베딩 {len(cache)}개 다시 사용")


def embed_input(r):
    # DNA: 질문 + 응답 / Jigsaw·ProsocialDialog: 판정할 글
    return f"Q: {r.question}\nA: {r.text}" if r.source == "do_not_answer" else str(r.text)


todo = target[~target["text_id"].isin(cache)]
print(f"새로 임베딩할 텍스트 {len(todo)}개 (CPU 기준 DNA 전체면 1시간 이상, 표본만이면 몇 분)")
if len(todo):
    new = model.encode([embed_input(r) for r in todo.itertuples()], batch_size=32,
                       show_progress_bar=True, normalize_embeddings=True)
    cache.update(dict(zip(todo["text_id"], new)))
all_ids = list(cache)
np.save("text_emb.npy", np.stack([cache[i] for i in all_ids]))
pd.DataFrame({"text_id": all_ids}).to_csv("text_emb_ids.csv", index=False)

E = np.stack([cache[i] for i in target["text_id"]])
R = model.encode(rules["text"].tolist(), normalize_embeddings=True)
sim = pd.DataFrame((E @ R.T).round(4), columns=rules["node_id"], index=target["text_id"])
sim.reset_index().to_csv("similarity.csv", index=False)
print("similarity.csv 저장:", sim.shape)

# ---------------------------------------------------------------
# 3) 2차원 좌표 (UMAP, 없으면 PCA)
# ---------------------------------------------------------------
try:
    import umap
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(E)
except ImportError:
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=42).fit_transform(E)
pd.DataFrame({"text_id": target["text_id"], "x": xy[:, 0].round(4), "y": xy[:, 1].round(4)}) \
    .to_csv("coords.csv", index=False)
print("coords.csv 저장")

# ---------------------------------------------------------------
# 4) Colab 판정 입력: 텍스트마다 정답 규칙 + BFS 상위 5개 후보 규칙
# ---------------------------------------------------------------
tree = Tree(rules)
jtext = dict(zip(rules["node_id"], rules["judge_text"]))
cand_rows = []
for i, r in target.iterrows():
    ranked, _, _ = tree.bfs(sim.iloc[i], r["source"], TAU)
    g = gold_leaf(r)
    row = {"text_id": r["text_id"], "source": r["source"], "mode": r["mode"], "question": r["question"],
           "text": r["text"], "human_harmful": r["human_harmful"], "gold_rule": g,
           "gold_rule_text": jtext.get(g, "")}
    for k in range(TOP_K):
        row[f"cand_{k + 1}"] = ranked[k] if k < len(ranked) else ""
        row[f"cand_{k + 1}_text"] = jtext.get(ranked[k], "") if k < len(ranked) else ""
    cand_rows.append(row)
pd.DataFrame(cand_rows).to_csv("judge_input.csv", index=False)
print(f"judge_input.csv 저장 (τ={TAU}, 상위 {TOP_K}개 후보)")
print("\n완료!")
