"""
2단계: 조교 파일 대신 쓸 파일을 직접 만들기
- rules.csv      : 규칙 트리 (대분류 5 → 중분류 12 → 소분류 61)
- splits.csv     : dev / test 나누기 (같은 질문의 응답 6개는 같은 쪽으로)
- similarity.csv : 텍스트와 규칙 노드 사이의 유사도 (all-roberta-large-v1)
- coords.csv     : 텍스트 지도용 2차원 좌표 (P3 뷰에서 사용)

실행: python make_assets.py
      빠르게 시험만 할 때: python make_assets.py --fast   (작은 모델 사용)
입력: dna_long.csv (1단계 결과), judgments.csv (승민 님 브랜치 results/ 에서 받은 파일)
"""
import sys
import numpy as np
import pandas as pd

FAST = "--fast" in sys.argv
MODEL = ("sentence-transformers/all-MiniLM-L6-v2" if FAST
         else "sentence-transformers/all-roberta-large-v1")   # PDF에 나온 규칙 찾기 모델

df = pd.read_csv("dna_long.csv")

# ---------------------------------------------------------------
# 1) 규칙 트리 만들기
#    노드마다 id, 단계(level), 부모(parent), 이름, 유사도 계산에 쓸 문장(text)
# ---------------------------------------------------------------
rows = []
for name in sorted(df["rule_l1"].unique()):
    rows.append({"node_id": "L1:" + name, "level": 1, "parent": "ROOT",
                 "name": name, "text": name})
for (l1, l2), _ in df.groupby(["rule_l1", "rule_l2"]):
    rows.append({"node_id": "L2:" + l2, "level": 2, "parent": "L1:" + l1,
                 "name": l2, "text": l2})
for (l2, l3, desc), _ in df.groupby(["rule_l2", "rule_l3", "rule_l3_desc"]):
    text = l3 if l3 == desc else f"{l3}: {desc}"
    rows.append({"node_id": "L3:" + l3, "level": 3, "parent": "L2:" + l2,
                 "name": l3, "text": text})
rules = pd.DataFrame(rows).drop_duplicates("node_id")
rules.to_csv("rules.csv", index=False)
print("규칙 노드 수:", rules["level"].value_counts().sort_index().to_dict())

# 정답 노드 id 를 텍스트 표에도 붙여 둔다 (나중에 BFS 결과와 비교할 때 사용)
df["gold_l1"] = "L1:" + df["rule_l1"]
df["gold_l2"] = "L2:" + df["rule_l2"]
df["gold_l3"] = "L3:" + df["rule_l3"]

# ---------------------------------------------------------------
# 2) dev / test 나누기
#    승민 님 judgments.csv 가 있으면, 거기서 판정된 텍스트 = dev 로 맞춘다
#    (승민 님은 dev 만 판정했으므로 이렇게 해야 팀과 분할이 같아진다)
#    없으면 질문 단위로 50:50 나눈다
#    ※ 같은 질문의 응답 6개는 항상 같은 쪽에 둔다 (test 정보가 새지 않게)
# ---------------------------------------------------------------
import os
if os.path.exists("judgments.csv"):
    judged = set(pd.read_csv("judgments.csv", usecols=["text_id"])["text_id"])
    df["split"] = np.where(df["text_id"].isin(judged), "dev", "test")
    print("분할 기준: 승민 님 judgments.csv (판정된 텍스트 = dev)")
else:
    rng = np.random.default_rng(42)
    qids = df["question_id"].unique()
    dev_q = set(rng.choice(qids, size=len(qids) // 2, replace=False))
    df["split"] = np.where(df["question_id"].isin(dev_q), "dev", "test")
    print("분할 기준: 무작위 50:50 (judgments.csv 없음)")
df[["text_id", "split"]].to_csv("splits.csv", index=False)
print("분할:", df["split"].value_counts().to_dict())

# ---------------------------------------------------------------
# 3) 문장 임베딩 → 유사도
#    판정 대상 텍스트 = 질문 + 응답
# ---------------------------------------------------------------
from sentence_transformers import SentenceTransformer

print("모델 불러오는 중:", MODEL, "(처음 한 번은 다운로드로 시간이 걸려요)")
model = SentenceTransformer(MODEL)

texts = ("Q: " + df["question"] + "\nA: " + df["response"].fillna("")).tolist()
print("텍스트", len(texts), "개 임베딩 중... (노트북 CPU 기준 10~30분)")
text_emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                        normalize_embeddings=True)
rule_emb = model.encode(rules["text"].tolist(), normalize_embeddings=True)
np.save("text_emb.npy", text_emb)

# 정규화했으므로 내적 = 코사인 유사도
sim = text_emb @ rule_emb.T
similarity = pd.DataFrame(sim.round(4), columns=rules["node_id"])
similarity.insert(0, "text_id", df["text_id"].values)
similarity.to_csv("similarity.csv", index=False)
print("similarity.csv 저장:", similarity.shape)

# ---------------------------------------------------------------
# 4) 2차원 좌표 (비슷한 텍스트끼리 가깝게 놓이도록)
# ---------------------------------------------------------------
try:
    import umap
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine",
                   random_state=42).fit_transform(text_emb)
    method = "UMAP"
except ImportError:
    from sklearn.decomposition import PCA
    xy = PCA(n_components=2, random_state=42).fit_transform(text_emb)
    method = "PCA (umap-learn 이 없어서 대신 사용)"
pd.DataFrame({"text_id": df["text_id"], "x": xy[:, 0].round(4),
              "y": xy[:, 1].round(4)}).to_csv("coords.csv", index=False)
print("coords.csv 저장 -", method)

# 정답 노드와 분할 정보를 붙인 표도 다시 저장
df.to_csv("dna_long.csv", index=False)
print("\n완료!")
