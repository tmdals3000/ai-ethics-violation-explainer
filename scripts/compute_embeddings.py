"""규칙 찾기용 임베딩 → similarity.parquet, coords.csv, embeddings.npz  (PDF의 similarity·coords 파일을 직접 만드는 단계)

모델: sentence-transformers/all-roberta-large-v1 (PDF 지정, 공개 모델이라 로그인 불필요)
  - 텍스트와 규칙 노드 문장을 1024차원 벡터로 바꾸고, 길이 1로 정규화해서 내적 = 코사인 유사도
  - 이 모델은 128토큰까지만 읽음 → 앞부분에 핵심이 오도록 입력을 구성

텍스트 입력 (--dna-input)
  dna    : 기본 both = "질문 + 응답". 거절 응답("I can't assist with that.")에는 주제가 없어서
           응답만 넣으면 어떤 규칙인지 찾을 수 없음. 128토큰 제한 때문에 질문을 앞에 둠
  jigsaw : 댓글
만드는 것 (data/processed/)
  similarity.parquet  텍스트 × 규칙 노드 유사도 (행 = text_id, 열 = node_id)
  coords.csv          텍스트 2D 좌표 (P3 지도용): text_id, x, y
  embeddings.npz      텍스트·노드 임베딩 (P4에서 규칙 문장을 고치면 그 노드만 다시 인코딩하면 됨)

    python -m pip install sentence-transformers
    python scripts/compute_embeddings.py
    python scripts/compute_embeddings.py --dna-input question --coords pca
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from common import PROCESSED

MODEL = "sentence-transformers/all-roberta-large-v1"


def get_encoder(batch_size):
    if os.environ.get("FAKE_ENCODER"):                       # 테스트용: 모델 없이 단어 해시 벡터
        import zlib

        def enc(texts):
            m = np.zeros((len(texts), 512), np.float32)
            for i, s in enumerate(texts):
                for w in str(s).lower().split()[:128]:
                    m[i, zlib.crc32(w.strip(".,!?:;\"'()").encode()) % 512] += 1
            return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        return enc, 128

    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device)
    if device == "cuda":
        model.half()
    print(f"모델 {MODEL} ({device}), 최대 {model.max_seq_length}토큰")

    def enc(texts):
        return model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True,
                            show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    return enc, model.max_seq_length


def text_input(t: pd.DataFrame, dna_input: str) -> pd.Series:
    s = t["text"].astype(str)
    dna = t["source"] == "dna"
    if dna_input == "question":
        s = s.where(~dna, t["context"])
    elif dna_input == "both":
        s = s.where(~dna, "Question: " + t["context"].astype(str) + "\nResponse: " + t["text"].astype(str))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=["dna", "jigsaw"],
                    help="임베딩할 출처 (P1·P3은 정답 규칙이 있는 dna, jigsaw만 씀)")
    ap.add_argument("--dna-input", choices=["both", "question", "response"], default="both")
    ap.add_argument("--coords", choices=["tsne", "pca", "none"], default="tsne")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    tree = pd.read_csv(PROCESSED / "rule_tree.csv", keep_default_na=False)
    t = pd.read_csv(PROCESSED / "texts_all.csv", keep_default_na=False)
    t = t[t.source.isin(args.sources)].reset_index(drop=True)
    print(f"텍스트 {len(t):,}건 ({', '.join(f'{s} {n:,}' for s, n in t.source.value_counts().items())}), "
          f"규칙 노드 {len(tree)}개, DNA 입력 = {args.dna_input}")

    enc, max_tokens = get_encoder(args.batch_size)
    t0 = time.time()
    node_emb = enc(tree["node_text"].tolist())
    text_emb = enc(text_input(t, args.dna_input).tolist())
    print(f"인코딩 {time.time() - t0:.0f}초")

    # 출처가 다른 규칙과는 비교하지 않음 (DNA 텍스트는 DNA 규칙 트리 안에서만 찾음) → NaN
    sim = text_emb @ node_emb.T
    same = t["source"].to_numpy()[:, None] == tree["source"].to_numpy()[None, :]
    sim = np.where(same, sim, np.nan).astype(np.float32)
    simdf = pd.DataFrame(sim, columns=tree["node_id"].tolist())
    simdf.insert(0, "text_id", t["text_id"].to_numpy())
    simdf.to_parquet(PROCESSED / "similarity.parquet", index=False)

    np.savez_compressed(PROCESSED / "embeddings.npz", text_ids=t["text_id"].to_numpy(), text_emb=text_emb,
                        node_ids=tree["node_id"].to_numpy(), node_emb=node_emb,
                        dna_input=np.array(args.dna_input), model=np.array(MODEL))

    long = simdf.melt(id_vars="text_id", var_name="node_id", value_name="sim").dropna()
    print(f"similarity.parquet: {sim.shape[0]:,} × {sim.shape[1]} (비교한 쌍 {len(long):,}개), "
          f"유사도 범위 {long.sim.min():.3f} ~ {long.sim.max():.3f}, 중앙값 {long.sim.median():.3f}")

    if args.coords != "none":
        t0 = time.time()
        if args.coords == "pca":
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=42).fit_transform(text_emb)
        else:
            from sklearn.manifold import TSNE
            xy = TSNE(n_components=2, metric="cosine", init="pca", random_state=42,
                      perplexity=min(30, max(5, len(t) // 4))).fit_transform(text_emb)
        pd.DataFrame({"text_id": t["text_id"], "x": xy[:, 0].round(4), "y": xy[:, 1].round(4),
                      "method": args.coords}).to_csv(PROCESSED / "coords.csv", index=False)
        print(f"coords.csv: {args.coords}, {time.time() - t0:.0f}초")


if __name__ == "__main__":
    main()
