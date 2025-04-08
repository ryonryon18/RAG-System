# rag_lab/retrievers/hybrid.py

from rank_bm25 import BM25Okapi
from retrievers.topk import get_query_embedding
import faiss
import pickle
import numpy as np

def retrieve_hybrid(query, k=5, alpha=0.5):
    with open("models/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # BM25用データ
    tokenized_corpus = [m["text"].split() for m in metadata]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.split())

    # Denseベクトル
    index = faiss.read_index("models/faiss_index.bin")
    query_vec = np.array(get_query_embedding(query)).astype("float32")
    all_vectors = index.reconstruct_n(0, index.ntotal)

    dense_scores = [np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v)) for v in all_vectors]

    # スコア融合
    hybrid_scores = alpha * np.array(bm25_scores) + (1 - alpha) * np.array(dense_scores)
    top_indices = hybrid_scores.argsort()[::-1][:k]

    return [metadata[i] for i in top_indices]

# テスト
if __name__ == "__main__":
    res = retrieve_hybrid("RAGの仕組みとは？", k=3)
    for i, chunk in enumerate(res, 1):
        print(f"\n--- Hybrid Top {i} ---\n{chunk['text'][:200]}...")
