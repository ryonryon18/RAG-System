# rag_lab/retrievers/mmr.py

import numpy as np
# 相対インポートに変更
from .topk import get_query_embedding  
import faiss
import pickle

def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b_norm)

def retrieve_mmr(query, k=5, lambda_param=0.6):
    # データ読み込み
    index = faiss.read_index("models/faiss_index.bin")
    with open("models/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    query_vec = np.array(get_query_embedding(query)).astype("float32")
    all_vectors = index.reconstruct_n(0, index.ntotal)  # 全ベクトル取得

    selected = []
    candidate_indices = list(range(len(all_vectors)))

    for _ in range(k):
        mmr_scores = []
        for idx in candidate_indices:
            relevance = cosine_similarity(query_vec, all_vectors[idx])
            diversity = max([cosine_similarity(all_vectors[idx], all_vectors[s]) for s in selected] or [0])
            mmr = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append((mmr, idx))

        mmr_scores.sort(reverse=True)
        best_score, selected_idx = mmr_scores[0]
        selected.append(selected_idx)
        candidate_indices.remove(selected_idx)

    return [metadata[i] for i in selected]