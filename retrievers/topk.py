# rag_lab/retrievers/topk.py

from openai import OpenAI
import faiss
import numpy as np
import pickle
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_query_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(
        input=[text],  # ← 必ずリストにする（複数対応）
        model=model
    )
    return response.data[0].embedding


# Top-k検索本体
def retrieve_top_k(query, k=5):
    # Faissインデックスとメタ情報読み込み
    index = faiss.read_index("models/faiss_index.bin")
    with open("models/metadata.pkl", "rb") as f:
        metadata = pickle.load(f)

    # クエリのEmbeddingを取得
    query_vector = np.array([get_query_embedding(query)]).astype("float32")

    # Faiss検索
    distances, indices = index.search(query_vector, k)

    results = []
    for i in indices[0]:
        results.append(metadata[i])
    return results

# 動作確認（スクリプト単体テスト）
if __name__ == "__main__":
    query = "RAGの仕組みとは？"
    results = retrieve_top_k(query, k=3)
    for i, item in enumerate(results, 1):
        print(f"\n--- Top {i} ---\n{item['text'][:200]}...")
