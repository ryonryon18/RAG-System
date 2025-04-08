# rag_lab/scripts/vector_store.py

import json
import faiss
import numpy as np
import os
import pickle

def build_faiss_index(embedded_data, dim=3072):
    index = faiss.IndexFlatL2(dim)  # L2距離で近さを測る
    vectors = []
    metadata = []

    for item in embedded_data:
        vectors.append(item["embedding"])
        metadata.append({"source": item["source"], "text": item["text"]})

    vectors_np = np.array(vectors).astype('float32')
    index.add(vectors_np)

    return index, metadata

if __name__ == "__main__":
    input_path = "data/embedded_chunks.json"
    index_path = "models/faiss_index.bin"
    metadata_path = "models/metadata.pkl"

    with open(input_path, "r", encoding="utf-8") as f:
        embedded_data = json.load(f)

    index, metadata = build_faiss_index(embedded_data)

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Faissインデックス作成完了: {len(metadata)}件 → {index_path}")
