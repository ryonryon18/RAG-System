# rag_lab/retrievers/__init__.py

from retrievers.topk import retrieve_top_k
from retrievers.mmr import retrieve_mmr
from retrievers.hybrid import retrieve_hybrid

def get_retriever(method="topk"):
    if method == "topk":
        return retrieve_top_k
    elif method == "mmr":
        return retrieve_mmr
    elif method == "hybrid":
        return retrieve_hybrid
    else:
        raise ValueError(f"Unknown retriever method: {method}")
