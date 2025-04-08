import json

with open("data/embedded_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

lens = [len(item["embedding"]) for item in data]
print("📏 ベクトル次元の一覧:", set(lens))
