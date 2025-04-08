from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-large"):
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding

if __name__ == "__main__":
    input_path = "data/chunks.json"
    output_path = "data/embedded_chunks.json"

    with open(input_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedded_chunks = []
    for item in tqdm(chunks, desc="Embedding中"):
        embedding = get_embedding(item["text"])
        item["embedding"] = embedding
        embedded_chunks.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Embedding完了：{len(embedded_chunks)}件 → {output_path}")

