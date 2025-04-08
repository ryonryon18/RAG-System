# rag_lab/scripts/chunking.py

import os
import json

def split_into_chunks(text, chunk_size=300, overlap=50):
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(' '.join(chunk))
    return chunks

def process_file(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = split_into_chunks(text)
    return chunks

if __name__ == "__main__":
    input_dir = "data/raw_docs"
    output_file = "data/chunks.json"
    all_chunks = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".md") or filename.endswith(".txt"):
            chunks = process_file(os.path.join(input_dir, filename))
            for chunk in chunks:
                all_chunks.append({"source": filename, "text": chunk})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ チャンク化完了：{len(all_chunks)}チャンク -> {output_file}")

