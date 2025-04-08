# rag_lab/rag_engine.py

import os
from dotenv import load_dotenv
from openai import OpenAI
from retrievers.topk import retrieve_top_k
from retrievers import get_retriever

# APIキーを読み込み
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PROMPT_TEMPLATE = """
あなたは知識豊富な医療アシスタントです。
以下の情報を元に、質問に対して専門的でわかりやすい日本語の回答を行ってください。

### 参考情報:
{context}

### 質問:
{question}

### 回答:
"""

# チャンクをプロンプトにまとめる関数
def build_prompt(question, context_chunks):
    context_text = "\n\n".join(chunk["text"] for chunk in context_chunks)
    return PROMPT_TEMPLATE.format(context=context_text, question=question)

# 実行：Retrieve → Prompt → GPT生成
def generate_answer(question, retriever_method="topk", k=3):
    retriever = get_retriever(retriever_method)
    context_chunks = retriever(question, k=k)
    prompt = build_prompt(question, context_chunks)

    response = client.chat.completions.create(
        model="gpt-4",  # 必要に応じて gpt-3.5 に変更OK
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# 動作確認（直接実行時）
if __name__ == "__main__":
    question = "RAGとは何ですか？"
    answer = generate_answer(question)
    print("🧠 回答:\n", answer)
