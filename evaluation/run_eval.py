# rag_lab/evaluation/run_eval.py

import json
from tqdm import tqdm
import pandas as pd
import sys; sys.path.append(".")
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag_engine import generate_answer
from evaluation.metrics import evaluate_bleu, evaluate_rouge, evaluate_exact_match
from feedback.error_db import save_error
from evaluation.metrics import evaluate_bleu, evaluate_rouge, evaluate_exact_match

input_questions = "data/questions.json"
output_results = "results/retriever_comparison.json"
output_csv = "results/retriever_comparison.csv"

retrievers = ["topk", "mmr", "hybrid"]
k = 3  # 取得数

with open(input_questions, "r", encoding="utf-8") as f:
    questions = json.load(f)

results = []
# 既存のコードに誤答収集を追加
for retriever in retrievers:
    for q in tqdm(questions, desc=f"{retriever}"):
        pred = generate_answer(q["question"], retriever_method=retriever, k=k)
        ref = q.get("ideal_answer", "")

        bleu = evaluate_bleu(pred, ref)
        rouge1, rougeL = evaluate_rouge(pred, ref)
        em = evaluate_exact_match(pred, ref)

        # 誤答例をDBに保存
        if em == 0:
            error_type = "incorrect"
            save_error(q["question"], pred, ref, error_type)

        results.append({
            "id": q["id"],
            "question": q["question"],
            "retriever": retriever,
            "prediction": pred,
            "reference": ref,
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "ROUGE-L": rougeL,
            "ExactMatch": em
        })
# 保存
os.makedirs("results", exist_ok=True)
with open(output_results, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
pd.DataFrame(results).to_csv(output_csv, index=False)



print("✅ 評価完了！")
