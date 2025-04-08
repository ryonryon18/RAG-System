import sqlite3
import pandas as pd
import sys
import os

# パス設定（これ大事）
sys.path.append(os.path.abspath("."))

from evaluation.metrics import evaluate_bleu, evaluate_rouge, evaluate_exact_match

def analyze_errors():
    conn = sqlite3.connect("data/error_db.sqlite")
    df = pd.read_sql_query("SELECT * FROM errors", conn)
    conn.close()

    analysis = []
    for _, row in df.iterrows():
        bleu = evaluate_bleu(row["prediction"], row["reference"])
        rouge1, rougeL = evaluate_rouge(row["prediction"], row["reference"])
        em = evaluate_exact_match(row["prediction"], row["reference"])

        analysis.append({
            "question": row["question"],
            "prediction": row["prediction"],
            "reference": row["reference"],
            "error_type": row["error_type"],
            "BLEU": bleu,
            "ROUGE-1": rouge1,
            "ROUGE-L": rougeL,
            "ExactMatch": em
        })

    return pd.DataFrame(analysis)

if __name__ == "__main__":
    df = analyze_errors()
    df.to_csv("results/error_analysis.csv", index=False)
    print("✅ 誤答分析CSVを出力しました：results/error_analysis.csv")
