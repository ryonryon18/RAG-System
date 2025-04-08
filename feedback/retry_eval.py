import sys
import os
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import evaluate_bleu, evaluate_rouge, evaluate_exact_match

# 読み込む元データ（プロンプト改善後の予測など）
df = pd.read_csv("results/revised_predictions.csv")

# スコア列を計算して追加
scores = []
for _, row in df.iterrows():
    bleu = evaluate_bleu(row['new_prediction'], row['reference'])
    rouge1, rougeL = evaluate_rouge(row['new_prediction'], row['reference'])
    em = evaluate_exact_match(row['new_prediction'], row['reference'])
    scores.append({
        'question': row['question'],
        'new_prediction': row['new_prediction'],
        'reference': row['reference'],
        'BLEU': bleu,
        'ROUGE-1': rouge1,
        'ROUGE-L': rougeL,
        'ExactMatch': em
    })

# 結果を保存
result_df = pd.DataFrame(scores)
result_df.to_csv("results/revised_evaluation.csv", index=False)

print("✅ 再評価完了：results/revised_evaluation.csv に保存されました。")
