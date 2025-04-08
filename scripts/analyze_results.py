import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ファイルパス指定
results_path = "results/retriever_comparison.csv"

# 評価結果読み込み
results_df = pd.read_csv(results_path)

# Retrieverごとの平均スコアを算出
summary_df = results_df.groupby("retriever").agg({
    'BLEU': 'mean',
    'ROUGE-1': 'mean',
    'ROUGE-L': 'mean',
    'ExactMatch': 'mean'
}).reset_index()

# 可視化（棒グラフ）
fig, ax = plt.subplots(figsize=(10, 6))
summary_df.plot(x="retriever", kind="bar", ax=ax)
plt.title("Retriever Performance Comparison")
plt.ylabel("Score")
plt.xlabel("Retriever Type")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# 各評価指標ごとのマトリックス作成
heatmap_df = results_df.pivot_table(index='retriever', columns='question', values='BLEU', aggfunc='mean')

# ヒートマップ可視化
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", fmt='.2f', linewidths=.5)
plt.title("BLEU Score Heatmap by Retriever and Question")
plt.xlabel("Question ID")
plt.ylabel("Retriever Type")
plt.tight_layout()
plt.show()