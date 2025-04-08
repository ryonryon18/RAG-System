# feedback/visualize_retry.py（例として）

import pandas as pd
import matplotlib.pyplot as plt

# 再評価データの読み込み
df = pd.read_csv("results/revised_evaluation.csv")

# 指標ごとの平均値を集計
metrics = ["BLEU", "ROUGE-1", "ROUGE-L", "ExactMatch"]
summary = df[metrics].mean()

# 可視化
plt.figure(figsize=(8, 5))
summary.plot(kind='bar', color='skyblue')
plt.title("再生成による評価指標の平均値")
plt.ylabel("スコア")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis="y")
plt.show()


# 並列で比較できるようにする（元評価 vs 再生成）

df_orig = pd.read_csv("results/error_analysis.csv")
df_retry = pd.read_csv("results/revised_evaluation.csv")

avg_orig = df_orig[metrics].mean()
avg_retry = df_retry[metrics].mean()

comparison_df = pd.DataFrame({
    "Before": avg_orig,
    "After": avg_retry
})

comparison_df.plot(kind="bar", figsize=(10, 6))
plt.title("誤答再生成前後のスコア比較")
plt.ylabel("平均スコア")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

