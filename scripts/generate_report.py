# rag_lab/scripts/generate_report.py

import pandas as pd
from pathlib import Path

results_path = "results/retriever_comparison.csv"
output_path = "docs/final_report.md"

df = pd.read_csv(results_path)

# 平均スコアを計算
summary = df.groupby("retriever").agg({
    "BLEU": "mean",
    "ROUGE-1": "mean",
    "ROUGE-L": "mean",
    "ExactMatch": "mean"
}).round(3).reset_index()

# レポート文字列生成
report_lines = [
    "# 最終評価レポート\n",
    "## 1. 評価結果のまとめ\n",
    "### 平均スコア\n",
    summary.to_markdown(index=False),  # Markdown形式の表で出力
    "\n",
    "## 2. 考察\n",
    "- **Top-k**: 単純な類似度検索で安定した回答。構造は自然だが重複が多い傾向。\n",
    "- **MMR**: 多様性重視。冗長性は低いが回答の精度がやや低め。\n",
    "- **Hybrid**: 意味とキーワードのバランスが取れており、最も自然な回答傾向。\n",
    "\n",
    "## 3. 次のステップ\n",
    "- 誤答分析（Step4）とプロンプト改善（Step5）による再評価\n",
    "- Hybrid Retriever にLoRA最適化済モデルを適用して再検証\n",
    "- 人手評価との比較・乖離分析の導入\n"
]

# 保存
Path("docs").mkdir(exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"✅ Markdownレポートを生成しました：{output_path}")
