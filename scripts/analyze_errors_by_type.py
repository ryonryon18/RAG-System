import pandas as pd
import json

# 誤答評価結果を読み込み
errors_df = pd.read_csv("results/error_analysis.csv")

# 質問タイプを読み込み
with open("data/questions.json", encoding="utf-8") as f:
    questions = json.load(f)
type_map = {q["question"]: q.get("type", "unknown") for q in questions}

# 質問タイプを誤答DFにマージ
errors_df["question_type"] = errors_df["question"].map(type_map)

# 集計：質問タイプごとの平均スコア
summary_by_type = errors_df.groupby("question_type")[["BLEU", "ROUGE-1", "ROUGE-L", "ExactMatch"]].mean().round(3)
print("📊 質問タイプ別スコア（平均）:\n", summary_by_type)

# 保存
summary_by_type.to_csv("results/error_summary_by_type.csv")
