import pandas as pd
from pathlib import Path

# 改善ロジック：誤答に対してプロンプトを改良（シンプル例）
def improve_prompt(question):
    return f"次の質問に、より詳細かつ専門的に答えてください：{question}"

# 疑似的に改善後の予測を生成（本番ではLLMに投げる）
def simulate_prediction(improved_prompt):
    return f"【生成】{improved_prompt} に対する回答。"

# ファイル読み込み（前段階で export された誤答ファイルを想定）
df = pd.read_csv("results/error_analysis.csv")  # または "results/errors.csv"

# 新しい列：改善後プロンプトと予測を追加
df["improved_prompt"] = df["question"].apply(improve_prompt)
df["new_prediction"] = df["improved_prompt"].apply(simulate_prediction)

# 保存
Path("results").mkdir(parents=True, exist_ok=True)
df[["question", "reference", "new_prediction"]].to_csv("results/revised_predictions.csv", index=False)

print("✅ 改善プロンプトで再生成 → 保存完了：results/revised_predictions.csv")
