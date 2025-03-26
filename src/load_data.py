import pandas as pd

# データの読み込み
df = pd.read_csv("data/health_fitness_dataset.csv")

# データの確認
print(df.head())

# 必要に応じてCSVとして保存
df.to_csv("data/processed_data.csv", index=False)
