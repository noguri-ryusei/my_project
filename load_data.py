# 1. ライブラリのインポート
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

# データの読み込み
df = pd.read_csv("data/health_fitness_dataset.csv")

# データの確認
print(df.head())

# 必要に応じてCSVとして保存
df.to_csv("data/processed_data.csv", index=False)

# メッツ表をもとに分類
print(df['activity_type'].unique())
# 置き換え
activity_mapping = {'Walking': 1,'Tennis': 1,'Yoga': 1,'Dancing': 1, 'Weight Training': 2,'Swimming': 2,'Basketball': 2, 'Cycling': 3,'Running': 3,'HIIT': 3}
df['activity_type'] = df['activity_type'].map(activity_mapping)

# Hypertension（高血圧）をターゲット（1）、それ以外を（0）に変換
df['hypertension'] = (df['health_condition'] == 'Hypertension').astype(int)
# 元の health_condition は不要になるため削除
df = df.drop(columns=['health_condition'])

# 不要なカラムの削除
df = df.drop(columns=['participant_id', 'date'])

# カテゴリ変数のエンコーディング
cat_cols = ['gender', 'intensity']
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# 三分位でカテゴリに変換（1: 低, 2: 中, 3: 高, 4: 超高）
columns_to_transform = ['height_cm', 'weight_kg', 'calories_burned', 'hours_sleep', 'hydration_level', 'bmi', 'resting_heart_rate', 
                        'blood_pressure_systolic','blood_pressure_diastolic','fitness_level', 'daily_steps','avg_heart_rate', 'duration_minutes']

# すべてのカラムに対してqcutを適用
for col in columns_to_transform:
    df[col] = pd.qcut(df[col], q=4, labels=[1, 2, 3, 4])

# カテゴリ変数のリスト（既にqcutでカテゴリ化された変数）
cat_vars = ['height_cm', 'weight_kg', 'calories_burned', 'hours_sleep', 'hydration_level', 'bmi', 
            'resting_heart_rate', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'fitness_level', 'daily_steps', 'avg_heart_rate', 'duration_minutes']

# 高血圧（hypertension）とのクロス集計と可視化
for var in cat_vars:
    print(f"クロス集計: {var} と 高血圧の関係")
    cross_tab = pd.crosstab(df['hypertension'], df[var])
    print(cross_tab)
    print("\n")
    
    # クロス集計結果を可視化
    plt.figure(figsize=(6, 4))
    sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='d', cbar=False, xticklabels=cross_tab.columns, 
                yticklabels=['No Hypertension', 'Hypertension'])
    plt.title(f'{var} vs Hypertension')
    plt.xlabel(var)
    plt.ylabel('Hypertension Status')
    plt.show()
