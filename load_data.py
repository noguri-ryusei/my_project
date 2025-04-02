# 1. ライブラリのインポート
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pandas.api.types import CategoricalDtype

# データの読み込み
df = pd.read_csv("data/health_fitness_dataset.csv")
# データの確認
print(df.head())
df.to_csv("data/processed_data.csv", index=False)

# 2. 特徴量エンジニアリング
# メッツ表をもとに分類
print(df['activity_type'].unique())
# 置き換え
activity_mapping = {'Walking': 1,'Tennis': 1,'Yoga': 1,'Dancing': 1, 'Weight Training': 2,'Swimming': 2,'Basketball': 2, 'Cycling': 3,'Running': 3,'HIIT': 3}
df['activity_type'] = df['activity_type'].map(activity_mapping)

print(df['smoking_status'].unique())
# 置き換え
smoking_mapping = {'Never': 1, 'Former': 2, 'Current': 3}
df['smoking_status'] = df['smoking_status'].map(smoking_mapping)

# 不要なカラムの削除
df = df.drop(columns=['participant_id', 'date','health_condition'])
# カテゴリ変数のエンコーディング
cat_cols = ['gender', 'intensity']
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

# 3. 三分位でカテゴリに変換（1: 低, 2: 中, 3: 高）
columns_to_transform = ['height_cm', 'weight_kg', 'calories_burned', 'hours_sleep', 'hydration_level', 'bmi', 'resting_heart_rate',
                        'fitness_level', 'daily_steps','avg_heart_rate', 'duration_minutes']
# すべてのカラムに対してqcutを適用
for col in columns_to_transform:
    df[col] = pd.qcut(df[col], q=3, labels=[1, 2, 3])

# 米国心臓協会（AHA）基準
def classify_bp(systolic, diastolic):
    if systolic >= 180 or diastolic >= 120:
        return 'Hypertensive Crisis'
    elif systolic >= 140 or diastolic >= 90:
        return 'Hypertension Stage 2'
    elif systolic >= 130 or diastolic >= 80:
        return 'Hypertension Stage 1'
    elif systolic >= 120 and diastolic < 80:
        return 'Elevated'
    else:
        return 'Normal'
# 高血圧のカテゴリ順を定義
bp_order = ["Normal", "Elevated", "Hypertension Stage 1", "Hypertension Stage 2", "Hypertensive Crisis"]
df['bp_category'] = df.apply(lambda row: classify_bp(row['blood_pressure_systolic'], row['blood_pressure_diastolic']), axis=1)
# カテゴリ型に変換（順序を指定）
bp_dtype = CategoricalDtype(categories=bp_order, ordered=True)
df['bp_category'] = df['bp_category'].astype(bp_dtype)

# 4. 分析
# # カテゴリ変数のリスト（既にqcutでカテゴリ化された変数）
cat_vars = ['height_cm', 'weight_kg', 'calories_burned', 'hours_sleep', 'hydration_level', 'bmi', 
            'resting_heart_rate', 'smoking_status','intensity',
            'fitness_level', 'daily_steps', 'avg_heart_rate', 'duration_minutes']
# 各カテゴリ変数と resting_heart_rate_q のクロス集計
for var in cat_vars:
    print(f"クロス集計: {var} と 高血圧の関係")
    # クロス集計表（行ごとの割合を計算）
    cross_tab = pd.crosstab(df['bp_category'], df[var], normalize='index') * 100
    print(cross_tab)
    print("\n")
    # クロス集計結果をヒートマップで可視化
    plt.figure(figsize=(6, 4))
    sns.heatmap(cross_tab, annot=True, cmap='Blues', fmt='.1f', cbar=False, xticklabels=cross_tab.columns)
    plt.title(f'{var} vs Heart Rate Quartile')
    plt.xlabel(var)
    plt.ylabel('Herat Rate Quartile')
    plt.show()