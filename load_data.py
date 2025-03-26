# 1. ライブラリのインポート
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# ランダムフォレスト
from sklearn.ensemble import RandomForestClassifier
# ロジスティック回帰
from sklearn.linear_model import LogisticRegression
# 決定木分析
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

