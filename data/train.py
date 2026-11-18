import joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

df = pd.read_csv("data/dataset_scoring.csv").sort_values("application_date").reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

ignore_cols = ["loan_id", "user_id", "application_date", "duration_days", "default"]
features = [c for c in df.columns if c not in ignore_cols]

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(train_df[features], train_df["default"])

auc = roc_auc_score(test_df["default"], model.predict_proba(test_df[features])[:, 1])
joblib.dump({"model": model, "features": features}, "models/scoring_model.pkl")
print(f"OK train - ROC-AUC : {auc:.3f}")