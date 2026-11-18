import pandas as pd
import numpy as np

df_tx = pd.read_csv("data/transactions.csv")
df_loans = pd.read_csv("data/loans.csv")
df_tx["timestamp"] = pd.to_datetime(df_tx["timestamp"])
df_loans["application_date"] = pd.to_datetime(df_loans["application_date"])

merged = pd.merge(df_tx, df_loans[["user_id", "application_date"]], on="user_id", how="inner")
merged["days_before_loan"] = (merged["application_date"] - merged["timestamp"]).dt.days
merged = merged[merged["days_before_loan"] >= 0]

merged["is_in"] = merged["tx_type"].isin(["CASH_IN", "P2P_IN"]).astype(int)
merged["amount_in"] = np.where(merged["is_in"] == 1, merged["amount_fcfa"], 0)
merged["amount_out"] = np.where(merged["is_in"] == 0, merged["amount_fcfa"], 0)

agg_180d = merged.groupby("user_id").agg(
    nb_tx_180d=("tx_type", "count"), total_in_180d=("amount_in", "sum"), total_out_180d=("amount_out", "sum"),
    avg_balance_180d=("balance_after", "mean"), min_balance_180d=("balance_after", "min"), std_balance_180d=("balance_after", "std"),
    nb_bills_180d=("tx_type", lambda x: (x == "BILL_PAYMENT").sum()), nb_airtime_180d=("tx_type", lambda x: (x == "AIRTIME").sum()),
    active_days_180d=("timestamp", lambda x: x.dt.date.nunique())
).reset_index()

recent_30d = merged[merged["days_before_loan"] <= 30]
agg_30d = recent_30d.groupby("user_id").agg(nb_tx_30d=("tx_type", "count"), total_in_30d=("amount_in", "sum"), avg_balance_30d=("balance_after", "mean")).reset_index()

dataset = pd.merge(df_loans, agg_180d, on="user_id", how="left")
dataset = pd.merge(dataset, agg_30d, on="user_id", how="left").fillna(0)
dataset["ratio_in_out_180d"] = (dataset["total_in_180d"] + 1) / (dataset["total_out_180d"] + 1)
dataset["loan_to_income_ratio"] = dataset["loan_amount_fcfa"] / (dataset["total_in_180d"] / 6 + 1)
dataset["activity_trend_30d_vs_avg"] = dataset["nb_tx_30d"] / (dataset["nb_tx_180d"] / 6 + 0.1)

dataset.to_csv("data/dataset_scoring.csv", index=False)
print("OK features")