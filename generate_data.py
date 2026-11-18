import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)
N_USERS = 2000
START_DATE = datetime(2026, 1, 1)
DAYS_HISTORY = 180
transactions, loans = [], []

for user_id in range(1, N_USERS + 1):
    regularity = np.random.beta(2, 2)
    income_level = np.random.choice(["low", "medium", "high"], p=[0.5, 0.35, 0.15])
    base_amount = {"low": 15000, "medium": 50000, "high": 150000}[income_level]
    n_tx = int(np.random.poisson(lam=15 + regularity * 45))
    balance = np.random.uniform(500, base_amount * 0.2)
    total_in, total_out, bill_count = 0, 0, 0
    tx_days = sorted(np.random.randint(0, DAYS_HISTORY, size=n_tx))
    
    for day in tx_days:
        tx_date = START_DATE + timedelta(days=int(day), hours=np.random.randint(7, 22))
        if balance < 2000:
            tx_type = np.random.choice(["CASH_IN", "P2P_IN"], p=[0.7, 0.3])
        else:
            tx_type = np.random.choice(["CASH_IN", "P2P_IN", "CASH_OUT", "P2P_OUT", "AIRTIME", "BILL_PAYMENT"], p=[0.25, 0.15, 0.25, 0.15, 0.12, 0.08])
        if tx_type in ["CASH_IN", "P2P_IN"]:
            amount = max(500, round(np.random.lognormal(np.log(base_amount * 0.4), 0.5), -2))
            balance += amount
            total_in += amount
        else:
            if tx_type == "AIRTIME":
                amount = int(np.random.choice([500, 1000, 2000, 5000]))
            elif tx_type == "BILL_PAYMENT":
                amount = round(np.random.uniform(5000, 25000), -2)
                bill_count += 1
            else:
                amount = round(np.random.uniform(1000, balance * 0.8), -2)
            amount = min(amount, balance)
            if amount < 200:
                continue
            balance -= amount
            total_out += amount
        transactions.append({"user_id": f"USR_{user_id:04d}", "timestamp": tx_date.strftime("%Y-%m-%d %H:%M:%S"), "tx_type": tx_type, "amount_fcfa": amount, "balance_after": round(balance, 2)})
        
    loan_date = START_DATE + timedelta(days=DAYS_HISTORY + np.random.randint(1, 15))
    loan_amount = round(base_amount * np.random.uniform(0.3, 0.8), -3)
    ratio_io = (total_in + 1) / (total_out + 1)
    risk_score = -1.8 * regularity - 0.4 * min(ratio_io, 2.0) - 0.3 * min(bill_count, 5) + (0.5 if balance < 1000 else -0.3) + np.random.normal(0, 0.4)
    default = int(np.random.rand() < (1 / (1 + np.exp(-risk_score))))
    loans.append({"loan_id": f"LN_{user_id:04d}", "user_id": f"USR_{user_id:04d}", "application_date": loan_date.strftime("%Y-%m-%d"), "loan_amount_fcfa": loan_amount, "duration_days": 30, "default": default})

pd.DataFrame(transactions).to_csv("data/transactions.csv", index=False)
pd.DataFrame(loans).to_csv("data/loans.csv", index=False)
print("1/3 - Données générées !")
