import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Scoring Mobile Money", page_icon="💳", layout="wide")
saved = joblib.load("models/scoring_model.pkl")
df = pd.read_csv("data/dataset_scoring.csv")
model, features = saved["model"], saved["features"]

st.title("💳 Moteur de Scoring Crédit - Mobile Money")
tab1, tab2 = st.tabs(["🔍 Simulateur Client", "📈 Simulateur Portefeuille"])

with tab1:
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amount = st.number_input("Montant demandé (FCFA)", 5000, 500000, 25000, 5000)
        nb_tx_180d = st.slider("Transactions (6 mois)", 1, 150, 45)
        nb_tx_30d = st.slider("Transactions (30j)", 0, 50, 10)
        active_days = st.slider("Jours actifs (6 mois)", 1, 180, 35)
    with c2:
        total_in = st.number_input("Entrées 6 mois (FCFA)", 0, value=180000, step=10000)
        total_out = st.number_input("Sorties 6 mois (FCFA)", 0, value=150000, step=10000)
        total_in_30d = st.number_input("Entrées 30j (FCFA)", 0, value=35000, step=5000)
        avg_bal_180d = st.number_input("Solde moyen 6 mois", 0, value=12000, step=1000)
    with c3:
        avg_bal_30d = st.number_input("Solde moyen 30j", 0, value=15000, step=1000)
        min_bal = st.number_input("Solde minimum", 0, value=1000, step=500)
        nb_bills = st.slider("Factures payées", 0, 20, 4)
        nb_airtime = st.slider("Achats crédit tél.", 0, 50, 12)

    inp = {
        "loan_amount_fcfa": loan_amount, "nb_tx_180d": nb_tx_180d, "total_in_180d": total_in, "total_out_180d": total_out,
        "avg_balance_180d": avg_bal_180d, "min_balance_180d": min_bal, "std_balance_180d": avg_bal_180d * 0.4,
        "nb_bills_180d": nb_bills, "nb_airtime_180d": nb_airtime, "active_days_180d": active_days,
        "nb_tx_30d": nb_tx_30d, "total_in_30d": total_in_30d, "avg_balance_30d": avg_bal_30d,
        "ratio_in_out_180d": (total_in + 1) / (total_out + 1), "loan_to_income_ratio": loan_amount / (total_in / 6 + 1),
        "activity_trend_30d_vs_avg": nb_tx_30d / (nb_tx_180d / 6 + 0.1)
    }
    prob = float(model.predict_proba(pd.DataFrame([{c: inp.get(c, 0) for c in features}]))[0, 1])
    score = int(np.round(300 + (1 - prob) * 550))
    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Score (300 - 850)", f"{score} / 850")
    r2.metric("Risque de défaut", f"{prob:.1%}")
    if score >= 600: r3.success("✅ PRÊT ACCORDÉ")
    elif score >= 500: r3.warning("⚠️ ACCORD SOUS CONDITIONS")
    else: r3.error("❌ PRÊT REFUSÉ")

with tab2:
    cutoff = st.slider("Seuil d'acceptation (Cut-off)", 350, 800, 550, 10)
    rate = st.slider("Taux d'intérêt (%)", 5, 30, 15) / 100
    df["score"] = np.round(300 + (1 - model.predict_proba(df[features])[:, 1]) * 550).astype(int)
    app_df = df[df["score"] >= cutoff]
    profit = app_df[app_df["default"] == 0]["loan_amount_fcfa"].sum() * rate - app_df[app_df["default"] == 1]["loan_amount_fcfa"].sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Taux d'acceptation", f"{len(app_df)/len(df):.1%}")
    m2.metric("Taux de défaut", f"{app_df['default'].mean() if len(app_df)>0 else 0:.1%}")
    m3.metric("Profit Net (FCFA)", f"{profit:,.0f} FCFA".replace(",", " "))
