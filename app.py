import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Scoring Mobile Money Pro", page_icon="💳", layout="wide")

@st.cache_resource
def load_assets():
    saved = joblib.load("models/scoring_model.pkl")
    df = pd.read_csv("data/dataset_scoring.csv")
    return saved, df

saved, df = load_assets()
model, features = saved["model"], saved["features"]

# Récupération des poids du modèle pour expliquer chaque décision
scaler = model.named_steps["standardscaler"]
clf = model.named_steps["logisticregression"]

LABELS = {
    "loan_amount_fcfa": "Montant du prêt demandé",
    "nb_tx_180d": "Volume de transactions (6 mois)",
    "total_in_180d": "Total des entrées d'argent (6 mois)",
    "total_out_180d": "Total des sorties d'argent (6 mois)",
    "avg_balance_180d": "Solde moyen conservé (6 mois)",
    "min_balance_180d": "Solde minimum atteint",
    "std_balance_180d": "Stabilité du solde",
    "nb_bills_180d": "Paiements de factures réguliers",
    "nb_airtime_180d": "Achats de crédit téléphonique",
    "active_days_180d": "Fréquence de jours actifs",
    "nb_tx_30d": "Activité récente (30 derniers jours)",
    "total_in_30d": "Revenus récents (30 derniers jours)",
    "avg_balance_30d": "Liquidité récente (30 jours)",
    "ratio_in_out_180d": "Ratio Entrées / Sorties",
    "loan_to_income_ratio": "Taux d'endettement (Prêt / Revenu)",
    "activity_trend_30d_vs_avg": "Dynamique d'activité récente"
}

st.title("💳 Moteur de Scoring Crédit - Mobile Money Pro")
st.caption("Plateforme B2B d'aide à la décision et de gestion du risque pour la microfinance (FCFA)")

tab1, tab2, tab3 = st.tabs([
    "🔍 Évaluation Individuelle & Explicabilité",
    "📂 Analyse de Relevés / Portefeuille (CSV)",
    "📈 Simulateur de Rentabilité (Directeur Crédit)"
])

# --- ONGLET 1 : ÉVALUATION INDIVIDUELLE + RAISONS ---
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
        avg_bal_180d = st.number_input("Solde moyen 6 mois (FCFA)", 0, value=12000, step=1000)
    with c3:
        avg_bal_30d = st.number_input("Solde moyen 30j (FCFA)", 0, value=15000, step=1000)
        min_bal = st.number_input("Solde minimum (FCFA)", 0, value=1000, step=500)
        nb_bills = st.slider("Factures payées (6 mois)", 0, 20, 4)
        nb_airtime = st.slider("Achats crédit tél. (6 mois)", 0, 50, 12)

    inp = {
        "loan_amount_fcfa": loan_amount, "nb_tx_180d": nb_tx_180d, "total_in_180d": total_in, "total_out_180d": total_out,
        "avg_balance_180d": avg_bal_180d, "min_balance_180d": min_bal, "std_balance_180d": avg_bal_180d * 0.4,
        "nb_bills_180d": nb_bills, "nb_airtime_180d": nb_airtime, "active_days_180d": active_days,
        "nb_tx_30d": nb_tx_30d, "total_in_30d": total_in_30d, "avg_balance_30d": avg_bal_30d,
        "ratio_in_out_180d": (total_in + 1) / (total_out + 1), "loan_to_income_ratio": loan_amount / (total_in / 6 + 1),
        "activity_trend_30d_vs_avg": nb_tx_30d / (nb_tx_180d / 6 + 0.1)
    }
    X_client = pd.DataFrame([{c: inp.get(c, 0) for c in features}])
    prob = float(model.predict_proba(X_client)[0, 1])
    score = int(np.round(300 + (1 - prob) * 550))

    st.divider()
    r1, r2, r3 = st.columns(3)
    r1.metric("Score de Solvabilité (300 - 850)", f"{score} / 850")
    r2.metric("Risque de défaut estimé", f"{prob:.1%}")
    if score >= 600:
        r3.success("✅ DÉCISION : PRÊT ACCORDÉ")
    elif score >= 500:
        r3.warning("⚠️ DÉCISION : ACCORD SOUS CONDITIONS")
    else:
        r3.error("❌ DÉCISION : PRÊT REFUSÉ")

    # Calcul des facteurs explicatifs (XAI)
    X_scaled = scaler.transform(X_client)[0]
    impacts = -X_scaled * clf.coef_[0]  # Positif = améliore le score
    reasons = sorted(zip(features, impacts), key=lambda x: abs(x[1]), reverse=True)[:4]

    st.subheader("📋 Facteurs clés justifiant cette décision (Audit & Conformité)")
    exp_cols = st.columns(2)
    for idx, (feat, val) in enumerate(reasons):
        label = LABELS.get(feat, feat)
        if val >= 0:
            exp_cols[idx % 2].info(f"🟢 **Point fort :** {label} (Impact positif sur la solvabilité)")
        else:
            exp_cols[idx % 2].error(f"🔴 **Point de vigilance :** {label} (Facteur de risque détecté)")

# --- ONGLET 2 : ANALYSE DE FICHIER CLIENT / LOT (CSV) ---
with tab2:
    st.subheader("Scoring automatisé par importation de fichier (Batch Scoring)")
    st.markdown("Importez un fichier CSV d'indicateurs clients ou téléchargez notre modèle d'exemple pour tester le traitement en masse.")
    
    sample_csv = df[features].head(10).to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger un exemple de fichier client (CSV)", sample_csv, "exemple_clients_mobile_money.csv", "text/csv")
    
    uploaded = st.file_uploader("Importer votre fichier CSV de demandes de crédit", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        missing = [c for c in features if c not in batch_df.columns]
        if missing:
            st.error(f"Colonnes manquantes dans le fichier : {missing}")
        else:
            probs = model.predict_proba(batch_df[features])[:, 1]
            batch_df["Score_Credit"] = np.round(300 + (1 - probs) * 550).astype(int)
            batch_df["Risque_Defaut_%"] = np.round(probs * 100, 1)
            batch_df["Decision"] = np.where(batch_df["Score_Credit"] >= 600, "ACCORDE", np.where(batch_df["Score_Credit"] >= 500, "SOUS_CONDITIONS", "REFUSE"))
            st.success(f"✅ {len(batch_df)} dossiers analysés instantanément !")
            st.dataframe(batch_df[["Score_Credit", "Risque_Defaut_%", "Decision"] + features[:4]])
            result_csv = batch_df.to_csv(index=False).encode("utf-8")
            st.download_button("📤 Exporter les décisions pour la banque (CSV)", result_csv, "decisions_scoring_credit.csv", "text/csv")

# --- ONGLET 3 : SIMULATEUR DE RENTABILITÉ ---
with tab3:
    st.subheader("Pilotage du risque et de la rentabilité (Outil Directeur Crédit)")
    cutoff = st.slider("Seuil minimum d'acceptation (Cut-off Score)", 350, 800, 550, 10)
    rate = st.slider("Taux d'intérêt / commission sur le prêt (%)", 5, 30, 15) / 100
    df["score"] = np.round(300 + (1 - model.predict_proba(df[features])[:, 1]) * 550).astype(int)
    app_df = df[df["score"] >= cutoff]
    profit = app_df[app_df["default"] == 0]["loan_amount_fcfa"].sum() * rate - app_df[app_df["default"] == 1]["loan_amount_fcfa"].sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Taux d'acceptation clients", f"{len(app_df)/len(df):.1%} ({len(app_df)} prêts)")
    m2.metric("Taux de défaut du portefeuille", f"{app_df['default'].mean() if len(app_df)>0 else 0:.1%}")
    m3.metric("Profit Net Estimé (FCFA)", f"{profit:,.0f} FCFA".replace(",", " "))
