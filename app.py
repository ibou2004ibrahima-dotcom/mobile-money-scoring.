import io
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

scaler = model.named_steps["standardscaler"]
clf = model.named_steps["logisticregression"]

LABELS = {
    "loan_amount_fcfa": "Montant du prêt demandé (FCFA)",
    "nb_tx_180d": "Nombre de transactions (6 mois)",
    "total_in_180d": "Total entrées d'argent 6 mois (FCFA)",
    "total_out_180d": "Total sorties d'argent 6 mois (FCFA)",
    "avg_balance_180d": "Solde moyen 6 mois (FCFA)",
    "min_balance_180d": "Solde minimum atteint (FCFA)",
    "std_balance_180d": "Stabilité du solde (FCFA)",
    "nb_bills_180d": "Factures payées (6 mois)",
    "nb_airtime_180d": "Achats crédit téléphonique (6 mois)",
    "active_days_180d": "Nombre de jours actifs (6 mois)",
    "nb_tx_30d": "Transactions récentes (30 jours)",
    "total_in_30d": "Entrées d'argent 30 jours (FCFA)",
    "avg_balance_30d": "Solde moyen 30 jours (FCFA)",
    "ratio_in_out_180d": "Ratio Entrées / Sorties",
    "loan_to_income_ratio": "Taux d'endettement (Prêt / Revenu)",
    "activity_trend_30d_vs_avg": "Dynamique d'activité récente"
}

# Dictionnaire de correspondance exacte pour Excel en français
FR_COLS = {
    "loan_amount_fcfa": "Montant_Demande_FCFA",
    "nb_tx_180d": "Nb_Transactions_6_Mois",
    "total_in_180d": "Entrees_6_Mois_FCFA",
    "total_out_180d": "Sorties_6_Mois_FCFA",
    "avg_balance_180d": "Solde_Moyen_6_Mois_FCFA",
    "min_balance_180d": "Solde_Minimum_FCFA",
    "std_balance_180d": "Variabilite_Solde_FCFA",
    "nb_bills_180d": "Factures_Payees_6_Mois",
    "nb_airtime_180d": "Achats_Credit_Tel_6_Mois",
    "active_days_180d": "Jours_Actifs_6_Mois",
    "nb_tx_30d": "Nb_Transactions_30_Jours",
    "total_in_30d": "Entrees_30_Jours_FCFA",
    "avg_balance_30d": "Solde_Moyen_30_Jours_FCFA",
    "ratio_in_out_180d": "Ratio_Entrees_Sorties",
    "loan_to_income_ratio": "Ratio_Endettement",
    "activity_trend_30d_vs_avg": "Tendance_Activite_Recente"
}
INV_FR_COLS = {v: k for k, v in FR_COLS.items()}

def to_clean_excel(dataframe, sheet_name="Scoring_Credit"):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        # Ajustement automatique de la largeur des colonnes pour un rendu propre
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col) + 4
            ws.column_dimensions[col[0].column_letter].width = max(max_len, 16)
    return output.getvalue()

st.title("💳 Moteur de Scoring Crédit - Mobile Money Pro")
st.caption("Plateforme B2B d'aide à la décision et de gestion du risque pour la microfinance (FCFA)")

tab1, tab2, tab3 = st.tabs([
    "🔍 Évaluation Individuelle & Explicabilité",
    "📊 Analyse de Portefeuille Excel (Français)",
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
    prob = float(model.predict_proba(X_client)[:, 1])
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

    X_scaled = scaler.transform(X_client)
    impacts = -X_scaled[0] * clf.coef_[0]
    reasons = sorted(zip(features, impacts), key=lambda x: abs(x[1]), reverse=True)[:4]

    st.subheader("📋 Facteurs clés justifiant cette décision (Audit & Conformité)")
    exp_cols = st.columns(2)
    for idx, (feat, val) in enumerate(reasons):
        label = LABELS.get(feat, feat)
        if val >= 0:
            exp_cols[idx % 2].info(f"🟢 **Point fort :** {label} (Impact positif)")
        else:
            exp_cols[idx % 2].error(f"🔴 **Point de vigilance :** {label} (Facteur de risque)")

# --- ONGLET 2 : ANALYSE DE PORTEFEUILLE EXCEL EN FRANÇAIS ---
with tab2:
    st.subheader("Traitement automatisé des demandes sous Excel (.xlsx)")
    st.markdown("1. Téléchargez le modèle Excel prêt à l'emploi en français.  \n2. Importez-le pour obtenir les scores et décisions de tout le portefeuille.")

    # Création d'un échantillon propre en français avec identifiant client
    sample_df = df[features].head(15).copy().round(0)
    sample_df.insert(0, "ID_Client", [f"CLT_{i:04d}" for i in range(1, len(sample_df) + 1)])
    sample_df_fr = sample_df.rename(columns=FR_COLS)

    excel_sample = to_clean_excel(sample_df_fr, sheet_name="Demandes_Clients")
    st.download_button(
        label="📥 Télécharger le modèle Excel en français (.xlsx)",
        data=excel_sample,
        file_name="Modele_Demandes_Credit_MobileMoney.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    st.divider()
    uploaded = st.file_uploader("Importer votre fichier Excel de demandes clients (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        batch_fr = pd.read_excel(uploaded)
        batch_en = batch_fr.rename(columns=INV_FR_COLS)
        missing = [FR_COLS.get(c, c) for c in features if c not in batch_en.columns]

        if missing:
            st.error(f"Colonnes manquantes dans votre fichier Excel : {missing}")
        else:
            probs = model.predict_proba(batch_en[features])[:, 1]
            scores = np.round(300 + (1 - probs) * 550).astype(int)
            decisions = np.where(scores >= 600, "ACCORDÉ", np.where(scores >= 500, "SOUS CONDITIONS", "REFUSÉ"))

            res_df = batch_fr.copy()
            res_df.insert(1 if "ID_Client" in res_df.columns else 0, "Score_Solvabilite_850", scores)
            res_df.insert(2 if "ID_Client" in res_df.columns else 1, "Risque_Defaut_%", np.round(probs * 100, 1))
            res_df.insert(3 if "ID_Client" in res_df.columns else 2, "Decision_Comite", decisions)

            st.success(f"✅ {len(res_df)} dossiers clients analysés et notés avec succès !")
            st.dataframe(res_df, use_container_width=True)

            excel_result = to_clean_excel(res_df, sheet_name="Decisions_Scoring")
            st.download_button(
                label="📤 Télécharger le rapport complet des décisions (.xlsx)",
                data=excel_result,
                file_name="Rapport_Decisions_Scoring_Credit.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

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
