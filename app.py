import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openpyxl.styles import Font, PatternFill, Alignment

BRAND_NAME = "TerangaScore"

st.set_page_config(page_title="TerangaScore PRO | Scoring Crédit Mobile Money", page_icon="🛡️", layout="wide")

if os.path.exists("logo_terangascore.png"):
    st.image("logo_terangascore.png", width=520)
else:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0F2C59 0%, #1E4B8F 100%); padding: 22px 28px; border-radius: 14px; border-left: 8px solid #F59E0B; margin-bottom: 22px;">
        <h1 style="color: #FFFFFF; margin: 0; font-size: 2.1rem; font-weight: 800;">
            🛡️ Teranga<span style="color: #F59E0B;">Score</span>
            <span style="background-color: #10B981; color: #FFFFFF; font-size: 0.9rem; padding: 4px 12px; border-radius: 20px; vertical-align: middle; margin-left: 8px;">PRO</span>
        </h1>
        <p style="color: #E5E7EB; margin: 6px 0 0 0; font-size: 1.05rem;">
            L'intelligence prédictive au service de l'inclusion financière • Dakar, Sénégal (FCFA)
        </p>
    </div>
    """, unsafe_allow_html=True)

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
        header_fill = PatternFill(start_color="0F2C59", end_color="0F2C59", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        for row in ws.iter_rows(min_row=1, max_row=1):
            for cell in row:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col) + 4
            ws.column_dimensions[col.column_letter].width = max(max_len, 18)
    return output.getvalue()

def parse_mobile_money_statement(raw_df, loan_amount):
    df_s = raw_df.copy()
    df_s.columns = [str(c).strip().lower() for c in df_s.columns]
    date_col = next((c for c in df_s.columns if "date" in c or "heure" in c or "time" in c), df_s.columns)
    type_col = next((c for c in df_s.columns if "type" in c or "libelle" in c or "operation" in c or "motif" in c), df_s.columns)
    amt_col = next((c for c in df_s.columns if "montant" in c or "amount" in c or "somme" in c), df_s.columns)
    bal_col = next((c for c in df_s.columns if "solde" in c or "balance" in c), None)

    df_s["dt"] = pd.to_datetime(df_s[date_col], errors="coerce")
    df_s = df_s.dropna(subset=["dt"]).sort_values("dt")
    df_s["amt"] = pd.to_numeric(df_s[amt_col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce").fillna(0)
    df_s["op"] = df_s[type_col].astype(str).str.upper()

    in_keywords = ["DEPOT", "RECU", "RECEPTION", "CASH_IN", "P2P_IN", "CREDIT", "ENTR", "VIREMENT RECU"]
    df_s["is_in"] = df_s["op"].apply(lambda x: 1 if any(k in x for k in in_keywords) else 0)
    df_s["amt_abs"] = df_s["amt"].abs()

    if bal_col is not None:
        df_s["bal"] = pd.to_numeric(df_s[bal_col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce").fillna(0)
    else:
        signed = np.where(df_s["is_in"] == 1, df_s["amt_abs"], -df_s["amt_abs"])
        df_s["bal"] = np.maximum(0, np.cumsum(signed) + 5000)

    max_date = df_s["dt"].max()
    df_180 = df_s[df_s["dt"] >= max_date - pd.Timedelta(days=180)]
    df_30 = df_s[df_s["dt"] >= max_date - pd.Timedelta(days=30)]

    nb_tx_180d = len(df_180)
    total_in_180d = df_180.loc[df_180["is_in"] == 1, "amt_abs"].sum()
    total_out_180d = df_180.loc[df_180["is_in"] == 0, "amt_abs"].sum()
    avg_balance_180d = df_180["bal"].mean() if nb_tx_180d > 0 else 0
    min_balance_180d = df_180["bal"].min() if nb_tx_180d > 0 else 0
    std_balance_180d = df_180["bal"].std() if nb_tx_180d > 1 else 0

    bill_kw = ["FACTURE", "SENELEC", "WOYOFAL", "SEN'EAU", "SENEAU", "CANAL", "BILL", "RAPIDO"]
    airtime_kw = ["CREDIT", "PASS", "RECHARGE", "AIRTIME", "ILLIMIX", "SEDDO"]
    nb_bills_180d = df_180["op"].apply(lambda x: 1 if any(k in x for k in bill_kw) else 0).sum()
    nb_airtime_180d = df_180["op"].apply(lambda x: 1 if any(k in x for k in airtime_kw) and not any(b in x for b in bill_kw) else 0).sum()
    active_days_180d = df_180["dt"].dt.date.nunique()

    nb_tx_30d = len(df_30)
    total_in_30d = df_30.loc[df_30["is_in"] == 1, "amt_abs"].sum()
    avg_balance_30d = df_30["bal"].mean() if nb_tx_30d > 0 else avg_balance_180d

    return {
        "loan_amount_fcfa": loan_amount,
        "nb_tx_180d": nb_tx_180d,
        "total_in_180d": total_in_180d,
        "total_out_180d": total_out_180d,
        "avg_balance_180d": avg_balance_180d,
        "min_balance_180d": min_balance_180d,
        "std_balance_180d": 0 if np.isnan(std_balance_180d) else std_balance_180d,
        "nb_bills_180d": nb_bills_180d,
        "nb_airtime_180d": nb_airtime_180d,
        "active_days_180d": active_days_180d,
        "nb_tx_30d": nb_tx_30d,
        "total_in_30d": total_in_30d,
        "avg_balance_30d": avg_balance_30d,
        "ratio_in_out_180d": (total_in_180d + 1) / (total_out_180d + 1),
        "loan_to_income_ratio": loan_amount / (total_in_180d / 6 + 1),
        "activity_trend_30d_vs_avg": nb_tx_30d / (nb_tx_180d / 6 + 0.1)
    }

def render_decision_and_xai(inp_dict, loan_amt):
    X_c = pd.DataFrame([{c: inp_dict.get(c, 0) for c in features}])
    probs_c = model.predict_proba(X_c)
    prob = float(probs_c)
    score = int(np.round(300 + (1 - prob) * 550))

    r1, r2, r3 = st.columns(3)
    r1.metric("Score de Solvabilité (300 - 850)", f"{score} / 850")
    r2.metric("Risque de défaut estimé", f"{prob:.1%}")
    if score >= 600:
        r3.success("✅ DÉCISION : PRÊT ACCORDÉ (100 % du montant)")
    elif score >= 500:
        r3.warning(f"⚠️ ACCORD SOUS CONDITIONS (Plafond conseillé : {int(loan_amt*0.5):,} FCFA)".replace(",", " "))
    else:
        r3.error("❌ DÉCISION : PRÊT REFUSÉ (Risque trop élevé)")

    X_scaled = scaler.transform(X_c)
    impacts = (-X_scaled * clf.coef_)
    reasons = sorted(zip(features, impacts), key=lambda item: abs(float(item)), reverse=True)[:4]

    st.subheader("📋 Facteurs clés justifiant cette décision (Audit & Conformité)")
    exp_cols = st.columns(2)
    for idx, (feat, val) in enumerate(reasons):
        label = LABELS.get(feat, feat)
        if val >= 0:
            exp_cols[idx % 2].info(f"🟢 **Point fort :** {label} (Impact positif)")
        else:
            exp_cols[idx % 2].error(f"🔴 **Point de vigilance :** {label} (Facteur de risque)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📱 Analyse Relevé Wave / Orange Money",
    "🔍 Évaluation Individuelle Manuelle",
    "📊 Analyse de Portefeuille Excel (.xlsx)",
    "📈 Simulateur de Rentabilité (Directeur Crédit)"
])

with tab1:
    st.subheader("Lecture automatique d'un relevé de transactions Wave ou Orange Money")
    st.markdown("Importez l'historique brut des transactions du client (Excel `.xlsx` ou `.csv`) : **TerangaScore PRO** extrait automatiquement ses flux sur 30 et 180 jours et calcule sa solvabilité.")

    sample_stmt = pd.DataFrame([
        {"Date": "2026-06-05 10:15", "Operateur": "Wave", "Type_Operation": "Dépôt marchand", "Montant_FCFA": 45000, "Solde_Apres_FCFA": 48000},
        {"Date": "2026-06-12 18:30", "Operateur": "Wave", "Type_Operation": "Paiement Facture Woyofal", "Montant_FCFA": 10000, "Solde_Apres_FCFA": 38000},
        {"Date": "2026-07-03 09:20", "Operateur": "Orange Money", "Type_Operation": "Transfert reçu", "Montant_FCFA": 60000, "Solde_Apres_FCFA": 98000},