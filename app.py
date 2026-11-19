import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openpyxl.styles import Font, PatternFill, Alignment

st.set_page_config(page_title="TerangaScore PRO | Scoring Crédit Mobile Money", page_icon="🛡️", layout="wide")

if os.path.exists("logo_terangascore.png"):
    st.image("logo_terangascore.png", width=520)
else:
    st.markdown("""
    <div style="background: linear-gradient(90deg, #0F2C59 0%, #1E4B8F 100%); padding: 20px 26px; border-radius: 12px; border-left: 8px solid #F59E0B; margin-bottom: 20px;">
        <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem;">🛡️ Teranga<span style="color: #F59E0B;">Score</span> <span style="background-color: #10B981; font-size: 0.85rem; padding: 3px 10px; border-radius: 15px;">PRO</span></h1>
        <p style="color: #E5E7EB; margin: 4px 0 0 0;">L'intelligence prédictive au service de l'inclusion financière • Dakar (FCFA)</p>
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
    "loan_amount_fcfa": "Montant du prêt demandé (FCFA)", "nb_tx_180d": "Transactions (6 mois)",
    "total_in_180d": "Total entrées 6 mois (FCFA)", "total_out_180d": "Total sorties 6 mois (FCFA)",
    "avg_balance_180d": "Solde moyen 6 mois (FCFA)", "min_balance_180d": "Solde minimum (FCFA)",
    "std_balance_180d": "Stabilité du solde", "nb_bills_180d": "Factures payées (6 mois)",
    "nb_airtime_180d": "Achats crédit téléphonique", "active_days_180d": "Jours actifs (6 mois)",
    "nb_tx_30d": "Activité récente (30 jours)", "total_in_30d": "Entrées 30 jours (FCFA)",
    "avg_balance_30d": "Solde moyen 30 jours (FCFA)", "ratio_in_out_180d": "Ratio Entrées / Sorties",
    "loan_to_income_ratio": "Taux d'endettement", "activity_trend_30d_vs_avg": "Dynamique récente"
}

FR_COLS = {
    "loan_amount_fcfa": "Montant_Demande_FCFA", "nb_tx_180d": "Nb_Transactions_6_Mois",
    "total_in_180d": "Entrees_6_Mois_FCFA", "total_out_180d": "Sorties_6_Mois_FCFA",
    "avg_balance_180d": "Solde_Moyen_6_Mois_FCFA", "min_balance_180d": "Solde_Minimum_FCFA",
    "std_balance_180d": "Variabilite_Solde_FCFA", "nb_bills_180d": "Factures_Payees_6_Mois",
    "nb_airtime_180d": "Achats_Credit_Tel_6_Mois", "active_days_180d": "Jours_Actifs_6_Mois",
    "nb_tx_30d": "Nb_Transactions_30_Jours", "total_in_30d": "Entrees_30_Jours_FCFA",
    "avg_balance_30d": "Solde_Moyen_30_Jours_FCFA", "ratio_in_out_180d": "Ratio_Entrees_Sorties",
    "loan_to_income_ratio": "Ratio_Endettement", "activity_trend_30d_vs_avg": "Tendance_Activite_Recente"
}
INV_FR_COLS = {v: k for k, v in FR_COLS.items()}

def to_clean_excel(dataframe, sheet_name="Scoring"):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name=sheet_name)
        ws = writer.sheets[sheet_name]
        for row in ws.iter_rows(min_row=1, max_row=1):
            for cell in row:
                cell.fill = PatternFill(start_color="0F2C59", end_color="0F2C59", fill_type="solid")
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(horizontal="center")
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 22
    return out.getvalue()

def _find_col(columns, keywords, default=None):
    """Retourne le premier nom de colonne contenant un des mots-clés."""
    return next((c for c in columns if any(k in c for k in keywords)), default)

def parse_statement(raw_df, loan_amt):
    s = raw_df.copy()
    s.columns = [str(c).strip().lower() for c in s.columns]
    cols = list(s.columns)

    d_col = _find_col(cols, ["date", "heure"], cols[0])
    t_col = _find_col(cols, ["type", "op", "libelle"], cols[1] if len(cols) > 1 else cols[0])
    a_col = _find_col(cols, ["montant", "amount"], cols[2] if len(cols) > 2 else cols[-1])
    b_col = _find_col(cols, ["solde", "balance"])

    s["dt"] = pd.to_datetime(s[d_col], errors="coerce", dayfirst=True)
    s = s.dropna(subset=["dt"]).sort_values("dt")
    if s.empty:
        raise ValueError("Aucune date valide trouvée dans le relevé.")

    s["amt"] = pd.to_numeric(
        s[a_col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce"
    ).fillna(0).abs()
    s["op"] = s[t_col].astype(str).str.upper()
    s["is_in"] = s["op"].apply(
        lambda x: 1 if any(k in x for k in ["DEPOT", "RECU", "CREDIT", "CASH_IN", "P2P_IN", "ENTR"]) else 0
    )

    if b_col is not None:
        s["bal"] = pd.to_numeric(
            s[b_col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce"
        ).fillna(0)
    else:
        s["bal"] = np.maximum(0, np.cumsum(np.where(s["is_in"] == 1, s["amt"], -s["amt"])) + 5000)

    m = s["dt"].max()
    d180 = s[s["dt"] >= m - pd.Timedelta(days=180)]
    d30 = s[s["dt"] >= m - pd.Timedelta(days=30)]
    t_in = d180.loc[d180["is_in"] == 1, "amt"].sum()
    t_out = d180.loc[d180["is_in"] == 0, "amt"].sum()
    bills = d180["op"].apply(lambda x: 1 if any(k in x for k in ["FACTURE", "SENELEC", "WOYOFAL", "EAU", "CANAL"]) else 0).sum()
    air = d180["op"].apply(lambda x: 1 if any(k in x for k in ["PASS", "ILLIMIX", "RECHARGE", "SEDDO"]) else 0).sum()
    return {
        "loan_amount_fcfa": loan_amt, "nb_tx_180d": len(d180), "total_in_180d": t_in, "total_out_180d": t_out,
        "avg_balance_180d": d180["bal"].mean() if len(d180) > 0 else 0,
        "min_balance_180d": d180["bal"].min() if len(d180) > 0 else 0,
        "std_balance_180d": d180["bal"].std() if len(d180) > 1 else 0,
        "nb_bills_180d": bills, "nb_airtime_180d": air, "active_days_180d": d180["dt"].dt.date.nunique(),
        "nb_tx_30d": len(d30), "total_in_30d": d30.loc[d30["is_in"] == 1, "amt"].sum(),
        "avg_balance_30d": d30["bal"].mean() if len(d30) > 0 else 0,
        "ratio_in_out_180d": (t_in + 1) / (t_out + 1), "loan_to_income_ratio": loan_amt / (t_in / 6 + 1),
        "activity_trend_30d_vs_avg": len(d30) / (len(d180) / 6 + 0.1)
    }

def display_scoring(inp_dict, loan_amt):
    X_c = pd.DataFrame([{c: inp_dict.get(c, 0) for c in features}])

    # Probabilité de la classe "défaut" (classe 1)
    idx_default = list(model.classes_).index(1)
    prob = float(model.predict_proba(X_c)[0, idx_default])
    score = int(np.round(300 + (1 - prob) * 550))

    r1, r2, r3 = st.columns(3)
    r1.metric("Score de Solvabilité (300 - 850)", f"{score} / 850")
    r2.metric("Risque de défaut estimé", f"{prob:.1%}")
    if score >= 600:
        r3.success("✅ DÉCISION : PRÊT ACCORDÉ (100 %)")
    elif score >= 500:
        r3.warning(f"⚠️ ACCORD SOUS CONDITIONS (Max : {int(loan_amt * 0.5):,} FCFA)".replace(",", " "))
    else:
        r3.error("❌ DÉCISION : PRÊT REFUSÉ")

    # Contribution de chaque variable (positive = favorable)
    impacts = (-scaler.transform(X_c) * clf.coef_)[0]
    reasons = sorted(zip(features, impacts), key=lambda item: abs(item[1]), reverse=True)[:4]

    st.subheader("📋 Facteurs clés justifiant la décision (Audit)")
    cols = st.columns(2)
    for i, (f, v) in enumerate(reasons):
        lbl = LABELS.get(f, f)
        if v >= 0:
            cols[i % 2].info(f"🟢 **Point fort :** {lbl}")
        else:
            cols[i % 2].error(f"🔴 **Point de vigilance :** {lbl}")

            # ============================================================
# TerangaScore PRO : bloc des onglets (version corrigée)
# Prérequis définis plus haut dans ton fichier :
# st, pd, np, model, df, features, FR_COLS, INV_FR_COLS,
# to_clean_excel(), parse_statement(), display_scoring()
# ============================================================

# Seuils de décision communs (onglets 3 et 4)
SEUIL_ACCORD = 600
SEUIL_CONDITIONS = 500
SCORE_MIN, SCORE_RANGE = 300, 550

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def calc_scores(data, mdl, feats):
    """Convertit la probabilité de défaut en score sur 850."""
    probs = mdl.predict_proba(data[feats])[:, 1]
    return np.round(SCORE_MIN + (1 - probs) * SCORE_RANGE).astype(int)


def decision(scores):
    return np.where(
        scores >= SEUIL_ACCORD, "ACCORDÉ",
        np.where(scores >= SEUIL_CONDITIONS, "SOUS CONDITIONS", "REFUSÉ"),
    )


tab1, tab2, tab3, tab4 = st.tabs([
    "📱 Relevé Wave / Orange Money",
    "🔍 Évaluation Individuelle",
    "📊 Portefeuille Excel (.xlsx)",
    "📈 Simulateur Rentabilité",
])

# ------------------------------------------------------------
# TAB 1 : Relevé Wave / Orange Money
# ------------------------------------------------------------
with tab1:
    st.subheader("Lecture automatique d'un relevé brut Wave ou Orange Money")

    ex_stmt = pd.DataFrame([
        {"Date": "2026-06-05", "Operateur": "Wave", "Type_Operation": "Dépôt marchand", "Montant_FCFA": 45000, "Solde_Apres_FCFA": 48000},
        {"Date": "2026-07-12", "Operateur": "Wave", "Type_Operation": "Paiement Facture Woyofal", "Montant_FCFA": 10000, "Solde_Apres_FCFA": 38000},
        {"Date": "2026-08-03", "Operateur": "Orange Money", "Type_Operation": "Transfert reçu", "Montant_FCFA": 60000, "Solde_Apres_FCFA": 98000},
        {"Date": "2026-09-14", "Operateur": "Orange Money", "Type_Operation": "Achat Pass Illimix", "Montant_FCFA": 2000, "Solde_Apres_FCFA": 96000},
        {"Date": "2026-10-02", "Operateur": "Wave", "Type_Operation": "Paiement Facture Sen'Eau", "Montant_FCFA": 12500, "Solde_Apres_FCFA": 83500},
        {"Date": "2026-11-12", "Operateur": "Wave", "Type_Operation": "Paiement Facture Senelec", "Montant_FCFA": 18000, "Solde_Apres_FCFA": 65500},
    ])
    st.download_button(
        "📥 Télécharger exemple relevé Wave/OM (.xlsx)",
        to_clean_excel(ex_stmt, "Releve_Client"),
        file_name="Exemple_Releve_Wave_OM.xlsx",
        mime=XLSX_MIME,
    )

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        s_loan = st.number_input("Montant demandé (FCFA)", 5000, 1000000, 50000, 5000, key="sl")
    with c_s2:
        s_file = st.file_uploader(
            "Importer relevé Wave / Orange Money (.xlsx ou .csv)",
            type=["xlsx", "csv"],
        )

    if s_file is not None:
        try:
            nom = s_file.name.lower()
            if nom.endswith(".xlsx"):
                rdf = pd.read_excel(s_file)
            else:
                # sep=None : détecte automatiquement "," ou ";"
                rdf = pd.read_csv(s_file, sep=None, engine="python")
            ext = parse_statement(rdf, s_loan)
        except Exception as e:
            st.error(f"❌ Fichier illisible ou colonnes manquantes : {e}")
        else:
            st.success(
                f"✅ {ext['nb_tx_180d']} transactions analysées "
                f"sur {ext['active_days_180d']} jours actifs."
            )
            display_scoring(ext, s_loan)

# ------------------------------------------------------------
# TAB 2 : Évaluation individuelle
# ------------------------------------------------------------
with tab2:
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
        "loan_amount_fcfa": loan_amount,
        "nb_tx_180d": nb_tx_180d,
        "total_in_180d": total_in,
        "total_out_180d": total_out,
        "avg_balance_180d": avg_bal_180d,
        "min_balance_180d": min_bal,
        "std_balance_180d": avg_bal_180d * 0.4,
        "nb_bills_180d": nb_bills,
        "nb_airtime_180d": nb_airtime,
        "active_days_180d": active_days,
        "nb_tx_30d": nb_tx_30d,
        "total_in_30d": total_in_30d,
        "avg_balance_30d": avg_bal_30d,
        "ratio_in_out_180d": (total_in + 1) / (total_out + 1),
        "loan_to_income_ratio": loan_amount / (total_in / 6 + 1),
        "activity_trend_30d_vs_avg": nb_tx_30d / (nb_tx_180d / 6 + 0.1),
    }
    st.divider()
    display_scoring(inp, loan_amount)

# ------------------------------------------------------------
# TAB 3 : Portefeuille Excel
# ------------------------------------------------------------
with tab3:
    sdf = df[features].head(15).copy().round(2)
    sdf.insert(0, "ID_Client", [f"CLT_{i:04d}" for i in range(1, len(sdf) + 1)])
    st.download_button(
        "📥 Télécharger modèle Excel français (.xlsx)",
        to_clean_excel(sdf.rename(columns=FR_COLS), "Clients"),
        file_name="TerangaScore_Modele.xlsx",
        mime=XLSX_MIME,
    )

    up = st.file_uploader("Importer fichier Excel (.xlsx)", type=["xlsx"])
    if up is not None:
        try:
            b_fr = pd.read_excel(up)
        except Exception as e:
            st.error(f"❌ Fichier illisible : {e}")
        else:
            b = b_fr.rename(columns=INV_FR_COLS)
            manquantes = [c for c in features if c not in b.columns]
            if manquantes:
                st.error("❌ Colonnes manquantes : " + ", ".join(manquantes))
            else:
                scores = calc_scores(b, model, features)
                res = b_fr.copy()
                res.insert(1, "Score_Solvabilite_850", scores)
                res.insert(2, "Decision_Comite", decision(scores))
                st.dataframe(res, width="stretch")
                st.download_button(
                    "📤 Télécharger rapport décisions (.xlsx)",
                    to_clean_excel(res, "Decisions"),
                    file_name="TerangaScore_Rapport.xlsx",
                    mime=XLSX_MIME,
                )

# ------------------------------------------------------------
# TAB 4 : Simulateur de rentabilité
# ------------------------------------------------------------
with tab4:
    cut = st.slider("Seuil acceptation (Cut-off)", 350, 800, 550, 10)
    rate = st.slider("Taux intérêt (%)", 5, 30, 15) / 100

    sim = df.copy()  # évite de modifier le DataFrame global à chaque rerun
    sim["score"] = calc_scores(sim, model, features)
    app_df = sim[sim["score"] >= cut]

    n, n_app = len(sim), len(app_df)
    acc_rate = n_app / n if n else 0
    def_rate = app_df["default"].mean() if n_app else 0
    profit = (
        app_df.loc[app_df["default"] == 0, "loan_amount_fcfa"].sum() * rate
        - app_df.loc[app_df["default"] == 1, "loan_amount_fcfa"].sum()
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Taux acceptation", f"{acc_rate:.1%}")
    m2.metric("Taux défaut", f"{def_rate:.1%}")
    m3.metric("Profit Net Estimé", f"{profit:,.0f} FCFA".replace(",", " "))

st.divider()
st.caption("© 2026 TerangaScore PRO — Développé par Ibrahima | Dakar, Sénégal")