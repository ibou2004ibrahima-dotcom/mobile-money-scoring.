# ============================================================
# TerangaScore PRO : partie 1 (imports, chargement, utilitaires)
# Corrections principales :
#  - extract_df_from_pdf : colonnes du DataFrame (columns=rows était faux),
#    lignes de tailles différentes, regex des nombres qui fusionnait
#    les montants, solde "10000" inventé quand il manquait
#  - build_sample_pdf : saut de page quand il y a beaucoup de lignes
#  - chargement du modèle/dataset sécurisé (message clair au lieu d'un crash)
#  - to_clean_excel : nom de feuille limité à 31 caractères (limite Excel)
# ============================================================
import io
import os
import re
import unicodedata

import joblib
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from openpyxl.styles import Alignment, Font, PatternFill
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

st.set_page_config(
    page_title="TerangaScore PRO | Scoring Crédit Mobile Money",
    page_icon="🛡️",
    layout="wide",
)

if os.path.exists("logo_terangascore.png"):
    st.image("logo_terangascore.png", width=520)
else:
    st.markdown(
        """
    <div style="background: linear-gradient(90deg, #0F2C59 0%, #1E4B8F 100%); padding: 20px 26px; border-radius: 12px; border-left: 8px solid #F59E0B; margin-bottom: 20px;">
        <h1 style="color: #FFFFFF; margin: 0; font-size: 2rem;">🛡️ Teranga<span style="color: #F59E0B;">Score</span> <span style="background-color: #10B981; font-size: 0.85rem; padding: 3px 10px; border-radius: 15px;">PRO</span></h1>
        <p style="color: #E5E7EB; margin: 4px 0 0 0;">L'intelligence prédictive au service de l'inclusion financière • Dakar (FCFA)</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_assets():
    saved_ = joblib.load("models/scoring_model.pkl")
    data_ = pd.read_csv("data/dataset_scoring.csv")
    return saved_, data_


try:
    saved, df = load_assets()
    df = df.copy()  # évite de modifier l'objet partagé par le cache
    model, features = saved["model"], saved["features"]
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
except FileNotFoundError as e:
    st.error(f"❌ Fichier introuvable : {e.filename}")
    st.stop()
except (KeyError, AttributeError) as e:
    st.error(f"❌ Modèle invalide (clé ou étape manquante) : {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erreur au chargement du modèle ou des données : {e}")
    st.stop()

LABELS = {
    "loan_amount_fcfa": "Montant du prêt demandé (FCFA)",
    "nb_tx_180d": "Transactions (6 mois)",
    "total_in_180d": "Total entrées 6 mois (FCFA)",
    "total_out_180d": "Total sorties 6 mois (FCFA)",
    "avg_balance_180d": "Solde moyen 6 mois (FCFA)",
    "min_balance_180d": "Solde minimum (FCFA)",
    "std_balance_180d": "Stabilité du solde",
    "nb_bills_180d": "Factures payées (6 mois)",
    "nb_airtime_180d": "Achats crédit téléphonique",
    "active_days_180d": "Jours actifs (6 mois)",
    "nb_tx_30d": "Activité récente (30 jours)",
    "total_in_30d": "Entrées 30 jours (FCFA)",
    "avg_balance_30d": "Solde moyen 30 jours (FCFA)",
    "ratio_in_out_180d": "Ratio Entrées / Sorties",
    "loan_to_income_ratio": "Taux d'endettement",
    "activity_trend_30d_vs_avg": "Dynamique récente",
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
    "activity_trend_30d_vs_avg": "Tendance_Activite_Recente",
}
INV_FR_COLS = {v: k for k, v in FR_COLS.items()}

STD_COLS = ["Date", "Operateur", "Type_Operation", "Montant_FCFA", "Solde_Apres_FCFA"]


def to_clean_excel(dataframe, sheet_name="Scoring"):
    sheet_name = str(sheet_name)[:31]  # limite Excel
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
            ws.column_dimensions[col[0].colu

            # -*- coding: utf-8 -*-
# NOTE : le début de votre fichier reste inchangé (imports pd, np, st, chargement de
# model / scaler / clf / features / df, LABELS, FR_COLS, INV_FR_COLS, build_sample_pdf,
# extract_df_from_pdf, to_clean_excel). Ajoutez seulement cet import en haut :
import unicodedata


# ----------------------------------------------------------------------
# Utilitaires
# ----------------------------------------------------------------------
def _norm(x):
    """Majuscules + suppression des accents (Dépôt -> DEPOT, reçu -> RECU)."""
    txt = unicodedata.normalize("NFKD", str(x))
    return "".join(ch for ch in txt if not unicodedata.combining(ch)).upper()


def _find_col(cols, keys, exclude=()):
    """Retourne la première colonne dont le nom (normalisé) contient un des mots-clés."""
    for c in cols:
        n = _norm(c)
        if any(k in n for k in keys) and not any(e in n for e in exclude):
            return c
    return None


def _to_number(series):
    """Convertit '45 000', '45\u202f000', '1250,50' ou '-2000 FCFA' en nombre."""
    cleaned = (
        series.astype(str)
        .str.replace("\u202f", "", regex=False)
        .str.replace("\xa0", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
        .str.replace(r"[^\d.\-]", "", regex=True)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _has(series, pattern):
    return series.str.contains(pattern, regex=True, na=False)


# ----------------------------------------------------------------------
# Lecture d'un relevé
# ----------------------------------------------------------------------
def parse_statement(raw_df, loan_amt):
    s = raw_df.copy()
    s.columns = [str(c).strip() for c in s.columns]

    d_col = _find_col(s.columns, ["DATE", "HEURE"])
    # "operateur" ne doit PAS être pris pour la colonne du type d'opération
    t_col = _find_col(s.columns, ["TYPE", "LIBELL", "OPERATION", "DESCRIPTION"])
    a_col = _find_col(s.columns, ["MONTANT", "AMOUNT"], exclude=["SOLDE", "BALANCE"])
    b_col = _find_col(s.columns, ["SOLDE", "BALANCE"])

    manquantes = [n for n, c in [("date", d_col), ("type d'opération", t_col), ("montant", a_col)] if c is None]
    if manquantes:
        raise ValueError("Colonnes introuvables : " + ", ".join(manquantes))

    s["dt"] = pd.to_datetime(s[d_col], errors="coerce", dayfirst=True)
    s = s.dropna(subset=["dt"]).sort_values("dt").copy()
    if s.empty:
        raise ValueError("Aucune date valide dans le fichier.")

    s["amt"] = _to_number(s[a_col]).fillna(0).abs()
    s["op"] = s[t_col].map(_norm)  # accents supprimés : "DÉPÔT" -> "DEPOT", "REÇU" -> "RECU"
    s["is_in"] = _has(s["op"], "DEPOT|RECU|CREDIT|CASH_IN|P2P_IN|ENTR").astype(int)

    if b_col is not None:
        s["bal"] = _to_number(s[b_col]).ffill().fillna(0)
    else:
        s["bal"] = np.maximum(0, np.cumsum(np.where(s["is_in"] == 1, s["amt"], -s["amt"])) + 5000)

    m = s["dt"].max()
    d180 = s[s["dt"] >= m - pd.Timedelta(days=180)]
    d30 = s[s["dt"] >= m - pd.Timedelta(days=30)]

    t_in = d180.loc[d180["is_in"] == 1, "amt"].sum()
    t_out = d180.loc[d180["is_in"] == 0, "amt"].sum()
    bills = int(_has(d180["op"], r"FACTURE|SENELEC|WOYOFAL|\bEAU\b|CANAL").sum())
    air = int(_has(d180["op"], r"\bPASS\b|ILLIMIX|RECHARGE|SEDDO").sum())

    return {
        "loan_amount_fcfa": loan_amt, "nb_tx_180d": len(d180), "total_in_180d": t_in, "total_out_180d": t_out,
        "avg_balance_180d": d180["bal"].mean() if len(d180) > 0 else 0,
        "min_balance_180d": d180["bal"].min() if len(d180) > 0 else 0,
        "std_balance_180d": d180["bal"].std() if len(d180) > 1 else 0,
        "nb_bills_180d": bills, "nb_airtime_180d": air, "active_days_180d": d180["dt"].dt.date.nunique(),
        "nb_tx_30d": len(d30), "total_in_30d": d30.loc[d30["is_in"] == 1, "amt"].sum(),
        "avg_balance_30d": d30["bal"].mean() if len(d30) > 0 else 0,
        "ratio_in_out_180d": (t_in + 1) / (t_out + 1), "loan_to_income_ratio": loan_amt / (t_in / 6 + 1),
        "activity_trend_30d_vs_avg": len(d30) / (len(d180) / 6 + 0.1),
    }


# ----------------------------------------------------------------------
# Affichage du score
# ----------------------------------------------------------------------
def display_scoring(inp_dict, loan_amt):
    X_c = pd.DataFrame([{c: inp_dict.get(c, 0) for c in features}])
    prob = float(model.predict_proba(X_c)[0, 1])
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

    # [0] : on prend la première (et seule) ligne, sinon zip() ne produit qu'un seul couple
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


XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

tab1, tab2, tab3, tab4 = st.tabs([
    "📱 Relevé Wave / Orange Money (PDF & Excel)",
    "🔍 Évaluation Individuelle",
    "📊 Portefeuille Excel (.xlsx)",
    "📈 Simulateur Rentabilité"
])

with tab1:
    st.subheader("Lecture automatique d'un relevé brut Wave ou Orange Money (PDF, Excel ou CSV)")
    ex_stmt = pd.DataFrame([
        {"Date": "2026-06-05", "Operateur": "Wave", "Type_Operation": "Dépôt marchand", "Montant_FCFA": 45000, "Solde_Apres_FCFA": 48000},
        {"Date": "2026-07-12", "Operateur": "Wave", "Type_Operation": "Paiement Facture Woyofal", "Montant_FCFA": 10000, "Solde_Apres_FCFA": 38000},
        {"Date": "2026-08-03", "Operateur": "Orange Money", "Type_Operation": "Transfert reçu", "Montant_FCFA": 60000, "Solde_Apres_FCFA": 98000},
        {"Date": "2026-09-14", "Operateur": "Orange Money", "Type_Operation": "Achat Pass Illimix", "Montant_FCFA": 2000, "Solde_Apres_FCFA": 96000},
        {"Date": "2026-10-02", "Operateur": "Wave", "Type_Operation": "Paiement Facture Sen'Eau", "Montant_FCFA": 12500, "Solde_Apres_FCFA": 83500},
        {"Date": "2026-11-12", "Operateur": "Wave", "Type_Operation": "Paiement Facture Senelec", "Montant_FCFA": 18000, "Solde_Apres_FCFA": 155500}
    ])
    b_dl1, b_dl2 = st.columns(2)
    with b_dl1:
        st.download_button("📥 Télécharger exemple relevé PDF (.pdf)", build_sample_pdf(ex_stmt),
                           "Exemple_Releve_Wave_OM.pdf", "application/pdf")
    with b_dl2:
        st.download_button("📥 Télécharger exemple relevé Excel (.xlsx)", to_clean_excel(ex_stmt, "Releve_Client"),
                           "Exemple_Releve_Wave_OM.xlsx", XLSX_MIME)
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        s_loan = st.number_input("Montant demandé (FCFA)", 5000, 1000000, 50000, 5000, key="sl")
    with c_s2:
        s_file = st.file_uploader("Importer relevé Wave / Orange Money (.pdf, .xlsx ou .csv)", type=["pdf", "xlsx", "csv"])
    if s_file is not None:
        try:
            nom = s_file.name.lower()
            if nom.endswith(".pdf"):
                rdf = extract_df_from_pdf(s_file)
            elif nom.endswith(".xlsx"):
                rdf = pd.read_excel(s_file)
            else:
                # sep=None : détecte automatiquement ; ou , (CSV français)
                rdf = pd.read_csv(s_file, sep=None, engine="python")
            if rdf is None or rdf.empty:
                st.error("Impossible de lire les transactions dans ce fichier. Vérifiez que le PDF contient du texte sélectionnable.")
            else:
                ext = parse_statement(rdf, s_loan)
                st.success(f"✅ {ext['nb_tx_180d']} transactions extraites automatiquement sur {ext['active_days_180d']} jours actifs.")
                display_scoring(ext, s_loan)
        except ValueError as e:
            st.error(f"Fichier non reconnu : {e}")
        except Exception as e:
            st.error(f"Erreur pendant la lecture du fichier : {e}")

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

    # Cohérence des saisies (le 30j ne peut pas dépasser le 6 mois)
    if nb_tx_30d > nb_tx_180d or total_in_30d > total_in:
        st.warning("⚠️ Saisies incohérentes : les valeurs sur 30 jours dépassent celles sur 6 mois.")

    inp = {
        "loan_amount_fcfa": loan_amount, "nb_tx_180d": nb_tx_180d, "total_in_180d": total_in, "total_out_180d": total_out,
        "avg_balance_180d": avg_bal_180d, "min_balance_180d": min_bal, "std_balance_180d": avg_bal_180d * 0.4,
        "nb_bills_180d": nb_bills, "nb_airtime_180d": nb_airtime, "active_days_180d": active_days,
        "nb_tx_30d": nb_tx_30d, "total_in_30d": total_in_30d, "avg_balance_30d": avg_bal_30d,
        "ratio_in_out_180d": (total_in + 1) / (total_out + 1), "loan_to_income_ratio": loan_amount / (total_in / 6 + 1),
        "activity_trend_30d_vs_avg": nb_tx_30d / (nb_tx_180d / 6 + 0.1)
    }
    st.divider()
    display_scoring(inp, loan_amount)

with tab3:
    sdf = df[features].head(15).copy().round(2)
    sdf.insert(0, "ID_Client", [f"CLT_{i:04d}" for i in range(1, len(sdf) + 1)])
    st.download_button("📥 Télécharger modèle Excel français (.xlsx)", to_clean_excel(sdf.rename(columns=FR_COLS), "Clients"),
                       "TerangaScore_Modele.xlsx", XLSX_MIME)
    up = st.file_uploader("Importer fichier Excel (.xlsx)", type=["xlsx"])
    if up is not None:
        try:
            b_fr = pd.read_excel(up)
            b_en = b_fr.rename(columns=INV_FR_COLS)
            manquantes = [c for c in features if c not in b_en.columns]
            if manquantes:
                st.error("Colonnes manquantes : " + ", ".join(FR_COLS.get(c, c) for c in manquantes))
            else:
                X_b = b_en[features].apply(pd.to_numeric, errors="coerce").fillna(0)
                probs = model.predict_proba(X_b)[:, 1]
                scores = np.round(300 + (1 - probs) * 550).astype(int)
                res = b_fr.copy()
                res.insert(1, "Score_Solvabilite_850", scores)
                res.insert(2, "Decision_Comite", np.where(scores >= 600, "ACCORDÉ", np.where(scores >= 500, "SOUS CONDITIONS", "REFUSÉ")))
                st.dataframe(res, use_container_width=True)
                st.download_button("📤 Télécharger rapport décisions (.xlsx)", to_clean_excel(res, "Decisions"),
                                   "TerangaScore_Rapport.xlsx", XLSX_MIME)
        except Exception as e:
            st.error(f"Erreur pendant la lecture du fichier : {e}")

with tab4:
    cut = st.slider("Seuil acceptation (Cut-off)", 350, 800, 550, 10)
    rate = st.slider("Taux intérêt (%)", 5, 30, 15) / 100
    sim = df.copy()  # ne jamais modifier df directement (il peut être mis en cache)
    sim["score"] = np.round(300 + (1 - model.predict_proba(sim[features])[:, 1]) * 550).astype(int)
    app_df = sim[sim["score"] >= cut]
    profit = (app_df[app_df["default"] == 0]["loan_amount_fcfa"].sum() * rate
              - app_df[app_df["default"] == 1]["loan_amount_fcfa"].sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Taux acceptation", f"{len(app_df) / len(sim):.1%}")
    m2.metric("Taux défaut", f"{app_df['default'].mean() if len(app_df) > 0 else 0:.1%}")
    m3.metric("Profit Net Estimé", f"{profit:,.0f} FCFA".replace(",", " "))

st.divider()
st.caption("© 2026 TerangaScore PRO — Développé par Ibrahima | Dakar, Sénégal")