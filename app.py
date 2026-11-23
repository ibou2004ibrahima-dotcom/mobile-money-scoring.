import io
import os
import re
import pdfplumber
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

st.set_page_config(
    page_title="TerangaScore PRO | Scoring Crédit Mobile Money",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B1F3A 0%, #153663 100%);
        color: #FFFFFF;
    }
    [data-testid="stSidebar"] * { color: #FFFFFF !important; }
    div[role="radiogroup"] > label {
        background-color: rgba(255, 255, 255, 0.07);
        padding: 12px 16px;
        border-radius: 10px;
        margin-bottom: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        font-weight: 600;
    }
    div[role="radiogroup"] > label:hover {
        background-color: rgba(245, 158, 11, 0.25);
        border-color: #F59E0B;
    }
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        padding: 18px 22px;
        border-radius: 12px;
        border: 1px solid #E2E8F0;
        border-top: 4px solid #0F2C59;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    [data-testid="stMetricValue"] { color: #0F2C59; font-weight: 800; }
    .stButton > button, .stDownloadButton > button {
        background-color: #0F2C59;
        color: #FFFFFF;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #1E4B8F;
        color: #F59E0B;
    }
</style>
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
        for idx in range(1, ws.max_column + 1):
            ws.column_dimensions[get_column_letter(idx)].width = 22
    return out.getvalue()

def build_sample_pdf(sample_df):
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, 800, "RELEVE OFFICIEL MOBILE MONEY - WAVE / ORANGE MONEY")
    c.setFont("Helvetica", 10)
    c.drawString(40, 782, "Titulaire : Client Demonstration Dakar | Devise : FCFA")
    y = 750
    c.setFont("Helvetica-Bold", 10)
    c.drawString(40, y, "Date | Operateur | Type Operation | Montant FCFA | Solde Apres FCFA")
    c.setFont("Helvetica", 10)
    for _, r in sample_df.iterrows():
        y -= 20
        c.drawString(40, y, f"{r['Date']} | {r['Operateur']} | {r['Type_Operation']} | {r['Montant_FCFA']} | {r['Solde_Apres_FCFA']}")
    c.save()
    return buf.getvalue()

def extract_df_from_pdf(uploaded_pdf):
    rows = []
    with pdfplumber.open(uploaded_pdf) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for tbl in tables:
                for row in tbl:
                    if row and len(row) >= 3:
                        rows.append([str(cell or "").strip() for cell in row])
            if not tables:
                text = page.extract_text() or ""
                for line in text.split("\n"):
                    m = re.search(r"(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})", line)
                    nums = re.findall(r"\b\d[\d\s.,]*\d|\b\d+\b", line[m.end():]) if m else []
                    if m and len(nums) >= 1:
                        dt_str = m.group(1)
                        amt_str = re.sub(r"[^\d]", "", nums[-2]) if len(nums) >= 2 else re.sub(r"[^\d]", "", nums[-1])
                        bal_str = re.sub(r"[^\d]", "", nums[-1]) if len(nums) >= 2 else "10000"
                        rows.append([dt_str, line, amt_str, bal_str])
    if len(rows) > 1 and any("date" in str(x).lower() for x in rows):
        return pd.DataFrame(rows[1:], columns=rows)
    return pd.DataFrame(rows, columns=["Date", "Type_Operation", "Montant_FCFA", "Solde_Apres_FCFA"][:len(rows)]) if rows else pd.DataFrame()

def parse_statement(raw_df, loan_amt):
    s = raw_df.copy()
    s.columns = [str(c).strip().lower() for c in s.columns]
    d_col = next((c for c in s.columns if "date" in c or "heure" in c), s.columns)
    t_col = next((c for c in s.columns if "type" in c or "op" in c or "libelle" in c), s.columns)
    a_col = next((c for c in s.columns if "montant" in c or "amount" in c), s.columns)
    b_col = next((c for c in s.columns if "solde" in c or "balance" in c), None)
    s["dt"] = pd.to_datetime(s[d_col], errors="coerce", dayfirst=True)
    s = s.dropna(subset=["dt"]).sort_values("dt")
    s["amt"] = pd.to_numeric(s[a_col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce").fillna(0).abs()
    s["op"] = s[t_col].astype(str).str.upper()
    s["is_in"] = s["op"].apply(lambda x: 1 if any(k in x for k in ["DEPOT", "RECU", "CREDIT", "CASH_IN", "P2P_IN", "ENTR"]) else 0)
    if b_col is not None:
        s["bal"] = pd.to_numeric(s[b_col].astype(str).str.replace(r"[^\d.-]", "", regex=True), errors="coerce").fillna(0)
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
    prob = float(model.predict_proba(X_c)[0][1])
    score = int(np.round(300 + (1 - prob) * 550))
    r1, r2, r3 = st.columns(3)
    r1.metric("Score de Solvabilité (300 - 850)", f"{score} / 850")
    r2.metric("Risque de défaut estimé", f"{prob:.1%}")
    with r3:
        if score >= 600:
            st.markdown('<div style="background:#ECFDF5;border-left:6px solid #10B981;padding:16px;border-radius:10px;color:#065F46;font-weight:700;">✅ DÉCISION : PRÊT ACCORDÉ (100 %)</div>', unsafe_allow_html=True)
        elif score >= 500:
            st.markdown(f'<div style="background:#FFFBEB;border-left:6px solid #F59E0B;padding:16px;border-radius:10px;color:#92400E;font-weight:700;">⚠️ ACCORD SOUS CONDITIONS<br>Plafond conseillé : {int(loan_amt*0.5):,} FCFA</div>'.replace(",", " "), unsafe_allow_html=True)
        else:st.markdown('<div style="background:#FEF2F2;border-left:6px solid #EF4444;padding:16px;border-radius:10px;color:#991B1B;font-weight:700;">❌ DÉCISION : PRÊT REFUSÉ</div>', unsafe_allow_html=True)

    impacts = (-scaler.transform(X_c) * clf.coef_).flatten()
reasons = sorted(zip(features, impacts), key=lambda pair: abs(pair.__getitem__(1)), reverse=True)[:4]
st.markdown("### 📋 Facteurs clés justifiant la décision (Audit & Comité)")
cols = st.columns(2)
    for i, (f, v) in enumerate(reasons):
        lbl = LABELS.get(f, f)
        if v >= 0:
            cols[i % 2].info(f"🟢 **Point fort :** {lbl}")
        else:
            cols[i % 2].error(f"🔴 **Point de vigilance :** {lbl}")

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 18px 0; border-bottom: 1px solid rgba(255,255,255,0.15); margin-bottom: 18px;">
        <div style="font-size: 2.4rem;">🛡️</div>
        <div style="font-size: 1.5rem; font-weight: 800; letter-spacing: 0.5px;">Teranga<span style="color:#F59E0B;">Score</span> <span style="background:#10B981;font-size:0.75rem;padding:2px 8px;border-radius:12px;">PRO</span></div>
        <div style="font-size: 0.8rem; color: #CBD5E1; margin-top: 4px;">Scoring Crédit Mobile Money • Dakar</div>
    </div>
    """, unsafe_allow_html=True)

    menu = st.radio("NAVIGATION", [
        "📱 Relevé Wave / Orange Money",
        "🔍 Évaluation Individuelle",
        "📊 Portefeuille Excel (.xlsx)",
        "📈 Simulateur Rentabilité",
        "🔒 Confidentialité (CDP Sénégal)"
    ], label_visibility="collapsed")

    st.markdown("""
    <div style="margin-top: 30px; padding: 14px; background: rgba(255,255,255,0.06); border-radius: 10px; font-size: 0.8rem; border-left: 3px solid #F59E0B;">
        <strong>Concepteur :</strong><br>Ibrahima DIALLO<br>
        <span style="color:#CBD5E1;">Élève Analyste Statisticien<br>ENSAE Pierre Ndiaye, Dakar</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"""
<div style="background: linear-gradient(90deg, #0F2C59 0%, #1E4B8F 100%); padding: 18px 24px; border-radius: 12px; border-left: 6px solid #F59E0B; margin-bottom: 22px; color: white;">
    <h2 style="margin:0; font-size: 1.45rem; color: white;">{menu}</h2>
    <p style="margin:4px 0 0 0; font-size: 0.9rem; color: #E2E8F0;">Plateforme d'aide à la décision crédit • Conforme Loi n° 2008-12 (CDP Sénégal)</p>
</div>
""", unsafe_allow_html=True)

if menu == "📱 Relevé Wave / Orange Money":
    consent_ok = st.checkbox("✅ Je certifie que le client a donné son consentement explicite pour l'analyse de son relevé (Loi n° 2008-12 - Sénégal).", value=True)
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
        st.download_button("📥 Télécharger exemple relevé PDF (.pdf)", build_sample_pdf(ex_stmt), "Exemple_Releve_Wave_OM.pdf", "application/pdf", use_container_width=True)
    with b_dl2:
        st.download_button("📥 Télécharger exemple relevé Excel (.xlsx)", to_clean_excel(ex_stmt, "Releve_Client"), "Exemple_Releve_Wave_OM.xlsx", use_container_width=True)
    st.divider()
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        s_loan = st.number_input("Montant demandé (FCFA)", 5000, 1000000, 50000, 5000, key="sl")
    with c_s2:
        s_file = st.file_uploader("Importer relevé Wave / Orange Money (.pdf, .xlsx ou .csv)", type=["pdf", "xlsx", "csv"], disabled=not consent_ok)
    if s_file is not None and consent_ok:
        if s_file.name.lower().endswith(".pdf"):
            rdf = extract_df_from_pdf(s_file)
        elif s_file.name.lower().endswith(".xlsx"):
            rdf = pd.read_excel(s_file)
        else:
            rdf = pd.read_csv(s_file)
        if rdf.empty:
            st.error("Impossible de lire les transactions dans ce fichier. Vérifiez que le PDF contient du texte sélectionnable.")
        else:
            ext = parse_statement(rdf, s_loan)
            st.success(f"✅ {ext['nb_tx_180d']} transactions extraites en mémoire vive (aucune donnée conservée après session).")
            display_scoring(ext, s_loan)

elif menu == "🔍 Évaluation Individuelle":
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
    st.divider()
    display_scoring(inp, loan_amount)

elif menu == "📊 Portefeuille Excel (.xlsx)":
    sdf = df[features].head(15).copy().round(2)
    sdf.insert(0, "ID_Client", [f"CLT_{i:04d}" for i in range(1, 16)])
    st.download_button("📥 Télécharger modèle Excel français (.xlsx)", to_clean_excel(sdf.rename(columns=FR_COLS), "Clients"), "TerangaScore_Modele.xlsx")
    st.divider()
    up = st.file_uploader("Importer fichier Excel (.xlsx)", type=["xlsx"])
    if up is not None:
        b_fr = pd.read_excel(up)
        probs = model.predict_proba(b_fr.rename(columns=INV_FR_COLS)[features])[:, 1]
        scores = np.round(300 + (1 - probs) * 550).astype(int)
        res = b_fr.copy()
        res.insert(1, "Score_Solvabilite_850", scores)
        res.insert(2, "Decision_Comite", np.where(scores >= 600, "ACCORDÉ", np.where(scores >= 500, "SOUS CONDITIONS", "REFUSÉ")))
        st.dataframe(res, use_container_width=True)
        st.download_button("📤 Télécharger rapport décisions (.xlsx)", to_clean_excel(res, "Decisions"), "TerangaScore_Rapport.xlsx")

elif menu == "📈 Simulateur Rentabilité":
    cut = st.slider("Seuil acceptation (Cut-off)", 350, 800, 550, 10)
    rate = st.slider("Taux intérêt (%)", 5, 30, 15) / 100
    df["score"] = np.round(300 + (1 - model.predict_proba(df[features])[:, 1]) * 550).astype(int)
    app_df = df[df["score"] >= cut]
    profit = app_df[app_df["default"] == 0]["loan_amount_fcfa"].sum() * rate - app_df[app_df["default"] == 1]["loan_amount_fcfa"].sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Taux acceptation", f"{len(app_df)/len(df):.1%}")
    m2.metric("Taux défaut", f"{app_df['default'].mean() if len(app_df)>0 else 0:.1%}")
    m3.metric("Profit Net Estimé", f"{profit:,.0f} FCFA".replace(",", " "))

elif menu == "🔒 Confidentialité (CDP Sénégal)":
    st.markdown("""
    <div style="background:white; padding:24px; border-radius:12px; border:1px solid #E2E8F0; line-height:1.7;">
        <h3 style="color:#0F2C59; margin-top:0;">Politique de Confidentialité & Protection des Données Personnelles</h3>
        <p><strong>1. Cadre Juridique :</strong> TerangaScore PRO est conçu dans le strict respect de la <strong>Loi n° 2008-12 du 25 janvier 2008</strong> relative à la protection des données à caractère personnel au Sénégal (supervisée par la <strong>CDP</strong>) et des directives de transparence bancaire de la <strong>BCEAO</strong>.</p>
        <p><strong>2. Minimisation et Anonymisation :</strong> L'algorithme fonctionne exclusivement sur des flux financiers anonymisés (identifiants numériques). Aucun nom, numéro de téléphone ou numéro CNI n'est requis ni exploité.</p>
        <p><strong>3. Traitement Éphémère (Zéro Conservation) :</strong> Les relevés Wave / Orange Money (PDF, Excel ou CSV) sont traités uniquement en mémoire vive (RAM) le temps du calcul et effacés immédiatement après la session.</p>
        <p><strong>4. Contrôle Humain :</strong> TerangaScore PRO est un outil d'aide à la décision : la validation finale d'octroi de crédit reste sous la responsabilité humaine du comité de crédit.</p>
        <p style="margin-bottom:0;"><strong>5. Concepteur & Responsable :</strong> Ibrahima DIALLO — Élève Analyste Statisticien à l'ENSAE Pierre Ndiaye de Dakar, Sénégal.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()
