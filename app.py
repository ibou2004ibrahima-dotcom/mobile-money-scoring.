import io, os, streamlit as st, pandas as pd, numpy as np, joblib
from openpyxl.styles import Font, PatternFill, Alignment

st.set_page_config(page_title="TerangaScore PRO | Scoring Crédit", page_icon="🛡️", layout="wide")
if os.path.exists("logo_terangascore.png"): st.image("logo_terangascore.png", width=520)
else: st.title("🛡️ TerangaScore PRO")

@st.cache_resource
def load_assets():
    return joblib.load("models/scoring_model.pkl"), pd.read_csv("data/dataset_scoring.csv")

saved, df = load_assets()
model, features = saved["model"], saved["features"]
scaler, clf = model.named_steps["standardscaler"], model.named_steps["logisticregression"]

LABELS = {"loan_amount_fcfa": "Montant demandé", "nb_tx_180d": "Transactions (6 mois)", "total_in_180d": "Entrées (6 mois)", "total_out_180d": "Sorties (6 mois)", "avg_balance_180d": "Solde moyen (6 mois)", "min_balance_180d": "Solde minimum", "std_balance_180d": "Stabilité solde", "nb_bills_180d": "Factures payées", "nb_airtime_180d": "Crédit téléphonique", "active_days_180d": "Jours actifs", "nb_tx_30d": "Transactions (30j)", "total_in_30d": "Entrées (30j)", "avg_balance_30d": "Solde moyen (30j)", "ratio_in_out_180d": "Ratio Entrées/Sorties", "loan_to_income_ratio": "Taux endettement", "activity_trend_30d_vs_avg": "Dynamique récente"}
FR_COLS = {"loan_amount_fcfa": "Montant_Demande_FCFA", "nb_tx_180d": "Nb_Transactions_6_Mois", "total_in_180d": "Entrees_6_Mois_FCFA", "total_out_180d": "Sorties_6_Mois_FCFA", "avg_balance_180d": "Solde_Moyen_6_Mois_FCFA", "min_balance_180d": "Solde_Minimum_FCFA", "std_balance_180d": "Variabilite_Solde_FCFA", "nb_bills_180d": "Factures_Payees_6_Mois", "nb_airtime_180d": "Achats_Credit_Tel_6_Mois", "active_days_180d": "Jours_Actifs_6_Mois", "nb_tx_30d": "Nb_Transactions_30_Jours", "total_in_30d": "Entrees_30_Jours_FCFA", "avg_balance_30d": "Solde_Moyen_30_Jours_FCFA", "ratio_in_out_180d": "Ratio_Entrees_Sorties", "loan_to_income_ratio": "Ratio_Endettement", "activity_trend_30d_vs_avg": "Tendance_Activite_Recente"}
INV_FR_COLS = {v: k for k, v in FR_COLS.items()}

def to_clean_excel(dataframe, sheet_name="Scoring"):
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        dataframe.to_excel(w, index=False, sheet_name=sheet_name)
        ws = w.sheets[sheet_name]
        for row in ws.iter_rows(min_row=1, max_row=1):
            for cell in row:
                cell.fill = PatternFill(start_color="0F2C59", end_color="0F2C59", fill_type="solid")
                cell.font = Font(color="FFFFFF", bold=True)
        for col in ws.columns: ws.column_dimensions[col.column_letter].width = 22
    return out.getvalue()

def parse_stmt(raw_df, loan_amt):
    s = raw_df.copy()
    s.columns = [str(c).strip().lower() for c in s.columns]
    d_col = next((c for c in s.columns if "date" in c), s.columns)
    t_col = next((c for c in s.columns if "type" in c or "op" in c), s.columns)
    a_col = next((c for c in s.columns if "montant" in c), s.columns)
    b_col = next((c for c in s.columns if "solde" in c), None)
    s["dt"] = pd.to_datetime(s[d_col], errors="coerce")
    s = s.dropna(subset=["dt"]).sort_values("dt")
    s["amt"] = pd.to_numeric(s[a_col], errors="coerce").fillna(0).abs()
    s["op"] = s[t_col].astype(str).str.upper()
    s["is_in"] = s["op"].apply(lambda x: 1 if any(k in x for k in ["DEPOT", "RECU", "CREDIT", "CASH_IN"]) else 0)
    s["bal"] = pd.to_numeric(s[b_col], errors="coerce").fillna(0) if b_col else 10000
    m = s["dt"].max()
    d180, d30 = s[s["dt"] >= m - pd.Timedelta(days=180)], s[s["dt"] >= m - pd.Timedelta(days=30)]
    t_in, t_out = d180.loc[d180["is_in"]==1, "amt"].sum(), d180.loc[d180["is_in"]==0, "amt"].sum()
    bills = d180["op"].apply(lambda x: 1 if any(k in x for k in ["FACTURE", "SENELEC", "WOYOFAL", "EAU"]) else 0).sum()
    air = d180["op"].apply(lambda x: 1 if any(k in x for k in ["PASS", "ILLIMIX", "RECHARGE"]) else 0).sum()
    return {"loan_amount_fcfa": loan_amt, "nb_tx_180d": len(d180), "total_in_180d": t_in, "total_out_180d": t_out, "avg_balance_180d": d180["bal"].mean() if len(d180)>0 else 0, "min_balance_180d": d180["bal"].min() if len(d180)>0 else 0, "std_balance_180d": d180["bal"].std() if len(d180)>1 else 0, "nb_bills_180d": bills, "nb_airtime_180d": air, "active_days_180d": d180["dt"].dt.date.nunique(), "nb_tx_30d": len(d30), "total_in_30d": d30.loc[d30["is_in"]==1, "amt"].sum(), "avg_balance_30d": d30["bal"].mean() if len(d30)>0 else 0, "ratio_in_out_180d": (t_in+1)/(t_out+1), "loan_to_income_ratio": loan_amt/(t_in/6+1), "activity_trend_30d_vs_avg": len(d30)/(len(d180)/6+0.1)}
    def show_result(inp_dict, loan_amt):
    X_c = pd.DataFrame([{c: inp_dict.get(c, 0) for c in features}])
    prob = float(model.predict_proba(X_c))
    score = int(np.round(300 + (1 - prob) * 550))
    r1, r2, r3 = st.columns(3)
    r1.metric("Score de Solvabilité (300 - 850)", f"{score} / 850")
    r2.metric("Risque de défaut", f"{prob:.1%}")
    if score >= 600: r3.success("✅ PRÊT ACCORDÉ (100 %)")
    elif score >= 500: r3.warning(f"⚠️ SOUS CONDITIONS (Max : {int(loan_amt*0.5):,} FCFA)".replace(",", " "))
    else: r3.error("❌ PRÊT REFUSÉ")
    impacts = (-scaler.transform(X_c) * clf.coef_)
    reasons = sorted(zip(features, impacts), key=lambda item: abs(float(item)), reverse=True)[:4]
    cols = st.columns(2)
    for i, (f, v) in enumerate(reasons):
        lbl = LABELS.get(f, f)
        if v >= 0: cols[i%2].info(f"🟢 **Point fort :** {lbl}")
        else: cols[i%2].error(f"🔴 **Point de vigilance :** {lbl}")

t1, t2, t3, t4 = st.tabs(["📱 Relevé Wave / Orange Money", "🔍 Évaluation Manuelle", "📊 Portefeuille Excel", "📈 Rentabilité"])

with t1:
    ex = pd.DataFrame([{"Date": "2026-06-05", "Operateur": "Wave", "Type_Operation": "Dépôt marchand", "Montant_FCFA": 45000, "Solde_Apres_FCFA": 48000}, {"Date": "2026-07-12", "Operateur": "Wave", "Type_Operation": "Paiement Facture Woyofal", "Montant_FCFA": 10000, "Solde_Apres_FCFA": 38000}, {"Date": "2026-08-03", "Operateur": "Orange Money", "Type_Operation": "Transfert reçu", "Montant_FCFA": 60000, "Solde_Apres_FCFA": 98000}, {"Date": "2026-11-12", "Operateur": "Wave", "Type_Operation": "Paiement Facture Senelec", "Montant_FCFA": 18000, "Solde_Apres_FCFA": 155500}])
    st.download_button("📥 Télécharger exemple relevé Wave/OM (.xlsx)", to_clean_excel(ex, "Releve"), "Exemple_Releve_Wave_OM.xlsx")
    s_loan = st.number_input("Montant demandé (FCFA)", 5000, 1000000, 50000, 5000, key="sl")
    s_file = st.file_uploader("Importer relevé Wave / Orange Money (.xlsx ou .csv)", type=["xlsx", "csv"])
    if s_file:
        rdf = pd.read_excel(s_file) if s_file.name.endswith(".xlsx") else pd.read_csv(s_file)
        ext = parse_stmt(rdf, s_loan)
        st.success(f"✅ {ext['nb_tx_180d']} transactions analysées sur {ext['active_days_180d']} jours actifs.")
        show_result(ext, s_loan)

with t2:
    c1, c2, c3 = st.columns(3)
    with c1:
        loan_amount = st.number_input("Montant (FCFA)", 5000, 500000, 25000, 5000)
        nb_tx_180d = st.slider("Transactions (6m)", 1, 150, 45)
        nb_tx_30d = st.slider("Transactions (30j)", 0, 50, 10)
        active_days = st.slider("Jours actifs (6m)", 1, 180, 35)
    with c2:
        total_in = st.number_input("Entrées 6m (FCFA)", 0, value=180000, step=10000)
        total_out = st.number_input("Sorties 6m (FCFA)", 0, value=150000, step=10000)
        total_in_30d = st.number_input("Entrées 30j (FCFA)", 0, value=35000, step=5000)
        avg_bal_180d = st.number_input("Solde moyen 6m", 0, value=12000, step=1000)
    with c3:
        avg_bal_30d = st.number_input("Solde moyen 30j", 0, value=15000, step=1000)
        min_bal = st.number_input("Solde minimum", 0, value=1000, step=500)
        nb_bills = st.slider("Factures payées", 0, 20, 4)
        nb_airtime = st.slider("Achats crédit tél.", 0, 50, 12)
    inp = {"loan_amount_fcfa": loan_amount, "nb_tx_180d": nb_tx_180d, "total_in_180d": total_in, "total_out_180d": total_out, "avg_balance_180d": avg_bal_180d, "min_balance_180d": min_bal, "std_balance_180d": avg_bal_180d*0.4, "nb_bills_180d": nb_bills, "nb_airtime_180d": nb_airtime, "active_days_180d": active_days, "nb_tx_30d": nb_tx_30d, "total_in_30d": total_in_30d, "avg_balance_30d": avg_bal_30d, "ratio_in_out_180d": (total_in+1)/(total_out+1), "loan_to_income_ratio": loan_amount/(total_in/6+1), "activity_trend_30d_vs_avg": nb_tx_30d/(nb_tx_180d/6+0.1)}
    show_result(inp, loan_amount)

with t3:
    sdf = df[features].head(15).copy().round(2)
    sdf.insert(0, "ID_Client", [f"CLT_{i:04d}" for i in range(1, 16)])
    st.download_button("📥 Modèle Excel français (.xlsx)", to_clean_excel(sdf.rename(columns=FR_COLS), "Clients"), "TerangaScore_Modele.xlsx")
    up = st.file_uploader("Importer fichier Excel (.xlsx)", type=["xlsx"])
    if up:
        b_fr = pd.read_excel(up)
        probs = model.predict_proba(b_fr.rename(columns=INV_FR_COLS)[features])[:, 1]
        scores = np.round(300 + (1 - probs) * 550).astype(int)
        res = b_fr.copy()
        res.insert(1, "Score_850", scores)
        res.insert(2, "Decision", np.where(scores>=600, "ACCORDÉ", np.where(scores>=500, "SOUS CONDITIONS", "REFUSÉ")))
        st.dataframe(res)
        st.download_button("📤 Télécharger rapport (.xlsx)", to_clean_excel(res, "Decisions"), "TerangaScore_Rapport.xlsx")

with t4:
    cut = st.slider("Seuil acceptation", 350, 800, 550, 10)
    rate = st.slider("Taux intérêt (%)", 5, 30, 15) / 100
    df["score"] = np.round(300 + (1 - model.predict_proba(df[features])[:, 1]) * 550).astype(int)
    app_df = df[df["score"] >= cut]
    profit = app_df[app_df["default"]==0]["loan_amount_fcfa"].sum()*rate - app_df[app_df["default"]==1]["loan_amount_fcfa"].sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("Acceptation", f"{len(app_df)/len(df):.1%}")
    m2.metric("Défaut", f"{app_df['default'].mean() if len(app_df)>0 else 0:.1%}")
    m3.metric("Profit Net", f"{profit:,.0f} FCFA".replace(",", " "))