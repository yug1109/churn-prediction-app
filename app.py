import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score, accuracy_score
)

st.set_page_config(page_title="ChurnSight · ABC Bank", page_icon="🔮",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background-color: #080B14; color: #E2E8F0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0D1120; border-right: 1px solid #1E2540; }
[data-testid="stSidebar"] * { color: #94A3B8 !important; }
[data-testid="stSidebar"] .sidebar-title { font-size:11px; letter-spacing:0.15em; text-transform:uppercase; color:#4B6BFB !important; font-weight:600; margin-bottom:0.5rem; }
.metric-card { background: linear-gradient(135deg,#0F1629,#111827); border:1px solid #1E2540; border-radius:16px; padding:1.4rem 1.6rem; position:relative; overflow:hidden; }
.metric-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,#4B6BFB,#7C3AED); border-radius:16px 16px 0 0; }
.metric-label { font-size:11px; letter-spacing:0.12em; text-transform:uppercase; color:#64748B; margin-bottom:0.5rem; font-weight:500; }
.metric-value { font-size:2rem; font-weight:700; letter-spacing:-0.02em; font-family:'JetBrains Mono',monospace; }
.metric-sub   { font-size:12px; color:#475569; margin-top:0.3rem; }
.metric-good  { color:#34D399; } .metric-warn { color:#FBBF24; } .metric-info { color:#60A5FA; } .metric-white { color:#F1F5F9; }
.section-header { display:flex; align-items:center; gap:10px; font-size:13px; letter-spacing:0.12em; text-transform:uppercase; color:#4B6BFB; font-weight:600; margin:2rem 0 1rem; }
.section-header::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1E2540,transparent); }
.verdict-churn { background:linear-gradient(135deg,#1A0A0A,#1F0F0F); border:1px solid #7F1D1D; border-left:4px solid #EF4444; border-radius:12px; padding:1.2rem 1.5rem; display:flex; align-items:center; gap:1rem; }
.verdict-safe  { background:linear-gradient(135deg,#021A0F,#051F15); border:1px solid #064E3B; border-left:4px solid #10B981; border-radius:12px; padding:1.2rem 1.5rem; display:flex; align-items:center; gap:1rem; }
.verdict-title { font-size:18px; font-weight:700; margin-bottom:2px; }
.verdict-sub   { font-size:13px; color:#94A3B8; }
.prob-bar-wrap { background:#1E2540; border-radius:99px; height:10px; overflow:hidden; margin:0.5rem 0; }
.prob-bar-fill { height:100%; border-radius:99px; }
.feature-pill  { display:inline-block; background:#1E2540; border:1px solid #2D3A5E; border-radius:8px; padding:4px 10px; font-size:12px; font-family:'JetBrains Mono',monospace; color:#94A3B8; margin:3px; }
.badge-high { background:#1F2937; border:1px solid #EF4444; color:#FCA5A5; border-radius:6px; padding:2px 8px; font-size:11px; }
.badge-med  { background:#1F2937; border:1px solid #FBBF24; color:#FDE68A; border-radius:6px; padding:2px 8px; font-size:11px; }
.badge-low  { background:#1F2937; border:1px solid #34D399; color:#A7F3D0; border-radius:6px; padding:2px 8px; font-size:11px; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor":"#0F1629","axes.facecolor":"#0F1629","axes.edgecolor":"#1E2540",
    "axes.labelcolor":"#94A3B8","xtick.color":"#475569","ytick.color":"#475569",
    "text.color":"#94A3B8","grid.color":"#1E2540","grid.linestyle":"--","font.family":"monospace",
})

# ── Load pkl ───────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_bundle():
    try:
        return joblib.load("churn_model.pkl")
    except FileNotFoundError:
        st.error("❌ `churn_model.pkl` not found. Place it in the same folder as app.py.")
        st.stop()

with st.spinner("🔮 Loading model…"):
    bundle = load_bundle()

pipe        = bundle["pipe"]
THRESHOLD   = bundle["threshold"]
X_test      = bundle["X_test"]
y_test      = bundle["y_test"]
y_prob_test = bundle["y_prob"]
FEAT_NAMES  = bundle["feature_names"]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:16px;padding-bottom:1.5rem;border-bottom:1px solid #1E2540;margin-bottom:0.5rem;">
  <div style="font-size:2rem;">🔮</div>
  <div>
    <div style="font-size:1.6rem;font-weight:700;letter-spacing:-0.02em;color:#F1F5F9;">Churn<span style="color:#4B6BFB;">Sight</span></div>
    <div style="font-size:13px;color:#475569;letter-spacing:0.05em;">ABC BANK · CUSTOMER RETENTION INTELLIGENCE</div>
  </div>
</div>""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", ["🔍  Predict Customer","📊  Model Dashboard","📁  Batch Scoring"],
                    label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<div class="sidebar-title">Threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("Classification threshold", 0.30, 0.80, float(THRESHOLD), 0.01,
                          help="Lower = more recall. Higher = more precision.")
    y_pred_live = (y_prob_test >= threshold).astype(int)
    st.markdown("---")
    st.markdown('<div class="sidebar-title">Live Metrics</div>', unsafe_allow_html=True)
    st.metric("Precision", f"{precision_score(y_test, y_pred_live):.3f}")
    st.metric("Recall",    f"{recall_score(y_test, y_pred_live):.3f}")
    st.metric("F1",        f"{f1_score(y_test, y_pred_live):.3f}")
    st.metric("AUC-ROC",   f"{roc_auc_score(y_test, y_prob_test):.3f}")

def add_features(d):
    d = d.copy()
    d["balance_zero"]         = (d["balance"] == 0).astype(int)
    d["age_bin"]              = pd.cut(d["age"], bins=[0,30,40,50,100], labels=[0,1,2,3]).astype(int)
    d["products_x_active"]    = d["products_number"] * d["active_member"]
    d["balance_per_product"]  = d["balance"] / (d["products_number"] + 1)
    d["credit_age"]           = d["credit_score"] * d["age"]
    d["salary_balance_ratio"] = d["estimated_salary"] / (d["balance"] + 1)
    return d

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
if "Predict" in page:
    st.markdown('<div class="section-header">Customer Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score    = st.number_input("Credit score",      300, 850,     650)
        age             = st.number_input("Age",                18,  92,      38)
        tenure          = st.number_input("Tenure (years)",      0,  10,       5)
        balance         = st.number_input("Account balance",   0.0, 300000.0, 75000.0, step=1000.0)
    with c2:
        products_number  = st.number_input("Number of products", 1, 4, 1)
        credit_card      = st.selectbox("Has credit card?", ["Yes","No"])
        active_member    = st.selectbox("Active member?",   ["Yes","No"])
        estimated_salary = st.number_input("Estimated salary", 0.0, 200000.0, 80000.0, step=1000.0)
    with c3:
        country = st.selectbox("Country", ["France","Germany","Spain"])
        gender  = st.selectbox("Gender",  ["Female","Male"])

    if st.button("🔮  Predict Churn Risk", use_container_width=True):
        row = add_features(pd.DataFrame([{
            "credit_score": credit_score, "country": country, "gender": gender,
            "age": age, "tenure": tenure, "balance": balance,
            "products_number": products_number,
            "credit_card":   1 if credit_card   == "Yes" else 0,
            "active_member": 1 if active_member  == "Yes" else 0,
            "estimated_salary": estimated_salary,
        }]))
        prob  = float(pipe.predict_proba(row)[0, 1])
        churn = prob >= threshold

        st.markdown('<div class="section-header">Prediction Result</div>', unsafe_allow_html=True)
        if churn:
            rl = "HIGH" if prob > 0.75 else "MEDIUM"
            bc = "badge-high" if prob > 0.75 else "badge-med"
            st.markdown(f'<div class="verdict-churn"><div style="font-size:2.5rem;">⚠️</div><div><div class="verdict-title" style="color:#FCA5A5;">Likely to Churn &nbsp;<span class="{bc}">{rl} RISK</span></div><div class="verdict-sub">Initiate retention action immediately.</div></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="verdict-safe"><div style="font-size:2.5rem;">✅</div><div><div class="verdict-title" style="color:#6EE7B7;">Likely to Stay &nbsp;<span class="badge-low">LOW RISK</span></div><div class="verdict-sub">Customer appears stable. Continue standard engagement.</div></div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        cp, cg = st.columns([1, 2])
        with cp:
            bc = "linear-gradient(90deg,#FBBF24,#EF4444)" if churn else "linear-gradient(90deg,#34D399,#10B981)"
            vc = "metric-warn" if churn else "metric-good"
            st.markdown(f'<div class="metric-card"><div class="metric-label">Churn Probability</div><div class="metric-value {vc}">{prob:.1%}</div><div class="prob-bar-wrap"><div class="prob-bar-fill" style="width:{prob*100:.1f}%;background:{bc};"></div></div><div class="metric-sub">Threshold: {threshold:.2f}</div></div>', unsafe_allow_html=True)
        with cg:
            fig, ax = plt.subplots(figsize=(4, 2.2))
            theta = np.linspace(np.pi, 0, 300)
            ax.plot(np.cos(theta), np.sin(theta), color="#1E2540", linewidth=18, solid_capstyle="round")
            fill = np.linspace(np.pi, np.pi - prob*np.pi, 300)
            col  = "#EF4444" if prob > threshold else "#34D399"
            ax.plot(np.cos(fill), np.sin(fill), color=col, linewidth=18, solid_capstyle="round")
            ax.set_xlim(-1.3,1.3); ax.set_ylim(-0.2,1.2); ax.axis("off")
            ax.text(0,-0.05,f"{prob:.1%}",ha="center",va="center",fontsize=20,fontweight="bold",color=col,fontfamily="monospace")
            ax.text(0,-0.2,"churn probability",ha="center",va="center",fontsize=9,color="#475569")
            st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown('<div class="section-header">Risk Factors</div>', unsafe_allow_html=True)
        factors = []
        if age > 45:               factors.append(("Age > 45","high"))
        if balance == 0:           factors.append(("Zero balance","high"))
        if products_number >= 3:   factors.append(("3+ products","med"))
        if active_member != "Yes": factors.append(("Inactive member","high"))
        if credit_score < 550:     factors.append(("Low credit score","med"))
        if country == "Germany":   factors.append(("Germany (higher base rate)","med"))
        if factors:
            st.markdown("".join(f'<span class="feature-pill"><span class="badge-{"high" if l=="high" else "med"}">{lb}</span></span>' for lb,l in factors), unsafe_allow_html=True)
        else:
            st.markdown('<span class="feature-pill"><span class="badge-low">No major risk signals detected</span></span>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Dashboard" in page:
    y_pred = (y_prob_test >= threshold).astype(int)
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    for col, (label, val, cls, sub) in zip(st.columns(5), [
        ("AUC-ROC",    f"{roc_auc_score(y_test,y_prob_test):.4f}", "metric-info",  "Discrimination power"),
        ("Accuracy",   f"{accuracy_score(y_test,y_pred):.4f}",      "metric-white", "Overall correctness"),
        ("F1 (churn)", f"{f1_score(y_test,y_pred):.4f}",            "metric-warn",  "Harmonic P/R mean"),
        ("Precision",  f"{precision_score(y_test,y_pred):.4f}",     "metric-good",  "Alarm accuracy"),
        ("Recall",     f"{recall_score(y_test,y_pred):.4f}",        "metric-good",  "Churners caught"),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {cls}">{val}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["  ROC Curve  ","  Confusion Matrix  ","  Threshold Analysis  "])

    with tab1:
        fpr, tpr, _ = roc_curve(y_test, y_prob_test)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(fpr, tpr, color="#4B6BFB", linewidth=2.5, label=f"AUC={roc_auc_score(y_test,y_prob_test):.4f}")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#4B6BFB")
        ax.plot([0,1],[0,1],"--",color="#2D3A5E",linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right",facecolor="#0F1629",edgecolor="#1E2540")
        ax.grid(True,alpha=0.3); st.pyplot(fig,use_container_width=True); plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(5,4))
        sns.heatmap(confusion_matrix(y_test,y_pred), annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"],
                    ax=ax, cbar=False, annot_kws={"size":16,"weight":"bold","color":"#F1F5F9"})
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig,use_container_width=True); plt.close()

    with tab3:
        ths = np.arange(0.25,0.80,0.01)
        precs=[]; recs=[]; f1s=[]
        for t in ths:
            yp=(y_prob_test>=t).astype(int)
            precs.append(precision_score(y_test,yp,zero_division=0))
            recs.append(recall_score(y_test,yp,zero_division=0))
            f1s.append(f1_score(y_test,yp,zero_division=0))
        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(ths,precs,color="#34D399",linewidth=2,label="Precision")
        ax.plot(ths,recs, color="#60A5FA",linewidth=2,label="Recall")
        ax.plot(ths,f1s,  color="#FBBF24",linewidth=2,label="F1")
        ax.axvline(threshold,color="#7C3AED",linestyle="--",linewidth=1.5,label=f"Current={threshold:.2f}")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
        ax.legend(facecolor="#0F1629",edgecolor="#1E2540"); ax.grid(True,alpha=0.3); ax.set_ylim(0,1)
        st.pyplot(fig,use_container_width=True); plt.close()

    st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
    try:
        imps = np.mean([e.estimator.feature_importances_
                        for e in pipe.named_steps["classifier"].calibrated_classifiers_], axis=0)
        feat_df = (pd.DataFrame({"feature":FEAT_NAMES,"importance":imps})
                     .sort_values("importance",ascending=True).tail(15))
        fig, ax = plt.subplots(figsize=(7,5))
        bars = ax.barh(feat_df["feature"], feat_df["importance"], color="#4B6BFB", alpha=0.85, height=0.6)
        for bar, val in zip(bars, feat_df["importance"]):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.3f}", va="center", fontsize=9, color="#64748B")
        ax.set_xlabel("Importance"); ax.grid(axis="x",alpha=0.3)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        st.pyplot(fig,use_container_width=True); plt.close()
    except Exception as e:
        st.warning(f"Feature importance unavailable: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — BATCH
# ══════════════════════════════════════════════════════════════════════════════
elif "Batch" in page:
    st.markdown('<div class="section-header">Batch CSV Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div style="background:#0F1629;border:1px solid #1E2540;border-radius:12px;padding:1rem 1.5rem;font-size:13px;color:#64748B;margin-bottom:1rem;">Upload a CSV — no <code style="color:#94A3B8;">churn</code> column needed. Required: <code style="color:#94A3B8;">credit_score, country, gender, age, tenure, balance, products_number, credit_card, active_member, estimated_salary</code></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        ids   = df_up["customer_id"].values if "customer_id" in df_up.columns else np.arange(len(df_up))
        df_up = df_up.drop(columns=[c for c in ["customer_id","churn"] if c in df_up.columns])
        probs = pipe.predict_proba(add_features(df_up))[:, 1]
        preds = (probs >= threshold).astype(int)
        df_out = df_up.copy()
        df_out.insert(0,"customer_id",ids)
        df_out["churn_probability"] = probs.round(4)
        df_out["predicted_churn"]   = preds
        df_out["risk_label"] = pd.cut(probs,bins=[0,0.4,0.6,1.0],labels=["Low","Medium","High"])

        n_churn=int(preds.sum()); high=int((probs>0.6).sum())
        for col,(lb,v,sub,cls) in zip(st.columns(3),[
            ("Total customers",   f"{len(df_out):,}","",                            "metric-info"),
            ("Predicted churners",f"{n_churn:,}",    f"{n_churn/len(df_out):.1%}", "metric-warn"),
            ("High risk (>60%)",  f"{high:,}",       "Immediate attention",         "metric-warn"),
        ]):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{lb}</div><div class="metric-value {cls}">{v}</div><div class="metric-sub">{sub}</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_out[["customer_id","churn_probability","predicted_churn","risk_label"]],
                     use_container_width=True, height=400)
        st.download_button("⬇️  Download Scored CSV", df_out.to_csv(index=False).encode(),
                           "churn_scored.csv","text/csv", use_container_width=True)
