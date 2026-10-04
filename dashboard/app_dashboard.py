"""
Streamlit Dashboard – Smart Financial Risk Intelligence System
Visually redesigned: luxury dark theme, refined typography, glowing accents,
compact mid-size charts, animated KPI cards.
Run: streamlit run dashboard/app_dashboard.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import json, warnings
warnings.filterwarnings("ignore")

from src.fraud_detection import predict_fraud
from src.credit_risk      import predict_credit
from src.preprocessing    import preprocess_fraud, preprocess_credit, preprocess_inflation
from src.economic_trend   import train_arima

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SFRIS · Financial Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Palette & design tokens ──────────────────────────────────────────────────
BG      = "#080c14"
SURFACE = "#0d1424"
CARD    = "#111827"
BORDER  = "#1e2d4a"
ACCENT  = "#3b82f6"
GOLD    = "#f59e0b"
TEAL    = "#14b8a6"
ROSE    = "#f43f5e"
MUTED   = "#64748b"
TEXT    = "#e2e8f0"
DIM     = "#94a3b8"

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&family=Syne:wght@600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
.stApp {{ background-color: {BG}; }}
.block-container {{ padding: 3.5rem 2rem 2rem; max-width: 1400px; }}

[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080c14 0%, #0a1120 60%, #050810 100%);
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .stRadio label {{
    color: {DIM} !important;
    font-size: 0.88rem;
    padding: 6px 0;
    transition: color 0.2s;
}}
[data-testid="stSidebar"] .stRadio label:hover {{ color: {TEXT} !important; }}

.brand-wrap {{ padding: 1.2rem 0 0.5rem; text-align: center; }}
.brand-name {{
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.25rem;
    letter-spacing: 0.04em;
    background: linear-gradient(135deg, {ACCENT}, {TEAL});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}}
.brand-sub {{
    font-size: 0.68rem; color: {MUTED};
    letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px;
}}
.sidebar-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {BORDER}, transparent);
    margin: 1rem 0;
}}
.nav-label {{
    font-size: 0.65rem; letter-spacing: 0.12em;
    text-transform: uppercase; color: {MUTED}; padding: 0 0 0.4rem;
}}
.page-title {{
    font-family: 'Syne', sans-serif; font-weight: 800;
    font-size: 1.9rem; letter-spacing: -0.01em;
    color: {TEXT}; margin: 0 0 0.2rem; line-height: 1.1;
}}
.page-sub {{
    font-size: 0.82rem; color: {MUTED};
    letter-spacing: 0.02em; margin-bottom: 1.5rem;
}}
.kpi-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 14px; margin-bottom: 1.8rem;
}}
.kpi-card {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 14px; padding: 18px 20px;
    position: relative; overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}}
.kpi-card:hover {{ transform: translateY(-2px); border-color: {ACCENT}44; }}
.kpi-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; border-radius: 14px 14px 0 0;
}}
.kpi-card.blue::before  {{ background: linear-gradient(90deg, {ACCENT}, {TEAL}); }}
.kpi-card.gold::before  {{ background: linear-gradient(90deg, {GOLD}, #fb923c); }}
.kpi-card.teal::before  {{ background: linear-gradient(90deg, {TEAL}, #06b6d4); }}
.kpi-card.rose::before  {{ background: linear-gradient(90deg, {ROSE}, #fb7185); }}
.kpi-icon {{ font-size: 1.3rem; margin-bottom: 10px; display: block; }}
.kpi-val {{
    font-family: 'Space Mono', monospace; font-size: 1.7rem;
    font-weight: 700; color: {TEXT}; line-height: 1; margin-bottom: 4px;
}}
.kpi-label {{
    font-size: 0.76rem; color: {DIM};
    letter-spacing: 0.03em; text-transform: uppercase;
}}
.kpi-badge {{
    position: absolute; top: 14px; right: 14px;
    font-size: 0.62rem; font-family: 'Space Mono', monospace;
    padding: 2px 7px; border-radius: 20px;
    background: rgba(59,130,246,0.12); color: {ACCENT};
    border: 1px solid rgba(59,130,246,0.25);
}}
.section-header {{
    font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700;
    color: {TEXT}; letter-spacing: 0.01em; margin: 0.2rem 0 1rem;
    display: flex; align-items: center; gap: 8px;
}}
.section-dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: {ACCENT}; display: inline-block;
    box-shadow: 0 0 8px {ACCENT};
}}
.model-card {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 12px; padding: 16px 18px; height: 100%;
}}
.model-title {{
    font-family: 'Syne', sans-serif; font-size: 0.82rem; font-weight: 700;
    color: {DIM}; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 10px;
}}
.model-name {{ font-size: 0.95rem; font-weight: 600; color: {TEXT}; margin-bottom: 12px; }}
.stat-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 5px 0; border-bottom: 1px solid {BORDER};
}}
.stat-row:last-child {{ border-bottom: none; }}
.stat-key {{ font-size: 0.76rem; color: {MUTED}; }}
.stat-val {{ font-family: 'Space Mono', monospace; font-size: 0.82rem; color: {TEXT}; font-weight: 700; }}
.stat-val.good {{ color: {TEAL}; }}
.prog-wrap {{ margin: 8px 0; }}
.prog-label {{
    font-size: 0.72rem; color: {DIM};
    display: flex; justify-content: space-between; margin-bottom: 4px;
}}
.prog-bar {{ height: 5px; background: {BORDER}; border-radius: 99px; overflow: hidden; }}
.prog-fill {{ height: 100%; border-radius: 99px; }}
.status-row {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 1.2rem; }}
.status-chip {{
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.7rem; padding: 3px 10px; border-radius: 20px;
    font-family: 'Space Mono', monospace; letter-spacing: 0.04em;
}}
.chip-green {{ background: rgba(20,184,166,0.1); border: 1px solid rgba(20,184,166,0.3); color: {TEAL}; }}
.chip-blue  {{ background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); color: {ACCENT}; }}
.chip-gold  {{ background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); color: {GOLD}; }}
.dot-pulse {{ width:6px;height:6px;border-radius:50%;background:currentColor; animation:pulse 1.8s ease-in-out infinite; }}
@keyframes pulse {{ 0%,100%{{opacity:1;transform:scale(1)}} 50%{{opacity:0.4;transform:scale(0.75)}} }}
.h-divider {{
    height: 1px;
    background: linear-gradient(90deg, transparent, {BORDER} 20%, {BORDER} 80%, transparent);
    margin: 1.4rem 0;
}}
.stButton>button {{
    background: linear-gradient(135deg, {ACCENT}, {TEAL}) !important;
    border: none !important; color: white !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    border-radius: 8px !important; font-size: 0.88rem !important;
    letter-spacing: 0.02em !important;
}}
[data-testid="stMetric"] {{
    background: {CARD}; border: 1px solid {BORDER};
    border-radius: 10px; padding: 12px 14px !important;
}}
[data-testid="stMetricLabel"] {{ color: {DIM} !important; font-size: 0.72rem !important; }}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important; font-family: 'Space Mono', monospace;
    font-size: 1.25rem !important;
}}
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="brand-wrap">
      <div class="brand-name">SFRIS</div>
      <div class="brand-sub">Financial Risk Intelligence</div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)
    page = st.radio("", [
        "📊  Overview",
        "🔍  Fraud Detection",
        "💳  Credit Risk",
        "📈  Economic Trends",
    ], label_visibility="collapsed")
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.68rem;color:{MUTED};line-height:1.8;padding:0 0 0.5rem;">
      <div style="margin-bottom:6px;color:{DIM};font-weight:600;text-transform:uppercase;
                  letter-spacing:0.1em;font-size:0.62rem;">System Status</div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
        <div style="width:6px;height:6px;border-radius:50%;background:{TEAL};
                    box-shadow:0 0 6px {TEAL};"></div>Fraud Engine · Online
      </div>
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:3px;">
        <div style="width:6px;height:6px;border-radius:50%;background:{TEAL};
                    box-shadow:0 0 6px {TEAL};"></div>Credit Scorer · Online
      </div>
      <div style="display:flex;align-items:center;gap:6px;">
        <div style="width:6px;height:6px;border-radius:50%;background:{TEAL};
                    box-shadow:0 0 6px {TEAL};"></div>ARIMA Engine · Online
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─── Load metrics ─────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    p = ROOT / "reports" / "metrics_report.json"
    return json.loads(p.read_text()) if p.exists() else None

metrics = load_metrics()

# ─── Matplotlib theme helper ──────────────────────────────────────────────────
def dark_fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=DIM, labelsize=8)
    ax.xaxis.label.set_color(DIM)
    ax.yaxis.label.set_color(DIM)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    return fig, ax

def prog_bar(label, val, max_val=1.0, color=None):
    pct = min(val / max_val, 1.0)
    fc = color or ACCENT
    return f"""
    <div class="prog-wrap">
      <div class="prog-label"><span>{label}</span><span style="font-family:'Space Mono',monospace;font-size:0.7rem;">{val}</span></div>
      <div class="prog-bar">
        <div class="prog-fill" style="width:{pct*100:.1f}%;background:linear-gradient(90deg,{fc},{TEAL});"></div>
      </div>
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊  Overview":
    st.markdown('<div class="page-title">Financial Risk Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Reserve Bank of India · Real-time Analytics Platform · 3 Active Modules</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-row">
      <div class="status-chip chip-green"><div class="dot-pulse"></div>All Systems Operational</div>
      <div class="status-chip chip-blue">5,000 Transactions Analyzed</div>
      <div class="status-chip chip-gold">6-Month Forecast Active</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card blue">
        <span class="kpi-icon">🛡️</span>
        <div class="kpi-badge">RF · ISO</div>
        <div class="kpi-val">98.3%</div>
        <div class="kpi-label">Fraud F1-Score</div>
      </div>
      <div class="kpi-card teal">
        <span class="kpi-icon">💳</span>
        <div class="kpi-badge">LR · DT</div>
        <div class="kpi-val">96.9%</div>
        <div class="kpi-label">Credit AUC-ROC</div>
      </div>
      <div class="kpi-card gold">
        <span class="kpi-icon">📈</span>
        <div class="kpi-badge">ARIMA(2,1,2)</div>
        <div class="kpi-val">0.557</div>
        <div class="kpi-label">Inflation RMSE</div>
      </div>
      <div class="kpi-card rose">
        <span class="kpi-icon">⚡</span>
        <div class="kpi-badge">Live</div>
        <div class="kpi-val">8,000+</div>
        <div class="kpi-label">Records Processed</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)

    # ── Model Performance Cards ──────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-dot"></span>Model Performance</div>', unsafe_allow_html=True)

    if metrics:
        fm = metrics["fraud"]["random_forest"]
        fi_iso = metrics["fraud"]["isolation_forest"]
        cm = metrics["credit"]["logistic_regression"]
        dt = metrics["credit"]["decision_tree"]
        im = metrics["inflation"]

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown(f"""
            <div class="model-card">
              <div class="model-title">Module 01 · Fraud Detection</div>
              <div class="model-name">Random Forest Classifier</div>
              {prog_bar("Precision", fm['precision'])}
              {prog_bar("Recall",    fm['recall'],    color=TEAL)}
              {prog_bar("F1-Score",  fm['f1_score'],  color=GOLD)}
              <div style="margin-top:14px;">
                <div class="model-title" style="margin-bottom:6px;">Isolation Forest</div>
                {prog_bar("Precision", fi_iso['precision'], color=ROSE)}
                {prog_bar("F1-Score",  fi_iso['f1_score'],  color=ACCENT)}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="model-card">
              <div class="model-title">Module 02 · Credit Risk</div>
              <div class="model-name">Logistic Regression</div>
              {prog_bar("Precision", cm['precision'])}
              {prog_bar("Recall",    cm['recall'],    color=TEAL)}
              {prog_bar("F1-Score",  cm['f1_score'],  color=GOLD)}
              {prog_bar("AUC-ROC",   cm['auc_roc'],   color=ROSE)}
              <div style="margin-top:14px;">
                <div class="model-title" style="margin-bottom:6px;">Decision Tree</div>
                {prog_bar("AUC-ROC", dt['auc_roc'], color=ACCENT)}
              </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            adf_color = TEAL if im['adf']['stationary'] else GOLD
            adf_label = "Stationary ✓" if im['adf']['stationary'] else "Differenced (d=1)"
            st.markdown(f"""
            <div class="model-card">
              <div class="model-title">Module 03 · Economic Trend</div>
              <div class="model-name">ARIMA (2, 1, 2)</div>
              <div class="stat-row">
                <span class="stat-key">AIC</span>
                <span class="stat-val">{im['aic']}</span>
              </div>
              <div class="stat-row">
                <span class="stat-key">BIC</span>
                <span class="stat-val">{im['bic']}</span>
              </div>
              <div class="stat-row">
                <span class="stat-key">MAE</span>
                <span class="stat-val good">{im['mae']}</span>
              </div>
              <div class="stat-row">
                <span class="stat-key">RMSE</span>
                <span class="stat-val good">{im['rmse']}</span>
              </div>
              <div class="stat-row">
                <span class="stat-key">ADF Stat</span>
                <span class="stat-val">{im['adf']['adf_stat']}</span>
              </div>
              <div class="stat-row">
                <span class="stat-key">Stationarity</span>
                <span class="stat-val" style="color:{adf_color};">{adf_label}</span>
              </div>
              <div class="stat-row">
                <span class="stat-key">Forecast Horizon</span>
                <span class="stat-val">6 months</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)

    # ── Visual Diagnostics (mid-size charts) ─────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-dot"></span>Visual Diagnostics</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        if metrics:
            fm = metrics["fraud"]["random_forest"]
            fi_iso = metrics["fraud"]["isolation_forest"]
            fig, ax = dark_fig(4.0, 2.8)
            cats  = ["Precision", "Recall", "F1"]
            rf_v  = [fm["precision"], fm["recall"], fm["f1_score"]]
            iso_v = [fi_iso["precision"], fi_iso["recall"], fi_iso["f1_score"]]
            x = np.arange(len(cats)); w = 0.32
            ax.bar(x-w/2, rf_v,  w, label="Random Forest",    color=ACCENT, alpha=0.9, edgecolor="none")
            ax.bar(x+w/2, iso_v, w, label="Isolation Forest", color=ROSE,   alpha=0.9, edgecolor="none")
            ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=8, color=DIM)
            ax.set_ylim(0, 1.12); ax.set_title("Fraud Models", color=TEXT, fontsize=9, fontweight="bold", pad=8)
            ax.tick_params(left=False); ax.yaxis.set_visible(False)
            for cont in [ax.containers[0], ax.containers[1]]:
                for bar in cont:
                    h = bar.get_height()
                    ax.text(bar.get_x()+bar.get_width()/2, h+0.01, f"{h:.2f}",
                            ha="center", va="bottom", fontsize=6.5, color=TEXT, fontfamily="monospace")
            ax.legend(fontsize=6.5, facecolor=CARD, labelcolor=DIM, edgecolor=BORDER, loc="lower right")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_edgecolor(BORDER)
            plt.tight_layout(pad=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with c2:
        if metrics:
            cm = metrics["credit"]["logistic_regression"]
            dt = metrics["credit"]["decision_tree"]
            fig, ax = dark_fig(4.0, 2.8)
            models_c = ["Logistic\nReg.", "Decision\nTree"]
            aucs     = [cm["auc_roc"], dt["auc_roc"]]
            bars = ax.bar(models_c, aucs, color=[TEAL, GOLD], alpha=0.9, edgecolor="none", width=0.42)
            ax.set_ylim(0.88, 1.01)
            ax.set_title("Credit Risk AUC-ROC", color=TEXT, fontsize=9, fontweight="bold", pad=8)
            ax.tick_params(left=False); ax.yaxis.set_visible(False)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h+0.001, f"{h:.4f}",
                        ha="center", va="bottom", fontsize=8, color=TEXT, fontfamily="monospace", fontweight="bold")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_edgecolor(BORDER)
            plt.tight_layout(pad=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with c3:
        if metrics:
            fi_data = metrics["fraud"]["feature_importance"]
            names   = list(fi_data.keys()); vals = list(fi_data.values())
            paired  = sorted(zip(vals, names)); vals, names = zip(*paired)
            fig, ax = dark_fig(4.0, 2.8)
            bar_c = [ACCENT if v < 0.2 else (TEAL if v < 0.35 else GOLD) for v in vals]
            bars  = ax.barh(names, vals, color=bar_c, alpha=0.9, edgecolor="none", height=0.55)
            ax.set_title("Fraud Feature Importance", color=TEXT, fontsize=9, fontweight="bold", pad=8)
            ax.xaxis.set_visible(False)
            ax.tick_params(labelsize=7.5)
            for bar, val in zip(bars, vals):
                ax.text(val+0.004, bar.get_y()+bar.get_height()/2, f"{val:.3f}",
                        va="center", fontsize=6.5, color=DIM, fontfamily="monospace")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.spines["left"].set_visible(True); ax.spines["left"].set_edgecolor(BORDER)
            plt.tight_layout(pad=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close()

    st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)

    # ── Architecture diagram ─────────────────────────────────────────────────
    st.markdown('<div class="section-header"><span class="section-dot"></span>System Architecture</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(12, 2.6))
    fig.patch.set_facecolor(CARD); ax.set_facecolor(CARD)
    ax.set_xlim(0, 13); ax.set_ylim(0, 3); ax.axis("off")

    nodes = [
        (1.0,  1.5, "📂  Raw Data",      "Fraud / Credit\n/ CPI Series",      ACCENT),
        (3.2,  1.5, "⚙️  Preprocess",    "Impute · Scale\nEncode",             TEAL),
        (5.5,  2.3, "🛡️  Fraud",         "ISO Forest\nRandom Forest",          ROSE),
        (5.5,  1.5, "💳  Credit",        "Logistic Reg.\nDecision Tree",       GOLD),
        (5.5,  0.7, "📈  ARIMA",         "Inflation\nForecaster",              ACCENT),
        (8.5,  1.5, "🌐  Flask API",     "/predict-fraud\n/predict-credit",    TEAL),
        (11.5, 1.5, "📊  Dashboard",     "Streamlit\nInteractive UI",          ROSE),
    ]
    for (x, y, title, sub, ac) in nodes:
        rect = FancyBboxPatch((x-0.9, y-0.5), 1.8, 1.0,
                               boxstyle="round,pad=0.06",
                               facecolor=BG, edgecolor=ac, linewidth=1.2, alpha=0.95)
        ax.add_patch(rect)
        ax.text(x, y+0.2,  title, ha="center", va="center", fontsize=6.5, color=ac, fontweight="bold")
        ax.text(x, y-0.18, sub,   ha="center", va="center", fontsize=5.5, color=DIM, linespacing=1.4)

    arrow_kw = dict(arrowstyle="-|>", color=MUTED, lw=1.1, mutation_scale=7)
    for (xs, xt, y) in [(1.95, 2.3, 1.5), (4.15, 4.6, 1.5)]:
        ax.annotate("", xy=(xt, y), xytext=(xs, y), arrowprops=arrow_kw)
    for yt in [2.3, 1.5, 0.7]:
        ax.annotate("", xy=(7.6, 1.5), xytext=(6.4, yt), arrowprops=arrow_kw)
    ax.annotate("", xy=(10.6, 1.5), xytext=(9.45, 1.5), arrowprops=arrow_kw)

    plt.tight_layout(pad=0.3)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown(f"""
    <div style="margin-top:1.5rem;padding-top:1rem;border-top:1px solid {BORDER};
         display:flex;justify-content:space-between;align-items:center;">
      <span style="font-size:0.7rem;color:{MUTED};font-family:'Space Mono',monospace;">
        SFRIS v2.0 · Python · Scikit-learn · Statsmodels · Flask · Streamlit
      </span>
      <span style="font-size:0.7rem;color:{MUTED};">
        Inspired by Reserve Bank of India Analytics Division
      </span>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – FRAUD DETECTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  Fraud Detection":
    st.markdown('<div class="page-title">Fraud Detection</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Real-time transaction scoring · Random Forest + Isolation Forest</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>Transaction Input</div>', unsafe_allow_html=True)
        amount      = st.number_input("Transaction Amount (₹)", 0.0, 1_000_000.0, 250.0, step=50.0)
        time_of_day = st.slider("Hour of Day (0–23)", 0, 23, 14)
        txn_freq    = st.slider("Transactions in Last Hour", 1, 50, 3)
        dist_home   = st.number_input("Distance from Home (km)", 0.0, 10000.0, 25.0)
        v1 = st.number_input("Velocity Feature V1", -5.0, 5.0, 0.0, 0.1)
        v2 = st.number_input("Velocity Feature V2", -5.0, 5.0, 0.0, 0.1)
        v3 = st.number_input("Velocity Feature V3", -5.0, 5.0, 0.0, 0.1)
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍  Analyze Transaction", use_container_width=True):
            try:
                result = predict_fraud({
                    "amount": amount, "time_of_day": time_of_day,
                    "transaction_freq": txn_freq, "distance_from_home": dist_home,
                    "v1": v1, "v2": v2, "v3": v3,
                })
                st.session_state["fraud_result"] = result
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>Prediction Result</div>', unsafe_allow_html=True)
        res = st.session_state.get("fraud_result")
        if res:
            if res["label"] == "FRAUD":
                st.error("🚨 **FRAUD DETECTED** — This transaction is flagged as suspicious.")
            else:
                st.success("✅ **LEGITIMATE** — Transaction cleared successfully.")
            m1, m2 = st.columns(2)
            m1.metric("RF Fraud Probability", f"{res['rf_probability']*100:.1f}%")
            m2.metric("Isolation Forest", "⚠️ Anomaly" if res["iso_prediction"] else "✅ Normal")

            fig, ax = dark_fig(5, 2.6)
            prob  = res["rf_probability"]
            color = ROSE if prob > 0.65 else (GOLD if prob > 0.35 else TEAL)
            ax.barh(["Risk"], [prob],      color=color,  height=0.35, alpha=0.9)
            ax.barh(["Risk"], [1-prob],    left=[prob], color=BORDER, height=0.35)
            ax.set_xlim(0, 1)
            ax.axvline(0.35, color=GOLD, lw=1, ls="--", alpha=0.6)
            ax.axvline(0.65, color=ROSE, lw=1, ls="--", alpha=0.6)
            ax.text(0.35, 0.62, "MED",  fontsize=6, color=GOLD, ha="center", transform=ax.get_xaxis_transform())
            ax.text(0.65, 0.62, "HIGH", fontsize=6, color=ROSE, ha="center", transform=ax.get_xaxis_transform())
            ax.text(prob, -0.5, f"{prob*100:.1f}%", fontsize=9, color=color, ha="center",
                    fontfamily="monospace", fontweight="bold", transform=ax.get_xaxis_transform())
            ax.set_title("Fraud Risk Score", color=TEXT, fontsize=9, fontweight="bold")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.tick_params(left=False, labelleft=False)
            plt.tight_layout(pad=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.markdown(f"""
            <div style="background:{CARD};border:1px dashed {BORDER};border-radius:12px;
                        padding:40px;text-align:center;color:{MUTED};font-size:0.85rem;">
              Fill in the transaction details and click<br>
              <strong style="color:{DIM};">Analyze Transaction</strong> to see results.
            </div>""", unsafe_allow_html=True)

    if metrics:
        st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>Feature Importance · Random Forest</div>', unsafe_allow_html=True)
        fi = metrics["fraud"]["feature_importance"]
        names = list(fi.keys()); vals = list(fi.values())
        paired = sorted(zip(vals, names)); vals, names = zip(*paired)
        fig, ax = dark_fig(8, 2.6)
        cmap_c = [ACCENT if v < 0.2 else (TEAL if v < 0.35 else GOLD) for v in vals]
        bars = ax.barh(names, vals, color=cmap_c, alpha=0.9, edgecolor="none", height=0.55)
        ax.xaxis.set_visible(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.spines["left"].set_visible(True); ax.spines["left"].set_edgecolor(BORDER)
        for bar, val in zip(bars, vals):
            ax.text(val+0.003, bar.get_y()+bar.get_height()/2, f"{val:.4f}",
                    va="center", fontsize=8, color=DIM, fontfamily="monospace")
        plt.tight_layout(pad=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – CREDIT RISK
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💳  Credit Risk":
    st.markdown('<div class="page-title">Credit Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">Loan default probability · Logistic Regression + Decision Tree ensemble</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>Applicant Profile</div>', unsafe_allow_html=True)
        age            = st.slider("Age", 18, 70, 35)
        income         = st.number_input("Annual Income (₹)", 10000, 2_000_000, 600_000, step=10_000)
        loan_amount    = st.number_input("Loan Amount (₹)", 1000, 5_000_000, 200_000, step=5_000)
        loan_tenure    = st.selectbox("Loan Tenure (months)", [12, 24, 36, 48, 60])
        credit_score   = st.slider("Credit Score (300–850)", 300, 850, 680)
        existing_loans = st.slider("Existing Loans", 0, 10, 1)
        emp_type       = st.selectbox("Employment Type", ["Salaried", "Self-employed", "Business"])
        education      = st.selectbox("Education", ["Graduate", "Post-Graduate", "Under-Graduate"])
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("💳  Assess Credit Risk", use_container_width=True):
            try:
                result = predict_credit({
                    "age": age, "income": income, "loan_amount": loan_amount,
                    "loan_tenure": loan_tenure, "credit_score": credit_score,
                    "existing_loans": existing_loans,
                    "employment_type": emp_type, "education": education,
                })
                st.session_state["credit_result"] = result
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>Risk Assessment</div>', unsafe_allow_html=True)
        res = st.session_state.get("credit_result")
        if res:
            risk = res["risk_category"]
            if risk == "Low":
                st.success("✅ **Low Risk** — Loan likely to be repaid on time.")
            elif risk == "Medium":
                st.warning("⚠️ **Medium Risk** — Proceed with additional verification.")
            else:
                st.error("🚨 **High Risk** — High default probability detected.")
            m1, m2, m3 = st.columns(3)
            m1.metric("Logistic Reg.", f"{res['lr_probability']:.1f}%")
            m2.metric("Decision Tree", f"{res['dt_probability']:.1f}%")
            m3.metric("Ensemble",      f"{res['avg_probability']:.1f}%")

            fig, ax = dark_fig(5, 2.6)
            mnames = ["Logistic\nReg.", "Decision\nTree", "Ensemble"]
            probs  = [res["lr_probability"], res["dt_probability"], res["avg_probability"]]
            bcolors= [TEAL if p<35 else (GOLD if p<65 else ROSE) for p in probs]
            ax.bar(mnames, probs, color=bcolors, alpha=0.9, edgecolor="none", width=0.42)
            ax.axhline(35, color=GOLD, lw=1, ls="--", alpha=0.6)
            ax.axhline(65, color=ROSE, lw=1, ls="--", alpha=0.6)
            ax.set_ylim(0, 110); ax.set_title("Default Probability", color=TEXT, fontsize=9, fontweight="bold")
            for sp in ax.spines.values(): sp.set_visible(False)
            ax.spines["bottom"].set_visible(True); ax.spines["bottom"].set_edgecolor(BORDER)
            ax.tick_params(left=False); ax.yaxis.set_visible(False)
            for i, p in enumerate(probs):
                ax.text(i, p+1.5, f"{p:.1f}%", ha="center", fontsize=8,
                        color=TEXT, fontfamily="monospace", fontweight="bold")
            plt.tight_layout(pad=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close()
        else:
            st.markdown(f"""
            <div style="background:{CARD};border:1px dashed {BORDER};border-radius:12px;
                        padding:40px;text-align:center;color:{MUTED};font-size:0.85rem;">
              Complete the applicant form and click<br>
              <strong style="color:{DIM};">Assess Credit Risk</strong> to see results.
            </div>""", unsafe_allow_html=True)

    if metrics:
        st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>Feature Importance · Decision Tree</div>', unsafe_allow_html=True)
        fi = metrics["credit"]["feature_importance_dt"]
        names = list(fi.keys()); vals = list(fi.values())
        paired = sorted(zip(vals, names)); vals, names = zip(*paired)
        fig, ax = dark_fig(8, 2.6)
        cmap_c = [TEAL if v<0.15 else (ACCENT if v<0.3 else GOLD) for v in vals]
        bars = ax.barh(names, vals, color=cmap_c, alpha=0.9, edgecolor="none", height=0.55)
        ax.xaxis.set_visible(False)
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.spines["left"].set_visible(True); ax.spines["left"].set_edgecolor(BORDER)
        for bar, val in zip(bars, vals):
            ax.text(val+0.002, bar.get_y()+bar.get_height()/2, f"{val:.4f}",
                    va="center", fontsize=8, color=DIM, fontfamily="monospace")
        plt.tight_layout(pad=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – ECONOMIC TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Economic Trends":
    st.markdown('<div class="page-title">Economic Trend Analyzer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">ARIMA(2,1,2) · India CPI Inflation · 6-Month Ahead Forecast</div>', unsafe_allow_html=True)

    @st.cache_data
    def load_arima():
        df = pd.read_csv(ROOT / "data" / "inflation_data.csv")
        ts = preprocess_inflation(df)
        m, r = train_arima(ts)
        return m, r

    try:
        am, ar = load_arima()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AIC",  am["aic"])
        c2.metric("RMSE", am["rmse"])
        c3.metric("MAE",  am["mae"])
        c4.metric("ADF p-value", am["adf"]["p_value"],
                  delta="Stationary" if am["adf"]["stationary"] else "Differenced")

        st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>CPI Trend & 6-Month Forecast</div>', unsafe_allow_html=True)

        fig, ax = dark_fig(11, 4.0)
        ax.plot(ar["actual"].index,   ar["actual"].values,   color=DIM,   lw=1.4, label="Actual CPI", zorder=2)
        ax.plot(ar["fitted"].index,   ar["fitted"].values,   color=TEAL,  lw=1.4, ls="--", label="ARIMA Fitted", zorder=3)
        ax.plot(ar["forecast"].index, ar["forecast"].values, color=ROSE,  lw=2, marker="o", ms=5, label="6-Month Forecast", zorder=4)
        ax.fill_between(ar["forecast"].index,
                        ar["conf_lower"].values, ar["conf_upper"].values,
                        alpha=0.15, color=ROSE, label="95% CI")
        ax.axvline(ar["actual"].index[-1], color=GOLD, lw=1, ls=":", alpha=0.7, label="Forecast Start")
        ax.set_ylabel("CPI Inflation (%)", fontsize=8)
        ax.set_title("India CPI Inflation — Historical Trend + 6-Month ARIMA Forecast",
                     color=TEXT, fontsize=10, fontweight="bold", pad=10)
        ax.legend(fontsize=7.5, facecolor=CARD, labelcolor=DIM, edgecolor=BORDER)
        ax.grid(alpha=0.07, color=DIM)
        for sp in ax.spines.values(): sp.set_edgecolor(BORDER)
        plt.tight_layout(pad=0.8)
        st.pyplot(fig, use_container_width=True)
        plt.close()

        st.markdown('<div class="h-divider"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-header"><span class="section-dot"></span>Forecast Table</div>', unsafe_allow_html=True)
        fc_df = pd.DataFrame({
            "Date":             ar["forecast"].index.strftime("%b %Y"),
            "Forecast CPI (%)": ar["forecast"].values.round(2),
            "Lower Bound":      ar["conf_lower"].values.round(2),
            "Upper Bound":      ar["conf_upper"].values.round(2),
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load ARIMA model: {e}")
        st.info("Run `python src/train_all.py` first to train models.")