"""
Generate all screenshots/visualizations for the project report.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import warnings; warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json

SHOTS = ROOT / "screenshots"
SHOTS.mkdir(exist_ok=True)

DARK_BG = "#0a0a1a"
CARD_BG = "#16213e"
RED     = "#e94560"
GREEN   = "#00b894"
YELLOW  = "#fdcb6e"
BLUE    = "#a8b2d8"

# Load report
metrics = json.loads((ROOT / "reports" / "metrics_report.json").read_text())


def dark_ax(fig, ax, title=""):
    fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(CARD_BG)
    ax.tick_params(colors="white"); ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    if title: ax.set_title(title, color="white", fontsize=13, fontweight="bold")
    for spine in ax.spines.values(): spine.set_edgecolor("#0f3460")


# ── 1. Dashboard KPI Overview ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.patch.set_facecolor(DARK_BG)
fig.suptitle("Smart Financial Risk Intelligence System – KPI Dashboard",
             color="white", fontsize=15, fontweight="bold", y=0.98)

kpis = [
    ("RF F1-Score",    metrics["fraud"]["random_forest"]["f1_score"],      RED),
    ("RF Precision",   metrics["fraud"]["random_forest"]["precision"],     GREEN),
    ("LR AUC-ROC",     metrics["credit"]["logistic_regression"]["auc_roc"], YELLOW),
    ("LR F1-Score",    metrics["credit"]["logistic_regression"]["f1_score"], BLUE),
    ("ARIMA RMSE",     metrics["inflation"]["rmse"],                        RED),
    ("ARIMA AIC",      metrics["inflation"]["aic"],                         GREEN),
]
for ax, (label, val, color) in zip(axes.flat, kpis):
    ax.set_facecolor(CARD_BG); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
    ax.text(0.5, 0.65, str(val), ha="center", va="center",
            fontsize=28, fontweight="bold", color=color,
            transform=ax.transAxes)
    ax.text(0.5, 0.35, label, ha="center", va="center",
            fontsize=11, color=BLUE, transform=ax.transAxes)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor(RED); sp.set_linewidth(1.5)

plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig(SHOTS / "01_kpi_dashboard.png", dpi=120, bbox_inches="tight")
plt.close(); print("✓ 01_kpi_dashboard.png")


# ── 2. Fraud Feature Importance ──────────────────────────────────────────────
fi = metrics["fraud"]["feature_importance"]
fig, ax = plt.subplots(figsize=(9, 4))
names = list(fi.keys()); vals = list(fi.values())
bars = ax.barh(names, vals, color=RED, edgecolor="#0f3460")
ax.set_xlabel("Importance Score")
dark_ax(fig, ax, "Fraud Detection – Feature Importance (Random Forest)")
for bar, val in zip(bars, vals):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center", color="white", fontsize=9)
plt.tight_layout()
plt.savefig(SHOTS / "02_fraud_feature_importance.png", dpi=120, bbox_inches="tight")
plt.close(); print("✓ 02_fraud_feature_importance.png")


# ── 3. Fraud – class distribution pie ────────────────────────────────────────
df = pd.read_csv(ROOT / "data" / "fraud_data.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.patch.set_facecolor(DARK_BG)
counts = df["is_fraud"].value_counts()
ax1.pie(counts, labels=["Legitimate","Fraud"], colors=[GREEN, RED],
        autopct="%1.1f%%", startangle=90, textprops={"color":"white"})
ax1.set_title("Class Distribution", color="white", fontweight="bold")
ax1.set_facecolor(CARD_BG)

# Model comparison bar
models = ["Random\nForest", "Isolation\nForest"]
f1s    = [metrics["fraud"]["random_forest"]["f1_score"],
          metrics["fraud"]["isolation_forest"]["f1_score"]]
ax2.bar(models, f1s, color=[GREEN, YELLOW], edgecolor=RED)
ax2.set_ylim(0, 1.05); ax2.set_ylabel("F1 Score")
dark_ax(fig, ax2, "Fraud Models – F1 Score Comparison")
for i, v in enumerate(f1s):
    ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", color="white", fontweight="bold")
plt.tight_layout()
plt.savefig(SHOTS / "03_fraud_analysis.png", dpi=120, bbox_inches="tight")
plt.close(); print("✓ 03_fraud_analysis.png")


# ── 4. Credit Risk – model metrics ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fig.patch.set_facecolor(DARK_BG)

cats   = ["Precision", "Recall", "F1", "AUC-ROC"]
lr_v   = [metrics["credit"]["logistic_regression"][k]
          for k in ["precision","recall","f1_score","auc_roc"]]
dt_v   = [metrics["credit"]["decision_tree"][k]
          for k in ["precision","recall","f1_score","auc_roc"]]
x = np.arange(len(cats)); w = 0.35
ax = axes[0]
ax.bar(x-w/2, lr_v, w, label="Logistic Reg.", color=GREEN, edgecolor=RED)
ax.bar(x+w/2, dt_v, w, label="Decision Tree", color=YELLOW, edgecolor=RED)
ax.set_xticks(x); ax.set_xticklabels(cats)
ax.set_ylim(0, 1.1); ax.legend(facecolor=CARD_BG, labelcolor="white")
dark_ax(fig, ax, "Credit Risk – Model Metrics")

# Feature importance (DT)
fi2 = metrics["credit"]["feature_importance_dt"]
ax2 = axes[1]
names2 = list(fi2.keys()); vals2 = list(fi2.values())
ax2.barh(names2, vals2, color=YELLOW, edgecolor=RED)
ax2.set_xlabel("Importance")
dark_ax(fig, ax2, "Credit Risk – Feature Importance (DT)")
plt.tight_layout()
plt.savefig(SHOTS / "04_credit_risk_analysis.png", dpi=120, bbox_inches="tight")
plt.close(); print("✓ 04_credit_risk_analysis.png")


# ── 5. Inflation / ARIMA forecast ────────────────────────────────────────────
from src.preprocessing import preprocess_inflation
from src.economic_trend import train_arima

df_inf = pd.read_csv(ROOT / "data" / "inflation_data.csv")
ts     = preprocess_inflation(df_inf)
_, res = train_arima(ts)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(res["actual"].index,   res["actual"].values,   color=BLUE,   lw=1.5, label="Actual CPI")
ax.plot(res["fitted"].index,   res["fitted"].values,   color=GREEN,  lw=1.5, ls="--", label="ARIMA Fitted")
ax.plot(res["forecast"].index, res["forecast"].values, color=RED,    lw=2, marker="o", ms=5, label="6-Month Forecast")
ax.fill_between(res["forecast"].index,
                res["conf_lower"].values, res["conf_upper"].values,
                alpha=0.2, color=RED, label="95% Confidence Interval")
ax.axvline(res["actual"].index[-1], color=YELLOW, linestyle=":", lw=1.5, label="Forecast Start")
ax.set_xlabel("Date"); ax.set_ylabel("CPI Inflation (%)")
ax.legend(facecolor=CARD_BG, labelcolor="white")
ax.grid(alpha=0.15)
dark_ax(fig, ax, "India CPI Inflation – Historical Trend & 6-Month ARIMA Forecast")
plt.tight_layout()
plt.savefig(SHOTS / "05_inflation_forecast.png", dpi=120, bbox_inches="tight")
plt.close(); print("✓ 05_inflation_forecast.png")


# ── 6. System architecture ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0,14); ax.set_ylim(0,5); ax.axis("off")
fig.patch.set_facecolor(DARK_BG); ax.set_facecolor(DARK_BG)
ax.set_title("System Architecture – Smart Financial Risk Intelligence System",
             color="white", fontsize=14, fontweight="bold")

boxes = [
    (1.1, 2.5, "Raw Data\n📂 fraud_data.csv\ncredit_data.csv\ninflation_data.csv", "#0f3460"),
    (3.5, 2.5, "Preprocessing\n⚙️ Impute / Scale\nLabel Encode\nStandardScaler", "#1a1a5e"),
    (6.0, 4.0, "Fraud Detection\n🔍 Isolation Forest\nRandom Forest", RED),
    (6.0, 2.5, "Credit Risk\n💳 Logistic Reg.\nDecision Tree",          GREEN),
    (6.0, 1.0, "Economic Trend\n📈 ARIMA\nInflation Forecast",          YELLOW),
    (9.0, 2.5, "Flask REST API\n🌐 /predict-fraud\n/predict-credit",    "#0f3460"),
    (12.0,2.5, "Streamlit\nDashboard\n📊 Interactive UI",               "#1a1a5e"),
]
for (x, y, label, color) in boxes:
    rect = mpatches.FancyBboxPatch((x-1.0, y-0.7), 2.0, 1.4,
                                    boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor=RED, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
            color="white", fontweight="bold")

# Arrows
for (xs, xt, y) in [(2.1,2.5,2.5),(4.5,5.0,2.5)]:
    ax.annotate("", xy=(xt,y), xytext=(xs,y),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
for yt in [4.0, 2.5, 1.0]:
    ax.annotate("", xy=(7.0,yt), xytext=(6.5,yt),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    ax.annotate("", xy=(8.0,2.5), xytext=(7.0,yt),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1))
ax.annotate("", xy=(11.0,2.5), xytext=(10.0,2.5),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))

plt.tight_layout()
plt.savefig(SHOTS / "06_system_architecture.png", dpi=120, bbox_inches="tight")
plt.close(); print("✓ 06_system_architecture.png")

print("\n✅ All screenshots saved to screenshots/")
