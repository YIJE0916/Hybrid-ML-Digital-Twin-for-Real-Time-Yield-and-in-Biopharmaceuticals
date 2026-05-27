"""
Layer 2 – Hybrid Soft-Sensor ML
XGBoost trained on mechanistic residuals (Titre_obs - Titre_mech).
Feature set is restricted to DAG-identified ancestors of Titre.
Hybrid prediction: Titre_pred = Titre_mech + XGBoost(residual)
SHAP analysis explains which upstream signal drives titre deviations.

Run mechanistic.py first to generate mechanistic_predictions.csv.
Install dependencies: pip install xgboost shap scikit-learn
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
    import shap
except ImportError:
    raise ImportError("Run: pip install xgboost shap")

# ── DAG-derived feature set ────────────────────────────────────────────────────
# Only ancestors of Titre in the causal DAG are valid predictors (see dag.py).
# Direct ancestors: Protein_amount → Titre
# Upstream: VCD, q_p → Protein_amount; mu → VCD, q_p; Glucose_conc → mu; etc.
DAG_FEATURES = ["mu", "q_p", "VCD", "Glucose_conc", "Lactate_conc", "Viability", "Temperature"]

# ── Data loading ───────────────────────────────────────────────────────────────
def load_data():
    raw = pd.read_excel("new.xlsx", sheet_name="Dataset", skiprows=1)
    raw.columns = [
        "Reactor", "Day", "Volume", "Glucose_Feed",
        "VCD", "Titre", "Temperature",
        "Glucose_conc", "Glucose_consumed", "Cumul_Glucose",
        "Lactate_conc", "Protein_amount", "Viability",
        "mu", "q_p", "q_s",
        "VCD_pred", "Titre_pred", "Cumul_Glucose_pred",
    ]
    df = raw.apply(pd.to_numeric, errors="coerce").dropna(
        subset=["Reactor", "Day", "Titre"] + DAG_FEATURES
    ).reset_index(drop=True)

    try:
        mech = pd.read_csv("mechanistic_predictions.csv")[
            ["Reactor", "Day", "Titre_mech", "mu_mech", "q_p_mech"]
        ]
        df = df.merge(mech, on=["Reactor", "Day"], how="left")
        df["Titre_mech"] = df["Titre_mech"].fillna(df["Titre"].mean())
        print("Mechanistic predictions loaded from mechanistic_predictions.csv")
    except FileNotFoundError:
        print("mechanistic_predictions.csv not found — run mechanistic.py first.")
        print("Using Titre mean as mechanistic baseline (poor substitute).")
        df["Titre_mech"] = df["Titre"].mean()
        df["mu_mech"]    = df["mu"].mean()
        df["q_p_mech"]   = df["q_p"].mean()

    df["Titre_residual"] = df["Titre"] - df["Titre_mech"]
    return df

# ── Reactor-aware cross-validation ────────────────────────────────────────────
def cross_validate(df, feature_cols, target_col, groups_col, n_splits=5):
    """GroupKFold so no reactor appears in both train and validation."""
    X      = df[feature_cols].fillna(df[feature_cols].median())
    y      = df[target_col]
    groups = df[groups_col]
    gkf    = GroupKFold(n_splits=n_splits)

    oof_preds = np.full(len(df), np.nan)
    models    = []

    for train_idx, val_idx in gkf.split(X, y, groups):
        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=0.1,
            random_state=42, verbosity=0,
        )
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=False,
        )
        oof_preds[val_idx] = model.predict(X.iloc[val_idx])
        models.append(model)

    return oof_preds, models

# ── Main training + evaluation ─────────────────────────────────────────────────
def train_hybrid(df):
    X_cols = DAG_FEATURES

    # 1. Pure-ML baseline: XGBoost predicts Titre directly
    oof_baseline, _ = cross_validate(df, X_cols, "Titre", "Reactor")
    r2_base  = r2_score(df["Titre"], oof_baseline)
    mae_base = mean_absolute_error(df["Titre"], oof_baseline)

    # 2. Hybrid: XGBoost predicts residual, adds mechanistic baseline
    oof_residual, residual_models = cross_validate(df, X_cols, "Titre_residual", "Reactor")
    oof_hybrid = df["Titre_mech"].values + oof_residual
    r2_hyb  = r2_score(df["Titre"], oof_hybrid)
    mae_hyb = mean_absolute_error(df["Titre"], oof_hybrid)

    print("\nModel Comparison (5-fold GroupKFold — held-out reactors)")
    print(f"  Pure-ML XGBoost  → R² = {r2_base:.4f},  MAE = {mae_base:.5f}")
    print(f"  Hybrid (ODE+XGB) → R² = {r2_hyb:.4f},  MAE = {mae_hyb:.5f}")
    print(f"  R² gain from mechanistic prior: Δ = {r2_hyb - r2_base:+.4f}")

    return residual_models[0], X_cols, oof_hybrid, oof_baseline

# ── SHAP analysis ──────────────────────────────────────────────────────────────
def shap_analysis(model, df, feature_cols):
    X = df[feature_cols].fillna(df[feature_cols].median())
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    plt.sca(axes[0])
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    axes[0].set_title("Feature Importance (Mean |SHAP|)\nfor Titre Residual", fontweight="bold")

    plt.sca(axes[1])
    shap.summary_plot(shap_values, X, show=False)
    axes[1].set_title("SHAP Beeswarm\n(Impact on Titre Deviation from Mechanistic Baseline)",
                       fontweight="bold")

    plt.tight_layout()
    plt.savefig("shap_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved shap_analysis.png")

# ── Diagnostic plots ───────────────────────────────────────────────────────────
def plot_results(df, oof_hybrid, oof_baseline):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Parity plot — hybrid
    ax = axes[0]
    ax.scatter(df["Titre"], oof_hybrid, alpha=0.35, s=12, color="#2980B9", label="Hybrid")
    ax.scatter(df["Titre"], oof_baseline, alpha=0.2,  s=8,  color="#E74C3C", label="Pure-ML")
    lo, hi = df["Titre"].min(), df["Titre"].max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("Observed Titre"); ax.set_ylabel("Predicted Titre")
    r2_h = r2_score(df["Titre"], oof_hybrid)
    r2_b = r2_score(df["Titre"], oof_baseline)
    ax.set_title(f"Parity Plot\nHybrid R²={r2_h:.3f}   Pure-ML R²={r2_b:.3f}", fontweight="bold")
    ax.legend(fontsize=9)

    # Time series — sample reactor
    ax = axes[1]
    rid = df["Reactor"].unique()[0]
    rdf = df[df["Reactor"] == rid].sort_values("Day")
    pos = rdf.index
    ax.plot(rdf["Day"], rdf["Titre"],              "o-",  color="#2980B9", ms=5, label="Observed")
    ax.plot(rdf["Day"], rdf["Titre_mech"],          "--",  color="#E74C3C", lw=2, label="Mechanistic ODE")
    ax.plot(rdf["Day"], oof_hybrid[pos],            "-",   color="#27AE60", lw=2, label="Hybrid")
    ax.set_xlabel("Day"); ax.set_ylabel("Titre (norm.)")
    ax.set_title(f"Reactor {rid} — Time Series", fontweight="bold")
    ax.legend(fontsize=9)

    # Residual distribution
    ax = axes[2]
    residual_mech   = df["Titre"] - df["Titre_mech"]
    residual_hybrid = df["Titre"] - oof_hybrid
    ax.hist(residual_mech,   bins=40, alpha=0.55, color="#E74C3C", label="Mech. residual")
    ax.hist(residual_hybrid, bins=40, alpha=0.55, color="#27AE60", label="Hybrid residual")
    ax.axvline(0, color="k", lw=1, ls="--")
    ax.set_xlabel("Residual (Observed − Predicted)")
    ax.set_title("Residual Distribution\nHybrid vs. Mechanistic Baseline", fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Hybrid ML Model Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("hybrid_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved hybrid_predictions.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} rows, {df['Reactor'].nunique()} reactors")

    model, feature_cols, oof_hybrid, oof_baseline = train_hybrid(df)

    shap_analysis(model, df, feature_cols)
    plot_results(df, oof_hybrid, oof_baseline)

    # Save predictions for supply_chain.py
    out = df[["Reactor", "Day", "Titre", "Titre_mech"]].copy()
    out["Titre_hybrid"] = oof_hybrid
    out.to_csv("hybrid_predictions.csv", index=False)
    print("Saved hybrid_predictions.csv")
