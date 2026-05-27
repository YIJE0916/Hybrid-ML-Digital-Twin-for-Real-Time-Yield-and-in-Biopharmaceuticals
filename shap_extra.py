"""
Analysis 2: SHAP importance vs. DAG causal layer
Analysis 4: SHAP comparison — Pure-ML vs. Hybrid
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold

DAG_FEATURES = ["mu", "q_p", "VCD", "Glucose_conc", "Lactate_conc", "Viability", "Temperature"]

# DAG layer for each feature (from dag.py causal order)
DAG_LAYER = {
    "Temperature":   "CPP",
    "Glucose_conc":  "Process State",
    "mu":            "Kinetic Rate",
    "q_p":           "Kinetic Rate",
    "VCD":           "Cell State",
    "Viability":     "Cell State",
    "Lactate_conc":  "Metabolic Byproduct",
}

LAYER_COLOR = {
    "CPP":                "#E74C3C",
    "Process State":      "#F39C12",
    "Kinetic Rate":       "#27AE60",
    "Cell State":         "#2980B9",
    "Metabolic Byproduct":"#8E44AD",
}

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
        mech = pd.read_csv("mechanistic_predictions.csv")[["Reactor", "Day", "Titre_mech"]]
        df = df.merge(mech, on=["Reactor", "Day"], how="left")
        df["Titre_mech"] = df["Titre_mech"].fillna(df["Titre"].mean())
    except FileNotFoundError:
        df["Titre_mech"] = df["Titre"].mean()

    df["Titre_residual"] = df["Titre"] - df["Titre_mech"]
    return df

def train_and_get_shap(df, target_col):
    X = df[DAG_FEATURES].fillna(df[DAG_FEATURES].median())
    y = df[target_col]
    groups = df["Reactor"]
    gkf = GroupKFold(n_splits=5)

    shap_all = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        model = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, gamma=0.1,
            random_state=42, verbosity=0,
        )
        model.fit(X.iloc[train_idx], y.iloc[train_idx], verbose=False)
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X.iloc[val_idx])
        shap_all.append(pd.DataFrame(shap_vals, columns=DAG_FEATURES))

    return pd.concat(shap_all, ignore_index=True)

def plot_analysis2(shap_hybrid):
    """SHAP importance colored by DAG causal layer."""
    mean_abs = shap_hybrid.abs().mean().sort_values(ascending=True)

    colors = [LAYER_COLOR[DAG_LAYER[f]] for f in mean_abs.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(mean_abs.index, mean_abs.values, color=colors)
    ax.set_xlabel("Mean |SHAP value| (impact on Titre residual)")
    ax.set_title("Analysis 2: SHAP Importance by DAG Causal Layer", fontweight="bold")

    # Legend for layers
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for l, c in LAYER_COLOR.items()
               if l in DAG_LAYER.values()]
    ax.legend(handles=handles, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig("shap_dag_layers.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved shap_dag_layers.png")

def plot_analysis4(shap_hybrid, shap_pureml):
    """Side-by-side mean |SHAP| for hybrid vs pure-ML."""
    hybrid_imp = shap_hybrid.abs().mean().rename("Hybrid (residual target)")
    pureml_imp = shap_pureml.abs().mean().rename("Pure-ML (Titre target)")

    comp = pd.concat([hybrid_imp, pureml_imp], axis=1)
    comp = comp.sort_values("Hybrid (residual target)", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(comp))
    w = 0.35
    ax.barh(x - w/2, comp["Hybrid (residual target)"], w,
            color="#27AE60", label="Hybrid (residual target)")
    ax.barh(x + w/2, comp["Pure-ML (Titre target)"], w,
            color="#E74C3C", label="Pure-ML (Titre target)")
    ax.set_yticks(x)
    ax.set_yticklabels(comp.index)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Analysis 4: Feature Importance — Pure-ML vs. Hybrid", fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig("shap_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved shap_comparison.png")

if __name__ == "__main__":
    df = load_data()
    print(f"Data: {df.shape[0]} rows, {df['Reactor'].nunique()} reactors")

    print("Computing SHAP for hybrid model (residual target)...")
    shap_hybrid = train_and_get_shap(df, "Titre_residual")

    print("Computing SHAP for pure-ML model (Titre target)...")
    shap_pureml = train_and_get_shap(df, "Titre")

    plot_analysis2(shap_hybrid)
    plot_analysis4(shap_hybrid, shap_pureml)
