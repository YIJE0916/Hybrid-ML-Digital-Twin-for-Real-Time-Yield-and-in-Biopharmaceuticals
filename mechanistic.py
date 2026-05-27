"""
Layer 1 – Mechanistic ODE
Fits Monod (μ_max, K_s) and Luedeking-Piret (α, β) from data.
Outputs per-row μ_mech, q_p_mech, and a cumulative-integration Titre_mech.
Saves mechanistic_predictions.csv for the hybrid ML layer.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

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
    return raw.apply(pd.to_numeric, errors="coerce").dropna(
        subset=["Reactor", "Day", "VCD", "Titre", "Glucose_conc", "mu", "q_p"]
    ).reset_index(drop=True)

# ── Kinetic models ─────────────────────────────────────────────────────────────
def monod(S, mu_max, Ks):
    """Monod 1949: μ = μ_max · S / (K_s + S)"""
    return mu_max * S / (Ks + S)

def luedeking_piret(mu, alpha, beta):
    """Luedeking & Piret 1959: q_p = α·μ + β"""
    return alpha * mu + beta

# ── Parameter fitting ──────────────────────────────────────────────────────────
def fit_kinetic_parameters(df):
    clean = df[["Glucose_conc", "mu", "q_p"]].dropna()
    S, mu_obs, qp_obs = clean["Glucose_conc"].values, clean["mu"].values, clean["q_p"].values

    (mu_max, Ks), _ = curve_fit(
        monod, S, mu_obs,
        p0=[mu_obs.max(), np.median(S)],
        bounds=([0, 1e-6], [mu_obs.max() * 3, S.max()])
    )

    (alpha, beta), _ = curve_fit(
        luedeking_piret, mu_obs, qp_obs, p0=[0.1, qp_obs.mean()]
    )

    mu_pred  = monod(S, mu_max, Ks)
    qp_pred  = luedeking_piret(mu_obs, alpha, beta)
    r2_mu    = r2_score(mu_obs, mu_pred)
    r2_qp    = r2_score(qp_obs, qp_pred)

    print("Kinetic parameters fitted:")
    print(f"  μ_max = {mu_max:.4f}   K_s = {Ks:.4f}   R²(Monod)  = {r2_mu:.4f}")
    print(f"  α     = {alpha:.4f}   β   = {beta:.4f}   R²(L-P)    = {r2_qp:.4f}")

    return {"mu_max": mu_max, "Ks": Ks, "alpha": alpha, "beta": beta}

# ── Build mechanistic features per reactor ─────────────────────────────────────
def build_mechanistic_features(df, params):
    """
    For each reactor (sorted by Day):
      - μ_mech   = Monod(Glucose_conc; fitted params)
      - q_p_mech = LP(μ_mech; fitted params)
      - cum_prod  = cumulative integral of q_p_mech × VCD × dt  [proxy for Titre_mech]
    """
    records = []
    for _, rdf in df.groupby("Reactor"):
        rdf = rdf.sort_values("Day").copy()
        S   = rdf["Glucose_conc"].values
        VCD = rdf["VCD"].values
        days = rdf["Day"].values
        dt   = np.diff(days, prepend=days[0])
        dt[0] = 0.0   # t=0 contributes no accumulation

        mu_mech  = monod(S, params["mu_max"], params["Ks"])
        q_p_mech = luedeking_piret(mu_mech, params["alpha"], params["beta"])
        cum_prod = np.cumsum(q_p_mech * VCD * dt)

        rdf["mu_mech"]   = mu_mech
        rdf["q_p_mech"]  = q_p_mech
        rdf["cum_prod"]  = cum_prod   # un-scaled integral → Titre_mech via Ridge below
        records.append(rdf)

    return pd.concat(records).reset_index(drop=True)

# ── Fit a linear scale for Titre_mech = a·cum_prod + b ────────────────────────
def fit_mechanistic_titre(df_mech):
    """
    Ridge regression maps the cumulative mechanistic production integral to Titre.
    This absorbs the normalisation constant hidden in the dataset's time/unit scaling.
    """
    X = df_mech[["cum_prod"]].values
    y = df_mech["Titre"].values
    ridge = Ridge(alpha=1.0).fit(X, y)
    df_mech["Titre_mech"] = ridge.predict(X)
    r2 = r2_score(y, df_mech["Titre_mech"])
    print(f"\nMechanistic Titre baseline  R² = {r2:.4f}")
    return df_mech, ridge

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_kinetic_fits(df, params):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Monod curve
    ax = axes[0]
    S_range = np.linspace(0, df["Glucose_conc"].max(), 300)
    ax.scatter(df["Glucose_conc"], df["mu"], s=8, alpha=0.3, color="#2980B9", label="Observed")
    ax.plot(S_range, monod(S_range, params["mu_max"], params["Ks"]),
            color="#E74C3C", lw=2, label=f"Monod fit (μ_max={params['mu_max']:.2f}, Ks={params['Ks']:.2f})")
    ax.set_xlabel("Glucose concentration (normalised)")
    ax.set_ylabel("Specific growth rate μ (normalised)")
    ax.set_title("Monod Kinetics Fit", fontweight="bold")
    ax.legend(fontsize=9)

    # Luedeking-Piret
    ax = axes[1]
    mu_range = np.linspace(df["mu"].min(), df["mu"].max(), 200)
    ax.scatter(df["mu"], df["q_p"], s=8, alpha=0.3, color="#2980B9", label="Observed")
    ax.plot(mu_range, luedeking_piret(mu_range, params["alpha"], params["beta"]),
            color="#E74C3C", lw=2,
            label=f"L-P fit (α={params['alpha']:.3f}, β={params['beta']:.3f})")
    ax.set_xlabel("Specific growth rate μ")
    ax.set_ylabel("Specific productivity q_p")
    ax.set_title("Luedeking-Piret Fit", fontweight="bold")
    ax.legend(fontsize=9)

    plt.suptitle("Mechanistic Parameter Estimation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mechanistic_fit.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved mechanistic_fit.png")

def plot_titre_trajectories(df_mech, n_reactors=6):
    reactors = df_mech["Reactor"].unique()[:n_reactors]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for ax, rid in zip(axes.flat, reactors):
        rdf = df_mech[df_mech["Reactor"] == rid].sort_values("Day")
        ax.plot(rdf["Day"], rdf["Titre"],      "o-", label="Observed",    color="#2980B9", ms=4)
        ax.plot(rdf["Day"], rdf["Titre_mech"], "--", label="Mechanistic", color="#E74C3C", lw=2)
        ax.set_title(f"Reactor {rid}", fontsize=9)
        ax.set_xlabel("Day"); ax.set_ylabel("Titre (norm.)")
        ax.legend(fontsize=7)

    plt.suptitle("Mechanistic ODE Titre vs. Observed — Sample Reactors",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mechanistic_trajectories.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved mechanistic_trajectories.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    print(f"Data loaded: {df.shape[0]} rows, {df['Reactor'].nunique()} reactors")

    params   = fit_kinetic_parameters(df)
    df_mech  = build_mechanistic_features(df, params)
    df_mech, _ = fit_mechanistic_titre(df_mech)

    out_cols = ["Reactor", "Day", "Titre", "Titre_mech", "mu_mech", "q_p_mech", "cum_prod"]
    df_mech[out_cols].to_csv("mechanistic_predictions.csv", index=False)
    print(f"Saved mechanistic_predictions.csv  ({len(df_mech)} rows)")

    plot_kinetic_fits(df, params)
    plot_titre_trajectories(df_mech)
