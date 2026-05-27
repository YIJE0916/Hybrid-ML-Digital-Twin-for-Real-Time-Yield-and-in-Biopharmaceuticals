"""
Layer 3 – Stochastic Supply Chain Optimization
Uses the hybrid model's Titre prediction distribution as f̂_t(y).
Solves the newsvendor problem to find the optimal inventory order.
Monte Carlo comparison: static (blind) policy vs. DAG-informed dynamic policy.

Objective (Proposal Eq. 2):
    min  Σ_t ∫ [C_h·I_t(y) + C_b·B_t(y) + C_p·ΔP_t] · f̂_t(y) dy
    subject to: I_t(y) − B_t(y) = I_{t−1} + y + ΔP_t − Demand_t

Run hybrid_ml.py first to generate hybrid_predictions.csv.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Cost parameters (normalised units, adjustable) ─────────────────────────────
C_h = 1.0   # holding cost per unit inventory per day
C_b = 5.0   # backorder penalty per unit shortage per day (> C_h so we avoid stockouts)
C_p = 2.0   # expedited-sourcing cost per unit of proactive order ΔP_t

DEMAND_MU  = 0.35   # normalised mean daily demand
DEMAND_STD = 0.04
BATCH_DAYS = 14

# ── Load hybrid predictions ────────────────────────────────────────────────────
def load_predictions():
    try:
        df = pd.read_csv("hybrid_predictions.csv")
        titre_vals = df["Titre_hybrid"].dropna().values
        print(f"Loaded hybrid predictions: {len(titre_vals)} observations")
    except FileNotFoundError:
        print("hybrid_predictions.csv not found — run hybrid_ml.py first.")
        print("Falling back to observed Titre from new.xlsx.")
        raw = pd.read_excel("new.xlsx", sheet_name="Dataset", skiprows=1)
        raw.columns = [
            "Reactor","Day","Volume","Glucose_Feed",
            "VCD","Titre","Temperature","Glucose_conc","Glucose_consumed","Cumul_Glucose",
            "Lactate_conc","Protein_amount","Viability","mu","q_p","q_s",
            "VCD_pred","Titre_pred","Cumul_Glucose_pred",
        ]
        titre_vals = raw["Titre"].dropna().values
    return titre_vals

# ── Yield distribution: f̂_t(y) ────────────────────────────────────────────────
def fit_yield_distribution(titre_vals):
    """Fit Gaussian to predicted titre distribution — f̂_t(y) in the proposal."""
    mu_y    = float(np.mean(titre_vals))
    sigma_y = float(np.std(titre_vals))
    return mu_y, sigma_y

def yield_cdf(y, mu_y, sigma_y):
    return norm.cdf(y, mu_y, sigma_y)

def yield_pdf(y, mu_y, sigma_y):
    return norm.pdf(y, mu_y, sigma_y)

# ── Newsvendor expected cost ───────────────────────────────────────────────────
def expected_period_cost(I_order, mu_y, sigma_y, demand):
    """
    E[cost] = ∫ [C_h·max(I+y−D, 0) + C_b·max(D−I−y, 0)] · f̂(y) dy
    Closed-form newsvendor solution using Gaussian CDF.
    """
    net = I_order - demand   # net position before yield realisation
    # Expected holding:  C_h · E[max(net + y, 0)]
    # Expected backorder: C_b · E[max(−net − y, 0)]
    z = (net + mu_y) / sigma_y  # standardised net inventory
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    exp_holding   = C_h * ((net + mu_y) * Phi + sigma_y * phi)
    exp_backorder = C_b * ((-(net + mu_y)) * (1 - Phi) + sigma_y * phi)
    return float(exp_holding + exp_backorder)

def optimal_order(mu_y, sigma_y, demand):
    """
    Optimal I* = D − μ_y + σ_y · Φ^{-1}(C_b / (C_b + C_h))
    Classic newsvendor critical ratio.
    """
    critical_ratio = C_b / (C_b + C_h)
    I_star = demand - mu_y + sigma_y * norm.ppf(critical_ratio)
    I_star = max(0.0, I_star)
    cost_star = expected_period_cost(I_star, mu_y, sigma_y, demand)
    return I_star, cost_star

# ── Simulate one batch under a given policy ────────────────────────────────────
def simulate_batch(titre_vals, demand_series, policy, mu_y, sigma_y):
    """
    Simulate one 14-day batch; return total cost.
    policy: 'static'  — order fixed = mean demand, no early warning
             'dynamic' — order optimal I* from yield distribution; trigger
                         expedited sourcing if deficit probability > threshold
                         at the midpoint of the batch.
    """
    I_fixed   = DEMAND_MU                # static order quantity
    total_cost = 0.0
    inventory  = 0.0
    expedited  = False

    for t, demand in enumerate(demand_series):
        y_realised = float(np.random.choice(titre_vals))

        if policy == "static":
            order_qty = I_fixed
        else:
            I_opt, _ = optimal_order(mu_y, sigma_y, demand)
            order_qty = I_opt

            # Early warning at batch midpoint: trigger expedited sourcing
            if not expedited and t == BATCH_DAYS // 2:
                p_deficit = 1 - yield_cdf(demand - I_opt, mu_y, sigma_y)
                if p_deficit > 0.30:
                    delta_P   = demand * 0.10          # order 10% extra from spot market
                    total_cost += C_p * delta_P
                    inventory  += delta_P
                    expedited   = True

        inventory += order_qty + y_realised - demand
        total_cost += C_h * max(inventory, 0) + C_b * max(-inventory, 0)
        inventory  = max(inventory, 0)   # backorders do not carry forward in this model

    return total_cost

# ── Monte Carlo comparison ─────────────────────────────────────────────────────
def monte_carlo_comparison(titre_vals, n_sims=1000):
    mu_y, sigma_y = fit_yield_distribution(titre_vals)
    demand_series = np.random.normal(DEMAND_MU, DEMAND_STD, (n_sims, BATCH_DAYS)).clip(0)

    static_costs  = [simulate_batch(titre_vals, demand_series[i], "static",  mu_y, sigma_y)
                     for i in range(n_sims)]
    dynamic_costs = [simulate_batch(titre_vals, demand_series[i], "dynamic", mu_y, sigma_y)
                     for i in range(n_sims)]

    s_mean, s_std = np.mean(static_costs),  np.std(static_costs)
    d_mean, d_std = np.mean(dynamic_costs), np.std(dynamic_costs)
    reduction     = (s_mean - d_mean) / s_mean * 100

    print(f"\nMonte Carlo Supply Chain Simulation  (n = {n_sims} batches)")
    print(f"  Static  policy → mean cost = {s_mean:.4f}  ±  {s_std:.4f}")
    print(f"  Dynamic policy → mean cost = {d_mean:.4f}  ±  {d_std:.4f}")
    print(f"  Expected cost reduction:      {reduction:.1f}%")

    return static_costs, dynamic_costs, mu_y, sigma_y

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_yield_distribution(mu_y, sigma_y):
    y_range = np.linspace(mu_y - 4*sigma_y, mu_y + 4*sigma_y, 400)
    pdf     = yield_pdf(y_range, mu_y, sigma_y)

    I_opt, _ = optimal_order(mu_y, sigma_y, DEMAND_MU)
    deficit_threshold = DEMAND_MU - I_opt   # yield level that triggers backorder

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Supply Chain Optimisation — Stochastic Yield Framework",
                 fontsize=13, fontweight="bold")

    # Yield PDF with risk zone
    ax = axes[0]
    ax.plot(y_range, pdf, color="#2980B9", lw=2.5, label="f̂_t(y)")
    ax.fill_between(y_range, pdf,
                    where=(y_range < deficit_threshold),
                    alpha=0.3, color="#E74C3C", label="Deficit risk zone")
    ax.axvline(DEMAND_MU, color="#27AE60", ls="--", lw=1.8,
               label=f"Demand μ = {DEMAND_MU:.2f}")
    ax.axvline(mu_y, color="#2980B9", ls=":", lw=1.5,
               label=f"Titre μ = {mu_y:.3f}")
    ax.set_xlabel("Titre (normalised)")
    ax.set_ylabel("Density")
    ax.set_title("Predicted Yield Distribution  f̂_t(y)", fontweight="bold")
    ax.legend(fontsize=9)

    # Expected cost vs. order quantity
    ax = axes[1]
    I_range = np.linspace(0, DEMAND_MU * 2, 200)
    costs   = [expected_period_cost(I, mu_y, sigma_y, DEMAND_MU) for I in I_range]
    ax.plot(I_range, costs, color="#8E44AD", lw=2.5)
    ax.axvline(I_opt, color="#E74C3C", ls="--", lw=1.8,
               label=f"I* = {I_opt:.3f}")
    ax.set_xlabel("Inventory order quantity I")
    ax.set_ylabel("Expected total cost")
    ax.set_title("Newsvendor Cost Curve  (C_h=1, C_b=5)", fontweight="bold")
    ax.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig("yield_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved yield_distribution.png")

def plot_cost_comparison(static_costs, dynamic_costs):
    s_mean = np.mean(static_costs)
    d_mean = np.mean(dynamic_costs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Supply Chain Policy Comparison — Monte Carlo Simulation",
                 fontsize=13, fontweight="bold")

    # Histogram
    ax = axes[0]
    ax.hist(static_costs,  bins=50, alpha=0.6, color="#E74C3C", label="Static policy")
    ax.hist(dynamic_costs, bins=50, alpha=0.6, color="#27AE60", label="Dynamic policy")
    ax.axvline(s_mean, color="#E74C3C", ls="--", lw=2)
    ax.axvline(d_mean, color="#27AE60", ls="--", lw=2)
    ax.set_xlabel("Total batch cost (normalised)")
    ax.set_ylabel("Frequency")
    ax.set_title("Monte Carlo Cost Distribution\nStatic vs. DAG-Informed Dynamic Policy",
                 fontweight="bold")
    ax.legend(fontsize=10)

    # Box plot
    ax = axes[1]
    bp = ax.boxplot(
        [static_costs, dynamic_costs],
        labels=["Static", "Dynamic"],
        patch_artist=True,
        notch=True,
        widths=0.4,
    )
    bp["boxes"][0].set_facecolor("#E74C3C")
    bp["boxes"][1].set_facecolor("#27AE60")
    for patch in bp["boxes"]: patch.set_alpha(0.7)
    reduction = (s_mean - d_mean) / s_mean * 100
    ax.set_ylabel("Total batch cost (normalised)")
    ax.set_title(f"Cost Box Plot\n{reduction:.1f}% expected cost reduction", fontweight="bold")

    fig.tight_layout()
    fig.savefig("supply_chain_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved supply_chain_comparison.png")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    titre_vals = load_predictions()

    mu_y, sigma_y = fit_yield_distribution(titre_vals)
    print(f"Titre distribution:  μ = {mu_y:.4f},  σ = {sigma_y:.4f}")

    I_opt, cost_opt = optimal_order(mu_y, sigma_y, DEMAND_MU)
    critical_ratio  = C_b / (C_b + C_h)
    print(f"\nNewsvendor solution (C_h={C_h}, C_b={C_b}, C_p={C_p})")
    print(f"  Critical ratio:   {critical_ratio:.2f}")
    print(f"  Optimal order I*: {I_opt:.4f}")
    print(f"  Expected cost:    {cost_opt:.4f}")

    static_costs, dynamic_costs, mu_y, sigma_y = monte_carlo_comparison(titre_vals)

    plot_yield_distribution(mu_y, sigma_y)
    plot_cost_comparison(static_costs, dynamic_costs)
