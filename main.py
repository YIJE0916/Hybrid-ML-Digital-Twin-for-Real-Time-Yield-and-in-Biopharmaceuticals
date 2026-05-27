"""
Hybrid-ML Digital Twin for Biopharma Yield & Supply Chain Prediction
Run order: DAG → Mechanistic → Hybrid ML → Supply Chain

Usage:
    python main.py              # run all four layers
    python main.py --dag        # causal structure only
    python main.py --mech       # mechanistic ODE only
    python main.py --ml         # hybrid ML only
    python main.py --sc         # supply chain only
"""

import argparse
import subprocess
import sys
import os
import time

# Force non-interactive matplotlib in all subprocesses so plt.show() never blocks
ENV = os.environ.copy()
ENV["MPLBACKEND"] = "Agg"

def run_script(name, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    result = subprocess.run(
        [sys.executable, name],
        env=ENV,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    if result.returncode != 0:
        print(f"\n[ERROR] {name} exited with code {result.returncode}. Stopping pipeline.")
        sys.exit(result.returncode)

def run_dag():
    run_script("dag.py", "LAYER 0: Causal DAG  (NOTEARS + Literature Skeleton)  ~3 min")

def run_mechanistic():
    run_script("mechanistic.py", "LAYER 1: Mechanistic ODE  (Monod + Luedeking-Piret)")

def run_hybrid_ml():
    run_script("hybrid_ml.py", "LAYER 2: Hybrid Soft-Sensor  (XGBoost on ODE Residuals)")

def run_supply_chain():
    run_script("supply_chain.py", "LAYER 3: Stochastic Supply Chain Optimisation")

# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Hybrid-ML Digital Twin pipeline")
    parser.add_argument("--dag",  action="store_true", help="Run DAG layer only")
    parser.add_argument("--mech", action="store_true", help="Run mechanistic ODE only")
    parser.add_argument("--ml",   action="store_true", help="Run hybrid ML only")
    parser.add_argument("--sc",   action="store_true", help="Run supply chain only")
    args = parser.parse_args()

    selected = any([args.dag, args.mech, args.ml, args.sc])
    t0 = time.time()

    if args.dag or not selected:
        run_dag()
    if args.mech or not selected:
        run_mechanistic()
    if args.ml or not selected:
        run_hybrid_ml()
    if args.sc or not selected:
        run_supply_chain()

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Output files:")
    print(f"    dag.png                      — causal graph")
    print(f"    mechanistic_predictions.csv  — ODE layer output")
    print(f"    mechanistic_fit.png          — Monod + L-P parameter fits")
    print(f"    mechanistic_trajectories.png — per-reactor Titre trajectories")
    print(f"    hybrid_predictions.csv       — hybrid model predictions")
    print(f"    shap_analysis.png            — SHAP feature importance")
    print(f"    hybrid_predictions.png       — parity + residual plots")
    print(f"    yield_distribution.png       — f̂_t(y) + newsvendor cost curve")
    print(f"    supply_chain_comparison.png  — Monte Carlo policy comparison")
    print("="*60)

if __name__ == "__main__":
    main()
