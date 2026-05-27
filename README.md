# Hybrid-ML Digital Twin for Real-Time Yield and Supply Chain Risk Prediction in Biopharmaceuticals

A four-layer computational pipeline that combines **causal structure learning**, **mechanistic ODEs**, **machine learning**, and **stochastic supply chain optimisation** to predict CHO fed-batch bioreactor yield and quantify downstream supply risk.

---

## Overview

| Layer | Script | Method |
|-------|--------|--------|
| 0 — Causal DAG | `dag.py` | NOTEARS + literature-enforced skeleton |
| 1 — Mechanistic ODE | `mechanistic.py` | Monod kinetics + Luedeking-Piret model |
| 2 — Hybrid Soft-Sensor | `hybrid_ml.py` | XGBoost on ODE residuals + SHAP analysis |
| 3 — Supply Chain | `supply_chain.py` | Monte Carlo stochastic optimisation |

**Dataset:** 30 CHO (Chinese Hamster Ovary) fed-batch reactors, 377 normalised observations

---

## Pipeline Architecture

```
CPPs (Temperature, Glucose_Feed)
        │
        ▼
[Layer 0] Causal DAG (NOTEARS)
        │  learns causal structure over process variables
        ▼
[Layer 1] Mechanistic ODE
        │  Monod: μ = μ_max · S / (K_s + S)
        │  Luedeking-Piret: q_p = α·μ + β
        ▼
[Layer 2] Hybrid ML (XGBoost)
        │  fits residuals not captured by ODE
        │  SHAP explains feature contributions
        ▼
[Layer 3] Supply Chain Optimisation
           Monte Carlo yield distribution → newsvendor cost minimisation
```

---

## Key Variables (DAG Nodes)

| Category | Variables |
|----------|-----------|
| **CPPs** (controlled inputs) | Temperature, Glucose_Feed |
| **Process state** | Glucose_conc, Glucose_consumed, Cumul_Glucose |
| **Kinetic rates** | μ (specific growth rate), q_p (specific productivity), q_s (glucose uptake) |
| **Cell state** | VCD (viable cell density), Viability |
| **Metabolic byproduct** | Lactate_conc |
| **CQA** (target) | Titre |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python main.py

# Run individual layers
python main.py --dag     # Causal DAG only (~3 min)
python main.py --mech    # Mechanistic ODE only
python main.py --ml      # Hybrid ML only
python main.py --sc      # Supply chain only
```

### Output Files

| File | Description |
|------|-------------|
| `Figures/dag.png` | Causal graph with node categories |
| `Figures/mechanistic_fit.png` | Monod + Luedeking-Piret parameter fits |
| `Figures/mechanistic_trajectories.png` | Per-reactor Titre trajectories |
| `Figures/hybrid_predictions.png` | Parity plot + residual analysis |
| `Figures/shap_analysis.png` | SHAP feature importance |
| `Figures/yield_distribution.png` | Yield distribution + newsvendor cost curve |
| `Figures/supply_chain_comparison.png` | Monte Carlo policy comparison |

---

## Causal DAG

The DAG encodes the causal structure of the CHO fed-batch process following the **Quality by Design (QbD)** taxonomy from ICH Q8(R2). Edges are either:
- **Literature-required** — grounded in established mechanistic/kinetic models (Monod 1949, Luedeking-Piret 1959)
- **Data-driven** — discovered by NOTEARS from the observational dataset

See [`dag_explanation.md`](dag_explanation.md) for full node and edge definitions with references.

---

## Project Structure

```
.
├── main.py                 # Pipeline entry point (runs all 4 layers)
├── dag.py                  # Layer 0: Causal structure learning
├── mechanistic.py          # Layer 1: Mechanistic ODE model
├── hybrid_ml.py            # Layer 2: XGBoost soft-sensor + SHAP
├── supply_chain.py         # Layer 3: Stochastic supply chain
├── shap_extra.py           # Additional SHAP visualisations
├── dag_explanation.md      # Detailed DAG documentation with references
└── Figures/                # All output figures
```

---

## Methods

### Causal Structure Learning
NOTEARS (Zheng et al., NeurIPS 2018) formulates DAG learning as a continuous optimisation problem, enforcing acyclicity via a smooth algebraic constraint. A literature-derived skeleton ensures biologically implausible edges are excluded.

### Mechanistic Model
- **Monod kinetics**: specific growth rate as a saturating function of substrate concentration
- **Luedeking-Piret model**: specific productivity linked to growth rate (growth-associated + non-growth-associated terms)
- **Mass balances**: VCD, Titre, Lactate, and Glucose dynamics

### Hybrid ML Layer
XGBoost is trained on the residuals between mechanistic ODE predictions and observed data. Only DAG-identified upstream variables (parents of Titre) are used as features, eliminating spurious correlates. SHAP values provide interpretable feature attribution.

### Supply Chain Optimisation
Monte Carlo simulation propagates uncertainty from CPP perturbations (e.g., temperature excursions) through the causal graph to generate a yield distribution $\hat{f}_t(y)$. A newsvendor-style cost function minimises holding + shortage costs, enabling early warning 7–10 days before batch end.

---

## References

1. Monod, J. (1949). The growth of bacterial cultures. *Annual Review of Microbiology*, 3, 371–394.
2. Luedeking, R. & Piret, E. L. (1959). A kinetic study of the lactic acid fermentation. *JBMTE*, 1(4), 393–412.
3. Zheng, X. et al. (2018). DAGs with NO TEARS. *NeurIPS 2018*.
4. ICH Q8(R2) (2009). *Pharmaceutical Development*. International Council for Harmonisation.
5. FDA (2004). *PAT — A Framework for Innovative Pharmaceutical Development*. US FDA.

Full reference list in [`dag_explanation.md`](dag_explanation.md).
