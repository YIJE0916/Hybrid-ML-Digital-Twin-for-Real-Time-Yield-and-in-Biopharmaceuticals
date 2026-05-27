"""
CHO Fed-Batch Bioprocess – Causal DAG
Dataset : cbeurrier/bioprocess-performance (Kaggle)
Node / edge construction grounded in QbD-PAT literature (see dag_explanation.md).
"""

import pandas as pd
import numpy as np
from causalnex.structure.notears import from_pandas
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

np.random.seed(42)

# ── 1. Load & rename columns ───────────────────────────────────────────────────
raw = pd.read_excel("new.xlsx", sheet_name="Dataset", skiprows=1)
raw.columns = [
    "Reactor", "Day", "Volume", "Glucose_Feed",
    # ML Inputs (measured telemetry)
    "VCD", "Titre", "Temperature",
    "Glucose_conc", "Glucose_consumed", "Cumul_Glucose",
    "Lactate_conc", "Protein_amount", "Viability",
    # ML→Mechanistic bridge (computed specific rates)
    "mu", "q_p", "q_s",
    # Mechanistic layer outputs (predicted quantities – excluded from DAG)
    "VCD_pred", "Titre_pred", "Cumul_Glucose_pred",
]

# ── 2. Select DAG node set (literature-grounded taxonomy) ──────────────────────
# Nodes follow ICH Q8(R2) CPP/CQA taxonomy and Monod kinetics framework.
# CPPs: Temperature, Glucose_Feed
# Process state: Glucose_conc, Glucose_consumed, Cumul_Glucose
# Specific rates (ML→Mechanistic bridge): mu, q_p, q_s
# Cell state: VCD, Viability
# Metabolic byproduct: Lactate_conc
# Intermediate product: Protein_amount
# CQA: Titre
DAG_COLS = [
    "Temperature", "Glucose_Feed",
    "Glucose_conc", "Glucose_consumed", "Cumul_Glucose",
    "mu", "q_p", "q_s",
    "VCD", "Viability",
    "Lactate_conc", "Protein_amount",
    "Titre",
]

df = raw[DAG_COLS].apply(pd.to_numeric, errors="coerce").dropna()
print(f"Data shape after cleaning: {df.shape}")

# ── 3. NOTEARS structure learning ──────────────────────────────────────────────
# Forbidden edges: effects cannot retroactively cause their inputs (acausality).
TABU = [
    ("Titre",         "Temperature"),  ("Titre",         "Glucose_Feed"),
    ("Titre",         "mu"),           ("Titre",         "VCD"),
    ("Titre",         "q_p"),
    ("VCD",           "Temperature"),  ("VCD",           "Glucose_Feed"),
    ("mu",            "Temperature"),  ("mu",            "Glucose_Feed"),
    ("Lactate_conc",  "Glucose_Feed"), ("Lactate_conc",  "mu"),
    ("Protein_amount","Temperature"),  ("Protein_amount","Glucose_Feed"),
    ("Cumul_Glucose", "Glucose_conc"), ("Cumul_Glucose", "Glucose_Feed"),
    ("Viability",     "Temperature"),  ("Viability",     "Glucose_Feed"),
    ("Viability",     "mu"),           ("Viability",     "Glucose_conc"),
    ("q_p",           "Glucose_Feed"), ("q_s",           "Glucose_Feed"),
]

sm = from_pandas(df, w_threshold=0.8, tabu_edges=TABU)

# ── 4. Enforce literature-required edges ───────────────────────────────────────
# Each edge is grounded in a specific mechanistic or kinetic model.
# See dag_explanation.md for full citation details.
REQUIRED = [
    # Substrate supply (mass balance)
    ("Glucose_Feed",    "Glucose_conc"),    # Fed-batch mass balance
    # Monod kinetics: μ = μmax · S / (Ks + S)   [Monod 1949]
    ("Glucose_conc",    "mu"),
    ("Temperature",     "mu"),              # Arrhenius / Q10 effect [Schmalzriedt 2003]
    # Cell mass balance: dX/dt = μX − DX         [Doyle & Stankovic 2005]
    ("mu",              "VCD"),
    # Growth-associated specific rates (Luedeking-Piret model) [Luedeking & Piret 1959]
    ("mu",              "q_p"),
    ("mu",              "q_s"),
    # Substrate consumption stoichiometry
    ("VCD",             "Glucose_consumed"),
    ("Glucose_consumed","Cumul_Glucose"),
    # Aerobic glycolysis / Warburg-like effect in CHO [Zagari et al. 2013]
    ("VCD",             "Lactate_conc"),
    ("Glucose_consumed","Lactate_conc"),
    # Cell viability inhibition
    ("Lactate_conc",    "Viability"),       # Lactate toxicity [Altamirano et al. 2001]
    ("Temperature",     "Viability"),       # Thermal stress   [Kaufmann et al. 1999]
    # Product accumulation: P = q_p · X · Δt
    ("VCD",             "Protein_amount"),
    ("q_p",             "Protein_amount"),
    # CQA: Titre = extracellular protein concentration [ICH Q8(R2)]
    ("Protein_amount",  "Titre"),
]

REQUIRED_SET = set(REQUIRED)

for src, dst in REQUIRED:
    if not sm.has_edge(src, dst):
        sm.add_edge(src, dst)

for src, dst in TABU:
    if sm.has_edge(src, dst):
        sm.remove_edge(src, dst)

# ── 5. Build NetworkX graph and assign positions ───────────────────────────────
G = nx.DiGraph(sm)

# Hierarchical layout – causal flow runs top → bottom
POS = {
    # Layer 0: CPPs
    "Temperature":      (-4.5, 4),
    "Glucose_Feed":     (3.5, 4),
    # Layer 1: Process state
    "Glucose_conc":     (0.5, 2.5),
    # Layer 2: Kinetic rates
    "mu":               (-2, 1),
    "Glucose_consumed": (3.5, 1.5),
    # Layer 3: Derived state
    "q_p":              (-4.5, -0.5),
    "q_s":              (-1,   -0.5),
    "VCD":              (-2.5, -1.5),
    "Cumul_Glucose":    (5.5,  0.5),
    # Layer 4: Metabolic/cell quality
    "Lactate_conc":     (2,    -1),
    "Viability":        (2,    -2.5),
    "Protein_amount":   (-2.5, -3),
    # Layer 5: CQA
    "Titre":            (-2.5, -4.5),
}

# ── 6. Color coding by biological role ────────────────────────────────────────
LAYER_COLOR = {
    "Temperature":    "#E74C3C",  # CPP
    "Glucose_Feed":   "#E74C3C",  # CPP
    "Glucose_conc":   "#F39C12",  # Process state
    "Glucose_consumed":"#F39C12", # Process state
    "Cumul_Glucose":  "#F39C12",  # Process state
    "mu":             "#27AE60",  # Kinetic rate
    "q_p":            "#27AE60",  # Kinetic rate
    "q_s":            "#27AE60",  # Kinetic rate
    "VCD":            "#2980B9",  # Cell state
    "Viability":      "#2980B9",  # Cell state
    "Lactate_conc":   "#8E44AD",  # Metabolic byproduct
    "Protein_amount": "#5D6D7E",  # Intermediate product
    "Titre":          "#1ABC9C",  # CQA
}

node_colors = [LAYER_COLOR.get(n, "#BDC3C7") for n in G.nodes()]

# Distinguish literature-required vs. NOTEARS-learned edges
solid_edges  = [(u, v) for u, v in G.edges() if (u, v) in REQUIRED_SET]
dashed_edges = [(u, v) for u, v in G.edges() if (u, v) not in REQUIRED_SET]

# ── 7. Draw ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 12))

nx.draw_networkx_nodes(G, POS, node_color=node_colors,
                       node_size=3000, alpha=0.92, ax=ax)
nx.draw_networkx_labels(G, POS, font_size=8.5, font_weight="bold", ax=ax)

nx.draw_networkx_edges(G, POS, edgelist=solid_edges, ax=ax,
                       edge_color="#1A252F", arrows=True,
                       arrowsize=22, width=2.2,
                       connectionstyle="arc3,rad=0.07")
nx.draw_networkx_edges(G, POS, edgelist=dashed_edges, ax=ax,
                       edge_color="#95A5A6", arrows=True,
                       arrowsize=16, width=1.3, style="dashed",
                       connectionstyle="arc3,rad=0.07")

# Legend
legend_handles = [
    mpatches.Patch(color="#E74C3C", label="CPP – Critical Process Parameter"),
    mpatches.Patch(color="#F39C12", label="Process State Variable"),
    mpatches.Patch(color="#27AE60", label="Specific Rate (Kinetic Bridge)"),
    mpatches.Patch(color="#2980B9", label="Cell State"),
    mpatches.Patch(color="#8E44AD", label="Metabolic Byproduct"),
    mpatches.Patch(color="#5D6D7E", label="Intermediate Product"),
    mpatches.Patch(color="#1ABC9C", label="CQA – Critical Quality Attribute"),
    Line2D([0], [0], color="#1A252F", linewidth=2,
           label="Literature-required edge"),
    Line2D([0], [0], color="#95A5A6", linewidth=1.5, linestyle="dashed",
           label="NOTEARS-learned edge"),
]
ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5, framealpha=0.9)

ax.set_title(
    "CHO Fed-Batch Bioprocess Causal DAG\n"
    "(Hybrid Mechanistic–ML | QbD/PAT Framework)",
    fontsize=13, fontweight="bold"
)
ax.axis("off")
plt.tight_layout()
plt.savefig("dag.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"\nDAG summary: {len(G.nodes())} nodes, {len(G.edges())} edges")
print(f"  Literature-required : {len(solid_edges)}")
print(f"  NOTEARS-learned     : {len(dashed_edges)}")

