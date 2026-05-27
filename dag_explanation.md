# CHO Fed-Batch Bioprocess – Causal DAG Explanation

**Project:** Hybrid-ML Digital Twin for Real-Time Yield and Supply Chain Risk Prediction in Biopharmaceuticals  
**Dataset:**  — 30 CHO reactors, 377 normalised observations  
**Method:** NOTEARS structure learning (Zheng et al. 2018) + literature-enforced causal skeleton

---

## 1. Overview

The DAG encodes the causal structure of a CHO (Chinese Hamster Ovary) fed-batch bioreactor using the **Quality by Design (QbD)** taxonomy established in ICH Q8(R2). Every node and directed edge is either (a) **literature-required** — grounded in a specific mechanistic or kinetic model — or (b) **data-driven** — discovered by NOTEARS from the dataset but not contradicted by known biology.

The graph is acyclic by construction: causal flow runs from upstream process inputs (CPPs) down through intracellular kinetics, cell state, and metabolic byproducts to the final product quality attribute (Titre).

---

## 2. Node Definitions

### 2.1 Critical Process Parameters (CPPs) — Red

CPPs are independent variables that are directly controlled by the operator and whose variation has a demonstrated impact on product quality (ICH Q8(R2), 2009; FDA PAT Guidance, 2004).

| Node | Variable | Role |
|------|----------|------|
| **Temperature** | Bioreactor temperature (normalised) | Thermal optimum for CHO growth and protein folding. Typically controlled at 36–37 °C during growth, shifted to 31–33 °C in production phase. |
| **Glucose_Feed** | Volume of glucose bolus added (normalised) | Fed-batch feeding strategy that sustains substrate concentration above the Monod half-saturation constant *Ks* throughout the culture. |

### 2.2 Process State Variables — Orange

These are directly measurable quantities that describe the current physiochemical state of the reactor medium.

| Node | Variable | Role |
|------|----------|------|
| **Glucose_conc** | Dissolved glucose concentration | The limiting substrate in Monod kinetics. Its trajectory is determined by the balance of feeding rate and cellular consumption. |
| **Glucose_consumed** | Glucose consumed per time step | Stoichiometric consumption by cells; primary precursor for both biomass synthesis and lactate production. |
| **Cumul_Glucose** | Cumulative glucose consumed | Integral of per-step consumption; captures total substrate demand over the batch and is used as a mechanistic model input. |

### 2.3 Specific Rates — Kinetic Bridge — Green

These specific rates are the quantitative bridge between the ML soft-sensor layer and the mechanistic ODE layer (see Proposal Section 3.1–3.2). They are computed from telemetry by the ML model and then used as inputs to the mechanistic equations.

| Node | Variable | Role |
|------|----------|------|
| **mu** (μ) | Specific growth rate | Central kinetic variable. Computed from the Monod equation: μ = μ_max · S / (K_s + S). Governs both cell proliferation (VCD) and growth-associated production (q_p). |
| **q_p** | Specific productivity | Rate of product synthesis per cell. In the Luedeking-Piret model: q_p = α·μ + β, where α captures growth-associated production and β non-growth-associated production. |
| **q_s** | Specific glucose consumption rate | Rate of glucose uptake per cell. Linked to μ through stoichiometric yield coefficients (Y_X/S): q_s ≈ μ / Y_X/S. |

### 2.4 Cell State — Blue

| Node | Variable | Role |
|------|----------|------|
| **VCD** | Viable cell density | Net number of living cells per unit volume. Governed by the mass balance: dX/dt = μX − DX, where D is the dilution rate. VCD is both a key process indicator and the direct driver of product accumulation. |
| **Viability** | Fraction of live cells | An indicator of culture health. Declines when inhibitory metabolites (Lactate, Ammonia) accumulate or when cells experience temperature stress. |

### 2.5 Metabolic Byproduct — Purple

| Node | Variable | Role |
|------|----------|------|
| **Lactate_conc** | Lactate concentration | Primary inhibitory byproduct of aerobic glycolysis in CHO cells (Warburg-like effect). At concentrations above ~20 mM (unnormalised), lactate directly reduces cell viability and growth rate. |

### 2.6 Intermediate Product — Gray

| Node | Variable | Role |
|------|----------|------|
| **Protein_amount** | Cumulative protein amount | Total secreted product accumulated in the reactor, equal to the integral of VCD × q_p × Δt. Serves as the mechanistic intermediate between specific productivity and final titre. |

### 2.7 Critical Quality Attribute (CQA) — Teal

| Node | Variable | Role |
|------|----------|------|
| **Titre** | Extracellular product concentration | The primary CQA of a biopharmaceutical production process (ICH Q8(R2)). Titre = Protein_amount / Volume. Maximising titre while maintaining product quality (glycosylation, charge heterogeneity) is the core objective. |

---

## 3. Causal Edge Definitions

### 3.1 Literature-Required Edges (solid black arrows)

These edges are enforced regardless of NOTEARS output because they correspond to established mechanistic laws.

#### Substrate Supply
- **Glucose_Feed → Glucose_conc**  
  Fed-batch mass balance: dS/dt = F·S_F/V − μX/Y_X/S. Each glucose bolus directly raises dissolved concentration. *(Mass balance; standard fed-batch control theory)*

#### Monod Kinetics
- **Glucose_conc → μ**  
  Monod equation: μ = μ_max · S / (K_s + S). The specific growth rate is a saturating function of substrate concentration. *(Monod 1949)*
  
- **Temperature → μ**  
  Temperature modulates μ through an Arrhenius-type relationship: μ(T) ∝ exp(−E_a / RT). Temperature shifts are a standard PAT strategy to switch CHO cultures from growth to production phase. *(Schmalzriedt et al. 2003; Kaufmann et al. 1999)*

#### Cell Mass Balance
- **μ → VCD**  
  Biomass balance: dX/dt = μX − DX. Growth rate is the sole positive driver of cell density increase in a fed-batch with no cell bleed. *(Doyle & Stankovic 2005)*

#### Growth-Associated Specific Rates (Luedeking-Piret)
- **μ → q_p**  
  Luedeking-Piret model: q_p = α·μ + β. The growth-associated term (α·μ) links productivity to growth rate; for many mAb processes α dominates. *(Luedeking & Piret 1959; Lim et al. 2010)*
  
- **μ → q_s**  
  Stoichiometric linkage: q_s = μ / Y_X/S + m_s, where m_s is maintenance energy. Higher growth rates require higher glucose uptake rates. *(Zupke & Stephanopoulos 1994)*

#### Substrate Consumption Stoichiometry
- **VCD → Glucose_consumed**  
  Absolute consumption rate = q_s · VCD. More viable cells = more glucose consumed per time step. *(Standard stoichiometry)*
  
- **Glucose_consumed → Cumul_Glucose**  
  Cumulative consumption is the running integral of per-step consumption; a purely deterministic relationship. *(Integral)*

#### Aerobic Glycolysis (Warburg-like Effect in CHO)
- **VCD → Lactate_conc**  
  Each viable cell secretes lactate as a byproduct of aerobic glycolysis (high glycolytic flux even under aerobic conditions, analogous to the Warburg effect). *(Zagari et al. 2013; Templeton et al. 2013)*
  
- **Glucose_consumed → Lactate_conc**  
  Stoichiometrically, lactate yield from glucose is approximately 1.3–1.8 mol/mol in CHO, depending on metabolic state. *(Zagari et al. 2013)*

#### Cell Viability Inhibition
- **Lactate_conc → Viability**  
  High lactate concentrations directly inhibit cell growth and increase apoptosis. The inhibitory effect follows an uncompetitive or non-competitive inhibition model. *(Altamirano et al. 2001; Mulukutla et al. 2012)*
  
- **Temperature → Viability**  
  Sub-optimal temperatures (>37 °C or temperature shifts that are too abrupt) reduce the live cell fraction by inducing heat-shock responses and apoptosis. *(Kaufmann et al. 1999)*

#### Product Accumulation
- **VCD → Protein_amount**  
  Total secreted protein = ∫ q_p · VCD dt. Higher cell density directly scales product accumulation. *(Lim et al. 2010)*
  
- **q_p → Protein_amount**  
  Higher per-cell productivity directly increases accumulation rate independent of cell count. *(Luedeking & Piret 1959)*

#### CQA
- **Protein_amount → Titre**  
  Titre is defined as extracellular protein concentration = Protein_amount / Volume. *(ICH Q8(R2) 2009)*

---

### 3.2 NOTEARS-Learned Edges (dashed gray arrows)

These are edges discovered by the NOTEARS algorithm from the observational data that were not explicitly required by the literature skeleton. They are kept if they are not physically forbidden and do not create cycles.

Common examples that may appear:
- **Cumul_Glucose → q_s**: Historical glucose depletion may condition current uptake rate — consistent with substrate limitation adaption.
- **Viability → Protein_amount**: Lower viability correlates with less protein secretion — a real biological association even though it is downstream in the mechanistic hierarchy.
- **q_s → VCD**: Glucose consumption rate is a proxy for metabolic activity, which correlates strongly with growth.

These data-driven edges should be interpreted cautiously: they reflect conditional dependence in the dataset, not necessarily direct physical causation. In downstream analysis, sensitivity (SHAP) should be used to check whether these edges contribute meaningfully to prediction.

---

## 4. DAG Structure Overview

```
CPPs (controlled inputs)
    Temperature ─────────────────────────────────────┐
    Glucose_Feed ──► Glucose_conc                    │
                           │                         │
                           ▼  (Monod kinetics)       ▼
                          mu ◄─────────────── Temperature
                     ┌─────┴──────┐
                     ▼            ▼
                   VCD           q_p
               ┌───┴───┐         │
               ▼       ▼         ▼
    Glucose_consumed  Lactate  Protein_amount ──► Titre (CQA)
               │       │
               ▼       ▼
         Cumul_Glucose  Viability
```

The graph has **5 natural causal layers**:
1. **CPPs** → operator-controlled inputs, no upstream causes within the process
2. **Substrate state** → determined by feeding and consumption
3. **Kinetic rates (mu, q_p, q_s)** → the ML-to-Mechanistic bridge layer
4. **Cell and metabolic state** → VCD, Lactate, Viability, Protein
5. **CQA** → Titre as the downstream quality objective

---

## 5. Integration with the Hybrid-ML Framework

The DAG serves three roles in the broader Hybrid-ML Digital Twin:

| Framework Layer | DAG Role |
|-----------------|----------|
| **Mechanistic ODE Layer** (Section 3.1) | Enforces the directed edges μ→VCD, Glucose→μ, VCD→Protein→Titre as hard structural constraints on the ODE system. |
| **Soft-Sensor ML Layer** (Section 3.2) | Defines the feature set for the XGBoost model: only parents of Titre in the DAG are legitimate predictors. Eliminates spurious correlates. Informs SHAP analysis by showing which upstream node is responsible for a titre deviation. |
| **Supply Chain Stochastic Layer** (Section 3.3) | The DAG propagates uncertainty: perturbations in CPPs (e.g., temperature excursion) can be traced forward through the causal graph to quantify their effect on the titre yield distribution *f̂_t(y)*, enabling early warning 7–10 days before batch end. |

---

## 6. References

| # | Citation |
|---|----------|
| 1 | Monod, J. (1949). The growth of bacterial cultures. *Annual Review of Microbiology*, 3, 371–394. |
| 2 | Luedeking, R., & Piret, E. L. (1959). A kinetic study of the lactic acid fermentation. *Journal of Biochemical and Microbiological Technology and Engineering*, 1(4), 393–412. |
| 3 | Kaufmann, H., Mazur, X., Marone, R., Bailey, J. E., & Fussenegger, M. (1999). Comparative analysis of two controlled proliferation strategies regarding product quality, influence on the cell physiology and process costs for CHO cells. *Biotechnology and Bioengineering*, 63(5), 573–582. |
| 4 | Altamirano, C., Illanes, A., Casablancas, A., Gàmez, X., Cairo, J. J., & Gódia, C. (2001). Analysis of CHO cells metabolic redistribution in a glutamate-based defined medium in continuous culture. *Biotechnology Progress*, 17(6), 1032–1041. |
| 5 | Doyle III, F. J., & Stankovic, A. M. (2005). Control of fed-batch fermentation. In *Chemical Process Control* (pp. 345–380). AIChE. |
| 6 | Zagari, F., Jordan, M., Stettler, M., Broly, H., & Wurm, F. M. (2013). Lactate metabolism shift in CHO cell culture: the role of mitochondrial oxidative activity. *New Biotechnology*, 30(2), 238–245. |
| 7 | Lim, Y., Kim, T. K., Lee, J. S., Lee, G. M., & Workman, C. T. (2010). Metabolic flux analysis of CHO cells in different culture media. *Metabolic Engineering*, 12, 277–293. |
| 8 | Schmalzriedt, S., Jenne, M., Mauch, K., & Reuss, M. (2003). Integration of physiology and fluid dynamics. *Advances in Biochemical Engineering/Biotechnology*, 80, 19–68. |
| 9 | Templeton, N., Dean, J., Reddy, P., & Young, J. D. (2013). Peak antibody production is associated with increased oxidative metabolism in an industrially relevant fed-batch CHO cell culture. *Biotechnology and Bioengineering*, 110(7), 2013–2024. |
| 10 | Mulukutla, B. C., Khan, S., Lange, A., & Hu, W. S. (2012). Glucose metabolism in mammalian cell culture: new insights for tweaking vintages. *Trends in Biotechnology*, 30(7), 371–377. |
| 11 | Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *NeurIPS 2018*. |
| 12 | ICH Q8(R2) (2009). *Pharmaceutical Development*. International Council for Harmonisation. |
| 13 | FDA (2004). *PAT — A Framework for Innovative Pharmaceutical Development, Manufacturing, and Quality Assurance*. US Food and Drug Administration. |
| 14 | Zupke, C., & Stephanopoulos, G. (1994). Modeling of isotope distributions and intracellular fluxes in metabolic networks using atom mapping matrices. *Biotechnology Progress*, 10(5), 489–498. |
