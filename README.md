# Bayesian GRN Inference: GCV-Calibrated Ridge Regression Outperforms GENIE3 in the Low-Sample Regime

> **Key finding:** GCV-calibrated ridge regression, applied to per-gene standardised expression data, beats GENIE3 on 3 of 4 DREAM5 gene regulatory network benchmarks — achieving +59% AUPR on *S. aureus*, +43% AUPR on *E. coli*, and +9% AUPR on *S. cerevisiae* — without any training loop, in under 2 seconds total.

This repository contains the full experimental codebase, results, and preprint draft sections for a study of Bayesian regression approaches to gene regulatory network (GRN) inference, benchmarked on the DREAM5 challenge (Marbach et al. 2012).

---

## Results at a Glance

**Primary metric: AUPR** (area under precision-recall curve). AUPR rewards precision at the top of the ranked edge list — the operationally relevant quantity for downstream experimental validation.

| Network | Organism | n/TF ratio | GENIE3 AUPR | **GCV-Ridge AUPR** | Δ |
|---------|----------|-----------|------------|-------------------|---|
| net1 | In silico | 4.1 | 0.259 | 0.237 | −8% |
| net2 | *S. aureus* | 1.6 | 0.011 | **0.018** | **+59%** |
| net3 | *E. coli* | 2.4 | 0.099 | **0.142** | **+43%** |
| net4 | *S. cerevisiae* | 1.6 | 0.022 | **0.024** | **+9%** |

The crossover from GENIE3-dominant to Ridge-dominant performance occurs between n/TF ≈ 2.4 and n/TF ≈ 4.1 — consistent with the data-scarcity regime where non-parametric ensemble methods overfit.

**Ablation summary (AUPR):**

| Method | net1 | net2 | net3 | net4 | Beats GENIE3 |
|--------|------|------|------|------|-------------|
| GENIE3 (baseline) | 0.259 | 0.011 | 0.099 | 0.022 | — |
| **GCV-Ridge** | **0.237** | **0.018** | **0.142** | **0.024** | **3/4** |
| VI Horseshoe (B-JEPA) | 0.204 | 0.016 | 0.080 | — | 1/3 |
| Ridge (hand-tuned) | 0.186 | 0.018 | 0.087 | 0.024 | 2/4 |
| MAP + JEPA prior | 0.080 | 0.002 | 0.037 | 0.019 | 0/4 |

---

## Method Overview

### The Core Method: GCV-Calibrated Ridge Regression

For each DREAM5 network, the pipeline is:

1. **Standardise** expression per gene across samples (zero mean, unit variance)
2. **Fit ridge regression** of all target genes simultaneously against TF expression:
   `W = (XᵀX + αI)⁻¹ XᵀY` — one batched LAPACK solve
3. **Calibrate α** using Generalised Cross-Validation (GCV) on the SVD of X — no gold standard labels used
4. **Rank edges** by absolute coefficient `|W[tf, gene]|`

Runtime: < 2 seconds across all 4 networks on CPU. No GPU. No gradient descent.

The GCV formula selects the ridge penalty that minimises a closed-form approximation to leave-one-out prediction error:

```
GCV(α) = ‖Y − X·Ŵ‖²_F / (n · (1 − tr(H_α)/n)²)
tr(H_α) = Σ_j d_j² / (d_j² + α)       # from SVD of X, computed once
```

### Why Horseshoe Shrinkage = Ridge Here

The analytical horseshoe shrinkage (Piironen & Vehtari 2017) applied post-hoc to ridge coefficients produces **identical AUROC and AUPR** on all four networks. This is provably expected: the shrinkage `|β_hs| = C|β_ols|³/(1 + C·β_ols²)` is a strictly monotone function of `|β_ols|` for fixed per-gene constants, so rank-based metrics are invariant. The horseshoe prior provides theoretical motivation for the regularisation scale (via τ₀) but no additional ranking power over ridge in the standardised expression setting.

### What Doesn't Work: The B-JEPA Architecture

We also implemented and evaluated a two-stage deep learning approach:
- **Stage 1**: I-JEPA-style encoder pre-training with EMA target encoder and linear predictor W_init
- **Stage 2**: Mean-field variational inference (VI) for the horseshoe prior over TF→gene regression weights

**Stage 1 failure:** W_init achieves AUROC = 0.495 (random) as edge scores — the JEPA pretext task encodes co-expression correlation, not causal regulation. Using W_init as a MAP prior mean degrades all four networks by 21%–89% AUPR.

**Stage 2 failure:** Mean-field VI for the horseshoe exhibits a degenerate fixed point — local scale parameters λ̃ collapse to 1 for all edges (κ ≈ 0.49, std < 0.001), because the KL gradient is ~500× larger than the regression gradient. The VI reduces to implicit ridge regression through its KL regularisation term, and underperforms explicit GCV-calibrated ridge.

These are honest negative results. The B-JEPA architecture as designed does not deliver its intended sparse Bayesian regularisation. The GCV-ridge solution — simpler, faster, and analytically grounded — is the primary contribution.

---

## Repository Structure

```
.
├── configs/
│   ├── bjepa_default.yaml          # B-JEPA training hyperparameters
│   └── bjepa_smoke.yaml            # Fast smoke-test config (few epochs)
│
├── data/                           # DREAM5 benchmark data (TSV files)
│   ├── net{1,2,3,4}_expression_data.tsv
│   ├── net{1,2,3,4}_transcription_factors.tsv
│   └── DREAM5_NetworkInference_GoldStandard_Network{1,2,3,4}.*
│
├── docs/
│   ├── methods.md                  # Full methods draft (preprint)
│   ├── results.md                  # Full results draft (preprint)
│   └── discussions.md              # Full discussion draft (preprint)
│
├── experiments/
│   ├── exp1_genie3_baseline/run.py # GENIE3 baseline runner
│   ├── exp2_bjepa/run.py           # B-JEPA two-stage training runner
│   └── exp3_analytical_hs/run.py   # GCV-ridge + horseshoe ablation runner ← PRIMARY
│
├── notebooks/
│   └── 01_eda_dream5.ipynb         # Exploratory data analysis
│
├── results/
│   ├── genie3/                     # GENIE3 edge predictions + summary
│   ├── bjepa/                      # B-JEPA training checkpoints + results
│   └── analytical_hs/              # GCV-ridge results (OLS, MAP, GCV modes)
│       ├── ols/net{1,2,3,4}/       # Hand-tuned ridge results
│       ├── map/net{1,2,3,4}/       # MAP + W_init results
│       └── summary.{csv,json}
│
└── src/bjepa/
    ├── data/
    │   └── dream5.py               # DREAM5Network dataclass + loader
    ├── eval/
    │   └── metrics.py              # AUROC, AUPR, evaluate_predictions()
    ├── models/
    │   ├── analytical_horseshoe.py # GCV calibration + ridge + horseshoe ← PRIMARY
    │   ├── horseshoe.py            # Mean-field VI horseshoe (HorseshoeRegressor)
    │   ├── bjepa.py                # BJEPAStage1, BJEPAStage2
    │   ├── encoders.py             # Context/target encoder MLPs + EMA
    │   └── predictor.py            # Linear JEPA predictor head
    ├── baselines/
    │   └── genie3.py               # GENIE3 wrapper
    └── training/
        └── trainer.py              # BJEPATrainer + TrainerConfig
```

---

## Installation

Requires Python ≥ 3.10. All experiments were run in a conda environment named `bayes_stats`.

```bash
# Clone and enter repo
cd bayesian-jepa-gene-regulatory-inference

# Create environment (if not already present)
conda create -n bayes_stats python=3.11 -y
conda activate bayes_stats

# Install package in editable mode (installs all dependencies)
pip install -e ".[dev]"
```

**Key dependencies:** `torch >= 2.2`, `numpy`, `pandas`, `scikit-learn`, `scipy`, `pyro-ppl`, `pyyaml`.

---

## Quickstart: Reproducing the Main Results

All commands are run from the repo root with `bayes_stats` activated.

### 1. GENIE3 Baseline

```bash
python experiments/exp1_genie3_baseline/run.py
# Results → results/genie3/
```

### 2. GCV-Calibrated Ridge (Primary Method) — 2 seconds total

```bash
# All 4 networks with GCV-optimal ridge
python experiments/exp3_analytical_hs/run.py --calibrate

# Single network
python experiments/exp3_analytical_hs/run.py --network 2 --calibrate

# Manual ridge factor (skip GCV)
python experiments/exp3_analytical_hs/run.py --network 2 --ridge_factor 0.05
```

Output includes per-network GCV calibration summary, AUROC/AUPR, and the full comparison table against GENIE3 and B-JEPA.

### 3. B-JEPA Two-Stage Training (Comparison)

```bash
# All networks (slow: ~20 min per network on MPS/GPU)
python experiments/exp2_bjepa/run.py

# Single network
python experiments/exp2_bjepa/run.py --network 1

# Custom config
python experiments/exp2_bjepa/run.py --config configs/bjepa_smoke.yaml
```

Checkpoints saved to `results/bjepa/net{id}/`:
- `stage1_final.pt` — Stage 1 encoder weights + W_init
- `stage2_best.pt` — Stage 2 VI weights at best AUPR epoch
- `result.json` — AUROC, AUPR, best epoch, elapsed time

---

## Key API

### GCV-Calibrated Ridge (one function call)

```python
from bjepa.data import load_network
from bjepa.models.analytical_horseshoe import analytical_horseshoe_scores, gcv_ridge_factor
from bjepa.eval.metrics import evaluate_predictions

net = load_network("data/", network_id=2)          # S. aureus

# Find optimal ridge by GCV (no labels needed)
ridge_factor = gcv_ridge_factor(net, verbose=True)

# Score all TF→gene edges
scores_df = analytical_horseshoe_scores(net, ridge_factor=ridge_factor)
# scores_df: DataFrame with columns [tf, target, score], sorted desc

# Evaluate against gold standard
metrics = evaluate_predictions(scores_df, net.gold_standard)
print(metrics)  # {'auroc': 0.6504, 'aupr': 0.0175}
```

### Horseshoe Shrinkage (rank-neutral, shown for completeness)

```python
# horseshoe=True applies (1-κ)·β shrinkage after ridge; identical AUROC/AUPR
scores_hs = analytical_horseshoe_scores(net, ridge_factor=ridge_factor, horseshoe=True)
# horseshoe=False (default-equivalent) ranks by |β_ridge| directly
scores_ridge = analytical_horseshoe_scores(net, ridge_factor=ridge_factor, horseshoe=False)
# scores_hs == scores_ridge (to 4 decimal places on all networks)
```

### Data Loading

```python
from bjepa.data import load_network

net = load_network("data/", network_id=3)

net.n_samples        # 805
net.n_tfs            # 334
net.n_genes          # 4511
net.expression       # DataFrame (805 × 4511), log-normalised
net.tf_ids           # list of 334 TF gene IDs
net.gold_standard    # DataFrame [tf, target, label]
net.n_positive_edges # 2066
```

---

## Experimental Design and Findings

### Why GCV Matters: Net3 Case Study

With ridge\_factor = 1e-6 (essentially OLS), net3 yields AUPR = 0.087. GCV selects ridge\_factor = 0.067 (α\* = 53.7), improving to AUPR = 0.142 — a **63% gain from regularisation alone**, with no gold standard labels used in calibration.

| Ridge factor | net3 AUPR | Note |
|-------------|-----------|------|
| 1e-6 (OLS) | 0.087 | Catastrophically under-regularised |
| 0.067 (GCV) | **0.142** | GCV-optimal |
| 25.8 | 0.020 | Over-regularised |

### The n/TF Crossover

| Network | n/TF | GCV-Ridge vs GENIE3 |
|---------|------|---------------------|
| net4 (*S. cerevisiae*) | 1.6 | +9% AUPR |
| net2 (*S. aureus*) | 1.6 | +59% AUPR |
| net3 (*E. coli*) | 2.4 | +43% AUPR |
| net1 (in silico) | 4.1 | −8% AUPR |

The in silico network uses Boolean regulatory logic (genuinely non-linear), favouring GENIE3 by construction. On all three biological networks, the linear model dominates.

### B-JEPA Failure Modes (Diagnosed)

| Component | Symptom | Root cause |
|-----------|---------|------------|
| Stage 1 W_init | AUROC = 0.495 (random) | JEPA encodes co-expression, not causality |
| MAP prior from W_init | AUPR −21% to −89% | Co-expression prior biases toward noise |
| VI horseshoe | κ ≈ 0.49 for all edges (std < 0.001) | KL gradient 500× larger than regression gradient |
| VI horseshoe | Performs like ridge | Lambda collapse → implicit ridge via KL term |
| VI implementation | c² (slab width) not used | `prior_var = λ̃²τ²` drops c², implements unbounded horseshoe |

---

## Preprint Sections

Draft sections are in `docs/`:

| File | Content |
|------|---------|
| [docs/methods.md](docs/methods.md) | Problem formulation, standardisation, ridge derivation, GCV formula, horseshoe rank-neutrality proof, B-JEPA Stage 1/2 spec, VI failure analysis, GENIE3 description, evaluation protocol |
| [docs/results.md](docs/results.md) | Network statistics, primary AUPR/AUROC tables, GCV calibration table, four ablation tables, recommended figures |
| [docs/discussions.md](docs/discussions.md) | n/TF crossover mechanism, GCV necessity, conditions for horseshoe to help, JEPA failure modes, mean-field VI structural limitations, practical implications, limitations, future directions |

---

## Future Directions

Concrete next steps emerging from the negative results:

1. **Inverted JEPA masking** — Mask target gene columns and predict from TF context expression, aligning the pretext task with the causal TF→gene direction. This would make W_init a meaningful prior for downstream regression.

2. **Per-gene GCV** — Fit separate α per target gene, accounting for heterogeneous TF-explainability across genes. Remains fully analytical and parallelisable.

3. **HMC for the regularised horseshoe** — Use GCV-ridge as warm start for No-U-Turn Sampler (NumPyro/Stan). The analytical solution lies near the posterior mode, reducing required burn-in. Would provide calibrated posterior uncertainty and potentially the bimodal λ profile that mean-field VI cannot represent.

4. **Signed edge evaluation** — Regression coefficient sign (positive: activation, negative: repression) is biologically interpretable and evaluable against the DREAM5 signed gold standard for net1.

5. **Single-cell extension** — Apply GCV-ridge with weighted regression to large scRNA-seq compendium datasets. GCV extends naturally to weighted least squares.

---

## References

- Marbach et al. (2012). Wisdom of crowds for robust gene network inference. *Nature Methods*, 9, 796–804.
- Huynh-Thu et al. (2010). Inferring regulatory networks from expression data using tree-based methods. *PLOS ONE*, 5, e12776. (GENIE3)
- Carvalho, Polson & Scott (2010). The horseshoe estimator for sparse signals. *Biometrika*, 97(2), 465–480.
- Piironen & Vehtari (2017). Sparsity information and regularization in the horseshoe and other shrinkage priors. *Electronic Journal of Statistics*, 11(2), 5018–5051.
- Assran et al. (2023). Self-supervised learning from images with a joint-embedding predictive architecture. *CVPR 2023*. (I-JEPA)
- Golub, Heath & Wahba (1979). Generalized cross-validation as a method for choosing a good ridge parameter. *Technometrics*, 21(2), 215–223.
