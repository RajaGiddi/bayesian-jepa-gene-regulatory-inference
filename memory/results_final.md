---
name: Final experimental results
description: GCV-calibrated ridge final AUPR/AUROC numbers across all 4 DREAM5 networks, ablation table, key findings for preprint
type: project
---

# Final Results (as of 2026-04-06)

## Primary result: GCV-calibrated ridge regression

Method: per-gene z-score standardization → ridge regression (GCV-optimal α) → rank by |coefficient|.
No training loop. Runtime: <2s total across all 4 networks.

### AUPR (primary metric)

| Network | n/p | GENIE3 | GCV-Ridge | Δ vs GENIE3 |
|---------|-----|--------|-----------|-------------|
| net1 (in_silico) | 4.1 | 0.259 | 0.237 | −8% |
| net2 (s_aureus) | 1.6 | 0.011 | **0.018** | **+59%** |
| net3 (e_coli) | 2.4 | 0.099 | **0.142** | **+43%** |
| net4 (s_cerevisiae) | 1.6 | 0.022 | **0.024** | **+9%** |

GCV-ridge beats GENIE3 on 3/4 networks (all biological, not in silico).

### GCV-selected ridge factors

| Network | GCV ridge_factor | alpha* |
|---------|-----------------|--------|
| net1 | 0.102 | 82.2 |
| net2 | 0.044 | 7.0 |
| net3 | 0.067 | 53.7 |
| net4 | 0.053 | 28.5 |

All networks benefit from meaningful regularization. Hand-tuned ridge=1e-6 for net1/net3 was catastrophically under-regularized.

## Ablation table (AUPR)

| Method | net1 | net2 | net3 | net4 | Beats GENIE3 |
|--------|------|------|------|------|-------------|
| GENIE3 | 0.259 | 0.011 | 0.099 | 0.022 | — |
| GCV-Ridge | 0.237 | 0.018 | 0.142 | 0.024 | 3/4 |
| VI+KL (B-JEPA) | 0.204 | 0.016 | 0.080 | — | 1/3 |
| Ridge (hand-tuned) | 0.186 | 0.018 | 0.087 | 0.024 | 2/4 |
| MAP+W_init | 0.080 | 0.002 | 0.037 | 0.019 | 0/4 |

## Key findings

1. **GCV calibration is essential**: net3 AUPR went 0.087→0.142 just from proper ridge tuning (no gold standard used).
2. **Horseshoe shrinkage = ridge**: analytical horseshoe shrinkage is a monotone function of |β_ols| → rank-invariant. Ridge and OLS-HS produce identical AUROC/AUPR on all networks.
3. **JEPA warm start hurts**: MAP-HS with W_init prior mean degrades all networks. W_init AUROC=0.495 (random signal). Stage 1 JEPA encodes co-expression, not causal regulation.
4. **VI horseshoe ≈ ridge with early stopping**: Lambda collapse (κ≈0.49 for all edges) means VI horseshoe is implicit ridge. Worse than GCV-ridge on all measured networks.
5. **n/p crossover**: Regularized linear regression beats tree ensembles (GENIE3) when n/p ≲ 2–4 (biological data scarcity regime). GENIE3 wins only on in silico net1 (n/p=4.1).

## Code locations

- Analytical horseshoe + GCV: `src/bjepa/models/analytical_horseshoe.py`
- Experiment runner: `experiments/exp3_analytical_hs/run.py`
- Results: `results/analytical_hs/`
- Run GCV: `python experiments/exp3_analytical_hs/run.py --calibrate`
