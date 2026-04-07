# Results

## 3.1 DREAM5 Benchmark Networks

We evaluate on all four DREAM5 GRN inference networks, spanning one in silico and three organism-specific datasets. Network characteristics are summarised in Table 1. The n/p ratio (samples to TFs) varies from 1.6 to 4.1, spanning the under- to over-determined regimes. The fraction of positive edges in the gold standard ranges from 0.14% to 1.25%, reflecting the extreme class imbalance typical of GRN data.

**Table 1. DREAM5 network statistics.**

| Network | Organism | n (samples) | D (TFs) | G (genes) | n/D | Positive edges | Edge density |
|---------|----------|-------------|---------|-----------|-----|----------------|--------------|
| net1 | In silico | 805 | 195 | 1,643 | 4.13 | 4,012 | 1.25% |
| net2 | S. aureus | 160 | 99 | 2,810 | 1.62 | 428 | 0.15% |
| net3 | E. coli | 805 | 334 | 4,511 | 2.41 | 2,066 | 0.14% |
| net4 | S. cerevisiae | 536 | 333 | 5,950 | 1.61 | 3,940 | 0.20% |

## 3.2 GCV-Calibrated Ridge Regression Beats GENIE3 on 3/4 Networks

Table 2 presents the primary comparison. GCV-calibrated ridge regression achieves AUPR of 0.237, 0.018, 0.142, and 0.024 on net1–net4. GENIE3 achieves AUPR of 0.259, 0.011, 0.099, and 0.022. On the three biological networks (net2, net3, net4), GCV-ridge outperforms GENIE3 by +59%, +43%, and +9% in AUPR respectively. On the in silico network (net1), GCV-ridge achieves 91.5% of GENIE3 AUPR (0.237 vs 0.259).

**Table 2. Primary results: AUPR and AUROC across DREAM5 networks.**

| Method | net1 AUPR | net2 AUPR | net3 AUPR | net4 AUPR | Beats GENIE3 |
|--------|-----------|-----------|-----------|-----------|-------------|
| GENIE3 | 0.259 | 0.011 | 0.099 | 0.022 | — |
| **GCV-Ridge** | **0.237** | **0.018** | **0.142** | **0.024** | **3/4** |
| VI+KL (B-JEPA) | 0.204 | 0.016 | 0.080 | — | 1/3† |
| Ridge (hand-tuned) | 0.186 | 0.018 | 0.087 | 0.024 | 2/4 |
| MAP-HS (+W_init) | 0.080 | 0.002 | 0.037 | 0.019 | 0/4 |

†Net4 B-JEPA training did not produce a saved evaluation result.

| Method | net1 AUROC | net2 AUROC | net3 AUROC | net4 AUROC |
|--------|------------|------------|------------|------------|
| GENIE3 | 0.830 | 0.678 | 0.694 | 0.541 |
| GCV-Ridge | 0.751 | 0.650 | 0.657 | 0.541 |
| VI+KL (B-JEPA) | 0.744 | 0.599 | 0.600 | — |

The GCV method runs in under 2 seconds total across all four networks (net1: 0.2s, net2: 0.1s, net3: 0.8s, net4: 1.0s) on a standard laptop CPU, without any GPU or iterative optimisation.

## 3.3 GCV Calibration: The Ridge Parameter Is Not Trivially Set

Table 3 shows the GCV-selected ridge factors and corresponding $\alpha^* = \text{ridge\_factor} \times n$ values for each network.

**Table 3. GCV-optimal ridge factors.**

| Network | n/D | GCV ridge\_factor | $\alpha^*$ | GCV score |
|---------|-----|-------------------|-----------|-----------|
| net1 | 4.13 | 0.102 | 82.2 | 634 |
| net2 | 1.62 | 0.044 | 7.0 | 397 |
| net3 | 2.41 | 0.067 | 53.7 | 845 |
| net4 | 1.61 | 0.053 | 28.5 | 636 |

All four networks require ridge\_factor in the range 0.04–0.10 — three to five orders of magnitude larger than the numerical floor (ridge\_factor = $10^{-6}$). The cost of under-regularisation is severe: using ridge\_factor = $10^{-6}$ on net3 yields AUPR = 0.087; GCV calibration to ridge\_factor = 0.067 gives AUPR = 0.142, a 64% improvement without any model change. The GCV objective is purely unsupervised — gold standard labels are not used in calibration.

## 3.4 Ablation 1: Horseshoe Shrinkage Is Rank-Neutral

Applying the Piironen-Vehtari closed-form horseshoe shrinkage on top of the ridge coefficients (OLS-HS, Section 2.5) produces AUROC and AUPR identical to pure ridge on all four networks under both hand-tuned and GCV-calibrated settings.

**Table 4. Ridge vs OLS-HS (identical on all networks).**

| | net1 AUPR | net2 AUPR | net3 AUPR | net4 AUPR |
|--|-----------|-----------|-----------|-----------|
| Ridge (GCV) | 0.237 | 0.018 | 0.142 | 0.024 |
| OLS-HS (GCV) | 0.237 | 0.018 | 0.142 | 0.024 |

This result is theoretically expected (Section 2.5.1): the horseshoe shrinkage $|\hat{\beta}^{\text{hs}}| = C|\hat{\beta}^{\text{ols}}|^3 / (1 + C\hat{\beta}^{{\text{ols}},2})$ is a strictly monotone function of $|\hat{\beta}^{\text{ols}}|$ for fixed $C = n/(\sigma_g^2 \tau_{0g}^2)$. Rank-based metrics (AUROC, AUPR) are invariant to monotone score transformations. With per-gene unit-variance standardisation, the cross-gene variation in $C_g$ is insufficient to materially reorder the global edge ranking.

The horseshoe prior provides a principled Bayesian justification for the regularisation scale through $\tau_0$ (which GCV replaces with a data-driven estimate), but contributes no additional ranking power beyond ridge regression in this experimental setting.

## 3.5 Ablation 2: JEPA Warm Start Degrades Inference

We evaluate the MAP horseshoe (MAP-HS), which replaces OLS with a Gaussian MAP estimate centred at the Stage 1 JEPA weight matrix $\mathbf{W}_{\text{init}}$:

$$\hat{\mathbf{W}}_{\text{map}} = (\mathbf{X}_{\text{tf}}^\top \mathbf{X}_{\text{tf}} + \alpha \mathbf{I})^{-1}(\mathbf{X}_{\text{tf}}^\top \mathbf{Y} + \alpha \mathbf{W}_{\text{init}})$$

The MAP prior effectively regularises $\mathbf{W}$ toward the JEPA co-expression weights. Table 5 shows that MAP-HS consistently underperforms plain ridge on all four networks, with the degradation most severe where it matters most (net2: AUPR 0.002 vs 0.018; net3: AUPR 0.037 vs 0.142).

**Table 5. MAP-HS vs GCV-Ridge (AUPR).**

| Network | GCV-Ridge | MAP-HS | $\Delta$ |
|---------|-----------|--------|---------|
| net1 | 0.237 | 0.080 | −66% |
| net2 | 0.018 | 0.002 | −89% |
| net3 | 0.142 | 0.037 | −74% |
| net4 | 0.024 | 0.019 | −21% |

The cause is that $\mathbf{W}_{\text{init}}$ carries co-expression structure, not causal regulatory signal. When evaluated directly as edge scores, $\mathbf{W}_{\text{init}}$ achieves AUROC = 0.495 — indistinguishable from random. Using such a prior mean biases all coefficients toward random directions, overwhelming the data-driven regression signal especially in networks where $n/D$ is small and each additional constraint from the prior is influential.

## 3.6 Ablation 3: VI Horseshoe Reduces to Implicit Ridge

The B-JEPA Stage 2 VI horseshoe achieves AUPR of 0.204 (net1), 0.016 (net2), and 0.080 (net3). These results are below GCV-ridge on all measured networks by 14%, 11%, and 44% respectively.

Diagnostic inspection reveals a degenerate VI fixed point. After 200 training epochs, the local scale parameters $\tilde{\lambda}_{dg} = \exp(m_{\lambda,dg})$ satisfy $\kappa_{dg} \approx 0.49$ with std $< 0.001$ across all $D \times G$ edges on net1 — the horseshoe's signature U-shaped shrinkage profile does not emerge. All edges receive near-identical shrinkage, consistent with ridge regression rather than sparse selection.

The mechanism is a gradient magnitude imbalance: the KL gradient on $m_\lambda$ (from the Half-Cauchy prior) is approximately 500× larger than the regression gradient flowing through $\mathbf{W}_{\text{eff}} = \mu_\mathbf{W} \odot \tilde{\boldsymbol{\lambda}} \odot (\tau/\tau_0)$. The KL equilibrium drives $\tilde{\lambda}^* = |\mu_W|/\tau \approx 1$ for all edges, since $\mu_\mathbf{W}$ is initialised at scale $\tau_0 \approx \tau$. Under lambda collapse, the VI horseshoe provides ridge-equivalent regularisation through its KL term, with effective ridge strength $\alpha_{\text{eff}} \approx n \cdot \beta_{\text{KL}} / (2 D \tau_0^2)$.

The B-JEPA VI result on net2 (AUPR = 0.016) still exceeds GENIE3 (AUPR = 0.011), because even implicit ridge from KL regularisation stabilises the ill-conditioned $n/D = 1.6$ regime. However, the analytical GCV-ridge (AUPR = 0.018) — which makes the same ridge mechanism explicit and data-adaptive — outperforms it.

## 3.7 Summary

The complete ablation hierarchy reveals a monotone ordering: GCV calibration is the most important design decision, and all architectural elaborations beyond explicit ridge regression do not improve — and often degrade — performance.

**Figure 1 (recommended).** AUPR vs n/D ratio for GENIE3 and GCV-Ridge across all four networks. The crossover from GENIE3-dominant to Ridge-dominant performance occurs between n/D = 2.4 and n/D = 4.1, consistent with the data-scarcity regime where non-parametric ensemble methods overfit.
