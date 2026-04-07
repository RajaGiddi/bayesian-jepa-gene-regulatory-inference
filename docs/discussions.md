# Discussion

## 4.1 Why Regularised Linear Regression Outperforms Tree Ensembles in the Low-Sample Regime

The central empirical finding — that GCV-calibrated ridge regression outperforms GENIE3 on 3/4 DREAM5 networks — is consistent with a fundamental statistical tradeoff between bias and variance in function approximation.

GENIE3 fits an independent Extra-Trees regressor per target gene. Each model has $O(\sqrt{D})$ free parameters per split and up to 1000 trees, making it highly flexible but also high-variance when $n$ is small relative to $D$. For a target gene with $D = 334$ TF predictors and $n = 805$ samples (net3, $n/D = 2.4$), each tree uses $\approx 18$ features per split; the ensemble can overfit co-expression patterns specific to the training samples that do not generalise to the held-out gold standard edges. Regularised linear regression, by contrast, has $D$ parameters and imposes a strong global shrinkage via ridge, dramatically reducing variance at the cost of a linear inductive bias.

The crossover occurs between $n/D = 2.4$ (net3, GCV-ridge wins) and $n/D = 4.1$ (net1, GENIE3 wins). This is consistent with classical results showing that linear models outperform flexible non-parametric methods when the number of observations per free parameter falls below a critical threshold, typically $n/D \lesssim 3$–5 for random forests and tree ensembles (Hastie, Tibshirani & Friedman 2009). Additionally, net1 is an in silico network with Boolean logic gates — a setting where the ground truth is genuinely non-linear, favouring GENIE3 by construction. The biological networks (net2, net3, net4) have smoother regulatory functions arising from real transcriptional kinetics, making the linear model more appropriate.

The implication for GRN inference practice: when designing experiments, the number of distinct conditions $n$ relative to the number of TFs $D$ is a better predictor of method choice than total sample size alone. Bulk RNA-seq datasets with $n/D \lesssim 3$ — which describe the majority of published perturbation or time-series compendia — are likely better served by regularised linear regression than by ensemble tree methods, even without hyperparameter tuning if GCV calibration is applied.

## 4.2 GCV Calibration Is Non-Trivially Necessary

The magnitude of the improvement from GCV calibration — net3 AUPR from 0.087 to 0.142 (+63%) with no change to the model class — underscores that ridge regression is not merely "linear regression with a small stabilisation term." The optimal ridge factors (0.044–0.102 across networks, Table 3) represent substantial regularisation: at $\alpha^* = 82.2$ for net1, the effective degrees of freedom $\text{tr}(\mathbf{H}_\alpha) = \sum_j d_j^2/(d_j^2 + \alpha^*)$ is substantially less than $D = 195$, meaning the model is operating well below its nominal dimensionality.

GCV is particularly well-suited to this problem for three reasons. First, it requires no held-out gold standard labels, making it applicable even in zero-label settings (new organisms, novel conditions). Second, the computational cost of GCV calibration is dominated by the SVD of $\mathbf{X}_{\text{tf}}$ ($O(nD^2)$), which is identical to the cost of solving the regression itself — there is no separate cross-validation loop. Third, GCV is consistent for ridge in the fixed-design linear model (Li 1986), meaning it asymptotically selects the same $\alpha$ as the optimal LOO-CV predictor.

A practical alternative is generalised information criteria (AIC/BIC with effective degrees of freedom), which would give similar results. The key message is that any principled unsupervised selection of $\alpha$ — rather than a fixed small value — is necessary for good GRN inference with ridge regression.

## 4.3 Horseshoe Shrinkage Does Not Improve Ranking Under Standardisation

The analytical horseshoe shrinkage is rank-neutral relative to ridge (Section 3.4). This is not a failure of the horseshoe prior — it is a consequence of the combination of per-gene unit-variance standardisation and a fixed $p_0$ parameter. Under standardisation, per-gene residual variances $\sigma_g^2 \approx (1 - R_g^2)$ where $R_g^2$ is the coefficient of determination for gene $g$. The shrinkage constant $C_g = n/(\sigma_g^2 \tau_{0g}^2)$ varies across genes, and the horseshoe could in principle reorder cross-gene edge comparisons. Empirically, this reordering is negligible.

For the horseshoe to provide ranking benefits beyond ridge, one of two conditions would need to hold:

1. **Strong cross-gene heterogeneity in signal-to-noise.** If some genes are almost entirely explained by TFs ($R_g^2 \approx 1$, $\sigma_g^2 \approx 0$) while others are almost entirely noise ($R_g^2 \approx 0$, $\sigma_g^2 \approx 1$), then the per-gene shrinkage constants $C_g$ would differ by orders of magnitude, causing the horseshoe to amplify well-explained gene scores relative to noise-dominated ones. In the DREAM5 data after standardisation, the range of $\sigma_g^2$ is not extreme enough to induce this.

2. **Gene-specific $p_0$ (sparsity prior).** If different target genes are expected to have different numbers of regulators — for instance, master regulators receive many inputs while housekeeping genes receive few — then per-gene $\tau_{0g}$ could differ substantially. Using a global $p_0 = 10$ assumes identical expected sparsity across all $G$ target genes, which is biologically unrealistic but computationally convenient.

These two extensions — heteroskedastic residuals and gene-specific sparsity priors — represent concrete directions for improving ranking quality beyond what ridge achieves. They would require either cross-validation of $p_0$ per gene, or incorporating prior knowledge from databases of known regulatory complexity.

## 4.4 Why the JEPA Warm Start Fails

The MAP horseshoe with $\mathbf{W}_{\text{init}}$ prior mean degraded performance on all four networks by 21%–89% in AUPR (Table 5). Understanding this failure is important for the design of future pre-training strategies for GRN inference.

Stage 1 JEPA encodes co-expression structure: the weight matrix $\mathbf{W}_{\text{init}}$ predicts target gene latents from TF latents, minimising a contrastive MSE loss in $d$-dimensional latent space. Co-expression (correlation across conditions) is not the same as causal regulation (TF binding to a gene's promoter and driving its transcription). A TF and a target gene may be co-expressed because they are both regulated by a third party, or because they respond similarly to environmental signals, without any direct regulatory interaction. The JEPA pretext task optimises co-expression prediction, not causal identification. The AUROC of $\mathbf{W}_{\text{init}}$ as a regulatory edge predictor (0.495) quantifies this: Stage 1 learns no useful regulatory signal.

When $\mathbf{W}_{\text{init}}$ is used as a MAP prior mean, the regression is biased toward these co-expression weights. In the low-$n/D$ regime where each additional constraint is influential, this bias is severe: the data cannot fully correct the prior misinformation within the $n$ available samples.

A correct pre-training strategy for GRN inference should align the pretext task with causal directionality. One concrete design: the *inverted JEPA masking* scheme, where TF columns are provided as context and target gene columns are masked. The predictor then learns weights $\mathbf{W}$ that predict target gene expression from TF expression, directly in expression space, as a pretext task. This aligns with the causal direction TF→gene and would make $\mathbf{W}_{\text{init}}$ a meaningful prior mean for downstream regression. We leave implementation and evaluation of this design to future work.

## 4.5 Mean-Field VI Is Poorly Suited to the Horseshoe Prior

The failure of the B-JEPA VI horseshoe to produce sparse edge differentiation (Section 3.6) reflects a known difficulty with mean-field VI applied to heavy-tailed priors. The horseshoe's Half-Cauchy local scales require the posterior to be multimodal: most $\lambda_{dg}$ should collapse near zero (noise edges), while a few should be large (true regulatory edges). Mean-field VI, which optimises a unimodal LogNormal approximation per $\lambda_{dg}$ independently, cannot represent this bimodal structure. The factorised approximation $q(\boldsymbol{\lambda}) = \prod_{d,g} q(\lambda_{dg})$ prevents the model from representing the dependence structure in which a few edges are selected jointly while the rest are suppressed.

Additionally, the gradient imbalance (KL gradient 500× larger than regression gradient for $m_\lambda$) means the variational optimisation is dominated by the prior during training, preventing the likelihood from driving sparsification. This imbalance is not a bug that can be fixed by increasing `n_mc`; it is a structural consequence of the relative magnitudes of the horseshoe KL and the regression MSE under the chosen $\tau_0$ and `kl_weight`.

Full Bayesian inference via No-U-Turn Sampler (NUTS/HMC), as in Stan or NumPyro, avoids both issues: HMC samples the full joint posterior without factorisation assumptions and explores the multimodal $\lambda$ landscape via Hamiltonian dynamics. Piironen & Vehtari (2017) demonstrate that HMC for the regularised horseshoe achieves the intended U-shaped shrinkage profile. For GRN inference specifically, where $D \times G$ parameters can reach $\sim 10^6$, HMC is computationally challenging but tractable with the analytical GCV solution as an initialisation (warm-starting HMC from a good point significantly reduces required MCMC chains).

## 4.6 Implications for GRN Inference Methodology

This work has three practical implications for practitioners:

**1. Benchmark against regularised linear regression before adopting complex models.** The strong performance of GCV-calibrated ridge — achieving or exceeding the DREAM5 challenge-winning GENIE3 on 3/4 networks with negligible computation — suggests that the GRN inference community routinely compares complex deep learning and graph-based methods against an under-regularised linear baseline. A properly calibrated ridge regression should be the minimum baseline.

**2. Report n/D ratio alongside performance numbers.** The crossover between linear and non-linear methods at $n/D \approx 3$ provides a practical decision rule: apply regularised linear regression for $n/D \lesssim 3$ (most biological experiments), and non-parametric methods only when data are abundant relative to TF count. This rule is directly applicable without tuning.

**3. Pre-training should encode causal, not correlational, signal.** Self-supervised pre-training frameworks (JEPA, masked autoencoders, contrastive learning) applied to expression data learn co-expression patterns — useful for cell type classification and trajectory analysis, but misleading for GRN inference where the target is regulatory causality. Future pre-training designs for GRN should explicitly mask TF expression and predict target genes, embedding causal directionality into the pretext task.

## 4.7 Limitations

**AUPR vs AUROC.** GCV-ridge achieves lower AUROC than GENIE3 on net1, net2, and net3, while winning on AUPR. AUROC at extreme class imbalance (0.14%–1.25% positives) is dominated by the large number of true negatives and is less informative for the biologically relevant question of "what are the top-ranked edges?" AUPR is the appropriate primary metric.

**Single gene regression model.** The model regresses each target gene independently on TF expression, ignoring co-regulatory relationships between target genes. Regulatory programs often involve coordinated expression changes across gene modules. Incorporating gene–gene covariance (e.g., via multivariate regression or graphical model structure) could improve both precision and interpretability.

**No uncertainty quantification reported.** The analytical horseshoe provides per-coefficient posterior variance estimates, but these are rank-neutral (the same monotone argument applies to posterior standard deviations). Reporting confidence intervals on edge scores could guide experimental prioritisation; this was not evaluated on DREAM5, which lacks an uncertainty-aware evaluation protocol.

**Net4 B-JEPA result missing.** Stage 2 VI training on net4 (5,950 genes, 333 TFs, 536 samples) did not complete within the evaluation window. The GCV-ridge result on net4 (AUPR = 0.024, beating GENIE3's 0.022) is available; the VI comparison for net4 is left for future work.

**DREAM5 gold standard completeness.** The DREAM5 gold standards for net2 and net4 list only confirmed positive edges; negatives are constructed as all remaining TF×gene pairs. This construction is inexact — some "negative" pairs may be true regulatory edges not yet experimentally confirmed — which systematically underestimates AUPR for all methods. The relative rankings between methods are nonetheless informative.

## 4.8 Future Directions

- **Per-gene GCV.** Fit separate ridge parameters per target gene, accounting for heterogeneous signal levels. This requires $G$ independent GCV runs but remains embarrassingly parallel and fast.
- **Inverted JEPA masking.** Redesign Stage 1 to mask target gene columns and predict from TF context, aligning the pretext task with causal TF→gene direction.
- **HMC for the regularised horseshoe.** Use the GCV-ridge estimate as a warm start for 4–8 NUTS chains in NumPyro. The analytical solution lies near the posterior mode, substantially reducing the required burn-in.
- **Signed edge scores.** The regression coefficient sign ($\hat{\beta}_{dg} > 0$: activation, $< 0$: repression) is biologically interpretable. DREAM5 provides a signed gold standard for net1; evaluating sign accuracy separately from magnitude ranking is a natural extension.
- **Transfer to single-cell data.** Apply GCV-ridge to large-scale scRNA-seq compendium datasets where $n$ is large but batch effects and dropout require modified standardisation. The GCV framework extends naturally to weighted ridge regression.
