# Methods

## 2.1 Problem Formulation

Gene regulatory network (GRN) inference is cast as a sparse regression problem. Given a gene expression matrix $\mathbf{X} \in \mathbb{R}^{n \times G}$ (n experimental conditions, G genes) and a set of D transcription factors (TFs) identified as a subset of the G genes, the task is to assign a confidence score $s_{dg} \geq 0$ to every directed edge TF$_d$ → gene$_g$, $d \neq g$. Edges are then ranked by score and evaluated against a held-out gold standard.

We assume that the expression of each target gene $g$ is a linear function of TF expression:

$$y_g = \mathbf{X}_{\text{tf}} \boldsymbol{\beta}_g + \boldsymbol{\varepsilon}_g, \quad \boldsymbol{\varepsilon}_g \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{I}_n)$$

where $\mathbf{X}_{\text{tf}} \in \mathbb{R}^{n \times D}$ is the submatrix of TF expression columns, $\boldsymbol{\beta}_g \in \mathbb{R}^D$ is the vector of TF regulatory coefficients for gene $g$, and $\sigma_g^2$ is the gene-specific noise variance. The magnitude $|\hat{\beta}_{dg}|$ serves as the edge score for TF$_d$ → gene$_g$.

All G regression problems share the same design matrix $\mathbf{X}_{\text{tf}}$, so they are solved jointly as a single matrix equation:

$$\mathbf{Y} = \mathbf{X}_{\text{tf}} \mathbf{W} + \mathbf{E}, \quad \mathbf{W} \in \mathbb{R}^{D \times G}$$

## 2.2 Expression Standardisation

Raw log-normalised expression values are standardised per gene across samples before regression. For gene $g$:

$$\tilde{x}_{ig} = \frac{x_{ig} - \bar{x}_g}{s_g}, \quad \bar{x}_g = \frac{1}{n}\sum_i x_{ig}, \quad s_g = \max\!\left(\sqrt{\frac{1}{n}\sum_i (x_{ig} - \bar{x}_g)^2},\ 10^{-8}\right)$$

Standardisation serves two purposes. First, it ensures that the regression coefficients $\hat{\beta}_{dg}$ are on a comparable scale across all target genes (unit-variance response), so that $|\hat{\beta}_{dg}|$ is a meaningful cross-gene ranking criterion. Second, it makes the OLS residuals approximately homoskedastic, satisfying the Gauss-Markov assumption and rendering per-gene $\sigma_g^2$ estimates comparable.

## 2.3 Ridge Regression

The joint ridge estimate for all G target genes is:

$$\hat{\mathbf{W}}_\alpha = \left(\mathbf{X}_{\text{tf}}^\top \mathbf{X}_{\text{tf}} + \alpha \mathbf{I}_D\right)^{-1} \mathbf{X}_{\text{tf}}^\top \mathbf{Y}$$

solved via a single Cholesky factorisation of the $D \times D$ positive-definite system, with all G right-hand sides handled in one LAPACK call. The edge score for TF$_d$ → gene$_g$ is $s_{dg} = |\hat{W}_{dg}|$. Self-loops ($d = g$) are excluded.

The regularisation parameter $\alpha$ controls the bias-variance trade-off and is critical: when $n/D$ is close to 1, the unregularised OLS matrix $\mathbf{X}_{\text{tf}}^\top \mathbf{X}_{\text{tf}}$ is ill-conditioned and small eigenvalues inflate coefficient estimates. Ridge shrinks all coefficients toward zero uniformly, stabilising the solution at the cost of bias.

## 2.4 GCV Calibration of the Ridge Parameter

We select $\alpha$ by minimising the Generalised Cross-Validation (GCV) criterion (Golub, Heath & Wahba 1979), which approximates leave-one-out prediction error without refitting:

$$\text{GCV}(\alpha) = \frac{\|\mathbf{Y} - \mathbf{X}_{\text{tf}} \hat{\mathbf{W}}_\alpha\|_F^2}{n \left(1 - \frac{\text{tr}(\mathbf{H}_\alpha)}{n}\right)^2}$$

where $\mathbf{H}_\alpha = \mathbf{X}_{\text{tf}}(\mathbf{X}_{\text{tf}}^\top \mathbf{X}_{\text{tf}} + \alpha \mathbf{I})^{-1} \mathbf{X}_{\text{tf}}^\top$ is the hat matrix. Its trace has a closed form via the thin SVD $\mathbf{X}_{\text{tf}} = \mathbf{U} \mathbf{D} \mathbf{V}^\top$, $\mathbf{d} = \text{diag}(\mathbf{D})$:

$$\text{tr}(\mathbf{H}_\alpha) = \sum_{j=1}^D \frac{d_j^2}{d_j^2 + \alpha}$$

Given the SVD (computed once, $O(nD^2)$), the GCV score at any $\alpha$ requires only $O(nGD)$ work for the residuals, making a grid search over 60 log-spaced candidates in $[10^{-4},\ 10^4 n]$ negligible in runtime. No gold standard labels are used in calibration; GCV is purely unsupervised. The optimal $\alpha^* = \arg\min_\alpha \text{GCV}(\alpha)$ is used directly without further tuning.

## 2.5 Horseshoe Prior: Theoretical Motivation and Closed-Form Shrinkage

The horseshoe prior (Carvalho, Polson & Scott 2010) places a regularised Normal prior on each coefficient $\beta_{dg}$ with a local scale $\lambda_{dg}$ and a global scale $\tau$:

$$\beta_{dg} \mid \lambda_{dg}, \tau \sim \mathcal{N}(0,\ \lambda_{dg}^2 \tau^2), \quad \lambda_{dg} \sim \text{Half-Cauchy}(0, 1), \quad \tau \sim \text{Half-Cauchy}(0, \tau_0)$$

The shrinkage coefficient $\kappa_{dg} = 1/(1 + \lambda_{dg}^2) \in (0,1)$ follows a Beta(1/2, 1/2) marginal — the U-shaped horseshoe profile that fully shrinks noise ($\kappa \approx 1$) while leaving signals unshrunk ($\kappa \approx 0$).

Piironen & Vehtari (2017) recommended setting the global scale based on prior sparsity belief: if $p_0$ TFs are expected to regulate each gene out of $D$ total,

$$\tau_0 = \frac{p_0}{D - p_0} \cdot \frac{\sigma_g}{\sqrt{n}}$$

In the analytical (closed-form) special case — when $n > D$ so OLS is well-defined — the posterior mode under the horseshoe likelihood takes the form:

$$\hat{\beta}_{dg}^{\text{hs}} = (1 - \hat{\kappa}_{dg}) \hat{\beta}_{dg}^{\text{ols}}, \quad \hat{\kappa}_{dg} = \frac{1}{1 + \frac{n \hat{\beta}_{dg}^2}{\sigma_g^2 \tau_{0g}^2}}$$

where $\hat{\beta}_{dg}^{\text{ols}}$ is the OLS estimate and $\sigma_g^2$ is estimated from OLS residuals with $\max(n - D, 1)$ degrees of freedom.

### 2.5.1 Rank-Neutrality of the Analytical Horseshoe

The analytical horseshoe shrinkage is a strictly monotone function of $|\hat{\beta}_{dg}^{\text{ols}}|$ within each target gene $g$ (fixed $\sigma_g^2$, $\tau_{0g}^2$):

$$|\hat{\beta}_{dg}^{\text{hs}}| = \frac{C_g |\hat{\beta}_{dg}^{\text{ols}}|^3}{1 + C_g \hat{\beta}_{dg}^{{\text{ols}},2}}, \quad C_g = \frac{n}{\sigma_g^2 \tau_{0g}^2} > 0$$

Since $\frac{d|\hat{\beta}^{\text{hs}}|}{d|\hat{\beta}^{\text{ols}}|} = \frac{C_g \hat{\beta}^2(3 + C_g \hat{\beta}^2)}{(1 + C_g \hat{\beta}^2)^2} > 0$ for all $\hat{\beta} \neq 0$, the horseshoe shrinkage preserves the rank ordering of coefficients within each gene. Across genes, the different $C_g$ values could in principle reorder cross-gene comparisons; empirically with per-gene unit-variance standardisation, $\sigma_g^2 \approx (1 - R_g^2)$ varies across genes but not enough to materially alter the global ranking (confirmed by ablation: Ridge = OLS-HS to four decimal places on all four DREAM5 networks). AUROC and AUPR, being rank-based metrics, are therefore identical for ridge and horseshoe-shrinkage scores in this setting.

This implies that GCV-calibrated ridge regression is the effective method, with the horseshoe prior providing the theoretical motivation for the regularisation scale via $\tau_0$.

## 2.6 B-JEPA: Joint Embedding Predictive Architecture (Comparison Method)

We also evaluate a two-stage deep learning approach, B-JEPA, as a comparison.

### Stage 1: JEPA Encoder Pre-training

Context and target encoders ($f_\theta$, $f_\xi$) map the $n$-dimensional expression vector of each gene to a $d$-dimensional latent representation. The context encoder is trained by an EMA (exponential moving average) target encoder — the I-JEPA framework (Assran et al. 2023) adapted for gene expression. A linear predictor $\mathbf{W}_{\text{init}} \in \mathbb{R}^{D \times G}$ reconstructs target gene latents from TF context latents:

$$\hat{\mathbf{s}}_g = \mathbf{W}_{\text{init}}^\top \mathbf{s}_{\text{context}}, \quad \mathcal{L}_1 = \text{MSE}(\hat{\mathbf{s}}_g,\ \text{stopgrad}(\mathbf{s}_g^{\text{target}}))$$

Stage 1 is trained for 200 epochs with AdamW; the EMA momentum is 0.996.

### Stage 2: Variational Horseshoe Regression

With encoders frozen, $\mathbf{W}_{\text{init}}$ warm-starts the variational posterior mean $\mu_\mathbf{W}$ for a mean-field VI horseshoe over the expression-space regression problem:

$$\mathcal{L}_2 = \text{MSE}(\mathbf{X}_{\text{tf}} \mathbf{W}_{\text{eff}},\ \mathbf{Y}) + \beta \cdot \text{KL}(q(\mathbf{W}, \boldsymbol{\lambda}, \tau) \| p_{\text{horseshoe}})$$

where $\mathbf{W}_{\text{eff}} = \mu_\mathbf{W} \odot \tilde{\boldsymbol{\lambda}} \odot (\tau / \tau_0)$ and the variational families are:

$$q(\beta_{dg}) = \mathcal{N}(\mu_{dg}, \sigma_{dg}^2), \quad q(\tilde{\lambda}_{dg}) = \text{LogNormal}(m_{\lambda,dg}, s_{\lambda,dg}^2), \quad q(\tau) = \text{LogNormal}(m_\tau, s_\tau^2)$$

The KL is estimated via Monte Carlo with $n_{\text{mc}} = 4$ samples. KL weight is warmed up linearly from 0 to $\beta_{\max} = 0.1$ over 50 epochs; Stage 2 runs for 200 epochs total.

**Diagnosed failure modes.** Empirical inspection revealed two issues in the VI implementation. First, mean-field VI for the horseshoe exhibits a degenerate fixed point: the KL gradient on the local scale parameters $m_\lambda$ is approximately 500× larger in magnitude than the regression gradient, causing $\tilde{\lambda} \to 1$ for all edges ($\kappa \approx 0.49$, std $< 0.001$). The horseshoe effectively reduces to ridge regression through its KL regularisation term. Second, the slab width parameter $c^2$ (Piironen & Vehtari 2017 regularised horseshoe) is maintained in the variational distribution but not propagated into the prior variance computation ($\text{prior\_var} = \tilde{\lambda}^2 \tau^2$, with $c^2$ dropped), so the implementation corresponds to the unbounded original horseshoe rather than the regularised variant.

Edge scores from B-JEPA are $|\mu_{dg} \cdot \tilde{\lambda}_{dg} \cdot \tau / \tau_0|$ evaluated at the variational posterior mode.

## 2.7 GENIE3 Baseline

GENIE3 (Huynh-Thu et al. 2010) trains an Extra-Trees regressor independently for each target gene, ranking TF importances by the decrease in impurity. We use the implementation from the original DREAM5 challenge with default hyperparameters ($K = \sqrt{D}$ features per split, 1000 trees). GENIE3 is a non-parametric ensemble method that makes no linearity assumption.

## 2.8 Evaluation Protocol

We follow the DREAM5 evaluation protocol (Marbach et al. 2012). For each network, predictions are a ranked list of TF→gene pairs. Evaluation is restricted to pairs present in the gold standard. AUROC (area under the ROC curve) and AUPR (area under the precision-recall curve) are computed against binary gold standard labels via scikit-learn. AUPR is the primary metric: at the extreme class imbalance present in GRN data (0.14%–1.25% positive edges), AUPR measures precision at the top of the ranked list, which is the operationally relevant quantity for downstream validation experiments.

Networks net2 and net4 provide only positive edges; negative pairs are constructed as all TF×gene pairs not listed, excluding self-loops, following standard practice.
