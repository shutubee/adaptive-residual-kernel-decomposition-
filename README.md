# adaptive-residual-kernel-decomposition-
Adaptive Residual Kernel Decomposition (ARKD) is an experimental machine-learning framework that combines multiple positive-semidefinite kernels to model global patterns, local relationships, and nonlinear feature interactions.

ARKD-Retail extends this framework to retail demand forecasting, incorporating temporal similarity, inventory censoring, probabilistic demand estimation, and replenishment optimization.

The research investigates whether adaptive kernel combinations can improve demand estimation and inventory decisions under changing retail conditions.

The framework is under active experimental development. It has not yet demonstrated consistent superiority over established forecasting methods.

1. Mathematical Framework

The original ARKD model combines three kernels:

[ K_{\mathrm{ARKD}}(x,x’)

\theta_GK_G(x,x’)
+
\theta_LK_L(x,x’)
+
\theta_RK_R(x,x’)
]

where:

• (K_G): Global kernel for broad, smooth relationships.
• (K_L): Local kernel for neighborhood-dependent patterns.
• (K_R): Residual kernel for nonlinear feature interactions.

The kernel weights satisfy:

[
\theta_j \geq 0,
\qquad
\sum_j\theta_j=1.
]

Provided the component kernels are positive semidefinite, their nonnegative weighted sum is also positive semidefinite.

ARKD-Retail Extension

ARKD-Retail introduces a temporal kernel:

[ K_{\mathrm{Retail}}

\theta_GK_G+
\theta_LK_L+
\theta_RK_R+
\theta_TK_T
]

with:

[ K_T(x_t,x_{t’})

\exp(-\rho|t-t’|)
K_G(x_t,x_{t’})
]

where (\rho\geq0) controls temporal decay.

The model is intended to capture:

• Seasonal and recurring demand patterns.
• Local store and product behavior.
• Promotion and price interactions.
• Time-dependent changes in customer demand.

2. Stockout-Aware Demand Modeling

Observed sales do not always equal customer demand.

For a simplified inventory-constrained sales process:

[
Y_t=\min(D_t,A_t)
]

where:

• (Y_t): Observed sales.
• (D_t): Latent customer demand.
• (A_t): Available inventory.

When inventory is exhausted, the observation provides a lower bound on demand rather than an exact measurement.

ARKD-Retail explores censored-demand estimation to account for this information.

Probabilistic Extension

Version 6 introduces a censored Gaussian demand model.

For a normally distributed latent demand variable:

[
D_t\mid x_t\sim
\mathcal N(\mu_t,\sigma_t^2)
]

uncensored observations contribute a probability density term, while censored observations contribute a survival-probability term.

This approach is intended to improve demand estimation when sales are limited by inventory.

3. Research Development

|Version       |Development                                         |
|--------------|----------------------------------------------------|
|ARKD v1       |Global, local, and residual kernel decomposition    |
|ARKD-Retail v2|Retail demand features and stockout-aware objectives|
|ARKD-Retail v3|Time-decayed temporal kernel                        |
|ARKD-Retail v4|Kernel-weight optimization using rolling validation |
|ARKD-Retail v5|Forecasting benchmarks and inventory cost simulation|
|ARKD-Retail v6|Probabilistic censored-demand estimation            |

4. Experimental Results

All reported results are from synthetic experiments and should not be interpreted as evidence of real-world forecasting performance.

Version 4 — Kernel Weight Optimization

The reported average test demand MAE across three synthetic datasets was:

MAE: 5.584

The optimized weights varied substantially between datasets.

|Kernel  |Seed 11|Seed 23|Seed 42|
|--------|------:|------:|------:|
|Global  |5.5%   |48.1%  |60.2%  |
|Local   |53.0%  |14.0%  |3.1%   |
|Residual|30.7%  |18.0%  |32.7%  |
|Temporal|10.7%  |19.9%  |4.1%   |

The results suggest that different synthetic datasets can favor different kernel mixtures. They do not yet establish the interpretability or statistical stability of those mixtures.

Version 5 — Forecasting Benchmarks

|Model            |Demand MAE|Inventory Cost / Day|
|-----------------|---------:|-------------------:|
|Random Forest    |**5.962** |**10.384**          |
|Gradient Boosting|6.111     |10.483              |
|ARKD-GLRT        |8.933     |19.323              |
|Seasonal Naïve   |9.834     |21.882              |
|RBF-KRR          |18.627    |55.882              |

Random Forest and gradient boosting outperformed ARKD in this experiment.

These results are specific to the synthetic data generation, preprocessing, hyperparameter choices, and evaluation protocol.

Version 6 — Probabilistic Demand Forecasting

|Metric              |ARKD v6 |Random Forest|
|--------------------|-------:|------------:|
|Demand MAE          |4.964   |**3.968**    |
|Inventory Cost / Day|8.031   |**6.224**    |
|Stockout Rate       |**9.1%**|15.8%        |

ARKD achieved a lower simulated stockout rate, but at a higher overall inventory cost.

This indicates that reducing stockouts alone is insufficient to demonstrate better inventory performance.

Results from different versions should not be directly compared because experimental settings and methodologies changed.