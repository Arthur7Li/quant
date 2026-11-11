# CS → Quant Glossary (`docs/glossary.md`)

A living reference mapping Computer Science concepts to their Quantitative Finance counterparts across Phases 1–3.

| Computer Science Concept | Quantitative Finance Counterpart | Where It Appears in `quant-lab` |
|---|---|---|
| **Dynamic Programming / Backward Induction** | Binomial Option Pricing (Cox-Ross-Rubinstein), American Option Early Exercise (Snell Envelope), Optimal Stopping | Phase 1 Notebook 05, `src/quantlab/pricing/binomial.py` |
| **Randomized Algorithms & Variance Reduction** | Monte Carlo Path Simulation (GBM, Stochastic Vol), Antithetic & Control Variates, Importance Sampling for Tail VaR | Phase 1 Notebook 03 & 06, `src/quantlab/pricing/monte_carlo.py` |
| **Numerical Linear Algebra (SVD, Eigendecomposition, Cholesky)** | Principal Component Analysis (PCA) Statistical Risk Factors, Random Matrix Theory (Marchenko–Pastur) Covariance Denoising, Correlated Path Generation | Phase 1 Notebook 08, Phase 2 Layer 1 (`portfolio/`) |
| **Regularization ($L_2$ Tikhonov / Shrinkage)** | Ledoit-Wolf & Oracle Approximating Shrinkage (OAS) Covariance Estimation, Ridge Return Forecasting | Phase 2 Layer 1 & Layer 4 |
| **Convex Optimization (QP, SOCP, Conic)** | Markowitz Mean-Variance, CVaR Optimization, Transaction-Cost-Aware Multi-Period Portfolio Optimization (`cvxpy`) | Phase 1 Notebook 07, Phase 2 Layer 3 (`portfolio/`) |
| **Hierarchical Clustering / Minimum Spanning Trees** | Hierarchical Risk Parity (HRP), Nested Clustered Optimization (NCO) | Phase 2 Layer 3 (`portfolio/`) |
| **Bayesian Inference (Conjugate Normal Updating)** | Black-Litterman Portfolio Model (combining market equilibrium prior with uncertain investor/model views) | Phase 2 Layer 3 (`portfolio/`) |
| **State Estimation / Linear Dynamical Systems** | Kalman Filter for Dynamic Hedge Ratios & Time-Varying Beta; Hidden Markov Models (HMM) for Market Regimes | Phase 2 Layer 4 (`models/`) |
| **Event-Driven Discrete-Event Simulation** | Event-Driven Backtesting Engine (`MarketEvent` → `SignalEvent` → `OrderEvent` → `FillEvent`) | Phase 1 Notebook 10, `src/quantlab/backtest/engine.py` |
| **Priority Queues / Heaps & Doubly Linked Lists** | Price-Time Priority (FIFO) Limit Order Book (LOB) Matching Engine | Phase 1 Notebook 11, `src/quantlab/backtest/orderbook.py`, Phase 3 Track G (Rust) |
| **Data Leakage & Temporal Isolation** | Point-in-Time (PIT) Dual Timestamps (`period_ts`, `available_ts`), Purged & Embargoed Cross-Validation, CPCV | Phase 1–2 `src/quantlab/validation/` |
| **Multiple Hypothesis Testing / Family-Wise Error Rate** | Backtest Overfitting, Deflated Sharpe Ratio (DSR with effective independent trials $\hat{N}_{\text{eff}}$), Probability of Backtest Overfitting (PBO) | Phase 1 Capstone, Phase 2 Layer 1–2 (`validation/`) |
| **Idempotency, Fault Tolerance & Circuit Breakers** | Scheduled Daily Paper-Trading Pipeline, Position Reconciliation, Automated Risk Kill Switches | Phase 2 `flagship/`, `src/quantlab/risk/` |
