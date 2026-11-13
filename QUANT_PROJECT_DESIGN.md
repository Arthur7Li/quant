# quant-lab — Design Document

> A three-phase GitHub project: learn quantitative finance as a computer scientist, build one genuinely production-grade US equity portfolio & risk system, then keep pushing at the research frontier with agentic AI.
>
> **Status:** Reviewed & Approved Design (v0.2 — Decisions Locked, Pre-Bootstrap)
> **Purpose of this file:** the source of truth for repo initialization (`QUANT_PROJECT_DESIGN.md` / `docs/DESIGN.md`). Antigravity agents scaffold from it, one phase and one task at a time.
> **Disclaimer:** Educational and research project. Nothing here is financial advice. No live trading with real capital is part of the design unless explicitly added later by a human decision (see Guardrails).

---

## 0. Read This First (Honest Framing)

A few facts shape every design choice below. They are here so that neither you nor the agents drift into optimistic narratives.

1. **Most backtests are wrong, and most "alpha" is noise.** With enough tried variants, some strategy will look great by chance. Phase 1 teaches this on purpose; Phase 2 builds the statistical machinery to defend against it; Phase 3 enforces it automatically on agent-generated research.
2. **Retail-scale, daily-frequency, public-data alpha is rare.** A realistic outcome of a well-run strategy at this scale is a modest net-of-cost Sharpe (roughly 0.3–0.8 if anything survives at all) or nothing. Success for this project is **not** defined by finding alpha. It is defined by correctness, rigor, reproducibility, and learning.
3. **"Meaningful real-world impact" in Phase 2 is interpreted as:** engineering quality at a standard a real desk or fund would respect (validated, monitored, reproducible, cost-aware), plus an externally verifiable artifact (a forward-tested paper-trading record and/or an open-source library other people can use). It is deliberately *not* defined as "makes money."
4. **Cutting-edge ≠ better.** In published and practitioner experience, simple, well-regularized models often match or beat deep models on noisy financial data. Every sophisticated technique in this repo must be benchmarked against a simple baseline, and the baseline wins ties.
5. **Data is the bottleneck.** Free data has survivorship bias, adjustment errors, and gaps. The design treats data quality as a first-class engineering problem.

---

## 1. Project Goals and Non-Goals

### Goals
- Build deep, working understanding of quantitative finance by connecting it to CS concepts you already know (dynamic programming, randomized algorithms, linear algebra, convex optimization, data structures, distributed/event-driven systems, ML).
- Focus tightly on the **North American (US) Stock Market (NYSE/NASDAQ)** so that calendar, currency, timezone, and microstructure semantics are uniform and well-specified.
- Produce a progressively more sophisticated codebase where each phase has independent value.
- Use phased complexity as **scaffolding for AI agents**: each phase raises the bar on tests, validation, and review, so agent output quality can be measured rather than assumed.
- Produce one production-grade result (Phase 2: **Option A — Cost-aware, regime-aware US equity & sector ETF portfolio construction & risk system**) and an open-ended research platform (Phase 3).

### Non-Goals
- Not a get-rich-quick trading bot.
- Not high-frequency trading infrastructure (no colocation, no microsecond latency claims). Microstructure is studied via simulation and historical L2/ITCH sample data.
- No multi-currency, non-US exchange, or 24/7 crypto portfolio plumbing in Phase 1–2 (keeps scope focused on US equities/ETFs).
- No custody of other people's money, no investment advice, no signal selling.

---

## 2. Repo Identity

| Item | Specification |
|---|---|
| Repo URL / Name | `https://github.com/Arthur7Li/quant` (`quant-lab`) |
| Visibility | **Public from Day 1** (enforced via `.gitignore` and secret/file-size pre-commit hooks; zero vendor data redistributed) |
| License | MIT |
| Primary language | Python 3.12 (100% Python in Phase 1–2) |
| Secondary (Phase 3 Track G) | Rust (LOB matching engine / fast event simulators via `PyO3` + `maturin`) |
| Package/env manager | `uv` |
| Layout | `src/` layout, single installable package `quantlab`, plus `notebooks/`, `docs/`, `experiments/`, `flagship/` |

---

## 3. Environment and Compute Plan

### Hardware roles

| Machine | Role | Hardware & Memory Guardrails |
|---|---|---|
| **macOS Apple M2 (8 GB Unified Memory)** (primary) | All Phase 1 work, most Phase 2 development, small-model training, notebooks, agent sessions | **8 GB Unified Memory constraint:** macOS + IDE + browser consume ~4–5 GB at idle. Never load large multi-year tick/panel datasets eagerly into pandas memory. Query local Parquet files via **DuckDB / Polars** (out-of-core/lazy) and pass compact **typed `float64` NumPy arrays** across module boundaries. PyTorch via **MPS** backend requires small batch sizes, `torch.mps.empty_cache()` between sweeps, and a tested CPU fallback path. |
| **Windows PC** (Ryzen 9 5900X, 32 GB RAM, RTX 3080 Ti 12 GB VRAM) | Heavy jobs: large CPCV hyperparameter sweeps, deep sequence models, GPU Monte Carlo, LOB deep learning, long walk-forward runs | **Use WSL2 (Ubuntu) for CUDA workloads**; it avoids Windows tooling friction. The 3080 Ti has 12 GB VRAM (use mixed precision and gradient accumulation where needed). 12 cores / 24 threads handle parallel CPU-bound CPCV and backtest sweeps. |

### Device-agnostic & memory-safe rules (enforced by agents)
- All ML code selects device through one helper: `quantlab.compute.get_device()` returning `cuda` → `mps` → `cpu`.
- Every experiment must be runnable on CPU and within **8 GB RAM** at reduced scale (a `--smoke` flag) so CI and the M2 can verify correctness before the 3080 Ti does the full run.
- **Zero Phase 1 → Phase 2 DataFrame rewrite:** Storage and ingestion use **Parquet + DuckDB/Polars (with PyArrow/pandas interop at notebook edges)** from Day 1 in Phase 1. Core numerical modules (`core/`, `pricing/`, `portfolio/`, `validation/`) accept and return **typed NumPy `ndarray`s** (or lightweight dataclasses wrapping `ndarray` + 1D UTC timestamp and ticker coordinate arrays) so numerical code is library-agnostic and zero-copy.
- Determinism: fixed seeds, logged library versions, lockfile committed (`uv.lock`).
- Sync strategy between machines: Git for code and the experiment ledger (`experiments/ledger.jsonl` + `experiments/trial_returns.parquet`); large raw/cached market data (`data/`) and model checkpoints are **not** committed and are reproducible via scripts or synced via `rsync`.

### Core stack by phase

| Layer | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Storage & Data frames | Parquet + Polars/DuckDB + pandas/numpy | + lazy streaming scans, point-in-time joins | + Arrow IPC, tick stores |
| Stats / econometrics | scipy, statsmodels | + arch (GARCH), scikit-learn, hmmlearn | + custom |
| Optimization | scipy.optimize | **cvxpy** (Clarabel/OSQP/SCS) | + custom solvers, GPU |
| ML | scikit-learn | + LightGBM, PyTorch | + JAX/PyTorch, foundation models |
| Testing | pytest | + hypothesis (property tests) | + fuzzing, statistical test suites |
| Quality | ruff, mypy, pre-commit | + GitHub Actions CI, mypy strict | + nightly research CI |
| Reporting | Jupyter / Quarto | + auto-generated tear sheets | + experiment ledger dashboards |
| Packaging/ops | uv | + Docker, scheduled paper-trading jobs | + Rust (`PyO3`/`maturin`) |

---

## 4. Data Strategy (Verified 2026 Terms & US Stock Market Scope)

Data quality problems silently invalidate results, so data sources, rate limits, and accounting semantics are specified up front.

### 4.1 Scope: North American (US) Stock Market
- **Exchange Calendar:** NYSE (`XNYS`), regular trading hours `09:30:00` to `16:00:00 America/New_York` (with early-close days respected via `exchange_calendars`).
- **Two-Tier US Universe (Phase 1–2):**
  1. **Tier 1 — US Market & Sector ETF Core (14 liquid ETFs, zero single-stock survivorship bias):** `SPY`, `QQQ`, `IWM`, plus the 11 Select Sector SPDRs (`XLK`, `XLF`, `XLV`, `XLE`, `XLI`, `XLY`, `XLP`, `XLU`, `XLB`, `XLRE`, `XLC`). Used as the clean baseline universe for portfolio construction, regime modeling, and paper trading.
  2. **Tier 2 — Liquid US Single-Stock Cross-Section (50–100 US equities):** Used for cross-sectional factor models (PCA, Fama-French regressions), high-dimensional covariance shrinkage & Random Matrix Theory (Marchenko–Pastur) denoising, corporate action testing, and survivorship-bias quantification.

### 4.2 Verified Data Sources (2026 Audit)

| Category | Approved Source(s) | 2026 Quotas & Technical Implementation Notes | Licensing & Repo Guardrails |
|---|---|---|---|
| **Primary Daily US Equities & ETFs** | **Tiingo REST API** (Primary EOD) + **Alpaca Market Data (`alpaca-py`)** | **Tiingo Free Starter Tier:** 50 req/hr, 1,000 req/day, 500 unique symbols/mo, 1 GB/mo, 30+ yrs history. Returns raw OHLCV, adjusted OHLCV, `splitFactor`, and `divCash` in one response.<br>**Alpaca Free Tier (\$0/mo):** 200 req/min, 7+ yrs history. **Critical SIP vs. IEX rule:** Real-time/recent 15-min data on the free plan is restricted to IEX (~2.5% volume). Historical queries older than 15 minutes support consolidated **SIP (`feed="sip"`, 100% US volume)** for free. Always pass `feed="sip"` and cap `end <= now - 16min`. | Internal/personal research only; **never commit downloaded data to Git**. |
| **Secondary / Exploratory Daily** | **Yahoo Finance (`yfinance`)** & **Stooq** | **`yfinance`:** Unofficial scraper; requires `curl_cffi` browser impersonation + $1.5\text{–}3.5\text{s}$ request pacing to avoid TLS-fingerprint `429` blocks. Useful in Phase 1 for pulling `Ticker.actions` (splits/dividends).<br>**Stooq:** Note that `pandas_datareader` **removed** Stooq in v0.11+. Fetch direct CSV URLs (`stooq.com/q/d/l/`) or bulk ZIPs. **Warning:** Stooq returns `HTTP 200 OK` with body `"Exceeded the daily hits limit"` when throttled, and its default CSVs are retroactively split- and dividend-adjusted. | Personal use only; no redistribution. |
| **Academic Survivorship-Free & L2/Options** | **Databento** (\$125 free credit + `.ca` academic access) & **WRDS / CRSP** (via `@ualberta.ca`) | **Databento:** \$125 free signup credit (pay-as-you-go, no monthly fee) + University of Alberta (`@ualberta.ca`) academic eligibility. Provides institutional NASDAQ TotalView-ITCH (`XNAS.ITCH`) L2/L3 and OPRA options (`OPRA.PILLAR`). Always run `.metadata.get_cost()` before fetching.<br>**WRDS / CRSP:** University of Alberta is a WRDS institution (`wrds-www.wharton.upenn.edu`); if account approval is granted, CRSP provides gold-standard survivorship-free US stock history back to 1926.<br>**LOBSTER:** Free 2012 single-day sample CSVs (`AAPL`, `AMZN`, `GOOG`, `INTC`, `MSFT`, `SPY`) in fixed-point ($\times 10^4$) format. | Strict non-redistribution; auto-download into gitignored `data/raw/`. |
| **Macro & Risk-Free Rate** | **FRED / ALFRED API** (St. Louis Fed) | Free API key; **120 req/min**. **Mandatory Point-in-Time Rule:** Use ALFRED parameters (`realtime_start`, `realtime_end`, `vintage_dates`) so macro series (CPI, GDP, Unemployment, Treasury yields `DGS3MO`/`DFF`) reflect the exact date a number was published, never its nominal observation month. | US Gov series public domain; 3rd-party series restricted. |
| **Factor Benchmarks** | **Kenneth French Data Library** | Direct ZIP-CSV downloads (`mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/*_CSV.zip`). Parser must split the daily/monthly table from the trailing annual summary table and **divide values by `100.0`** (Ken French reports percentage points `1.25` = $1.25\% = 0.0125$). | Free academic/research use; cite source. |
| **Explicitly Excluded Sources** | ~~Alpha Vantage~~, ~~Binance (`data.binance.vision`)~~ | **Alpha Vantage dropped:** Free tier cut to 25 req/day and `TIME_SERIES_DAILY_ADJUSTED` moved behind a \$49.99/mo paywall.<br>**Binance dropped:** Geo-blocked (`HTTP 451/403`) for US/North American IPs and out of scope for our US Stock Market focus. | N/A |

### 4.3 Data Architecture & Point-in-Time Rules
1. **Single `DataProvider` Protocol:** All sources implement a unified interface (`quantlab/data/providers/`). Strategies and models never call vendor HTTP APIs directly.
2. **Immutable Raw Store vs. Corporate Actions Table:**
   - Never rely exclusively on backward-adjusted `Adj Close` as the primary stored price, because every new dividend or split retroactively mutates the entire historical series (breaking content hashes, option strike alignment, and tick sizes).
   - Store **immutable raw (unadjusted) OHLCV** (`open, high, low, close, volume`) keyed by content hash, alongside an explicit **Corporate Actions table** (`ticker, ex_date, split_factor, div_cash`).
   - Derive total-return series and split-adjusted series on demand via one tested module (`quantlab/data/adjustments.py`).
3. **Dual-Timestamp Point-in-Time (PIT) Discipline:**
   - Every record stores both `period_ts` (the nominal bar date/period) and `available_ts` (the exact UTC timestamp when the datum became knowable to a market participant, e.g., `16:00:00 America/New_York` converted to UTC for US daily equity closes, or the ALFRED release timestamp for macro data).
   - Queries to the PIT store at simulation time $t$ strictly filter `available_ts <= t`.
4. **Survivorship Bias Strategy:**
   - Phase 1 (Notebook 10) demonstrates survivorship bias quantitatively by comparing a survivorship-biased single-stock subset against an unbiased baseline.
   - Phase 2 uses the survivorship-free **Tier 1 US Sector/Market ETF universe** as the primary benchmark and bounds/controls survivorship bias on **Tier 2 US Single Stocks** via point-in-time historical constituent lists (or CRSP if accessed via UAlberta WRDS).
5. **Automated Ingestion Validation Suite:**
   - Runs on every ingest (`quantlab/data/validation.py`): checks for missing `XNYS` trading days, negative/zero prices, extreme return outliers ($|r_t| > 50\%$ without a matching split factor), non-monotonic `available_ts`, and stale repeated prices.
6. **Zero Vendor Data in Git / Deterministic CI Fixtures:**
   - `data/` is gitignored. All unit tests and GitHub Actions CI runs execute exclusively against deterministic synthetic market data (`quantlab/data/synthetic.py`) and tiny hand-computed mathematical fixtures in `tests/fixtures/`.

---

## 5. AI-Agent Scaffolding Contract (for Antigravity)

This section is the heart of the "agent quality assurance" goal. It will be copied into `AGENTS.md` at initialization.

> **Tooling note:** this contract is model-agnostic: it relies on tests, CI, and human review rather than trusting any model's output. A fast model can handle boilerplate, tests, and refactors, while a stronger reasoning model handles design, numerical algorithms, and adversarial review. Measure actual performance in `docs/agent_scorecard.md`.

### 5.1 Standing rules for every agent task
1. **Work from a task spec.** No task starts without a file in `docs/tasks/` (template in §5.4) stating goal, inputs, outputs, acceptance tests, and out-of-scope items.
2. **Small, reviewable PRs / checkpoints.** After the initial Phase 0 repository bootstrap (§8 items 1–4), keep one concern per branch/PR. No drive-by refactors.
3. **Tests first for numerics.** Any numerical routine needs: a known-answer test, a property test, and a convergence/accuracy test where applicable.
4. **No look-ahead.** Any code touching time series must pass the leakage test suite (§6.2).
5. **No silent assumptions.** Assumptions, units, conventions (log vs. simple returns, annualization factor = 252 `XNYS` trading days, decimal vs. percentage returns, UTC timestamps) are documented in docstrings and in `docs/conventions.md`.
6. **Never fabricate results.** Reported numbers must come from committed code that reproduces them. Agents must not write result numbers into docs by hand.
7. **Experiment ledger is mandatory** (§6.3): every strategy/model variant tried is logged to `experiments/ledger.jsonl` and `experiments/trial_returns.parquet`, including failures.
8. **No secrets or vendor data in the repo.** API keys via environment variables / `.env` (gitignored); pre-commit hooks block secrets and large data files.
9. **Human gate:** merging to `main`, adding dependencies, changing the validation framework, and anything touching broker connectivity require human approval.
10. **Uncertainty reporting:** agents state what they verified vs. what they assumed at the end of each task.

### 5.2 Definition of Done (all phases)
- [ ] Code passes `uv run ruff check .`, `uv run mypy src tests`, and `uv run pytest`
- [ ] New logic has tests; numerics have known-answer + property tests
- [ ] Leakage tests pass for anything time-dependent
- [ ] Runs comfortably within the 8 GB M2 unified memory budget (via `--smoke` or out-of-core Parquet scans)
- [ ] Docs/docstrings updated; `docs/conventions.md` respected
- [ ] Experiment ledger (`ledger.jsonl` + `trial_returns.parquet`) updated if any model/strategy was evaluated
- [ ] Reproducible with a single documented command
- [ ] Task spec acceptance criteria all checked

### 5.3 Phase-scaled rigor (the "increasing sophistication" mechanism)

| Requirement | Phase 1 | Phase 2 | Phase 3 |
|---|---|---|---|
| Test coverage target (core `src/`) | ≥ 70% | ≥ 85% | ≥ 85% core, experimental code exempt but flagged |
| Type checking | mypy basic | mypy strict on core | strict on core |
| Validation | Simple chronological train/test split + demonstrated pitfalls | Cross-sectionally synchronized Purged K-Fold, CPCV, Deflated Sharpe ($N_{\text{eff}}$), PBO | Automated gatekeeper blocks any claim failing validation |
| Review | Human reads everything | Human reviews design + critical numerics; agent cross-review | Agent adversarial review + human spot-audit |
| Reproducibility | Notebook reruns + `uv.lock` | One-command reproducible pipeline, Docker | Versioned experiments, deterministic seeds, provenance |
| Agent autonomy | Single-task, tightly specced | Multi-step tasks with checkpoints | Longer-horizon research loops within budget/guardrails |

### 5.4 Task spec template (`docs/tasks/TEMPLATE.md`)
```markdown
# Task: <short name>
## Goal
## Context / links (design doc section, prior ADRs)
## Inputs
## Outputs (files, APIs)
## Acceptance tests (concrete, runnable)
## Out of scope
## Risks / assumptions to verify
## Phase & rigor level
```

### 5.5 Agent scorecard (measure, don't assume)
Track per task: first-pass test pass rate, number of review corrections, bugs found later, whether it hallucinated an API or a result. Review monthly; use it to decide which model gets which kind of task. Store in `docs/agent_scorecard.md`.

### 5.6 Anti-patterns to explicitly forbid in `AGENTS.md`
- Tuning on the test set or on the full sample, then reporting in-sample results.
- Fitting preprocessors (PCA, covariance shrinkage, volatility scaling, fractional differentiation) on the full dataset prior to CV splitting.
- Splitting multi-asset panels independently per ticker instead of synchronizing train/test/purge/embargo intervals across the cross-section by timestamp.
- Reporting Sharpe without costs, or without the trial return matrix logged in `experiments/`.
- "Fixing" a failing leakage/validation test by weakening it.
- Adding a heavyweight dependency to avoid writing 20 lines.
- Describing an unrun code path as working.

---

## 6. Cross-Cutting Infrastructure

### 6.1 Repo structure (target end-state; create incrementally)

```text
quant-lab/
├── README.md
├── QUANT_PROJECT_DESIGN.md        # canonical design doc (also linked/mirrored in docs/DESIGN.md)
├── AGENTS.md                      # agent rules (from §5)
├── LICENSE
├── pyproject.toml / uv.lock
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
├── docs/
│   ├── conventions.md             # returns, signal-to-fill lag, annualization (252), XNYS calendar, units
│   ├── glossary.md
│   ├── adr/                       # architecture decision records (0001-initial-architecture-decisions.md, ...)
│   ├── tasks/                     # task specs (TEMPLATE.md + instances)
│   ├── research_log.md
│   ├── agent_scorecard.md
│   └── risk_register.md
├── src/quantlab/
│   ├── compute.py                 # device selection (cuda -> mps -> cpu) & memory helpers
│   ├── data/                      # providers (tiingo, alpaca, fred, french, stooq, yfinance), adjustments, PIT store, synthetic
│   ├── core/                      # returns, stats, metrics, XNYS calendar alignment
│   ├── pricing/                   # BS, binomial, Monte Carlo, Greeks
│   ├── portfolio/                 # mean-variance, shrinkage/RMT, risk parity, HRP, BL, CVXPY cost-aware
│   ├── backtest/                  # event-driven engine (share/cash & total-return modes), costs, fills, orderbook
│   ├── validation/                # leakage tests, synchronized purged CV, CPCV, DSR (N_eff), PBO
│   ├── models/                    # factor, ML, regime, sequence models
│   ├── execution/                 # cost models, impact, schedulers
│   ├── risk/                      # VaR/CVaR, stress, exposures, kill switches
│   └── research/                  # (Phase 3) agent harness, ledger, gatekeeper
├── notebooks/phase1/              # curriculum notebooks (01..12)
├── experiments/                   # configs + ledger.jsonl + trial_returns.parquet
├── flagship/                      # (Phase 2) US Equity & ETF Portfolio Construction & Paper-Trading System
├── tests/
│   └── fixtures/                  # tiny hand-calculated & synthetic fixtures for CI
└── scripts/
```

### 6.2 Leakage, execution-lag, and sanity test suite (build in Phase 1, extend always)
- **Strict Signal-to-Fill Lag Convention:** A signal or target weight vector $w_t = f(\mathcal{F}_t)$ computed from data available up to bar $t$ (`available_ts <= close_ts[t]`) can at earliest execute at bar $t+1$ (`open[t+1]` or `close[t+1]`), earning return $r_{t+1}$. Executing at `close[t]` using a signal computed from `close[t]` is forbidden unless explicitly modeling a pre-close cutoff with closing-auction slippage.
- **Delay vs. Look-Ahead Probe (Shift Test):**
  - Delaying signals by $+1$ bar (`w_delayed[t] = w[t-1]`, i.e., `shift(+1)`) must degrade or leave flat a genuine signal; if injecting a $-1$ look-ahead probe (`w_peek[t] = w[t+1]`, i.e., `shift(-1)`) does not sharply change behavior or if an unshifted strategy matches the $-1$ probe, inspect for look-ahead.
- **Future-Poison Test:** Replace all input data where `available_ts > t` with `NaN` or extreme garbage values ($\pm 10^9$); the strategy signal and preprocessor state at time $\le t$ must be bitwise/numerically identical (`np.allclose`).
- **In-Fold Preprocessor Test:** Verify that fitting any covariance estimator, scaler, PCA, or fractional differentiator inside a CV fold produces zero dependence on test-fold rows.
- **Shuffled-Label / Random-Return Permutation Test:** A model trained on temporally permuted target returns must achieve an out-of-sample Sharpe distribution centered at zero.
- **Random-Signal Baseline:** Generate $M$ random weight/signal trajectories with matching turnover to establish the null distribution of Sharpe ratios on the exact same US equity universe and period.
- **Accounting Identities:**
  - *Mode A (Share & Cash):* $\text{Equity}_t = \text{Cash}_t + \sum_i \text{Shares}_{i,t} \cdot P_{i,t}^{\text{raw}}$, with splits multiplying $\text{Shares}_{i,t}$ and cash dividends credited to $\text{Cash}_t$ on `ex_date`.
  - *Mode B (Total-Return Weight):* Portfolio return nets exact transaction costs from weight turnover $\Delta w_t$.
- **Known-Answer Finance Tests:** Put-call parity, Black-Scholes vs. high-$N$ Cox-Ross-Rubinstein binomial tree, Monte Carlo vs. closed form within $3\sigma$ standard error.

### 6.3 Two-File Git-Tracked Experiment Ledger (critical for honest DSR & PBO)
Every strategy or model configuration evaluated is automatically logged to two Git-tracked files in `experiments/`:
1. `experiments/ledger.jsonl`: One JSON record per trial containing `trial_id`, timestamp, hypothesis ID, git commit SHA, config hash, dataset content hash, date window, in-sample/out-of-sample summary metrics (gross/net Sharpe, turnover, max drawdown), and skewness/kurtosis of returns.
2. `experiments/trial_returns.parquet`: A compact columnar table storing `(trial_id, period_ts, net_return)` for every trial.
   - **Why both are mandatory:** Computing the **Probability of Backtest Overfitting (PBO)** via Combinatorial Symmetric Cross-Validation (CSCV) and estimating the **effective number of independent trials ($\hat{N}_{\text{eff}}$)** via cross-trial return correlation for the **Deflated Sharpe Ratio (DSR)** mathematically require the $T \times K$ matrix of trial return series—scalar Sharpe ratios alone are insufficient when trials are correlated.

---

# PHASE 1 — Foundations: "Quant Finance for Computer Scientists"

**Theme:** Understanding over optimization. Every module links a CS idea you know to its financial counterpart, with a small, readable implementation on US equity/ETF data and an honest demonstration of what can go wrong.

**Estimated effort (part-time student):** ~6–8 weeks. **Compute:** M2 (8 GB RAM) only.

### 1.1 Deliverables
1. `quantlab` package with the core modules listed below, tested against known answers and synthetic CI fixtures.
2. A **curriculum of 12 notebooks** (`notebooks/phase1/`), each following: concept → CS analogy → math (minimal) → code → exercise → pitfall.
3. An **event-driven backtester** (`backtest/engine.py`) written from scratch supporting both Share/Cash and Total-Return accounting (later cross-checked against an established library in §1.6).
4. A capstone report, **"Why Most Backtests Lie,"** that deliberately overfits a US equity strategy, then exposes it with out-of-sample testing, transaction costs, survivorship bias analysis, and the random-signal baseline.
5. A one-page personal glossary mapping CS concepts → finance concepts (`docs/glossary.md`).

### 1.2 Curriculum (notebook plan & dependency path)

> **Critical Path Note:** Notebooks 01–10 and 12 form the daily-bar US equity critical path leading directly into Phase 2. Notebook 11 (`backtest/orderbook.py` — L2 limit order book matching) is a self-contained microstructure module that can be completed in parallel or after Notebook 12 without blocking Phase 2.

| # | Notebook | Core ideas | CS bridge | Hands-on (US Market Focus) |
|---|---|---|---|---|
| 01 | Markets, prices, returns | Simple vs. log returns, compounding, raw vs. split/dividend-adjusted prices, `XNYS` calendar | Data cleaning, dual-timestamp (`period_ts`, `available_ts`) indexing | Build the Parquet/PIT data layer + corporate actions + validation checks on US ETFs/stocks |
| 02 | Stylized facts | Fat tails, volatility clustering, weak return autocorrelation, leverage effect | Empirical distributions, hypothesis testing | Test normality on `SPY`/US equities; QQ plots; ACF of $r_t$ vs. $r_t^2$ and $|r_t|$ |
| 03 | Randomness and Monte Carlo | Brownian motion, Geometric Brownian Motion, variance reduction | Randomized algorithms, law of large numbers | Simulate paths; antithetic & control variates; verify $O(1/\sqrt{N})$ convergence |
| 04 | Option pricing I | Payoffs, no-arbitrage, put-call parity, Black-Scholes | Recurrence relations, closed forms | Implement BS; verify put-call parity; Greeks by closed form and finite differences |
| 05 | Option pricing II | Binomial trees (CRR), American options, early exercise boundary | **Dynamic programming** | Price American puts on US stocks by backward induction; verify convergence to European BS |
| 06 | Risk measures | Volatility, VaR, CVaR/Expected Shortfall, drawdown | Order statistics, tail estimation | Compute historical/parametric/MC VaR & CVaR on US sector ETFs; show VaR's subadditivity/tail failure |
| 07 | Portfolio theory | Mean-variance, efficient frontier, sample covariance fragility | **Linear algebra, convex (quadratic) optimization** | Build Markowitz on the 14 US Sector/Market ETFs; show condition-number instability vs. equal-weight (`1/N`) |
| 08 | Factor models | CAPM, Fama-French 3/5 factors, statistical PCA factors | **PCA / SVD**, linear regression | Regress US equities on Ken French factors; run SVD/PCA on a 50-stock US cross-section; interpret eigenportfolios |
| 09 | Strategies & signals | Cross-sectional & time-series momentum, mean reversion, pairs | Signal processing, state machines | Implement 2–3 US equity/ETF strategies strictly obeying the `signal[t] -> fill[t+1]` contract |
| 10 | Backtesting honestly | Event-driven loop, bid-ask + impact costs, slippage, bias catalog | **Event-driven simulation, queues** | Run strategies through `backtest/engine.py`; demonstrate look-ahead and single-stock survivorship bias on purpose |
| 11 | Market microstructure *(self-contained)* | Limit order book, bid-ask spread, price-time priority (FIFO) | **Heaps / priority queues, sorted maps** | Implement a price-time priority LOB (`orderbook.py`) + matching engine; replay synthetic/LOBSTER US equity flow |
| 12 | Capstone | Overfitting, multiple testing, random baselines, ledger logging | Statistical inference, honest evaluation | **"Why Most Backtests Lie"** report on US equities/ETFs |

### 1.3 CS → Quant mapping (the conceptual spine; expand in `docs/glossary.md`)

| CS concept | Quant counterpart |
|---|---|
| Dynamic programming | American option pricing (binomial), optimal stopping |
| Randomized algorithms / Monte Carlo | Path simulation, VaR, exotic option pricing |
| Linear algebra, SVD/PCA | Covariance structure, factor models, risk decomposition |
| Convex optimization | Portfolio construction, cost-aware rebalancing, calibration |
| Priority queues | Limit order book matching |
| Event-driven systems | Backtesting and paper/live trading engines |
| Overfitting / multiple hypothesis testing | Backtest overfitting, Deflated Sharpe Ratio, PBO |
| Time series / signal processing | Return modeling, Kalman filters, volatility models |
| Distributed systems, idempotency | Reliable scheduled trading pipelines (Phase 2) |
| Compression / information theory | Signal-to-noise in returns; weak-form market efficiency intuition |

### 1.4 Phase 1 module specs (for agent task generation)

- `data/providers/`: `DataProvider` protocol + Tiingo/Stooq/`yfinance` providers, `data/adjustments.py` (splits/dividends), `data/validation.py`, `data/synthetic.py` (deterministic fixtures for CI), and local Parquet PIT cache.
- `core/returns.py`: simple/log conversion, total-return computation from raw prices + corporate actions, resampling, annualization (`252` `XNYS` trading days).
- `core/metrics.py`: Sharpe (with Lo 2002 standard error), Sortino, max drawdown, Calmar, turnover; all accepting typed NumPy arrays.
- `pricing/black_scholes.py`, `pricing/binomial.py`, `pricing/monte_carlo.py`, `pricing/greeks.py`.
- `portfolio/markowitz.py` (closed-form + constrained via `scipy.optimize`), `portfolio/baselines.py` (equal weight `1/N`, inverse volatility).
- `backtest/engine.py`: event loop with `Strategy`, `Broker`/`FillModel` (enforcing `signal[t] -> fill[t+1]`), `CostModel`, `Portfolio` (Share/Cash and Total-Return modes); vectorized shortcut only as a cross-check.
- `backtest/orderbook.py`: limit order book with price-time priority (FIFO per price level).
- `validation/leakage.py`: future-poison, delay/look-ahead shift probe, shuffled-label, and in-fold preprocessor leakage tests (§6.2).

### 1.5 Phase 1 acceptance criteria ("done")
- [ ] All 12 notebooks run top-to-bottom reproducibly in a fresh environment within the 8 GB M2 memory budget.
- [ ] Known-answer tests pass: put-call parity, BS ≈ high-$N$ binomial, MC within $3\sigma$ confidence interval of BS.
- [ ] Backtester reproduces a hand-computed 5-bar US equity toy fixture (including a 2-for-1 split, a cash dividend, and transaction costs) to machine precision.
- [ ] Leakage tests exist and *fail correctly* on intentionally leaky strategies.
- [ ] Capstone report demonstrates: in-sample vs. out-of-sample decay, transaction cost erosion, single-stock survivorship bias impact, and the random-strategy Sharpe distribution.
- [ ] You can explain (to yourself, in writing) every module without reading its code.
- [ ] CI green on synthetic fixtures; coverage ≥ 70% on `src/`.

### 1.6 Phase 1 → 2 gate
Do not start Phase 2 until: leakage suite is trusted, `backtest/engine.py` is verified against at least one established library on a simple US ETF strategy (any accounting differences explained in writing), `docs/conventions.md` is stable, and you've written a short "what I now understand / what I'm unsure about" note.

---

# PHASE 2 — Sophistication: "One Production-Grade US Equity Portfolio & Risk System"

**Theme:** State-of-the-art portfolio construction, risk estimation, and validation techniques applied to the **US Stock Market**, culminating in **one** system that is deployable, monitored, reproducible, and statistically defensible.

**Estimated effort:** ~12–16 weeks. **Compute:** M2 (8 GB RAM) for development; RTX 3080 Ti (via WSL2) for CPCV sweeps and deep sequence models.

### 2.1 Locked Flagship Choice: Option A (US Equity & Sector ETF Portfolio Construction & Risk System)

Recorded in [ADR-0001](file:///Users/arthurli/quant/docs/adr/0001-initial-architecture-decisions.md):
- **What we are building:** A cost-aware, regime-aware US portfolio construction and risk management system operating on our two-tier US stock market universe (Tier 1: 14 liquid US Market & Sector ETFs; Tier 2: 50–100 liquid US equities).
- **Why this flagship is meaningful:** Return predictability on daily public US stock data is near zero, whereas covariance estimation, risk concentration control, regime-aware volatility scaling, and convex optimization under transaction costs deliver verifiable out-of-sample risk-adjusted and turnover improvements. Everything is forward-tested via an automated **Alpaca Paper Trading** pipeline (`feed="sip"` for historical validation).

### 2.2 Technique roadmap (implement in layers; each layer benchmarked against the previous)

**Layer 1 — Statistical foundations (fit strictly inside CV folds)**
- Shrinkage covariance (Ledoit-Wolf, OAS), Random Matrix Theory denoising (Marchenko–Pastur eigenvalue clipping/shrinkage on the Tier 2 US stock cross-section), and market-mode detoning.
- Robust return/location estimation (Huber / Student-$t$), EWMA and GARCH-family volatility (`arch`).
- Multiple-testing machinery: **Deflated Sharpe Ratio (DSR)** using effective independent trials $\hat{N}_{\text{eff}}$ estimated from `experiments/trial_returns.parquet`, **Probability of Backtest Overfitting (PBO)** via CSCV, and Minimum Track Record Length (MinTRL).

**Layer 2 — Validation framework (build before Layer 3/4 models)**
- **Cross-sectionally synchronized** Purged K-Fold with embargo; **Combinatorial Purged Cross-Validation (CPCV)**; walk-forward with anchored and rolling windows. (All tickers are split on identical time boundaries so market-factor leakage across correlated US equities is impossible.)
- Triple-barrier labeling, meta-labeling, and sample-uniqueness weights.
- Fractional differentiation (fixed-width window) for stationarity with memory preservation (fit strictly per training fold).
- Automated "Validation Tear Sheet" generator for any strategy/model: IS/OOS breakdown, CPCV path Sharpe distribution, DSR, PBO, and global trial count from `experiments/ledger.jsonl`.

**Layer 3 — Portfolio construction**
- Baselines: equal weight (`1/N`), inverse volatility, minimum variance, Equal Risk Contribution (ERC / risk parity).
- **Hierarchical Risk Parity (HRP)** and Nested Clustered Optimization (NCO) on US sectors and single stocks.
- **Black-Litterman** combining market-cap/ETF equilibrium priors with model views and view uncertainty.
- **Convex optimization with transaction costs (`cvxpy` with Clarabel/OSQP):** turnover penalties, spread + square-root market impact, long-only / dollar-neutral / sector-exposure / Fama-French factor-beta constraints, CVaR-constrained optimization, and Multi-Period Trading (Boyd et al. MPC formulation).
- Robust optimization (ellipsoidal uncertainty sets on expected returns and covariance).

**Layer 4 — Forecasting and regimes (challengers vs. risk-only baselines)**
- Cross-sectional & time-series features (momentum, reversal, realized vol, idiosyncratic vol, Fama-French residuals, ALFRED point-in-time macro regime indicators).
- Ridge / Elastic-Net linear baselines → Gradient Boosted Trees (LightGBM) → small sequence models (LSTM/GRU, PatchTST) evaluated strictly as *challengers* against a zero-forecast risk-based portfolio baseline.
- Regime detection: Hidden Markov Models (`hmmlearn`), volatility/correlation clustering on `SPY`/VIX/sector dispersion; test whether conditioning risk targets or turnover penalties on detected regimes improves **out-of-sample net-of-cost** performance.
- Kalman-filter dynamic hedge ratios for US equity/ETF relative-value pairs.
- Calibrated uncertainty: conformal prediction intervals feeding into Black-Litterman view confidence or robust optimization uncertainty sets.

**Layer 5 — Execution and costs**
- US Equity cost model: half-spread + SEC/TAF regulatory fees + square-root market impact ($\sigma_i \sqrt{Q_i / \text{ADV}_i}$); sensitivity analysis across $0.5\times, 1\times, 2\times, 5\times$ cost multipliers.
- Execution scheduling (TWAP, VWAP, Almgren-Chriss optimal trajectory) inside the backtester.

### 2.3 Production engineering requirements (`flagship/`)
- **Daily US Market Pipeline (post-`16:15 America/New_York`):** ingest Tiingo + Alpaca SIP bars + ALFRED macro → run validation suite → update PIT Parquet store → compute features & covariance → run model inference & CVXPY optimizer → generate target weights & share orders → submit to **Alpaca Paper Trading API** → reconcile fills & positions → append to monitoring log & tear sheet.
- **Idempotent, restartable jobs:** every daily run produces an immutable, hashed artifact bundle (data snapshot hash, git SHA, pydantic config hash, target weights, solver status).
- **Configuration-driven:** typed configs via `pydantic`; zero magic constants in code.
- **Dockerized:** one-command local run (`docker compose run flagship`); GitHub Actions CI builds and tests the container image.
- **Monitoring:** data freshness (`XNYS` calendar aware), feature & covariance condition-number drift, realized vs. predicted tracking error, turnover, realized Alpaca paper fill slippage vs. cost model, drawdown alarms.
- **Kill switches (tested in CI):** max single-stock/ETF weight, max gross/net leverage, max daily drawdown halt, stale-data halt, solver-infeasible fallback (hold current positions or fall back to `1/N`).
- **Model registry & rollback:** champion/challenger shadow evaluation.
- **Documentation:** model card, CPCV/DSR/PBO validation report, operational runbook, and known-limitations section.

### 2.4 Phase 2 acceptance criteria
- [ ] **Validation framework** passes synthetic ground-truth tests: pure-noise white-noise/GBM panels yield PBO $\approx 0.5$ and DSR rejecting skill; planted-signal synthetic panels are recovered.
- [ ] Flagship **beats its simple baselines (`1/N` and inverse-vol) net of costs under CPCV** *or* the repo publishes a rigorous, honest negative result explaining why simple baselines won.
- [ ] Reported performance includes: global trial count $K$, effective independent trials $\hat{N}_{\text{eff}}$, Deflated Sharpe Ratio, PBO, cost-multiplier sensitivity curve ($0\times \to 5\times$), subperiod/regime breakdown (e.g., 2020 COVID crash, 2022 rate-hike bear market, 2023–2026), turnover, and estimated dollar capacity.
- [ ] **Forward Alpaca paper-trading record** of at least 3 months (target 6), with realized paper PnL and slippage reconciled against backtest expectations.
- [ ] One-command reproducibility from cached/synthetic data to validation report in Docker.
- [ ] Monitoring and kill-switch tests pass.
- [ ] Coverage ≥ 85% on core; `mypy --strict` on core; CI green.
- [ ] Public artifacts: README with results, model card, technical writeup, and an independent adversarial review pass (`docs/research_log.md`) with all findings resolved or documented.

### 2.5 Risks (tracked in `docs/risk_register.md`)
- Overfitting through repeated reuse of the same US historical window (mitigate: two-file experiment ledger, DSR/PBO, and a sealed chronological holdout window untouched until final evaluation).
- Single-stock survivorship bias in Tier 2 universe (mitigate: benchmark everything on survivorship-free Tier 1 US ETFs first; use point-in-time constituent snapshots or UAlberta WRDS/CRSP for single stocks).
- Complexity creep (mitigate: baseline-wins-ties rule; ADR required for each new technique).
- Agent-introduced subtle numerical or leakage bugs (mitigate: known-answer tests, cross-checks against reference libraries, synchronized CV splits, human review of every numerical core change).

### 2.6 Phase 2 → 3 gate
Flagship delivered (or rigorous negative result published), validation framework trusted, two-file ledger populated, reproducibility proven on a clean machine.

---

# PHASE 3 — Frontier: "Open-Ended Research with Agentic AI"

**Theme:** No fixed endpoint. A research platform where humans and agents propose, test, and honestly evaluate ideas at the edge of quantitative finance, with the Phase 2 validation machinery acting as the immune system.

### 3.1 The research operating system (build first, then do research on top)
1. **Hypothesis registry:** each research idea is a structured pre-registered entry (hypothesis, economic/statistical rationale, expected mechanism, falsification criteria, US market dataset needed, pre-specified evaluation metric) committed **before** any experiment runs.
2. **Agentic research loop:** propose → implement → run CPU/M2 `--smoke` test → run full CPCV/validation on 3080 Ti → adversarial agent review → automatic `ledger.jsonl` + `trial_returns.parquet` update → human triage. Compute hours, trial count, and token budgets are capped per loop.
3. **Automated gatekeeper:** no result is promoted to "finding" unless it passes the validation suite (cross-sectionally synchronized CPCV, DSR using the global trial return matrix, realistic US equity cost model, leakage suite). Agents cannot modify `src/quantlab/validation/` or the gatekeeper without explicit human approval.
4. **Adversarial reviewer agent:** a separate agent persona instructed specifically to hunt for look-ahead leakage, survivorship bias, in-fold preprocessing leaks, and numerical bugs in candidate results.
5. **Global multiple-testing ledger:** DSR thresholds tighten automatically as total trials $K$ and effective independent trials $\hat{N}_{\text{eff}}$ grow.
6. **Provenance:** every claim links to git commit SHA, data content hash, config hash, and `trial_id`.
7. **Meta-evaluation:** periodically benchmark the research agents themselves (false discovery rate, bug injection rate) using synthetic US equity panels with planted signals and planted leakage bugs.

### 3.2 Research tracks (pick by interest; each is an independent, open-ended thread)

| Track | Ideas to explore | CS flavor |
|---|---|---|
| **A. Deep hedging & derivatives ML** | Deep hedging under transaction costs, neural SDEs, rough volatility (rough Bergomi), GPU Monte Carlo, differentiable vol-surface calibration on OPRA/Databento samples | Differentiable programming, GPU kernels |
| **B. Market microstructure & LOB learning** | LOB deep models (DeepLOB-style) on NASDAQ ITCH (LOBSTER / Databento `XNAS.ITCH`), queue-position modeling, Avellaneda–Stoikov market making, Hawkes processes, RL for optimal execution | Sequence models, simulation, systems |
| **C. Representation & foundation models** | Evaluating pretrained time-series foundation models (Chronos, TimesFM, Moirai) on US equities vs. Ridge/LightGBM baselines; self-supervised cross-sectional representations; path signatures | Transfer learning, representation learning |
| **D. Robustness & uncertainty** | Conformal risk control, Wasserstein distributionally robust portfolio optimization, optimal transport for regime stress testing, extreme value theory | Statistics, convex/robust optimization |
| **E. Generative market simulation** | Diffusion/VAE/flow matching for synthetic multi-asset return panels and realistic stress scenarios; testing whether synthetic augmentation improves out-of-sample portfolio robustness | Generative modeling |
| **F. Causal & alternative-data methods** | Causal discovery/inference for factor claims, SEC EDGAR 10-K/10-Q/8-K filing event studies (strictly stamped with SEC acceptance timestamp), LLM feature extraction with cutoff-date leakage controls | NLP, causal ML |
| **G. High-performance Rust systems** | **Rust (`PyO3` + `maturin`)** price-time priority matching engine and latency-aware event simulator with zero-copy Arrow/NumPy bindings, streaming `dbn`/Parquet tick replay | Systems programming, Rust/Python FFI |
| **H. Agentic AI for research itself** | Planted-signal benchmarks for research agents, multi-agent debate for hypothesis critique, automated literature-to-experiment pipelines, measuring false discovery rate vs. agent autonomy | AI agents, evaluation |

> **A note on LLMs and finance:** language models trained on web/news corpora that include the backtest period leak future knowledge (they "know" which US companies succeeded or crashed). Any LLM-in-the-loop feature experiment must enforce strict temporal cutoff discipline (evaluate exclusively on market data after the model's verified knowledge cutoff, or use anonymized synthetic inputs).

### 3.3 Continuous cadence (suggested)
- **Weekly:** pick 1–2 hypotheses; run through the loop; log everything including failures.
- **Monthly:** research review — what survived the gatekeeper, agent scorecard, update risk register, prune dead branches.
- **Quarterly:** write up one finding or one rigorous negative result publicly (blog/preprint-style). Negative results count.
- **Ongoing:** keep a reading list (`docs/reading_list.md`) tied to hypotheses, not accumulated for its own sake.

### 3.4 Phase 3 success metrics
- Number of hypotheses tested vs. promoted (and empirical false discovery estimate on planted benchmarks).
- Quality and clarity of the negative-result record.
- Agent reliability trend (first-pass test rate, leakage catches, review corrections).
- Reproducibility incidents (target: zero unexplained).
- Things learned that changed your mental model (a short running log in `docs/research_log.md`).

---

## 7. Reading List (starter, tied to phases)

**Phase 1**
- Hull, *Options, Futures, and Other Derivatives* (derivatives, binomial trees, Black-Scholes)
- Shreve, *Stochastic Calculus for Finance I & II* (binomial and continuous-time models)
- Ang, *Asset Management: A Systematic Approach to Factor Investing*
- Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing*

**Phase 2**
- López de Prado, *Advances in Financial Machine Learning* and *Machine Learning for Asset Managers* (Purged CV, CPCV, HRP, RMT denoising, triple-barrier)
- Grinold & Kahn, *Active Portfolio Management* (the Fundamental Law, cross-sectional portfolio construction)
- Boyd et al. (2017), *"Multi-Period Trading via Convex Optimization"* (Foundations and Trends in Optimization)
- Bailey & López de Prado (2014), *"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"*
- Bailey, Borwein, López de Prado & Zhu (2016), *"The Probability of Backtest Overfitting"*
- Almgren & Chriss (2000), *"Optimal Execution of Portfolio Transactions"*

**Phase 3**
- Bühler, Gonon, Teichmann & Wood (2019), *"Deep Hedging"*
- Gatheral, Jaisson & Rosenbaum (2018), *"Volatility is Rough"*
- Zhang, Zohren & Roberts (2019), *"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"*
- Cartea, Jaimungal & Penalva, *Algorithmic and High-Frequency Trading*
- Recent time-series foundation model and conformal risk control literature

---

## 8. Initialization Checklist

Execute **Phase 0 (Repo Bootstrap: items 1–4)** as a single cohesive foundation milestone so that `main` has a working Python 3.12 toolchain, test runner, linter, type-checker, and pre-commit guardrails. Then execute **items 5–9** one task spec (`docs/tasks/`) and one reviewable checkpoint/PR at a time:

### Phase 0 — Repo Bootstrap (Single Foundation Milestone)
1. [x] Install/verify `uv` + Python 3.12; create `pyproject.toml`, `uv.lock`, `src/quantlab/` package skeleton, `ruff`/`mypy`/`pytest` config, `.gitignore` (blocking `data/`, `.env`, `*.dbn`), and `.pre-commit-config.yaml` (secret & file-size guardrails).
2. [x] Add `AGENTS.md` (from §5), `docs/DESIGN.md` (or symlink/pointer to `QUANT_PROJECT_DESIGN.md`), `docs/conventions.md` (defining returns, `signal[t] -> fill[t+1]` lag, 252 `XNYS` annualization, dual timestamps), `docs/tasks/TEMPLATE.md`, `docs/adr/0001-initial-architecture-decisions.md`, `docs/risk_register.md`, `docs/agent_scorecard.md`, and empty `experiments/ledger.jsonl`.
3. [x] Add GitHub Actions CI workflow (`.github/workflows/ci.yml`): lint (`ruff`), type check (`mypy`), and unit tests (`pytest` on synthetic fixtures) on Python 3.12 (macOS and Ubuntu runners).
4. [x] Implement `quantlab.compute.get_device()` (`cuda` → `mps` → `cpu`) and 8 GB memory helper utilities with unit tests.

### Phase 1 Kickoff Tasks (One Task Spec per Item)
5. [x] `DataProvider` protocol + deterministic synthetic provider (`data/synthetic.py`) + Tiingo/Stooq/`yfinance` providers + `data/adjustments.py` (raw OHLCV + splits/dividends) + `data/validation.py` + local PIT Parquet cache.
6. [x] `core/returns.py` + `core/metrics.py` with known-answer tests and Lo (2002) Sharpe standard errors.
7. [x] `validation/leakage.py` suite (future-poison, $+1$/$-1$ shift probes, shuffled-label, and in-fold preprocessor tests) verified against deliberately leaky fixture strategies.
8. [x] Notebook 01 (`01_markets_prices_returns.ipynb`) and `README.md` "Getting Started."
9. [ ] Create task specs / GitHub Issues for the remainder of Phase 1 (§1.2 Notebooks 02–12 and §1.4 modules).

---

## 9. Resolved Architecture Decisions ([ADR-0001](file:///Users/arthurli/quant/docs/adr/0001-initial-architecture-decisions.md))

All seven open design questions were reviewed and resolved prior to repository initialization:

| # | Decision Area | Locked Choice | Summary Rationale |
|---|---|---|---|
| **1** | **Phase 2 Flagship** | **Option A: Cost-aware, regime-aware US portfolio construction & risk system** | Connects covariance shrinkage/RMT, synchronized Purged CV/CPCV/DSR/PBO, HRP/Black-Litterman/CVXPY convex optimization under transaction costs, regime detection, and live Alpaca paper trading. Delivers verifiable risk and cost improvements even when raw return alpha is near zero. |
| **2** | **Asset Universe (Phase 1–2)** | **North American (US) Stock Market (`XNYS`/`XNAS`) — Two-Tier Universe** | Limits scope and cuts cross-currency/multi-timezone noise. **Tier 1:** 14 liquid US Market & Sector ETFs (`SPY`, `QQQ`, `IWM`, 11 Select Sector SPDRs) as a survivorship-free core. **Tier 2:** 50–100 liquid US equities for high-dimensional RMT denoising, factor models, corporate actions, and survivorship-bias bounding (with optional UAlberta WRDS/CRSP access). |
| **3** | **Data Frequency** | **Daily bars (`1D`) for Phase 1–2 core; Intraday/L2 scoped to microstructure** | Aligns with 30+ years of free EOD history (Tiingo, FRED/ALFRED, Ken French), keeps CPCV sweeps fast within the 8 GB M2 memory budget, and isolates L2 tick complexity to Notebook 11 and Phase 3 Track B/G. |
| **4** | **Paper-Trading Broker** | **Alpaca Paper Trading API (`alpaca-py`)** | Stateless REST/WebSocket API that runs cleanly in Docker/cron using environment keys, plus free historical consolidated SIP data (`feed="sip"`, older than 15 min), avoiding IBKR's local Java gateway and weekly 2FA friction. |
| **5** | **Rust / C++ Component** | **Defer Rust (`PyO3` + `maturin`) to Phase 3 Track G; keep Phase 1–2 100% Python** | Phase 1–2 numerical heavy lifting is already C/C++/Fortran-backed via NumPy, Polars, DuckDB, Clarabel/OSQP (`cvxpy`), and PyTorch. Rust is reserved for Phase 3's high-throughput L2/L3 LOB matching engine and event simulator. |
| **6** | **Repo Visibility** | **Public from Day 1 with strict pre-commit & `.gitignore` guardrails** | Maximizes portfolio transparency and engineering accountability while protecting credentials and respecting vendor non-redistribution terms (CI uses synthetic fixtures only). |
| **7** | **Experiment Tracking** | **Git-tracked `experiments/ledger.jsonl` + `experiments/trial_returns.parquet`** | Zero external server state; syncs natively via Git between the macOS M2 and Windows RTX 3080 Ti machines; stores the full $T \times K$ out-of-sample trial return matrix required for CSCV/PBO and effective independent trial count $\hat{N}_{\text{eff}}$ in DSR. |

---

## 10. Glossary of Phase Names (for README)

- **Phase 1 — Foundations:** learn it, build it small, see it break.
- **Phase 2 — Production:** build it right, prove it honestly, ship one thing.
- **Phase 3 — Frontier:** keep asking better questions, with machines that are held to the same standard as the humans.
