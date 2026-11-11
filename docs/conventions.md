# Mathematical, Data, and Engineering Conventions (`docs/conventions.md`)

Every module, notebook, and test in `quant-lab` must adhere to the conventions defined below. Any deviation requires an explicit docstring note and an Architecture Decision Record (ADR).

---

## 1. Market Scope, Calendar, and Timezone

1. **Market Scope:** North American (US) Stock Market (NYSE / NASDAQ equities and ETFs).
2. **Primary Exchange Calendar:** `XNYS` (New York Stock Exchange) via `exchange_calendars`.
   - Regular trading session: `09:30:00` to `16:00:00 America/New_York` (early closes at `13:00:00 America/New_York` on designated half-days).
3. **Internal Timestamp Standard:**
   - All timestamps in storage (Parquet) and memory are **timezone-aware UTC (`datetime64[ns, UTC]`)**.
   - Every dataset record carries **two timestamps**:
     - `period_ts`: The nominal bar date or observation period (normalized to midnight UTC `00:00:00Z` for daily bars, or bar close timestamp for intraday bars).
     - `available_ts`: The exact UTC timestamp when the datum became knowable to a market participant:
       - For US daily equity/ETF bars (`1D`): `16:00:00 America/New_York` converted to UTC (`21:00:00Z` during EST, `20:00:00Z` during EDT).
       - For FRED/ALFRED macroeconomic series: the archival release date (`realtime_start` at `13:30:00Z` / `08:30:00 America/New_York`).
4. **Point-in-Time (PIT) Filtering Rule:**
   - At decision time $t$, any query to the data store must satisfy `available_ts <= t`.

---

## 2. Prices, Corporate Actions, and Returns

1. **Units:**
   - **Prices & Dividends:** US Dollars (`USD`), `float64`.
   - **Returns, Volatility, Weights, Interest Rates:** Decimal fractions (`0.01` means $1\%$, **never** `1.0`). External sources that report in percentage points (such as the Kenneth French Data Library or FRED yields) must be divided by `100.0` immediately upon ingestion.
2. **Raw Prices vs. Corporate Actions:**
   - Primary price storage uses **unadjusted (raw) OHLCV** (`open`, `high`, `low`, `close`, `volume`) so that historical content hashes are immutable.
   - Splits and dividends are stored in a companion **Corporate Actions table** with schema `(ticker, ex_date, split_factor, div_cash)`:
     - `split_factor`: Ratio of new shares to old shares on `ex_date` (e.g., `2.0` for a 2-for-1 split; `1.0` on normal days).
     - `div_cash`: Cash dividend per pre-split share going ex-dividend on `ex_date` (`0.0` on normal days).
3. **Return Definitions:**
   - **Simple Price Return (no corporate action on day $t$):**
     $$R_t = \frac{P_t}{P_{t-1}} - 1$$
   - **Total Simple Return (accounting for split $S_t$ and cash dividend $D_t$ on ex-date $t$):**
     $$R_t^{\text{tot}} = \frac{P_t^{\text{raw}} \cdot S_t + D_t}{P_{t-1}^{\text{raw}}} - 1$$
     *(Note: check vendor convention on `split_factor` definition; in `quantlab.data.adjustments`, `split_factor = 2.0` means 1 share at $t-1$ becomes 2 shares at $t$, each trading at roughly $\frac{1}{2} P_{t-1}^{\text{raw}}$).*
   - **Log (Continuously Compounded) Return:**
     $$r_t = \ln(1 + R_t) \quad \Longleftrightarrow \quad R_t = \exp(r_t) - 1$$
   - **Cross-Sectional vs. Time-Series Aggregations:**
     - Log returns add across **time**: $r_{t_0 \to t_k} = \sum_{s=1}^k r_{t_s}$.
     - Simple returns aggregate across **assets (portfolio weights)** at a single period $t$: $R_{p,t} = \sum_{i=1}^N w_{i,t-1} R_{i,t}$.
     - **Never** take a weighted sum of log returns across assets to compute portfolio return.

---

## 3. Signal-to-Fill Execution Lag & Shift Conventions

1. **Canonical Daily Timeline:**
   - At the close of trading day $t$ (`available_ts[t]`), the strategy observes information set $\mathcal{F}_t$ (which includes `close[t]`) and computes target portfolio weights $w_t = f(\mathcal{F}_t)$.
   - Those target weights $w_t$ are held over the interval $(t, t+1]$ and earn the **next-period return** $R_{t+1}$:
     - *Close-to-Close convention (with 1-bar signal lag):* $w_t$ is formed at `close[t]` and earns $R_{t+1}$ (or executes at `open[t+1]` earning open-to-open / open-to-close return, with explicit transaction cost deducted on weight change $\Delta w_t = w_t - w_{t-1}^{\text{drifted}}$).
   - **Forbidden Look-Ahead:** Forming $w_t$ from `close[t]` and applying it to $R_t = \frac{P_t}{P_{t-1}} - 1$ is contemporaneous look-ahead bias and must fail the leakage test suite.
2. **Shift Operator Notation:**
   - `shift(+1)` (Positive Lag / Delay): Moves a time series 1 step into the future ($x^{\text{lag}}_t = x_{t-1}$). This is Causal.
   - `shift(-1)` (Negative Lag / Look-Ahead Probe): Peeks 1 step into the future ($x^{\text{peek}}_t = x_{t+1}$). Used **only** inside `validation/leakage.py` to test leakage detection.

---

## 4. Annualization & Performance Metrics

1. **Annualization Factor:**
   - Daily US equity series use $A = 252$ trading days per year.
   - Annualized arithmetic mean excess return: $\mu_{\text{ann}} = 252 \cdot \bar{R}^e$.
   - Annualized compound growth rate (CAGR): $\text{CAGR} = \left(\prod_{t=1}^T (1 + R_t)\right)^{252 / T} - 1$.
   - Annualized volatility: $\sigma_{\text{ann}} = \sqrt{252} \cdot s_R$ (using sample standard deviation with `ddof=1`).
   - Annualized Sharpe Ratio: $\text{SR} = \frac{\bar{R}^e}{s_{R^e}} \sqrt{252}$, where $R^e_t = R_t - \frac{r_{f,t}}{252}$ and $r_{f,t}$ is the annualized risk-free rate known at $t-1$.
2. **Turnover:**
   - One-way turnover at rebalance $t$: $\tau_t = \frac{1}{2} \sum_{i=1}^N \left| w_{i,t} - w_{i,t-1}^+ \right|$, where $w_{i,t-1}^+ = \frac{w_{i,t-1}(1 + R_{i,t})}{1 + R_{p,t}}$ is the pre-trade drifted weight at the end of period $t$.

---

## 5. Array Shapes, Data Types, and Memory Conventions

1. **Standard Panel Shape:**
   - 2D price, return, and weight arrays have shape `(T, N)` where:
     - `axis=0` ($T$) is strictly chronological time (`period_ts`).
     - `axis=1` ($N$) is the asset cross-section (`ticker`, sorted lexicographically unless specified).
   - Covariance and correlation matrices have shape `(N, N)` and must be symmetric positive semi-definite (PSD within numerical tolerance $\epsilon = 10^{-10}$).
2. **Data Types:**
   - All numerical arrays in `core/`, `pricing/`, `portfolio/`, and `validation/` use `np.float64` (`dtype=np.float64`).
   - Deep learning tensors in PyTorch on MPS (`macOS M2`) or CUDA (`RTX 3080 Ti`) use `torch.float32` (since Apple MPS does not support `float64` natively).
