# Task: 0011 — Price-Time Priority Limit Order Book (`backtest/orderbook.py` & Notebook 11)

## Goal
Implement a self-contained price-time priority (FIFO per price level) limit order book matching engine (`src/quantlab/backtest/orderbook.py`) supporting limit, market, and cancel order flow, bid-ask spread, midprice/microprice, and order book imbalance analytics, alongside Notebook 11 (`11_market_microstructure_lob.ipynb`).

## Context / Links
- Design doc section: [QUANT_PROJECT_DESIGN.md](../../QUANT_PROJECT_DESIGN.md) §1.2 (Notebook 11), §1.4 (`backtest/orderbook.py`)
- Related ADRs: [ADR-0001](../adr/0001-initial-architecture-decisions.md)
- Mathematical conventions: [docs/conventions.md](../conventions.md) §5 (`np.float64` / integer tick prices)

## Inputs
- Python `collections.deque` (FIFO queue per price level) and `bisect` / `heapq` (sorted price levels)
- Synthetic Poisson/LOBSTER-style order flow fixture generator

## Outputs (Files & APIs)
- `src/quantlab/backtest/orderbook.py` (exported via `src/quantlab/backtest/__init__.py`):
  - `Side` (`"bid"` / `"ask"`), `Order`, `Fill`, `BookSnapshot` dataclasses
  - `LimitOrderBook`:
    - `submit_limit(order_id, side, price_ticks, qty, timestamp_ns) -> list[Fill]` (crossing limit orders execute immediately against passive resting liquidity; residual qty rests FIFO)
    - `submit_market(order_id, side, qty, timestamp_ns) -> list[Fill]`
    - `cancel_order(order_id) -> bool`
    - Properties/methods: `best_bid()`, `best_ask()`, `spread()`, `midprice()`, `microprice()`, `imbalance(levels=1)`, `snapshot(depth=5)`
  - `replay_message_stream(messages) -> list[BookSnapshot]`
- `tests/test_orderbook.py`
- `notebooks/phase1/11_market_microstructure_lob.ipynb` (registered in `tests/test_notebooks.py`)

## Acceptance Tests (Concrete, Runnable)
- [ ] **Known-answer FIFO & crossing tests:**
  - Hand-verified sequence of bids, asks, partial fills, full level sweeps, and middle-of-queue cancellations matches expected `Fill` prices/quantities, best bid/ask, spread, microprice, and remaining book state exactly.
  - Strict FIFO time priority verified: two limit orders at the same price level fill strictly in arrival timestamp order.
- [ ] **Property / invariant tests (`hypothesis`):**
  - Book is never crossed after any sequence of valid limit/market/cancel events (`best_bid < best_ask` whenever both sides are non-empty).
  - Total traded buy volume equals total traded sell volume across all fills; open interest + filled + canceled == total submitted limit quantity.
  - Replay determinism: identical message sequences always yield bitwise identical `Fill` and `BookSnapshot` sequences.
- [ ] Passes `.venv/bin/ruff format --check .`, `.venv/bin/ruff check .`, `.venv/bin/mypy src tests`, and `.venv/bin/pytest --cov=quantlab --cov-report=term-missing`.

## Out of Scope
- Rust `PyO3` / `maturin` high-throughput LOB engine and live NASDAQ TotalView-ITCH `.dbn` binary replay (Phase 3 Track B & Track G).

## Risks / Assumptions to Verify
- Floating-point rounding errors on price level keys: represent price levels strictly as integer ticks (`price_ticks: int`) internally and convert to USD via a configurable `tick_size: float = 0.01`.

## Phase & Rigor Level
- **Phase:** Phase 1 (Foundations — self-contained microstructure module)
- **Coverage target:** ≥ 95% on `src/quantlab/backtest/orderbook.py`
- **Type checking:** `mypy --strict`
