# portfolio_main.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load .env variables (local development)
load_dotenv()

# ---------- CONFIG ----------
# No default values for sensitive info
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432"))   # port can have a default
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")

ASSET_TYPES = ("EQUITY", "INDEX", "MUTUALFUND")
MIN_HISTORY_DAYS = 252

START_DATE = None
END_DATE   = None

CONF_LEVEL = 0.95


def compute_portfolio_daily_returns(
    asset_types=ASSET_TYPES,
    min_history_days=MIN_HISTORY_DAYS,
    start_date=START_DATE,
    end_date=END_DATE,
    conf_level=CONF_LEVEL,
):

    # ---------- CONNECT ----------
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True,
    )

    # ---------- 1) UNIVERSE ----------
    with engine.begin() as conn:
        sec = pd.read_sql(
            text("""
                SELECT symbol, name, asset_type, currency
                FROM security
                WHERE asset_type = ANY(:asset_types)
            """),
            conn,
            params={"asset_types": list(asset_types)},
        )

    if sec.empty:
        raise SystemExit("No securities for chosen ASSET_TYPES.")

    symbols = sec["symbol"].unique().tolist()

    # ---------- 2) PRICES ----------
    with engine.begin() as conn:
        query = """
            SELECT symbol, date, close_price
            FROM security_prices
            WHERE symbol = ANY(:syms)
        """
        if start_date:
            query += " AND date >= :start_date"
        if end_date:
            query += " AND date <= :end_date"
        query += " ORDER BY date"

        prices_raw = pd.read_sql(
            text(query),
            conn,
            params={"syms": symbols, "start_date": start_date, "end_date": end_date},
            parse_dates=["date"],
        )

    if prices_raw.empty:
        raise SystemExit("No price history found for the universe/date filters.")

    prices = prices_raw.pivot(index="date", columns="symbol", values="close_price").sort_index()
    prices = prices.ffill().dropna(how="any")

    # Keep assets with enough history
    enough = prices.count() >= min_history_days
    prices = prices.loc[:, enough]

    if prices.shape[1] == 0:
        raise SystemExit(f"No symbols with ≥ {min_history_days} days after cleaning.")

    symbols = prices.columns.tolist()

    # ---------- 4) METRICS + OPTIMIZATION ----------
    returns = prices.pct_change().dropna()
    mu = returns.mean().values * 252
    Sigma = np.cov(returns.values, rowvar=False) * 252
    n = len(mu)

    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - 5.0 * cp.quad_form(w, Sigma))  # gamma=5
    constraints = [cp.sum(w) == 1, w >= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: {prob.status}")

    weights = np.maximum(w.value, 0)
    weights = weights / weights.sum()
    w_series = pd.Series(weights, index=returns.columns)

    port_daily = (returns * w_series).sum(axis=1)
    return port_daily


# ---- RUN ONLY IF EXECUTED DIRECTLY ----
if __name__ == "__main__":
    port_daily = compute_portfolio_daily_returns()
    losses = -port_daily

    VaR = np.quantile(losses, CONF_LEVEL)
    ES = losses[losses >= VaR].mean()

    print("\n📉 Daily Risk (Historical):")
    print(f"VaR 95% = {(-VaR)*100:.2f}%")
    print(f"ES  95% = {(-ES)*100:.2f}%")
    print(f"Sample size: {len(port_daily)} days")

    # Optional plots
    plt.figure(figsize=(10,6))
    plt.hist(port_daily, bins=50, density=True, alpha=0.6)
    plt.axvline(-VaR, linestyle="--", linewidth=2, label=f"VaR={(-VaR):.3%}")
    plt.axvline(-ES,  linestyle="-.", linewidth=2, label=f"ES={(-ES):.3%}")
    plt.title("Portfolio Daily Returns with VaR/ES")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
