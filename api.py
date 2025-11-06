from fastapi import FastAPI
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd

app = FastAPI()

# Import or wrap your existing logic -------------------
from main import compute_portfolio_daily_returns  # you create this

CONF_LEVEL = 0.95

@app.get("/risk")
def get_daily_risk():
    """
    Returns VaR, ES and sample size as JSON.
    """

    # This function should return a pandas Series of portfolio daily returns
    port_daily = compute_portfolio_daily_returns()   # <— you plug your logic here

    losses = -port_daily
    VaR = np.quantile(losses, CONF_LEVEL)
    ES = losses[losses >= VaR].mean()
    sample = len(port_daily)

    result = {
        "VaR_95": float(-VaR),
        "ES_95": float(-ES),
        "SampleSize": int(sample)
    }
    return JSONResponse(content=result)



@app.get("/")
def root():
    return {"message": "API is running!"}
