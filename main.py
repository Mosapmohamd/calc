from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import t

# =========================
# CONFIG
# =========================
CSV_PATH = "test.csv"
MILEAGE_RATE = 0.10

app = FastAPI(title="Vehicle Fair Value API")

# =========================
# DATA LOADING
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    df = df.rename(columns={
        "Year": "year",
        "Make": "make",
        "Model": "model",
        "Trim": "trim",
        "Odometer": "odometer",
        "Price": "price",
        "Sale Date": "sale_date",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["make"] = df["make"].astype(str).str.upper().str.strip()
    df["model"] = df["model"].astype(str).str.upper().str.strip()
    df["trim"] = df["trim"].astype(str).str.upper().str.strip()

    df = df.dropna(subset=["year", "make", "model", "odometer", "price"])
    return df.reset_index(drop=True)

df = load_data(CSV_PATH)

# =========================
# REQUEST MODELS
# =========================
class EstimateRequest(BaseModel):
    year: int
    make: str
    model: str
    trim: Optional[str] = None
    odometer: int
    confidence: float = 0.9


class BatchEstimateRequest(BaseModel):
    vehicles: List[EstimateRequest]

# =========================
# CORE LOGIC
# =========================
def train_regression(sub_df: pd.DataFrame):
    X = sub_df[["odometer"]].values
    y = sub_df["price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    residuals = y - model.predict(X_scaled)
    sigma = np.std(residuals, ddof=1)

    return model, scaler, sigma

def estimate_value(payload: EstimateRequest):
    year = payload.year
    make = payload.make.upper().strip()
    model = payload.model.upper().strip()
    trim = payload.trim.upper().strip() if payload.trim else None
    confidence = min(max(payload.confidence, 0.5), 0.99)

    comps = df[
        (df.year == year) &
        (df.make == make) &
        (df.model == model)
    ]

    base_comps = comps.copy()

    if trim:
        comps = comps[comps.trim.str.contains(trim, regex=False, na=False)]

    if comps.empty:
        comps = base_comps

    n = len(comps)
    title = f"{year} {make} {model}" + (f" {trim}" if trim else "")

    if n == 0:
        return {
            "title": title,
            "price": None,
            "low": None,
            "high": None,
            "comparables": 0,
            "note": "No comparable vehicles found"
        }

    if n == 1:
        base = comps.iloc[0]
        km_diff = payload.odometer - base.odometer
        adjusted_price = base.price - (km_diff * MILEAGE_RATE)

        return {
            "title": title,
            "price": round(adjusted_price, 0),
            "low": None,
            "high": None,
            "comparables": 1,
            "note": "Single comparable. Mileage adjusted"
        }

    if n == 2:
        mean_price = comps.price.mean()
        return {
            "title": title,
            "price": round(mean_price, 0),
            "low": None,
            "high": None,
            "comparables": 2,
            "note": "Two comparables. Average price"
        }

    lr, scaler, sigma = train_regression(comps)
    X_new = scaler.transform([[payload.odometer]])
    pred = lr.predict(X_new)[0]

    if not np.isfinite(sigma) or sigma == 0:
        return {
            "title": title,
            "price": round(pred, 0),
            "low": None,
            "high": None,
            "comparables": n,
            "note": "Insufficient variance"
        }

    t_val = t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * sigma

    return {
        "title": title,
        "price": round(pred, 0),
        "low": round(pred - margin, 0),
        "high": round(pred + margin, 0),
        "comparables": n,
        "note": None
    }

# =========================
# ROUTES
# =========================
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/estimate")
def estimate(payload: EstimateRequest):
    return estimate_value(payload)

@app.post("/estimate/batch")
def estimate_batch(payload: BatchEstimateRequest):
    results = []
    for vehicle in payload.vehicles:
        results.append(estimate_value(vehicle))
    return results
