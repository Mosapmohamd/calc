from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
def load_data(path):
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

    df = df.dropna(
        subset=["year", "make", "model", "trim", "odometer", "price"]
    )

    return df

df = load_data(CSV_PATH)

# =========================
# MODELS
# =========================
class EstimateRequest(BaseModel):
    year: int
    make: str
    model: str
    trim: str
    odometer: int
    confidence: float = 0.9

# =========================
# LOGIC
# =========================
def train_regression(sub_df):
    X = sub_df[["odometer"]].values
    y = sub_df["price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    residuals = y - model.predict(X_scaled)
    sigma = np.std(residuals, ddof=1)

    return model, scaler, sigma, len(sub_df)

def estimate_value(payload: EstimateRequest):
    comps = df[
        (df.year == payload.year) &
        (df.make == payload.make.upper()) &
        (df.model == payload.model.upper())
    ]
    if payload.trim:
    comps = comps[df.trim.str.contains(payload.trim.upper(), na=False)]
    n = len(comps)

    if n == 0:
        raise HTTPException(status_code=404, detail="No comparable vehicles found")

    if n == 1:
        base = comps.iloc[0]
        km_diff = payload.odometer - base.odometer
        adjusted_price = base.price - (km_diff * MILEAGE_RATE)

        return {
            "price": round(adjusted_price, 0),
            "low": None,
            "high": None,
            "comparables": 1,
            "note": "Single comparable. Mileage adjusted."
        }

    if n == 2:
        return {
            "price": round(comps.price.mean(), 0),
            "low": None,
            "high": None,
            "comparables": 2,
            "note": "Two comparables. Average price."
        }

    lr, scaler, sigma, n = train_regression(comps)
    X_new = scaler.transform([[payload.odometer]])
    pred = lr.predict(X_new)[0]

    if not np.isfinite(sigma):
        return {
            "price": round(pred, 0),
            "low": None,
            "high": None,
            "comparables": n,
            "note": "Insufficient variance."
        }

    t_val = t.ppf((1 + payload.confidence) / 2, df=n - 1)
    margin = t_val * sigma

    return {
        "price": round(pred, 0),
        "low": round(pred - margin, 0),
        "high": round(pred + margin, 0),
        "comparables": n,
        "note": None
    }
    class EstimateRequest(BaseModel):
    year: int
    make: str
    model: str
    trim: str | None = None
    odometer: int
    confidence: float = 0.9
# =========================
# ROUTES
# =========================
@app.get("/")
def health_check():
    return {"status": "ok"}

@app.post("/estimate")
def estimate(payload: EstimateRequest):
    return estimate_value(payload)

