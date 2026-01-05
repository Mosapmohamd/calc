from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
import re
from collections import defaultdict

# =========================
# CONFIG
# =========================
CSV_PATH = "test.csv"
MILEAGE_RATE = 0.10

app = FastAPI(title="Vehicle Fair Value API")

# =========================
# NORMALIZATION HELPERS
# =========================
def norm(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    return re.sub(r"[^A-Z0-9\- ]", "", s.upper()).strip()

def match_value(value: Optional[str], valid_set: set[str]) -> Optional[str]:
    if not value:
        return None

    value = norm(value)
    if value in valid_set:
        return value

    for v in valid_set:
        if value in v or v in value:
            return v

    return None

# =========================
# LOAD CSV
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Year": "year",
        "Make": "make",
        "Model": "model",
        "Trim": "trim",
        "Odometer": "odometer",
        "Price": "price",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    df["make"] = df["make"].astype(str).str.upper().str.strip()
    df["model"] = df["model"].astype(str).str.upper().str.strip()
    df["trim"] = df["trim"].astype(str).str.upper().str.strip()

    return df.dropna(subset=["year", "make", "model", "odometer", "price"])

df = load_data(CSV_PATH)

# =========================
# CSV-DRIVEN LOOKUPS
# =========================
VALID_MAKES = set(df.make.unique())
VALID_MODELS = set(df.model.unique())

TRIM_MAP = defaultdict(set)
for _, row in df.iterrows():
    TRIM_MAP[(row.make, row.model)].add(row.trim)

print(f"Loaded {len(df)} rows")

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
# TRIM VALIDATION (STRICT)
# =========================
def match_trim(trim: Optional[str], make: str, model: str) -> Optional[str]:
    if not trim:
        return None

    trim = norm(trim)
    valid_trims = TRIM_MAP.get((make, model), set())

    # EXACT MATCH ONLY
    return trim if trim in valid_trims else None

# =========================
# REGRESSION
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

    return model, scaler, sigma

# =========================
# CORE LOGIC
# =========================
def estimate_value(p: EstimateRequest):
    confidence = min(max(p.confidence, 0.5), 0.99)

    make = match_value(p.make, VALID_MAKES)
    model = match_value(p.model, VALID_MODELS)

    if not make or not model:
        return {
            "title": f"{p.year} {p.make} {p.model}",
            "price": None,
            "comparables": 0,
            "note": "Make or model not found"
        }

    trim = match_trim(p.trim, make, model)

    comps = df[
        (df.year == p.year) &
        (df.make == make) &
        (df.model == model)
    ]

    base_comps = comps.copy()

    if trim:
        comps = comps[comps.trim == trim]

    if comps.empty:
        comps = base_comps

    n = len(comps)
    title = f"{p.year} {make} {model}" + (f" {trim}" if trim else "")

    if n == 0:
        return {
            "title": title,
            "price": None,
            "comparables": 0,
            "note": "No comparable vehicles found"
        }

    if n == 1:
        base = comps.iloc[0]
        adj = base.price - (p.odometer - base.odometer) * MILEAGE_RATE
        return {
            "title": title,
            "price": round(adj, 0),
            "comparables": 1,
            "note": "Single comparable"
        }

    if n == 2:
        return {
            "title": title,
            "price": round(comps.price.mean(), 0),
            "comparables": 2,
            "note": "Average of two"
        }

    lr, scaler, sigma = train_regression(comps)
    pred = lr.predict(scaler.transform([[p.odometer]]))[0]

    if not np.isfinite(sigma):
        return {
            "title": title,
            "price": round(pred, 0),
            "comparables": n,
            "note": "Low variance"
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
def health():
    return {"status": "ok"}

@app.post("/estimate")
def estimate(p: EstimateRequest):
    return estimate_value(p)

@app.post("/estimate/batch")
def estimate_batch(p: BatchEstimateRequest):
    return [estimate_value(v) for v in p.vehicles]
