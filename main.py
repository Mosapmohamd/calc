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

CSV_PATH = "test.csv"
MILEAGE_RATE = 0.10

app = FastAPI(title="Vehicle Fair Value API")

# =========================
# HELPERS
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
# LOAD DATA
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
        "Province": "province",
    })

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    for c in ["make", "model", "trim", "province"]:
        df[c] = df[c].astype(str).str.upper().str.strip()

    return df.dropna(subset=["year", "make", "model", "odometer", "price", "province"])

df = load_data(CSV_PATH)

VALID_MAKES = set(df.make.unique())
VALID_MODELS = set(df.model.unique())
VALID_PROVINCES = set(df.province.unique())

TRIM_MAP = defaultdict(set)
for _, r in df.iterrows():
    TRIM_MAP[(r.make, r.model)].add(r.trim)

# =========================
# REQUEST MODELS
# =========================
class EstimateRequest(BaseModel):
    year: int
    make: str
    model: str
    trim: Optional[str] = None
    province: Optional[str] = None
    odometer: int
    confidence: float = 0.9

class BatchEstimateRequest(BaseModel):
    vehicles: List[EstimateRequest]

# =========================
# TRIM
# =========================
def match_trim(trim: Optional[str], make: str, model: str) -> Optional[str]:
    if not trim:
        return None
    trim = norm(trim)
    return trim if trim in TRIM_MAP.get((make, model), set()) else None

# =========================
# REGRESSION WITH PROVINCE
# =========================
def train_regression(sub_df: pd.DataFrame):
    X = sub_df[["odometer", "province"]]

    X = pd.get_dummies(X, columns=["province"], drop_first=True)

    y = sub_df["price"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    model.fit(X_scaled, y)

    residuals = y - model.predict(X_scaled)
    sigma = np.std(residuals, ddof=1)

    return model, scaler, sigma, X.columns

# =========================
# CORE LOGIC
# =========================
def estimate_value(p: EstimateRequest):
    confidence = min(max(p.confidence, 0.5), 0.99)

    make = match_value(p.make, VALID_MAKES)
    model = match_value(p.model, VALID_MODELS)
    province = match_value(p.province, VALID_PROVINCES)

    if not make or not model:
        return {
            "title": f"{p.year} {p.make} {p.model}",
            "price": None,
            "comparables": 0,
            "note": "Make or model not found"
        }

    comps = df[
        (df.year == p.year) &
        (df.make == make) &
        (df.model == model)
    ]

    trim = match_trim(p.trim, make, model)
    if trim:
        comps = comps[comps.trim == trim]

    n = len(comps)

    title = f"{p.year} {make} {model}"
    if trim:
        title += f" {trim}"
    if province:
        title += f" ({province})"

    if n == 0:
        return {
            "title": title,
            "price": None,
            "comparables": 0,
            "note": "No comparable vehicles found"
        }

    if n < 3:
        return {
            "title": title,
            "price": round(comps.price.mean(), 0),
            "comparables": n,
            "note": "Not enough data for regression"
        }

    lr, scaler, sigma, cols = train_regression(comps)

    input_row = pd.DataFrame([{
        "odometer": p.odometer,
        "province": province if province else comps.province.mode()[0]
    }])

    input_row = pd.get_dummies(input_row, columns=["province"])
    input_row = input_row.reindex(columns=cols, fill_value=0)

    pred = lr.predict(scaler.transform(input_row))[0]

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
