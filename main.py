from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import t
from fuzzywuzzy import fuzz, process

CSV_PATH = "test.csv"
FUZZY_THRESHOLD = 80
MIN_SAMPLES = 2
YEAR_DELTA = 2
MILEAGE_RATE = 0.1
CONFIDENCE = 0.9

app = FastAPI(title="Vehicle Price Prediction API")

# =========================
# CLEANING
# =========================
def clean(s: Optional[str]) -> Optional[str]:
    if s is None or pd.isna(s):
        return None
    s = str(s).lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s if s else None

def match_make_smart(value: Optional[str], makes: List[str]):
    if not value or not makes:
        return None
    val = clean(value)
    for m in makes:
        if val == m:
            return m
    for m in makes:
        if val in m or m in val:
            return m
    hit = process.extractOne(val, makes, scorer=fuzz.partial_ratio)
    if hit and hit[1] >= FUZZY_THRESHOLD:
        return hit[0]
    return None

def match_model_smart(value: Optional[str], models: List[str]):
    if not value or not models:
        return None
    val = clean(value)
    for m in models:
        if val == m:
            return m
    for m in models:
        if val in m or m in val:
            return m
    hit = process.extractOne(val, models, scorer=fuzz.partial_ratio)
    if hit and hit[1] >= FUZZY_THRESHOLD:
        return hit[0]
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
    for c in ["year", "odometer", "price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["make", "model", "trim", "province"]:
        df[c] = df[c].apply(clean)
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
    odometer: int
    trim: Optional[str] = None
    province: Optional[str] = None

# =========================
# WEIGHTED REGRESSION
# =========================
def train_weighted_regression(comps: pd.DataFrame, target_year: int):
    X = comps[["odometer"]].values
    y = comps.price.values
    year_diff = (comps.year - target_year).abs()
    weights = 1 / (1 + year_diff)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    lr = LinearRegression()
    lr.fit(Xs, y, sample_weight=weights)
    preds = lr.predict(Xs)
    residuals = y - preds
    sigma = residuals.std(ddof=1) if len(residuals) > 1 else np.nan
    return lr, scaler, sigma

# =========================
# CORE LOGIC
# =========================
def estimate_value(p: EstimateRequest):
    make = match_make_smart(p.make, df.make.unique().tolist())
    if not make:
        return {"status": "error", "note": "make not found"}

    model = match_model_smart(
        p.model,
        df[df.make == make].model.unique().tolist()
    )
    if not model:
        return {"status": "error", "note": "model not found"}

    title = f"{p.year} {make.title()} {model.title()}"

    comps = df[
        (df.make == make) &
        (df.model == model) &
        (df.year >= p.year - YEAR_DELTA) &
        (df.year <= p.year + YEAR_DELTA)
    ]

    trim = clean(p.trim)
    if trim and "trim" in comps.columns:
        similar = comps[comps.trim.apply(
            lambda x: fuzz.partial_ratio(trim, x) >= FUZZY_THRESHOLD if isinstance(x, str) else False
        )]
        if len(similar) >= MIN_SAMPLES:
            comps = similar

    n = len(comps)

    if n == 1:
        base = comps.iloc[0]
        adj = base.price - (p.odometer - base.odometer) * MILEAGE_RATE
        return {
            "status": "success",
            "mode": "single_comparable",
            "title": title,
            "price": round(max(adj, 0), 0),
            "comparables": 1,
            "used_years": [int(base.year)]
        }

    if n == 2:
        base_price = comps.price.mean()
        base_odo = comps.odometer.mean()
        adj = base_price - (p.odometer - base_odo) * MILEAGE_RATE
        return {
            "status": "success",
            "mode": "two_comparables_adjusted",
            "title": title,
            "price": round(max(adj, 0), 0),
            "comparables": 2,
            "used_years": sorted(comps.year.unique().tolist())
        }

    if n >= 3:
        lr, scaler, sigma = train_weighted_regression(comps, p.year)
        pred = lr.predict(scaler.transform([[p.odometer]]))[0]
        if pred <= 0:
            pred = comps.price.median()
        if not np.isfinite(sigma):
            return {
                "status": "success",
                "mode": "weighted_regression_low_variance",
                "title": title,
                "price": round(pred, 0),
                "comparables": n,
                "used_years": sorted(comps.year.unique().tolist())
            }
        t_val = t.ppf((1 + CONFIDENCE) / 2, df=n - 1)
        margin = t_val * sigma
        return {
            "status": "success",
            "mode": "weighted_regression",
            "title": title,
            "price": round(pred, 0),
            "low": round(max(pred - margin, 0), 0),
            "high": round(pred + margin, 0),
            "comparables": n,
            "used_years": sorted(comps.year.unique().tolist())
        }

    all_data = df[(df.make == make) & (df.model == model)]
    if len(all_data) < MIN_SAMPLES:
        return {"status": "no_comparables", "mode": "none"}

    lr, scaler, _ = train_weighted_regression(all_data, p.year)
    price = lr.predict(scaler.transform([[p.odometer]]))[0]
    if price <= 0:
        price = all_data.price.median()

    return {
        "status": "success",
        "mode": "ml_fallback_weighted",
        "title": title,
        "price": round(price, 0),
        "comparables": len(all_data),
        "used_years": sorted(all_data.year.unique().tolist())
    }

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {"status": "ok", "records": len(df)}

@app.post("/estimate")
def estimate(p: EstimateRequest):
    return estimate_value(p)

# accepts LIST directly
@app.post("/estimate/batch")
def estimate_batch(vehicles: List[EstimateRequest]):
    return [estimate_value(v) for v in vehicles]
