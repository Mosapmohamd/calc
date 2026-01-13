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
MIN_MODEL_YEAR = 2010
HYBRID_BONUS = 3000

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

def is_hybrid(p: BaseModel) -> bool:
    text = clean(f"{p.make} {p.model} {p.trim or ''}")
    if not text:
        return False
    return "hybrid" in text

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
# REQUEST MODEL
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

    if p.year < MIN_MODEL_YEAR:
        return {
            "status": "not_supported",
            "title": f"{p.year} {p.make} {p.model}",
            "price": None,
            "note": "Model year below supported range"
        }

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

    def apply_hybrid(price: float) -> float:
        return price + HYBRID_BONUS if is_hybrid(p) else price

    if n == 0:
        return {
            "status": "no_comparables",
            "title": title,
            "price": None,
            "comparables": 0
        }

    if n == 1:
        base = comps.iloc[0]
        adj = base.price - (p.odometer - base.odometer) * MILEAGE_RATE
        price = apply_hybrid(max(adj, 0))
        return {
            "status": "success",
            "mode": "single_comparable",
            "title": title,
            "price": round(price, 0),
            "comparables": 1,
            "used_years": [int(base.year)]
        }

    if n == 2:
        base_price = comps.price.mean()
        base_odo = comps.odometer.mean()
        adj = base_price - (p.odometer - base_odo) * MILEAGE_RATE
        price = apply_hybrid(max(adj, 0))
        return {
            "status": "success",
            "mode": "two_comparables_adjusted",
            "title": title,
            "price": round(price, 0),
            "comparables": 2,
            "used_years": sorted(comps.year.unique().tolist())
        }

    lr, scaler, sigma = train_weighted_regression(comps, p.year)
    pred = lr.predict(scaler.transform([[p.odometer]]))[0]
    if pred <= 0:
        pred = comps.price.median()

    pred = apply_hybrid(pred)

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

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {"status": "ok", "records": len(df)}

@app.post("/estimate")
def estimate(p: EstimateRequest):
    return estimate_value(p)

@app.post("/estimate/batch")
def estimate_batch(vehicles: List[EstimateRequest]):
    return [estimate_value(v) for v in vehicles]
