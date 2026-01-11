from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from fuzzywuzzy import fuzz, process

CSV_PATH = "test.csv"
FUZZY_THRESHOLD = 80
MIN_SAMPLES = 2
YEAR_DELTA = 2

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

def fuzzy_match(value: Optional[str], choices: List[str], threshold=80):
    if not value or not choices:
        return None
    match = process.extractOne(value, choices, scorer=fuzz.ratio)
    if match and match[1] >= threshold:
        return match[0]
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
    df = df.reset_index(drop=True)
    return df

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
    confidence: Optional[float] = 0.9

class BatchEstimateRequest(BaseModel):
    vehicles: List[EstimateRequest]

# =========================
# CORE LOGIC
# =========================
def estimate_value(p: EstimateRequest):
    make = fuzzy_match(clean(p.make), df.make.unique().tolist())
    if not make:
        return {"status": "error", "note": "make not found"}

    model = fuzzy_match(
        clean(p.model),
        df[df.make == make].model.unique().tolist()
    )
    if not model:
        return {"status": "error", "note": "model not found"}

    # =========================
    # STRICT YEAR WINDOW ±2
    # =========================
    comps = df[
        (df.make == make) &
        (df.model == model) &
        (df.year >= p.year - YEAR_DELTA) &
        (df.year <= p.year + YEAR_DELTA)
    ]

    # Optional trim filter
    trim = clean(p.trim)
    if trim and "trim" in comps.columns:
        trimmed = comps[comps.trim == trim]
        if len(trimmed) >= MIN_SAMPLES:
            comps = trimmed

    if len(comps) < MIN_SAMPLES:
        return {
            "status": "no_comparables",
            "comparables": len(comps),
            "used_years": sorted(comps.year.unique().tolist())
        }

    # =========================
    # FEATURES (SAFE)
    # =========================
    X = comps[["odometer", "year"]].copy()
    y = comps.price.values

    # province يدخل فقط لو الداتا كفاية
    if len(comps) >= 5 and comps.province.notna().any():
        counts = comps.province.value_counts()
        valid = counts[counts >= 2].index.tolist()

        if valid:
            prov = comps.province.where(
                comps.province.isin(valid),
                other="other"
            )
            dummies = pd.get_dummies(prov, prefix="province")
            X = pd.concat([X, dummies], axis=1)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LinearRegression()
    lr.fit(Xs, y)

    # =========================
    # PREDICTION
    # =========================
    input_data = {
        "odometer": p.odometer,
        "year": p.year
    }

    for col in X.columns:
        if col.startswith("province_"):
            prov = col.replace("province_", "")
            input_data[col] = 1 if clean(p.province) == prov else 0

    input_df = pd.DataFrame([input_data])

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]
    price = lr.predict(scaler.transform(input_df))[0]
    price = max(price, 0)

    return {
        "status": "success",
        "title": f"{p.year} {make.title()} {model.title()}",
        "price": round(price, 0),
        "comparables": len(comps),
        "used_years": sorted(comps.year.unique().tolist()),
        "trim_used": trim
    }

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "records": len(df)
    }

@app.post("/estimate")
def estimate(p: EstimateRequest):
    return estimate_value(p)

@app.post("/estimate/batch")
def estimate_batch(p: BatchEstimateRequest):
    return [estimate_value(v) for v in p.vehicles]
