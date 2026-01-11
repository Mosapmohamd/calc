from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import pandas as pd
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
    # COMPARABLES MODE ±2 YEARS
    # =========================
    comps = df[
        (df.make == make) &
        (df.model == model) &
        (df.year >= p.year - YEAR_DELTA) &
        (df.year <= p.year + YEAR_DELTA)
    ]

    trim = clean(p.trim)
    if trim and "trim" in comps.columns:
        trimmed = comps[comps.trim == trim]
        if len(trimmed) >= MIN_SAMPLES:
            comps = trimmed

    if len(comps) >= MIN_SAMPLES:
        X = comps[["odometer", "year"]].copy()
        y = comps.price.values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        lr = LinearRegression()
        lr.fit(Xs, y)

        input_df = pd.DataFrame([{
            "odometer": p.odometer,
            "year": p.year
        }])

        price = lr.predict(scaler.transform(input_df))[0]

        return {
            "status": "success",
            "mode": "comparables",
            "title": f"{p.year} {make.title()} {model.title()}",
            "price": round(max(price, 0), 0),
            "comparables": len(comps),
            "used_years": sorted(comps.year.unique().tolist())
        }

    # =========================
    # ML FALLBACK MODE
    # =========================
    all_data = df[(df.make == make) & (df.model == model)]

    if len(all_data) < MIN_SAMPLES:
        return {
            "status": "no_comparables",
            "mode": "none",
            "comparables": len(all_data)
        }

    X = all_data[["odometer", "year"]].copy()
    y = all_data.price.values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    lr = LinearRegression()
    lr.fit(Xs, y)

    input_df = pd.DataFrame([{
        "odometer": p.odometer,
        "year": p.year
    }])

    price = lr.predict(scaler.transform(input_df))[0]

    return {
        "status": "success",
        "mode": "ml_fallback",
        "title": f"{p.year} {make.title()} {model.title()}",
        "price": round(max(price, 0), 0),
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

@app.post("/estimate/batch")
def estimate_batch(p: BatchEstimateRequest):
    return [estimate_value(v) for v in p.vehicles]
