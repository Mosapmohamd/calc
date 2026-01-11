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
from difflib import SequenceMatcher

CSV_PATH = "test.csv"
YEAR_RANGE = 1
MILEAGE_RATE = 0.10

app = FastAPI(title="Vehicle Fair Value API")

# =========================
# MODEL ALIASES
# =========================
MODEL_ALIASES = {
    "f150": "f150",
    "slvrdo": "silverado",
    "silvrdo": "silverado",
    "silverado": "silverado",
    "grndcrvn": "grandcaravan",
    "grandcrvn": "grandcaravan",
    "grandcaravan": "grandcaravan",
}

# =========================
# HELPERS
# =========================
def norm(s: Optional[str]):
    if not s:
        return None
    return re.sub(r"[^a-z0-9]", "", s.lower())

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def match_value(value: Optional[str], valid_set: set, threshold: float = 0.6):
    if not value:
        return None

    value = norm(value)
    best_match = None
    best_score = 0

    for v in valid_set:
        v_norm = norm(v)

        if value == v_norm:
            return v

        score = similarity(value, v_norm)
        if score > best_score:
            best_score = score
            best_match = v

    return best_match if best_score >= threshold else None

def normalize_model(raw: Optional[str]):
    if not raw:
        return None
    key = norm(raw)
    return MODEL_ALIASES.get(key, key)

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

    cat_cols = ["make", "model", "trim", "province"]
    for col in cat_cols:
        df[col] = df[col].astype(str)
        df[col] = df[col].str.lower()
        df[col] = df[col].str.replace(r"[^a-z0-9]", "", regex=True)

    return df.dropna()

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
# TRIM MATCH
# =========================
def match_trim(trim, make, model):
    if not trim:
        return None
    trims = TRIM_MAP.get((make, model), set())
    return match_value(trim, trims)

# =========================
# FIND NEAREST YEAR DATA
# =========================
def find_nearest_year_data(make, model, target_year):
    for d in range(0, YEAR_RANGE + 1):
        years = [target_year] if d == 0 else [target_year + d, target_year - d]
        for y in years:
            comps = df[
                (df.make == make) &
                (df.model == model) &
                (df.year == y)
            ]
            if not comps.empty:
                return comps, y
    return pd.DataFrame(), None

# =========================
# REGRESSION
# =========================
def train_regression(sub_df):
    X = sub_df[["odometer", "year", "province"]]
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
    model_norm = normalize_model(p.model)
    model = match_value(model_norm, VALID_MODELS)
    province = match_value(p.province, VALID_PROVINCES)

    if not make or not model:
        return {
            "title": f"{p.year} {p.make} {p.model}",
            "price": None,
            "comparables": 0,
            "note": "make or model not found"
        }

    comps, used_year = find_nearest_year_data(make, model, p.year)

    trim = match_trim(p.trim, make, model)
    if trim:
        comps_trim = comps[comps.trim == trim]
        if not comps_trim.empty:
            comps = comps_trim

    n = len(comps)
    title = f"{p.year} {make} {model}" + (f" {trim}" if trim else "")

    if n == 0:
        return {
            "title": title,
            "price": None,
            "comparables": 0,
            "note": "no comparable vehicles"
        }

    if n == 1:
        base = comps.iloc[0]
        adj_price = base.price - (p.odometer - base.odometer) * MILEAGE_RATE
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(adj_price, 0),
            "comparables": 1,
            "note": "single comparable"
        }

    if n == 2:
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(comps.price.mean(), 0),
            "comparables": 2,
            "note": "two comparables average"
        }

    lr, scaler, sigma, cols = train_regression(comps)

    input_row = pd.DataFrame([{
        "odometer": p.odometer,
        "year": used_year,
        "province": province if province else comps.province.mode()[0]
    }])

    input_row = pd.get_dummies(input_row, columns=["province"])
    input_row = input_row.reindex(columns=cols, fill_value=0)

    pred = lr.predict(scaler.transform(input_row))[0]

    if not np.isfinite(sigma):
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(pred, 0),
            "comparables": n,
            "note": "low variance"
        }

    t_val = t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_val * sigma

    return {
        "title": title,
        "estimated_year": used_year,
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
