# from fastapi import FastAPI
# from pydantic import BaseModel
# from typing import Optional, List
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# from scipy.stats import t
# import re
# from collections import defaultdict
# import warnings
# warnings.filterwarnings('ignore')

# CSV_PATH = "test.csv"
# YEAR_RANGE = 1
# MILEAGE_RATE = 0.10
# FUZZY_THRESHOLD = 80
# MIN_COMPARABLES_FOR_REGRESSION = 3

# app = FastAPI(title="Vehicle Fair Value API")

# # =========================
# # MODEL ALIASES (expanded with common variations)
# # =========================
# MODEL_ALIASES = {
#     "slvrdo": "silverado",
#     "silvrdo": "silverado",
#     "silverado": "silverado",
#     "silverad": "silverado",
    
#     "grndcrvn": "grand caravan",
#     "grandcrvn": "grand caravan",
#     "grand caravan": "grand caravan",
#     "grandcaravan": "grand caravan",
    
#     "f150": "f-150",
#     "f 150": "f-150",
#     "f-150": "f-150",
#     "f150xlt": "f-150",
    
#     "corolla": "corolla",
#     "corola": "corolla",
#     "corolla le": "corolla",
    
#     "civic": "civic",
#     "civic lx": "civic",
#     "civic ex": "civic",
    
#     "accord": "accord",
#     "acord": "accord",
    
#     "camry": "camry",
#     "camry le": "camry",
# }

# # Common manufacturer aliases
# MAKE_ALIASES = {
#     "gm": "general motors",
#     "gmc": "general motors",
#     "chev": "chevrolet",
#     "chevy": "chevrolet",
#     "vw": "volkswagen",
#     "vwvolkswagen": "volkswagen",
#     "toy": "toyota",
#     "hon": "honda",
#     "for": "ford",
#     "niss": "nissan",
# }

# # =========================
# # HELPERS
# # =========================
# def clean_string(s: str) -> str:
#     """Clean string according to the specified rules"""
#     if pd.isna(s) or s is None:
#         return ""
    
#     s = str(s)
#     s = s.lower()
#     s = s.strip()
#     s = re.sub(r'\s+', ' ', s)
#     s = re.sub(r'[^a-z0-9 ]', '', s)
#     return s

# def norm(s: Optional[str]):
#     """Normalize string: apply cleaning and return None if empty"""
#     if not s:
#         return None
#     cleaned = clean_string(s)
#     return cleaned if cleaned else None

# def normalize_model(raw: Optional[str]):
#     """Normalize model name with aliases"""
#     if not raw:
#         return None
    
#     cleaned = clean_string(raw)
#     if not cleaned:
#         return None
    
#     # Check exact aliases first
#     for alias, normalized in MODEL_ALIASES.items():
#         if cleaned == alias:
#             return normalized
    
#     # Check for partial matches
#     for alias in MODEL_ALIASES.keys():
#         if alias in cleaned or cleaned in alias:
#             return MODEL_ALIASES[alias]
    
#     return cleaned

# def normalize_make(raw: Optional[str]):
#     """Normalize manufacturer name with aliases"""
#     if not raw:
#         return None
    
#     cleaned = clean_string(raw)
#     if not cleaned:
#         return None
    
#     # Check aliases
#     for alias, normalized in MAKE_ALIASES.items():
#         if cleaned == alias:
#             return normalized
    
#     return cleaned

# def find_best_match(value: Optional[str], valid_set: set):
#     """Find best match in a set of valid values"""
#     if not value or not valid_set:
#         return None
    
#     value_norm = norm(value)
#     if not value_norm:
#         return None
    
#     # Check for exact match
#     for valid in valid_set:
#         if valid == value_norm:
#             return valid
    
#     # Check for substring match
#     for valid in valid_set:
#         if valid and value_norm and (value_norm in valid or valid in value_norm):
#             return valid
    
#     return None

# # =========================
# # LOAD AND CLEAN DATA
# # =========================
# def load_data(path: str) -> pd.DataFrame:
#     """Load and clean the CSV data"""
#     df = pd.read_csv(path)

#     # Rename columns to standard names
#     df = df.rename(columns={
#         "Year": "year",
#         "Make": "make",
#         "Model": "model",
#         "Trim": "trim",
#         "Odometer": "odometer",
#         "Price": "price",
#         "Province": "province",
#     })

#     # Clean numeric columns
#     for c in ["year", "odometer", "price"]:
#         df[c] = pd.to_numeric(df[c], errors="coerce")

#     # Clean categorical columns
#     cat_cols = ["make", "model", "trim", "province"]
    
#     for col in cat_cols:
#         df[col] = df[col].astype(str)
#         df[col] = df[col].str.lower()
#         df[col] = df[col].str.strip()
#         df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
#         df[col] = df[col].str.replace(r'[^a-z0-9 ]', '', regex=True)
#         df[col] = df[col].replace('', np.nan)

#     # Store original cleaned values
#     for col in cat_cols:
#         df[f"{col}_original"] = df[col]

#     # Drop rows with missing critical data
#     critical_cols = ["year", "make", "model", "odometer", "price"]
#     df = df.dropna(subset=critical_cols)

#     # Reset index
#     df = df.reset_index(drop=True)

#     return df

# # Load the data
# try:
#     df = load_data(CSV_PATH)
#     print(f"Successfully loaded data from {CSV_PATH}")
#     print(f"Data shape: {df.shape}")
# except FileNotFoundError:
#     print(f"Warning: {CSV_PATH} not found. Creating empty dataset for testing.")
#     df = pd.DataFrame(columns=[
#         'year', 'make', 'model', 'trim', 'odometer', 'price', 'province',
#         'make_original', 'model_original', 'trim_original', 'province_original'
#     ])

# # Store valid values
# VALID_MAKES = {make: make for make in df.make_original.unique() if pd.notna(make)} if len(df) > 0 else {}
# VALID_MODELS = {model: model for model in df.model_original.unique() if pd.notna(model)} if len(df) > 0 else {}
# VALID_PROVINCES = {prov: prov for prov in df.province_original.unique() if pd.notna(prov)} if len(df) > 0 else {}

# print(f"Loaded {len(df)} records")
# print(f"Unique makes: {len(VALID_MAKES)}")
# print(f"Unique models: {len(VALID_MODELS)}")
# print(f"Unique provinces: {len(VALID_PROVINCES)}")

# # =========================
# # REQUEST MODELS
# # =========================
# class EstimateRequest(BaseModel):
#     year: int
#     make: str
#     model: str
#     trim: Optional[str] = None
#     province: Optional[str] = None
#     odometer: int
#     confidence: float = 0.9

# class BatchEstimateRequest(BaseModel):
#     vehicles: List[EstimateRequest]

# # =========================
# # MATCHING FUNCTIONS
# # =========================
# def match_make(make_input: str) -> Optional[str]:
#     """Match manufacturer"""
#     make_norm = normalize_make(make_input)
#     return find_best_match(make_norm, set(VALID_MAKES.keys()))

# def match_model(model_input: str, make_matched: Optional[str] = None) -> Optional[str]:
#     """Match model"""
#     model_norm = normalize_model(model_input)
    
#     if make_matched:
#         make_models = df[df.make_original == make_matched].model_original.unique()
#         valid_models_for_make = {model: model for model in make_models if pd.notna(model)}
#         matched = find_best_match(model_norm, set(valid_models_for_make.keys()))
#         if matched:
#             return valid_models_for_make[matched]
    
#     return find_best_match(model_norm, set(VALID_MODELS.keys()))

# def match_province(province_input: Optional[str]) -> Optional[str]:
#     """Match province"""
#     if not province_input:
#         return None
#     return find_best_match(province_input, set(VALID_PROVINCES.keys()))

# # =========================
# # FIND COMPARABLE DATA
# # =========================
# def find_comparable_data(make: str, model: str, target_year: int = None):
#     """Find comparable data for specific make and model, optionally filtered by year range"""
#     if target_year:
#         # Find data within year range
#         for d in range(0, YEAR_RANGE + 1):
#             years = [target_year] if d == 0 else [target_year + d, target_year - d]
#             for y in years:
#                 comps = df[
#                     (df.make_original == make) &
#                     (df.model_original == model) &
#                     (df.year == y)
#                 ]
#                 if not comps.empty:
#                     return comps, y
#         return pd.DataFrame(), None
#     else:
#         # Get all data for this make/model regardless of year
#         comps = df[
#             (df.make_original == make) &
#             (df.model_original == model)
#         ]
#         return comps, None

# # =========================
# # IMPROVED REGRESSION FUNCTION
# # =========================
# def train_and_predict(comps: pd.DataFrame, target_odometer: int, 
#                      target_year: int = None, target_province: str = None,
#                      confidence: float = 0.9):
#     """Train regression model and make prediction for target vehicle"""
    
#     if len(comps) < MIN_COMPARABLES_FOR_REGRESSION:
#         return None, None, None, None, None, "Not enough comparables for regression"
    
#     try:
#         # Prepare features
#         X = comps[["odometer", "year"]].copy()
        
#         # Add province if available
#         if "province_original" in comps.columns and comps.province_original.notna().any():
#             province_dummies = pd.get_dummies(comps.province_original, prefix="province", drop_first=True)
#             if not province_dummies.empty:
#                 X = pd.concat([X, province_dummies], axis=1)
        
#         y = comps["price"].values
        
#         # Check if we have enough data points for the features
#         if len(X) <= len(X.columns):
#             return None, None, None, None, None, "Too few data points for the number of features"
        
#         # Train model
#         scaler = StandardScaler()
#         X_scaled = scaler.fit_transform(X)
        
#         model = LinearRegression()
#         model.fit(X_scaled, y)
        
#         # Prepare target for prediction
#         target_features = {"odometer": target_odometer, "year": target_year if target_year else comps.year.mean()}
        
#         # Add province features if they exist in the model
#         for col in X.columns:
#             if col.startswith("province_"):
#                 province_name = col.replace("province_", "")
#                 target_features[col] = 1 if target_province == province_name else 0
        
#         # Create target dataframe
#         target_df = pd.DataFrame([target_features])
        
#         # Ensure all columns are present
#         for col in X.columns:
#             if col not in target_df.columns:
#                 target_df[col] = 0
        
#         # Reorder columns to match training data
#         target_df = target_df[X.columns]
        
#         # Make prediction
#         target_scaled = scaler.transform(target_df)
#         pred = model.predict(target_scaled)[0]
        
#         # Calculate confidence interval
#         residuals = y - model.predict(X_scaled)
#         sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 0
#         n = len(X)
#         p = len(X.columns)
        
#         if sigma > 0 and n > p:
#             t_val = t.ppf((1 + confidence) / 2, df=n - p - 1)
#             margin = t_val * sigma * np.sqrt(1 + 1/n)
#             low = max(pred - margin, 0)
#             high = pred + margin
#         else:
#             low = high = pred
        
#         return pred, low, high, len(comps), model.score(X_scaled, y), None
        
#     except Exception as e:
#         return None, None, None, None, None, f"Regression error: {str(e)}"

# # =========================
# # CORE ESTIMATION FUNCTION
# # =========================
# def estimate_value(p: EstimateRequest):
#     """Main estimation function"""
#     confidence = min(max(p.confidence, 0.5), 0.99)
    
#     # Match make and model
#     make_matched = match_make(p.make)
#     model_matched = match_model(p.model, make_matched)
#     province_matched = match_province(p.province)
    
#     # Build title
#     title = f"{p.year} {p.make.title()} {p.model.title()}"
#     if p.trim:
#         title += f" {p.trim.title()}"
    
#     # Check if make/model found
#     if not make_matched or not model_matched:
#         return {
#             "title": title,
#             "price": None,
#             "comparables": 0,
#             "note": f"Make or model not found. Make: {p.make}, Model: {p.model}",
#             "matched_make": make_matched.title() if make_matched else None,
#             "matched_model": model_matched.title() if model_matched else None,
#             "matched_province": province_matched.title() if province_matched else None,
#             "status": "error_no_match"
#         }
    
#     # Find comparable data
#     comps, matched_year = find_comparable_data(make_matched, model_matched, p.year)
    
#     n = len(comps)
    
#     # Handle different cases based on number of comparables
#     if n == 0:
#         # Try to find any data for this make/model regardless of year
#         comps_all, _ = find_comparable_data(make_matched, model_matched, None)
#         n_all = len(comps_all)
        
#         if n_all >= MIN_COMPARABLES_FOR_REGRESSION:
#             # Use all available data for regression
#             pred, low, high, r2, error = train_and_predict(
#                 comps_all, p.odometer, p.year, province_matched, confidence
#             )[0:5]
            
#             if pred is not None:
#                 return {
#                     "title": title,
#                     "estimated_year": p.year,
#                     "price": round(pred, 0),
#                     "low": round(low, 0) if low is not None else None,
#                     "high": round(high, 0) if high is not None else None,
#                     "comparables": n_all,
#                     "r_squared": round(r2, 3) if r2 is not None else None,
#                     "note": "Using all available years for this make/model",
#                     "matched_make": make_matched.title(),
#                     "matched_model": model_matched.title(),
#                     "matched_province": province_matched.title() if province_matched else None,
#                     "confidence_interval": f"{confidence*100:.0f}%",
#                     "status": "success_all_years"
#                 }
        
#         return {
#             "title": title,
#             "price": None,
#             "comparables": 0,
#             "note": f"No data found for {make_matched.title()} {model_matched.title()}",
#             "matched_make": make_matched.title(),
#             "matched_model": model_matched.title(),
#             "matched_province": province_matched.title() if province_matched else None,
#             "status": "no_data"
#         }
    
#     elif n == 1:
#         # Single comparable
#         base = comps.iloc[0]
#         adj_price = base.price - (p.odometer - base.odometer) * MILEAGE_RATE
#         adj_price = max(adj_price, 0)
        
#         return {
#             "title": title,
#             "estimated_year": matched_year,
#             "price": round(adj_price, 0),
#             "comparables": 1,
#             "note": f"Single comparable with mileage adjustment (${MILEAGE_RATE}/km)",
#             "matched_make": make_matched.title(),
#             "matched_model": model_matched.title(),
#             "matched_province": province_matched.title() if province_matched else None,
#             "status": "single_comparable"
#         }
    
#     elif n == 2:
#         # Two comparables - use average
#         avg_price = comps.price.mean()
        
#         return {
#             "title": title,
#             "estimated_year": matched_year,
#             "price": round(avg_price, 0),
#             "comparables": 2,
#             "note": "Average of two comparables",
#             "matched_make": make_matched.title(),
#             "matched_model": model_matched.title(),
#             "matched_province": province_matched.title() if province_matched else None,
#             "status": "two_comparables"
#         }
    
#     else:
#         # Three or more comparables - use regression
#         pred, low, high, r2, error = train_and_predict(
#             comps, p.odometer, matched_year, province_matched, confidence
#         )[0:5]
        
#         if pred is not None:
#             return {
#                 "title": title,
#                 "estimated_year": matched_year,
#                 "price": round(pred, 0),
#                 "low": round(low, 0) if low is not None else None,
#                 "high": round(high, 0) if high is not None else None,
#                 "comparables": n,
#                 "r_squared": round(r2, 3) if r2 is not None else None,
#                 "note": None,
#                 "matched_make": make_matched.title(),
#                 "matched_model": model_matched.title(),
#                 "matched_province": province_matched.title() if province_matched else None,
#                 "confidence_interval": f"{confidence*100:.0f}%",
#                 "status": "success"
#             }
#         else:
#             # Regression failed, fallback to mean
#             avg_price = comps.price.mean()
#             return {
#                 "title": title,
#                 "estimated_year": matched_year,
#                 "price": round(avg_price, 0),
#                 "comparables": n,
#                 "note": f"Regression failed, using mean price. {error}",
#                 "matched_make": make_matched.title(),
#                 "matched_model": model_matched.title(),
#                 "matched_province": province_matched.title() if province_matched else None,
#                 "status": "regression_fallback"
#             }

# # =========================
# # ROUTES
# # =========================
# @app.get("/")
# def health():
#     return {
#         "status": "ok", 
#         "data_records": len(df),
#         "makes_available": len(VALID_MAKES),
#         "models_available": len(VALID_MODELS),
#         "year_range": YEAR_RANGE,
#         "mileage_rate": MILEAGE_RATE
#     }

# @app.post("/estimate")
# def estimate(p: EstimateRequest):
#     """Estimate value for a single vehicle"""
#     return estimate_value(p)

# @app.post("/estimate/batch")
# def estimate_batch(p: BatchEstimateRequest):
#     """Estimate value for multiple vehicles"""
#     return [estimate_value(v) for v in p.vehicles]

# @app.get("/makes")
# def get_makes():
#     """Get list of available makes"""
#     return {"makes": sorted(list(VALID_MAKES.keys()))}

# @app.get("/models/{make}")
# def get_models(make: str):
#     """Get list of available models for a make"""
#     make_matched = match_make(make)
#     if not make_matched:
#         return {"models": [], "make": make, "status": "not_found"}
    
#     models = df[df.make_original == make_matched].model_original.unique()
#     return {"models": sorted(list(models)), "make": make_matched, "status": "found"}

# @app.get("/stats")
# def get_stats():
#     """Get statistics about the data"""
#     if len(df) == 0:
#         return {"status": "no_data"}
    
#     return {
#         "total_records": len(df),
#         "year_range": f"{df.year.min()} - {df.year.max()}",
#         "price_stats": {
#             "min": int(df.price.min()),
#             "max": int(df.price.max()),
#             "mean": int(df.price.mean()),
#             "median": int(df.price.median())
#         },
#         "odometer_stats": {
#             "min": int(df.odometer.min()),
#             "max": int(df.odometer.max()),
#             "mean": int(df.odometer.mean()),
#             "median": int(df.odometer.median())
#         }
#     }

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
MIN_SAMPLES = 3
YEAR_WINDOW = 3

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
    province: Optional[str] = None

class BatchEstimateRequest(BaseModel):
    vehicles: List[EstimateRequest]

# =========================
# DATA SELECTION
# =========================
def get_comparables(make: str, model: str, target_year: int) -> pd.DataFrame:
    base = df[(df.make == make) & (df.model == model)]

    if base.empty:
        return base

    same_year = base[base.year == target_year]
    if len(same_year) >= MIN_SAMPLES:
        return same_year

    years = sorted(base.year.unique())
    for d in range(1, YEAR_WINDOW + 1):
        near = base[base.year.isin([target_year - d, target_year + d])]
        if len(near) >= MIN_SAMPLES:
            return near

    return base

# =========================
# CORE LOGIC
# =========================
def estimate_value(p: EstimateRequest):
    make_in = clean(p.make)
    model_in = clean(p.model)
    province_in = clean(p.province)

    make_matched = fuzzy_match(make_in, df.make.unique().tolist())
    if not make_matched:
        return {"status": "error", "note": "make not found"}

    model_matched = fuzzy_match(
        model_in,
        df[df.make == make_matched].model.unique().tolist()
    )
    if not model_matched:
        return {"status": "error", "note": "model not found"}

    comps = get_comparables(make_matched, model_matched, p.year)

    if len(comps) < MIN_SAMPLES:
        return {
            "status": "no_comparables",
            "comparables": len(comps),
            "note": "not enough data"
        }

    # =========================
    # FEATURES
    # =========================
    X = comps[["odometer", "year"]].copy()

    if comps.province.notna().any() and comps.province.nunique() > 1:
        dummies = pd.get_dummies(comps.province, prefix="province")
        X = pd.concat([X, dummies], axis=1)

    y = comps.price.values

    if len(X) <= X.shape[1]:
        return {
            "status": "error",
            "note": "invalid regression matrix"
        }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LinearRegression()
    lr.fit(X_scaled, y)

    # =========================
    # PREDICTION INPUT
    # =========================
    input_data = {
        "odometer": p.odometer,
        "year": p.year
    }

    for col in X.columns:
        if col.startswith("province_"):
            prov = col.replace("province_", "")
            input_data[col] = 1 if province_in == prov else 0

    input_df = pd.DataFrame([input_data])

    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X.columns]
    input_scaled = scaler.transform(input_df)

    price = lr.predict(input_scaled)[0]
    price = max(price, 0)

    return {
        "status": "success",
        "title": f"{p.year} {make_matched.title()} {model_matched.title()}",
        "price": round(price, 0),
        "comparables": len(comps),
        "used_years": sorted(comps.year.unique().tolist()),
        "features": list(X.columns)
    }

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {
        "status": "ok",
        "records": len(df),
        "makes": int(df.make.nunique()),
        "models": int(df.model.nunique())
    }

@app.post("/estimate")
def estimate(p: EstimateRequest):
    return estimate_value(p)

@app.post("/estimate/batch")
def estimate_batch(p: BatchEstimateRequest):
    return [estimate_value(v) for v in p.vehicles]
