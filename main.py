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
from difflib import get_close_matches
import fuzzywuzzy
from fuzzywuzzy import fuzz, process

CSV_PATH = "test.csv"
YEAR_RANGE = 1
MILEAGE_RATE = 0.10
FUZZY_THRESHOLD = 80  # Minimum fuzzy match score (0-100)
MIN_COMPARABLES_FOR_REGRESSION = 3

app = FastAPI(title="Vehicle Fair Value API")

# =========================
# MODEL ALIASES (expanded with common variations)
# =========================
MODEL_ALIASES = {
    "slvrdo": "silverado",
    "silvrdo": "silverado",
    "silverado": "silverado",
    "silverad": "silverado",
    
    "grndcrvn": "grand caravan",
    "grandcrvn": "grand caravan",
    "grand caravan": "grand caravan",
    "grandcaravan": "grand caravan",
    
    "f150": "f-150",
    "f 150": "f-150",
    "f-150": "f-150",
    "f150xlt": "f-150",
    
    "corolla": "corolla",
    "corola": "corolla",
    "corolla le": "corolla",
    
    "civic": "civic",
    "civic lx": "civic",
    "civic ex": "civic",
    
    "accord": "accord",
    "acord": "accord",
    
    "camry": "camry",
    "camry le": "camry",
}

# Common manufacturer aliases
MAKE_ALIASES = {
    "gm": "general motors",
    "gmc": "general motors",
    "chev": "chevrolet",
    "chevy": "chevrolet",
    "vw": "volkswagen",
    "vwvolkswagen": "volkswagen",
    "toy": "toyota",
    "hon": "honda",
    "for": "ford",
    "niss": "nissan",
}

# =========================
# HELPERS
# =========================
def clean_string(s: str) -> str:
    """Clean string according to the specified rules"""
    if pd.isna(s) or s is None:
        return ""
    
    s = str(s)
    # Convert to lowercase
    s = s.lower()
    # Strip whitespace
    s = s.strip()
    # Replace multiple spaces with single space
    s = re.sub(r'\s+', ' ', s)
    # Remove special characters, keep only a-z, 0-9, and space
    s = re.sub(r'[^a-z0-9 ]', '', s)
    return s

def norm(s: Optional[str]):
    """Normalize string: apply cleaning and return None if empty"""
    if not s:
        return None
    cleaned = clean_string(s)
    return cleaned if cleaned else None

def normalize_model(raw: Optional[str]):
    """Normalize model name with aliases and fuzzy matching"""
    if not raw:
        return None
    
    # Clean the input
    cleaned = clean_string(raw)
    if not cleaned:
        return None
    
    # Check exact aliases first
    for alias, normalized in MODEL_ALIASES.items():
        if cleaned == alias:
            return normalized
    
    # Check if cleaned is a close match to any alias
    for alias in MODEL_ALIASES.keys():
        if fuzz.ratio(cleaned, alias) > FUZZY_THRESHOLD:
            return MODEL_ALIASES[alias]
    
    # Return cleaned version
    return cleaned

def normalize_make(raw: Optional[str]):
    """Normalize manufacturer name with aliases"""
    if not raw:
        return None
    
    cleaned = clean_string(raw)
    if not cleaned:
        return None
    
    # Check aliases
    for alias, normalized in MAKE_ALIASES.items():
        if cleaned == alias:
            return normalized
    
    return cleaned

def find_best_match(value: Optional[str], valid_set: set, threshold: int = 80):
    """Find best fuzzy match in a set of valid values"""
    if not value or not valid_set:
        return None
    
    value_norm = norm(value)
    if not value_norm:
        return None
    
    # Check for exact match
    for valid in valid_set:
        if valid == value_norm:
            return valid
    
    # Check for substring match
    for valid in valid_set:
        if valid and value_norm and (value_norm in valid or valid in value_norm):
            return valid
    
    # Use fuzzy matching
    if len(valid_set) > 0:
        # Convert set to list for fuzzy matching
        valid_list = list(valid_set)
        # Use fuzzywuzzy to find best match
        match, score = process.extractOne(value_norm, valid_list)
        if score >= threshold:
            return match
    
    return None

def match_trim_fuzzy(trim: Optional[str], valid_trims: set, threshold: int = 75):
    """Fuzzy match for trim levels"""
    if not trim or not valid_trims:
        return None
    
    trim_norm = norm(trim)
    if not trim_norm:
        return None
    
    # Check for exact match
    for valid in valid_trims:
        if valid == trim_norm:
            return valid
    
    # Check for common trim patterns
    common_patterns = {
        "base": ["base", "standard"],
        "lx": ["lx"],
        "ex": ["ex", "executive"],
        "le": ["le", "limited edition"],
        "se": ["se", "special edition"],
        "xlt": ["xlt"],
        "lariat": ["lariat"],
        "platinum": ["platinum"],
        "limited": ["limited"],
        "premium": ["premium"],
        "luxury": ["luxury"],
        "sport": ["sport"],
        "touring": ["touring"],
    }
    
    # Check if input matches any common trim pattern
    for pattern_key, pattern_list in common_patterns.items():
        for pattern in pattern_list:
            if pattern in trim_norm:
                # Look for trims containing this pattern
                for valid in valid_trims:
                    if pattern_key in valid:
                        return valid
    
    # Fuzzy match as fallback
    if len(valid_trims) > 0:
        valid_list = list(valid_trims)
        match, score = process.extractOne(trim_norm, valid_list)
        if score >= threshold:
            return match
    
    return None

# =========================
# LOAD AND CLEAN DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    """Load and clean the CSV data"""
    df = pd.read_csv(path)

    # Rename columns to standard names
    df = df.rename(columns={
        "Year": "year",
        "Make": "make",
        "Model": "model",
        "Trim": "trim",
        "Odometer": "odometer",
        "Price": "price",
        "Province": "province",
    })

    # Clean numeric columns
    for c in ["year", "odometer", "price"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Clean categorical columns (apply your cleaning logic)
    cat_cols = ["make", "model", "trim", "province"]
    
    for col in cat_cols:
        # Convert to string and apply your cleaning logic
        df[col] = df[col].astype(str)
        df[col] = df[col].str.lower()
        df[col] = df[col].str.strip()
        df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        df[col] = df[col].str.replace(r'[^a-z0-9 ]', '', regex=True)
        
        # Replace empty strings with NaN
        df[col] = df[col].replace('', np.nan)

    # Store original cleaned values
    for col in cat_cols:
        df[f"{col}_original"] = df[col]

    # Drop rows with missing critical data
    critical_cols = ["year", "make", "model", "odometer", "price"]
    df = df.dropna(subset=critical_cols)

    # Reset index
    df = df.reset_index(drop=True)

    return df

# Load the data
try:
    df = load_data(CSV_PATH)
    print(f"Successfully loaded data from {CSV_PATH}")
    print(f"Data shape: {df.shape}")
except FileNotFoundError:
    print(f"Warning: {CSV_PATH} not found. Creating empty dataset for testing.")
    # Create empty dataframe with expected columns
    df = pd.DataFrame(columns=[
        'year', 'make', 'model', 'trim', 'odometer', 'price', 'province',
        'make_original', 'model_original', 'trim_original', 'province_original'
    ])

# Store both cleaned and original values
VALID_MAKES = {make: make for make in df.make_original.unique() if pd.notna(make)} if len(df) > 0 else {}
VALID_MODELS = {model: model for model in df.model_original.unique() if pd.notna(model)} if len(df) > 0 else {}
VALID_PROVINCES = {prov: prov for prov in df.province_original.unique() if pd.notna(prov)} if len(df) > 0 else {}

# Build trim mapping
TRIM_MAP = defaultdict(set)
if len(df) > 0:
    for _, r in df.iterrows():
        make_norm = r.make_original
        model_norm = r.model_original
        trim_val = r.trim_original
        if pd.notna(make_norm) and pd.notna(model_norm) and pd.notna(trim_val):
            TRIM_MAP[(make_norm, model_norm)].add(trim_val)

print(f"Loaded {len(df)} records")
print(f"Unique makes: {len(VALID_MAKES)}")
print(f"Unique models: {len(VALID_MODELS)}")
print(f"Unique provinces: {len(VALID_PROVINCES)}")

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
# UPDATED MATCHING FUNCTIONS
# =========================
def match_make(make_input: str) -> Optional[str]:
    """Match manufacturer with fuzzy logic"""
    make_norm = normalize_make(make_input)
    return find_best_match(make_norm, set(VALID_MAKES.keys()))

def match_model(model_input: str, make_matched: Optional[str] = None) -> Optional[str]:
    """Match model with fuzzy logic"""
    model_norm = normalize_model(model_input)
    
    # If we have a make, filter models by that make
    if make_matched:
        # Get all models for this make
        make_models = df[df.make_original == make_matched].model_original.unique()
        valid_models_for_make = {model: model for model in make_models if pd.notna(model)}
        matched = find_best_match(model_norm, set(valid_models_for_make.keys()))
        if matched:
            return valid_models_for_make[matched]
    
    # Fallback to all models
    return find_best_match(model_norm, set(VALID_MODELS.keys()))

def match_province(province_input: Optional[str]) -> Optional[str]:
    """Match province with fuzzy logic"""
    if not province_input:
        return None
    return find_best_match(province_input, set(VALID_PROVINCES.keys()))

def match_trim(trim_input: Optional[str], make: str, model: str) -> Optional[str]:
    """Match trim with fuzzy logic for specific make/model"""
    if not trim_input or not make or not model:
        return None
    
    if (make, model) not in TRIM_MAP:
        return None
    
    valid_trims = TRIM_MAP[(make, model)]
    return match_trim_fuzzy(trim_input, valid_trims)

# =========================
# FIND NEAREST YEAR DATA
# =========================
def find_nearest_year_data(make, model, target_year):
    """Find comparable data for the nearest available year"""
    for d in range(0, YEAR_RANGE + 1):
        years = [target_year] if d == 0 else [target_year + d, target_year - d]
        for y in years:
            # Filter by make and model (already cleaned)
            comps = df[
                (df.make_original == make) &
                (df.model_original == model) &
                (df.year == y)
            ]
            if not comps.empty:
                return comps, y
    return pd.DataFrame(), None

# =========================
# IMPROVED REGRESSION WITH BETTER ERROR HANDLING
# =========================
def train_regression(sub_df):
    """Train regression model with robust error handling"""
    if len(sub_df) < MIN_COMPARABLES_FOR_REGRESSION:
        return None, None, None, None
    
    try:
        X = sub_df[["odometer", "year"]].copy()
        
        # Add province if available and has variation
        if ("province_original" in sub_df.columns and 
            sub_df.province_original.notna().any() and
            sub_df.province_original.nunique() > 1):
            
            # Filter out rare provinces (less than 2 occurrences)
            province_counts = sub_df.province_original.value_counts()
            common_provinces = province_counts[province_counts >= 2].index.tolist()
            
            if common_provinces:
                # Create province indicator for common provinces
                for province in common_provinces:
                    X[f"province_{province}"] = (sub_df.province_original == province).astype(int)
        else:
            # Add constant column if no province info
            X["constant"] = 1
        
        y = sub_df["price"].values
        
        # Check for sufficient variation in features
        if len(X) < len(X.columns) + 1:
            return None, None, None, None
        
        # Remove constant columns
        X = X.loc[:, X.nunique() > 1]
        
        if len(X.columns) == 0:
            return None, None, None, None
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        residuals = y - model.predict(X_scaled)
        sigma = np.std(residuals, ddof=1) if len(residuals) > 1 else 0
        
        return model, scaler, sigma, X.columns.tolist()
    
    except Exception as e:
        print(f"Regression training failed: {e}")
        return None, None, None, None

# =========================
# CORE LOGIC WITH IMPROVED ERROR HANDLING
# =========================
def estimate_value(p: EstimateRequest):
    confidence = min(max(p.confidence, 0.5), 0.99)

    # Use fuzzy matching for all fields
    make_matched = match_make(p.make)
    model_matched = match_model(p.model, make_matched)
    province_matched = match_province(p.province)

    if not make_matched or not model_matched:
        return {
            "title": f"{p.year} {p.make} {p.model}",
            "price": None,
            "comparables": 0,
            "note": f"Make or model not found. Make: {p.make}, Model: {p.model}",
            "status": "error"
        }

    # Get display values
    make_display = make_matched.title() if make_matched else p.make
    model_display = model_matched.title() if model_matched else p.model

    comps, used_year = find_nearest_year_data(make_matched, model_matched, p.year)

    # Try to match trim if provided
    trim_matched = None
    if p.trim:
        trim_matched = match_trim(p.trim, make_matched, model_matched)
        if trim_matched and "trim_original" in comps.columns:
            comps_trim = comps[comps["trim_original"] == trim_matched]
            if not comps_trim.empty:
                comps = comps_trim

    n = len(comps)

    # Build title with matched values
    title = f"{p.year} {make_display} {model_display}"
    if trim_matched:
        title += f" {trim_matched.title()}"

    if n == 0:
        return {
            "title": title,
            "price": None,
            "comparables": 0,
            "note": "No comparable vehicles found",
            "matched_make": make_display,
            "matched_model": model_display,
            "matched_trim": trim_matched.title() if trim_matched else None,
            "status": "no_comparables"
        }

    if n == 1:
        base = comps.iloc[0]
        adj_price = base.price - (p.odometer - base.odometer) * MILEAGE_RATE
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(max(adj_price, 0), 0),  # Ensure non-negative price
            "comparables": 1,
            "note": "Single comparable with mileage adjustment",
            "matched_make": make_display,
            "matched_model": model_display,
            "matched_trim": trim_matched.title() if trim_matched else None,
            "status": "single_comparable"
        }

    if n == 2:
        avg_price = comps.price.mean()
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(avg_price, 0),
            "comparables": 2,
            "note": "Average of two comparables",
            "matched_make": make_display,
            "matched_model": model_display,
            "matched_trim": trim_matched.title() if trim_matched else None,
            "status": "two_comparables"
        }

    # Try regression if we have enough comparables
    lr, scaler, sigma, cols = train_regression(comps)
    
    if lr is None:
        # Fallback to mean price
        avg_price = comps.price.mean()
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(avg_price, 0),
            "comparables": n,
            "note": "Regression not possible, using mean price",
            "matched_make": make_display,
            "matched_model": model_display,
            "matched_trim": trim_matched.title() if trim_matched else None,
            "status": "mean_fallback"
        }

    try:
        # Prepare input for prediction
        input_data = {}
        
        # Add base features
        input_data["odometer"] = p.odometer
        input_data["year"] = used_year
        
        # Add province features if present
        for col in cols:
            if col.startswith("province_"):
                province_name = col.replace("province_", "")
                input_data[col] = 1 if province_matched == province_name else 0
            elif col == "constant":
                input_data[col] = 1
        
        # Create input row with all columns
        input_row = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in cols:
            if col not in input_row.columns:
                input_row[col] = 0
        
        # Reorder columns to match training data
        input_row = input_row[cols]
        
        # Make prediction
        X_scaled = scaler.transform(input_row)
        pred = lr.predict(X_scaled)[0]
        
        # Ensure non-negative prediction
        pred = max(pred, 0)
        
        if sigma == 0 or not np.isfinite(sigma) or n <= len(cols):
            return {
                "title": title,
                "estimated_year": used_year,
                "price": round(pred, 0),
                "comparables": n,
                "note": "Prediction with limited confidence",
                "matched_make": make_display,
                "matched_model": model_display,
                "matched_trim": trim_matched.title() if trim_matched else None,
                "status": "regression_limited"
            }
        
        # Calculate confidence interval
        t_val = t.ppf((1 + confidence) / 2, df=max(1, n - len(cols) - 1))
        margin = t_val * sigma
        
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(pred, 0),
            "low": round(max(pred - margin, 0), 0),
            "high": round(pred + margin, 0),
            "comparables": n,
            "note": None,
            "matched_make": make_display,
            "matched_model": model_display,
            "matched_trim": trim_matched.title() if trim_matched else None,
            "confidence_interval": f"{confidence*100:.0f}%",
            "status": "success"
        }
    
    except Exception as e:
        # Fallback to mean if prediction fails
        avg_price = comps.price.mean()
        return {
            "title": title,
            "estimated_year": used_year,
            "price": round(avg_price, 0),
            "comparables": n,
            "note": f"Prediction failed: {str(e)[:50]}",
            "matched_make": make_display,
            "matched_model": model_display,
            "matched_trim": trim_matched.title() if trim_matched else None,
            "status": "prediction_error"
        }

# =========================
# ROUTES
# =========================
@app.get("/")
def health():
    return {
        "status": "ok", 
        "fuzzy_matching": "enabled",
        "data_records": len(df),
        "makes_available": len(VALID_MAKES),
        "models_available": len(VALID_MODELS)
    }

@app.post("/estimate")
def estimate(p: EstimateRequest):
    return estimate_value(p)

@app.post("/estimate/batch")
def estimate_batch(p: BatchEstimateRequest):
    return [estimate_value(v) for v in p.vehicles]

@app.get("/valid-values")
def get_valid_values():
    """Endpoint to see what valid values are in the dataset"""
    return {
        "makes": sorted(list(VALID_MAKES.keys())),
        "models": sorted(list(VALID_MODELS.keys())),
        "provinces": sorted(list(VALID_PROVINCES.keys())),
        "count": len(df)
    }

@app.get("/search/{make}/{model}")
def search_models(make: str, model: str):
    """Search for specific make/model combinations"""
    make_matched = match_make(make)
    results = []
    
    if make_matched:
        # Get all models for this make
        make_models = df[df.make_original == make_matched].model_original.unique()
        model_matches = process.extract(norm(model) or "", make_models, limit=10)
        
        for model_match, score in model_matches:
            if score > 60:
                count = len(df[(df.make_original == make_matched) & (df.model_original == model_match)])
                results.append({
                    "model": model_match,
                    "score": score,
                    "count": count,
                    "years_available": df[(df.make_original == make_matched) & 
                                          (df.model_original == model_match)].year.unique().tolist()
                })
    
    return {"make": make_matched, "matches": results}

@app.get("/sample/{make}/{model}")
def get_sample_data(make: str, model: str, limit: int = 5):
    """Get sample data for a specific make/model"""
    make_matched = match_make(make)
    model_matched = match_model(model, make_matched)
    
    if not make_matched or not model_matched:
        return {"error": "Make or model not found"}
    
    sample = df[
        (df.make_original == make_matched) & 
        (df.model_original == model_matched)
    ].head(limit)
    
    return {
        "make": make_matched,
        "model": model_matched,
        "sample": sample[["year", "odometer", "price", "province_original", "trim_original"]].to_dict(orient="records")
    }