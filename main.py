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
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = "test.csv"
YEAR_RANGE = 2  # Increased for better matching
MILEAGE_RATE = 0.10
MIN_COMPARABLES_FOR_REGRESSION = 3

app = FastAPI(title="Vehicle Fair Value API", version="2.0")

# =========================
# MODEL ALIASES
# =========================
MODEL_ALIASES = {
    "slvrdo": "silverado",
    "silvrdo": "silverado",
    "silverado": "silverado",
    
    "grndcrvn": "grand caravan",
    "grandcrvn": "grand caravan",
    "grand caravan": "grand caravan",
    
    "f150": "f-150",
    "f 150": "f-150",
    "f-150": "f-150",
    
    "corolla": "corolla",
    "corola": "corolla",
    
    "civic": "civic",
    "cvic": "civic",
    
    "accord": "accord",
    "acord": "accord",
    
    "camry": "camry",
    "camy": "camry",
}

MAKE_ALIASES = {
    "gm": "general motors",
    "gmc": "general motors",
    "chev": "chevrolet",
    "chevy": "chevrolet",
    "vw": "volkswagen",
    "toy": "toyota",
    "hon": "honda",
    "for": "ford",
    "niss": "nissan",
    "bm": "bmw",
    "merc": "mercedes",
    "mercdes": "mercedes",
    "hyun": "hyundai",
    "hundai": "hyundai",
}

# =========================
# HELPERS
# =========================
def clean_string(s: str) -> str:
    """Clean string according to the specified rules"""
    if pd.isna(s) or s is None:
        return ""
    
    s = str(s)
    s = s.lower()
    s = s.strip()
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', '', s)
    return s

def norm(s: Optional[str]):
    """Normalize string"""
    if not s:
        return None
    cleaned = clean_string(s)
    return cleaned if cleaned else None

def normalize_model(raw: Optional[str]):
    """Normalize model name with aliases"""
    if not raw:
        return None
    
    cleaned = clean_string(raw)
    if not cleaned:
        return None
    
    # Remove spaces for alias matching
    cleaned_no_spaces = cleaned.replace(" ", "")
    
    # Check exact aliases first
    for alias, normalized in MODEL_ALIASES.items():
        if cleaned == alias or cleaned_no_spaces == alias:
            return normalized
    
    return cleaned

def normalize_make(raw: Optional[str]):
    """Normalize manufacturer name with aliases"""
    if not raw:
        return None
    
    cleaned = clean_string(raw)
    if not cleaned:
        return None
    
    cleaned_no_spaces = cleaned.replace(" ", "")
    
    for alias, normalized in MAKE_ALIASES.items():
        if cleaned == alias or cleaned_no_spaces == alias:
            return normalized
    
    return cleaned

def find_best_match(value: str, valid_set: set):
    """Find best match using various strategies"""
    if not value or not valid_set:
        return None
    
    value_norm = norm(value)
    if not value_norm:
        return None
    
    # 1. Exact match
    for valid in valid_set:
        if valid == value_norm:
            return valid
    
    # 2. Substring match
    for valid in valid_set:
        if value_norm in valid or valid in value_norm:
            return valid
    
    # 3. Partial word match
    value_words = set(value_norm.split())
    for valid in valid_set:
        valid_words = set(valid.split())
        if value_words.intersection(valid_words):
            return valid
    
    return None

# =========================
# LOAD AND CLEAN DATA
# =========================
def load_data(path: str) -> pd.DataFrame:
    """Load and clean the CSV data"""
    df = pd.read_csv(path)

    # Standardize column names
    column_mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'year' in col_lower:
            column_mapping[col] = 'year'
        elif 'make' in col_lower:
            column_mapping[col] = 'make'
        elif 'model' in col_lower:
            column_mapping[col] = 'model'
        elif 'trim' in col_lower:
            column_mapping[col] = 'trim'
        elif 'odometer' in col_lower or 'mileage' in col_lower:
            column_mapping[col] = 'odometer'
        elif 'price' in col_lower or 'value' in col_lower:
            column_mapping[col] = 'price'
        elif 'province' in col_lower or 'state' in col_lower or 'region' in col_lower:
            column_mapping[col] = 'province'
    
    df = df.rename(columns=column_mapping)

    # Ensure required columns exist
    required_cols = ['year', 'make', 'model', 'odometer', 'price']
    for col in required_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    # Clean numeric columns
    for c in ['year', 'odometer', 'price']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Clean categorical columns
    cat_cols = ['make', 'model', 'trim', 'province']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.lower()
            df[col] = df[col].str.strip()
            df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
            df[col] = df[col].str.replace(r'[^a-z0-9 ]', '', regex=True)
            df[col] = df[col].replace('', np.nan)
            df[col] = df[col].replace('nan', np.nan)
    
    # Apply aliases
    df['make_clean'] = df['make'].apply(lambda x: normalize_make(x) if pd.notna(x) else x)
    df['model_clean'] = df['model'].apply(lambda x: normalize_model(x) if pd.notna(x) else x)
    
    # Use cleaned versions
    df['make_final'] = df['make_clean'].combine_first(df['make'])
    df['model_final'] = df['model_clean'].combine_first(df['model'])
    
    # Drop rows with missing critical data
    df = df.dropna(subset=['year', 'make_final', 'model_final', 'odometer', 'price'])
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df

# Load data
try:
    df = load_data(CSV_PATH)
    print(f"✅ Data loaded: {len(df)} records")
    
    # Debug info
    print(f"   Makes: {df.make_final.nunique()}")
    print(f"   Models: {df.model_final.nunique()}")
    print(f"   Years range: {df.year.min()} - {df.year.max()}")
    print(f"   Price range: ${df.price.min():,.0f} - ${df.price.max():,.0f}")
    
except Exception as e:
    print(f"❌ Error loading data: {e}")
    df = pd.DataFrame()

# =========================
# VALID VALUES SETUP
# =========================
VALID_MAKES = set(df['make_final'].unique()) if len(df) > 0 else set()
VALID_MODELS = set(df['model_final'].unique()) if len(df) > 0 else set()
VALID_PROVINCES = set(df['province'].unique()) if 'province' in df.columns and len(df) > 0 else set()

# Build year ranges for each make-model
MAKE_MODEL_YEARS = {}
if len(df) > 0:
    for (make, model), group in df.groupby(['make_final', 'model_final']):
        MAKE_MODEL_YEARS[(make, model)] = {
            'min_year': group['year'].min(),
            'max_year': group['year'].max(),
            'years': sorted(group['year'].unique())
        }

# Build trim mapping
TRIM_MAP = defaultdict(set)
if len(df) > 0 and 'trim' in df.columns:
    for _, row in df.iterrows():
        make = row['make_final']
        model = row['model_final']
        trim = row['trim']
        if pd.notna(trim):
            TRIM_MAP[(make, model)].add(trim)

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

class PredictionResult(BaseModel):
    features: dict
    coefficients: Optional[dict]
    predictions: List[dict]

# =========================
# MATCHING FUNCTIONS
# =========================
def match_make(make_input: str) -> Optional[str]:
    """Match manufacturer"""
    make_norm = normalize_make(make_input)
    if not make_norm:
        return None
    return find_best_match(make_norm, VALID_MAKES)

def match_model(model_input: str, make_matched: str) -> Optional[str]:
    """Match model for specific make"""
    if not make_matched:
        return None
    
    model_norm = normalize_model(model_input)
    if not model_norm:
        return None
    
    # Get models for this specific make
    make_models = df[df['make_final'] == make_matched]['model_final'].unique()
    valid_models_for_make = set(make_models)
    
    return find_best_match(model_norm, valid_models_for_make)

def match_province(province_input: Optional[str]) -> Optional[str]:
    """Match province"""
    if not province_input:
        return None
    province_norm = norm(province_input)
    if not province_norm:
        return None
    return find_best_match(province_norm, VALID_PROVINCES)

# =========================
# DATA COLLECTION FOR REGRESSION
# =========================
def get_comparable_data(make: str, model: str, target_year: int):
    """Get all comparable data for regression training"""
    # First, try exact year
    comps = df[
        (df['make_final'] == make) & 
        (df['model_final'] == model) & 
        (df['year'] == target_year)
    ]
    
    # If not enough data, expand to nearby years
    if len(comps) < MIN_COMPARABLES_FOR_REGRESSION:
        for year_diff in range(1, YEAR_RANGE + 1):
            years_to_try = [target_year + year_diff, target_year - year_diff]
            for year in years_to_try:
                additional = df[
                    (df['make_final'] == make) & 
                    (df['model_final'] == model) & 
                    (df['year'] == year)
                ]
                comps = pd.concat([comps, additional])
                
                if len(comps) >= MIN_COMPARABLES_FOR_REGRESSION * 2:  # Get more for robustness
                    break
            if len(comps) >= MIN_COMPARABLES_FOR_REGRESSION * 2:
                break
    
    return comps

# =========================
# REGRESSION MODEL
# =========================
def train_price_regression(comps_df):
    """Train regression model to predict price based on features"""
    if len(comps_df) < MIN_COMPARABLES_FOR_REGRESSION:
        return None, None, None, None, None
    
    try:
        # Prepare features
        X = comps_df[['odometer', 'year']].copy()
        
        # Add province if available and has variation
        if 'province' in comps_df.columns and comps_df['province'].nunique() > 1:
            # Get top provinces (remove rare ones)
            province_counts = comps_df['province'].value_counts()
            common_provinces = province_counts[province_counts >= 2].index.tolist()
            
            if common_provinces:
                for province in common_provinces:
                    X[f'province_{province}'] = (comps_df['province'] == province).astype(int)
        
        # Add trim if available
        if 'trim' in comps_df.columns and comps_df['trim'].nunique() > 1:
            trim_counts = comps_df['trim'].value_counts()
            common_trims = trim_counts[trim_counts >= 2].index.tolist()
            
            if common_trims:
                for trim in common_trims:
                    if pd.notna(trim):
                        X[f'trim_{clean_string(trim)}'] = (comps_df['trim'] == trim).astype(int)
        
        y = comps_df['price'].values
        
        # Remove columns with no variation
        X = X.loc[:, X.nunique() > 1]
        
        if len(X.columns) == 0 or len(X) < len(X.columns) + 2:
            return None, None, None, None, None
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate statistics
        y_pred = model.predict(X_scaled)
        residuals = y - y_pred
        sigma = np.std(residuals, ddof=max(1, len(X.columns)))
        r_squared = model.score(X_scaled, y)
        
        # Get feature names
        feature_names = X.columns.tolist()
        
        # Get coefficients
        coefficients = dict(zip(feature_names, model.coef_))
        if hasattr(model, 'intercept_'):
            coefficients['intercept'] = model.intercept_
        
        return model, scaler, sigma, r_squared, coefficients
    
    except Exception as e:
        print(f"Regression training error: {e}")
        return None, None, None, None, None

def predict_price(model, scaler, features_dict, feature_names):
    """Predict price using trained model"""
    try:
        # Create input DataFrame
        input_df = pd.DataFrame([features_dict])
        
        # Ensure all feature columns exist
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder to match training
        input_df = input_df[feature_names]
        
        # Scale and predict
        X_scaled = scaler.transform(input_df)
        prediction = model.predict(X_scaled)[0]
        
        return max(prediction, 0)  # Ensure non-negative
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

# =========================
# CORE ESTIMATION LOGIC
# =========================
def estimate_value(request: EstimateRequest):
    """Main estimation function"""
    # Normalize inputs
    make_input = request.make
    model_input = request.model
    year_input = request.year
    odometer_input = request.odometer
    province_input = request.province
    trim_input = request.trim
    confidence = min(max(request.confidence, 0.5), 0.99)
    
    # Match make and model
    make_matched = match_make(make_input)
    model_matched = match_model(model_input, make_matched) if make_matched else None
    
    if not make_matched or not model_matched:
        return {
            "success": False,
            "title": f"{year_input} {make_input} {model_input}",
            "message": f"Make or model not found. Make: '{make_input}', Model: '{model_input}'",
            "matched_make": make_matched,
            "matched_model": model_matched,
            "comparables": 0
        }
    
    # Match province
    province_matched = match_province(province_input) if province_input else None
    
    # Get comparable data
    comps = get_comparable_data(make_matched, model_matched, year_input)
    
    if len(comps) == 0:
        # Check available years for this make-model
        key = (make_matched, model_matched)
        if key in MAKE_MODEL_YEARS:
            years_info = MAKE_MODEL_YEARS[key]
            return {
                "success": False,
                "title": f"{year_input} {make_matched} {model_matched}",
                "message": f"No data found for year {year_input}. Available years: {years_info['years']}",
                "matched_make": make_matched,
                "matched_model": model_matched,
                "available_years": years_info['years'],
                "comparables": 0
            }
        else:
            return {
                "success": False,
                "title": f"{year_input} {make_matched} {model_matched}",
                "message": "No comparable data found",
                "matched_make": make_matched,
                "matched_model": model_matched,
                "comparables": 0
            }
    
    # Prepare title
    title = f"{year_input} {make_matched.title()} {model_matched.title()}"
    if trim_input:
        title += f" {trim_input.title()}"
    
    # If we have only 1-2 comparables, use simple average
    if len(comps) < MIN_COMPARABLES_FOR_REGRESSION:
        if len(comps) == 1:
            base = comps.iloc[0]
            adj_price = base['price'] - (odometer_input - base['odometer']) * MILEAGE_RATE
            price = max(adj_price, 0)
            method = "single_comparable_adjusted"
        else:  # 2 comparables
            avg_price = comps['price'].mean()
            # Adjust for average odometer difference
            avg_odometer = comps['odometer'].mean()
            adj_price = avg_price - (odometer_input - avg_odometer) * MILEAGE_RATE
            price = max(adj_price, 0)
            method = "average_adjusted"
        
        return {
            "success": True,
            "title": title,
            "estimated_price": round(price, 2),
            "method": method,
            "comparables": len(comps),
            "matched_make": make_matched,
            "matched_model": model_matched,
            "matched_province": province_matched,
            "features_used": ["odometer", "year"],
            "confidence": confidence,
            "data_summary": {
                "years_used": sorted(comps['year'].unique()),
                "odometer_range": [comps['odometer'].min(), comps['odometer'].max()],
                "price_range": [comps['price'].min(), comps['price'].max()]
            }
        }
    
    # Train regression model
    model, scaler, sigma, r_squared, coefficients = train_price_regression(comps)
    
    if model is None:
        # Fallback to median with adjustment
        median_price = comps['price'].median()
        median_odometer = comps['odometer'].median()
        adj_price = median_price - (odometer_input - median_odometer) * MILEAGE_RATE
        price = max(adj_price, 0)
        
        return {
            "success": True,
            "title": title,
            "estimated_price": round(price, 2),
            "method": "median_adjusted",
            "comparables": len(comps),
            "matched_make": make_matched,
            "matched_model": model_matched,
            "matched_province": province_matched,
            "message": "Regression not possible, using median",
            "data_summary": {
                "years_used": sorted(comps['year'].unique()),
                "odometer_range": [comps['odometer'].min(), comps['odometer'].max()],
                "price_range": [comps['price'].min(), comps['price'].max()]
            }
        }
    
    # Prepare features for prediction
    features = {
        'odometer': odometer_input,
        'year': year_input
    }
    
    # Add province features
    if province_matched:
        province_features = [col for col in scaler.feature_names_in_ if col.startswith('province_')]
        for prov_feature in province_features:
            province_name = prov_feature.replace('province_', '')
            features[prov_feature] = 1 if province_matched == province_name else 0
    
    # Add trim features
    if trim_input:
        trim_clean = clean_string(trim_input)
        trim_features = [col for col in scaler.feature_names_in_ if col.startswith('trim_')]
        for trim_feature in trim_features:
            trim_name = trim_feature.replace('trim_', '')
            features[trim_feature] = 1 if trim_clean == trim_name else 0
    
    # Make prediction
    predicted_price = predict_price(model, scaler, features, scaler.feature_names_in_.tolist())
    
    if predicted_price is None:
        # Fallback to mean
        mean_price = comps['price'].mean()
        return {
            "success": True,
            "title": title,
            "estimated_price": round(mean_price, 2),
            "method": "mean_fallback",
            "comparables": len(comps),
            "matched_make": make_matched,
            "matched_model": model_matched,
            "message": "Prediction failed, using mean"
        }
    
    # Calculate confidence interval
    result = {
        "success": True,
        "title": title,
        "estimated_price": round(predicted_price, 2),
        "method": "regression",
        "comparables": len(comps),
        "matched_make": make_matched,
        "matched_model": model_matched,
        "matched_province": province_matched,
        "r_squared": round(r_squared, 3),
        "coefficients": coefficients,
        "features_used": scaler.feature_names_in_.tolist(),
        "confidence": confidence,
        "data_summary": {
            "years_used": sorted(comps['year'].unique()),
            "odometer_range": [comps['odometer'].min(), comps['odometer'].max()],
            "price_range": [comps['price'].min(), comps['price'].max()],
            "training_years_range": [comps['year'].min(), comps['year'].max()]
        }
    }
    
    # Add confidence interval if we have enough data
    if sigma > 0 and len(comps) > len(scaler.feature_names_in_) + 1:
        try:
            t_val = t.ppf((1 + confidence) / 2, df=len(comps) - len(scaler.feature_names_in_) - 1)
            margin = t_val * sigma
            
            result["confidence_interval"] = {
                "lower": round(max(predicted_price - margin, 0), 2),
                "upper": round(predicted_price + margin, 2),
                "margin": round(margin, 2)
            }
        except:
            pass
    
    return result

# =========================
# API ROUTES
# =========================
@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Vehicle Fair Value API",
        "version": "2.0",
        "description": "Predict vehicle prices using regression on comparable data",
        "endpoints": {
            "/": "This info",
            "/estimate": "POST - Estimate single vehicle",
            "/estimate/batch": "POST - Estimate multiple vehicles",
            "/stats": "GET - Dataset statistics",
            "/search/{make}": "GET - Search for models by make",
            "/models/{make}/{model}": "GET - Get model details"
        }
    }

@app.post("/estimate", response_model=dict)
async def estimate_vehicle(request: EstimateRequest):
    """Estimate value for a single vehicle"""
    return estimate_value(request)

@app.post("/estimate/batch", response_model=List[dict])
async def estimate_batch(request: BatchEstimateRequest):
    """Estimate values for multiple vehicles"""
    return [estimate_value(v) for v in request.vehicles]

@app.get("/stats")
async def get_stats():
    """Get dataset statistics"""
    if len(df) == 0:
        return {"error": "No data loaded"}
    
    stats = {
        "total_records": len(df),
        "makes_count": df['make_final'].nunique(),
        "models_count": df['model_final'].nunique(),
        "years_range": [int(df['year'].min()), int(df['year'].max())],
        "price_stats": {
            "min": float(df['price'].min()),
            "max": float(df['price'].max()),
            "mean": float(df['price'].mean()),
            "median": float(df['price'].median())
        },
        "odometer_stats": {
            "min": float(df['odometer'].min()),
            "max": float(df['odometer'].max()),
            "mean": float(df['odometer'].mean())
        }
    }
    
    if 'province' in df.columns:
        stats["provinces_count"] = df['province'].nunique()
    
    return stats

@app.get("/search/{make}")
async def search_make(make: str):
    """Search for models by make"""
    make_matched = match_make(make)
    
    if not make_matched:
        return {"error": f"Make '{make}' not found"}
    
    models = df[df['make_final'] == make_matched]['model_final'].unique()
    models_list = sorted(models.tolist())
    
    # Get stats for each model
    model_stats = []
    for model in models_list:
        model_data = df[(df['make_final'] == make_matched) & (df['model_final'] == model)]
        stats = {
            "model": model,
            "count": len(model_data),
            "years": sorted(model_data['year'].unique().tolist()),
            "min_year": int(model_data['year'].min()),
            "max_year": int(model_data['year'].max()),
            "price_range": [float(model_data['price'].min()), float(model_data['price'].max())]
        }
        model_stats.append(stats)
    
    return {
        "make": make_matched,
        "models_count": len(models_list),
        "models": model_stats
    }

@app.get("/models/{make}/{model}")
async def get_model_details(make: str, model: str):
    """Get detailed information about a specific model"""
    make_matched = match_make(make)
    model_matched = match_model(model, make_matched) if make_matched else None
    
    if not make_matched or not model_matched:
        return {"error": "Make or model not found"}
    
    model_data = df[(df['make_final'] == make_matched) & (df['model_final'] == model_matched)]
    
    if len(model_data) == 0:
        return {"error": "No data found for this model"}
    
    # Get year-wise statistics
    year_stats = []
    for year in sorted(model_data['year'].unique()):
        year_data = model_data[model_data['year'] == year]
        year_stats.append({
            "year": int(year),
            "count": len(year_data),
            "avg_price": float(year_data['price'].mean()),
            "min_price": float(year_data['price'].min()),
            "max_price": float(year_data['price'].max()),
            "avg_odometer": float(year_data['odometer'].mean())
        })
    
    # Get available features
    features = {
        "has_province": 'province' in model_data.columns and model_data['province'].notna().any(),
        "has_trim": 'trim' in model_data.columns and model_data['trim'].notna().any(),
        "provinces": sorted(model_data['province'].dropna().unique().tolist()) if 'province' in model_data.columns else [],
        "trims": sorted(model_data['trim'].dropna().unique().tolist()) if 'trim' in model_data.columns else []
    }
    
    return {
        "make": make_matched,
        "model": model_matched,
        "total_records": len(model_data),
        "year_range": [int(model_data['year'].min()), int(model_data['year'].max())],
        "price_range": [float(model_data['price'].min()), float(model_data['price'].max())],
        "odometer_range": [float(model_data['odometer'].min()), float(model_data['odometer'].max())],
        "year_stats": year_stats,
        "features": features,
        "regression_ready": len(model_data) >= MIN_COMPARABLES_FOR_REGRESSION
    }

# =========================
# PREDICTION TEST ENDPOINT
# =========================
@app.post("/predict/test")
async def test_prediction(request: EstimateRequest):
    """Test endpoint to see prediction details"""
    result = estimate_value(request)
    
    # Add additional debug info
    if result.get("success"):
        make_matched = result.get("matched_make")
        model_matched = result.get("matched_model")
        
        if make_matched and model_matched:
            # Get sample of training data
            comps = get_comparable_data(make_matched, model_matched, request.year)
            
            if len(comps) > 0:
                sample_data = comps[['year', 'odometer', 'price']].head(5).to_dict(orient='records')
                result["sample_training_data"] = sample_data
                result["training_data_shape"] = comps.shape
    
    return result
