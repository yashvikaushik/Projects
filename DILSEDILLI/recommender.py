"""
recommender.py
--------------
Core recommendation engine for the Delhi Tourism API.

Responsibilities:
  - Loading and preprocessing all CSV datasets at startup
  - Training KNN models for hotels, restaurants, and shopping markets
  - Exposing clean functions for each recommendation category
  - Providing the master get_recommendations() function
"""

import os
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from typing import Optional

from distance_utils import add_distance_column
from weather import get_weather, get_weather_recommendation

# ── Dataset Paths ────────────────────────────────────────────────────────────
# All CSV files are expected to live in the same directory as this script.
DATA_DIR = os.path.dirname(os.path.abspath(__file__))

TOURIST_CSV  = os.path.join(DATA_DIR, "delhi_tourist_places_100_fixed.csv")
HOTELS_CSV   = os.path.join(DATA_DIR, "delhi_hotels_final_processed.csv")
HOSPITAL_CSV = os.path.join(DATA_DIR, "final_hospital.csv")
SHOPPING_CSV = os.path.join(DATA_DIR, "delhi_unique_shopping_markets_dataset.csv")
RESTAURANTS_CSV = os.path.join(DATA_DIR, "zom.csv")

TOP_N = 5   # Default number of results per category


# ============================================================
#  DATA LOADING
# ============================================================

def _load_datasets():
    """
    Load and preprocess all CSV datasets.
    Called once at module import time.

    Returns:
        Tuple of (tourist_df, hotels_df, hospital_df, shopping_df,
                  restaurants_df, RESTAURANTS_AVAILABLE)
    """
    tourist_df  = pd.read_csv(TOURIST_CSV)
    hotels_df   = pd.read_csv(HOTELS_CSV)
    hospital_df = pd.read_csv(HOSPITAL_CSV)
    shopping_df = pd.read_csv(SHOPPING_CSV)

    # ── Clean hotels numeric columns ────────────────────────
    hotels_df["Price"] = (
        hotels_df["Price"]
        .astype(str).str.replace(",", "", regex=False)
    )
    for col in ["Price", "Rating", "Star Rating"]:
        hotels_df[col] = pd.to_numeric(hotels_df[col], errors="coerce")

    hotels_df["Distance to Landmark"] = (
        hotels_df["Distance to Landmark"]
        .astype(str).str.replace(" km", "", regex=False)
    )
    hotels_df["Distance to Landmark"] = pd.to_numeric(
        hotels_df["Distance to Landmark"], errors="coerce"
    )

    # ── Clean shopping numeric columns ──────────────────────
    for col in ["Price_Level", "Google_Rating"]:
        shopping_df[col] = pd.to_numeric(shopping_df[col], errors="coerce")

    # ── Restaurants (optional) ──────────────────────────────
    restaurants_df = None
    RESTAURANTS_AVAILABLE = False
    try:
        restaurants_df = pd.read_csv(RESTAURANTS_CSV)
        RESTAURANTS_AVAILABLE = True
        print("✅ Restaurants dataset loaded")
    except FileNotFoundError:
        print("⚠️  zom.csv not found — restaurant recommendations will be disabled")

    print(f"✅ Tourist Places   : {tourist_df.shape[0]} records")
    print(f"✅ Hotels           : {hotels_df.shape[0]} records")
    print(f"✅ Hospitals        : {hospital_df.shape[0]} records")
    print(f"✅ Shopping Markets : {shopping_df.shape[0]} records")

    return tourist_df, hotels_df, hospital_df, shopping_df, restaurants_df, RESTAURANTS_AVAILABLE


# ============================================================
#  KNN MODEL TRAINING
# ============================================================

def _train_knn_models(hotels_df, shopping_df, restaurants_df, RESTAURANTS_AVAILABLE):
    """
    Train KNN models for similarity-based recommendations.

    Returns:
        Dictionary containing trained models and scalers.
    """
    models = {}

    # ── Hotels KNN ──────────────────────────────────────────
    hotel_features = ["Rating", "Price", "Star Rating", "Distance to Landmark"]
    X_hotels = hotels_df[hotel_features].fillna(hotels_df[hotel_features].mean())
    scaler_hotels = StandardScaler()
    X_hotels_scaled = scaler_hotels.fit_transform(X_hotels)
    knn_hotels = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn_hotels.fit(X_hotels_scaled)
    models["hotels"] = {
        "knn": knn_hotels, "scaler": scaler_hotels,
        "X_scaled": X_hotels_scaled, "features": hotel_features
    }
    print("✅ Hotels KNN model trained")

    # ── Shopping Markets KNN ────────────────────────────────
    shop_features = ["Google_Rating", "Price_Level"]
    X_shop = shopping_df[shop_features].fillna(shopping_df[shop_features].mean())
    scaler_shop = StandardScaler()
    X_shop_scaled = scaler_shop.fit_transform(X_shop)
    knn_shop = NearestNeighbors(n_neighbors=6, metric="euclidean")
    knn_shop.fit(X_shop_scaled)
    models["shopping"] = {
        "knn": knn_shop, "scaler": scaler_shop,
        "X_scaled": X_shop_scaled, "features": shop_features
    }
    print("✅ Shopping Markets KNN model trained")

    # ── Restaurants KNN (optional) ──────────────────────────
    if RESTAURANTS_AVAILABLE:
        rest_features = ["Dining_Rating", "Pricing_for_2"]
        X_rest = restaurants_df[rest_features].fillna(restaurants_df[rest_features].mean())
        scaler_rest = StandardScaler()
        X_rest_scaled = scaler_rest.fit_transform(X_rest)
        knn_rest = NearestNeighbors(n_neighbors=6, metric="euclidean")
        knn_rest.fit(X_rest_scaled)
        models["restaurants"] = {
            "knn": knn_rest, "scaler": scaler_rest,
            "X_scaled": X_rest_scaled, "features": rest_features
        }
        print("✅ Restaurants KNN model trained")

    return models


# ── Module-level initialisation (runs once on import) ────────────────────────
(
    tourist_df,
    hotels_df,
    hospital_df,
    shopping_df,
    restaurants_df,
    RESTAURANTS_AVAILABLE,
) = _load_datasets()

knn_models = _train_knn_models(hotels_df, shopping_df, restaurants_df, RESTAURANTS_AVAILABLE)


# ============================================================
#  INDIVIDUAL RECOMMENDATION FUNCTIONS
# ============================================================

def get_place_details(place_name: str) -> Optional[dict]:
    """
    Look up a tourist place by name (case-insensitive) and return its details.

    Args:
        place_name: Name of the tourist place to search for

    Returns:
        Dictionary with place details, or None if not found
    """
    tourist_df["Name_lower"] = tourist_df["Name"].str.strip().str.lower()
    match = tourist_df[tourist_df["Name_lower"] == place_name.strip().lower()]

    if match.empty:
        return None

    row = match.iloc[0]
    return _clean_value({
        "name":          row["Name"].strip(),
        "type":          row.get("Type", "N/A"),
        "rating":        row.get("Google review rating", "N/A"),
        "visit_hours":   row.get("time needed to visit in hrs", "N/A"),
        "entrance_fee":  row.get("Entrance Fee in INR", 0),
        "best_time":     row.get("Best Time to visit", "N/A"),
        "weekly_off":    str(row.get("Weekly Off", "N/A")),
        "dslr_allowed":  row.get("DSLR Allowed", "N/A"),
        "latitude":      float(row["Latitude"]),
        "longitude":     float(row["Longitude"]),
    })


def get_nearby_hotels(lat: float, lon: float, top_n: int = TOP_N) -> list[dict]:
    """
    Return the nearest hotels to a given GPS coordinate, sorted by distance.

    Args:
        lat:   Reference latitude
        lon:   Reference longitude
        top_n: Number of results to return

    Returns:
        List of hotel dictionaries with distance information
    """
    df = hotels_df.copy()
    add_distance_column(df, lat, lon)
    nearest = df.sort_values("distance_km").head(top_n)

    results = []
    for _, row in nearest.iterrows():
        results.append(_clean_value({
            "name":         row.get("Hotel Name", "N/A"),
            "rating":       _safe_float(row.get("Rating")),
            "star_rating":  _safe_float(row.get("Star Rating")),
            "price_inr":    _safe_float(row.get("Price")),
            "distance_km":  round(float(row["distance_km"]), 3),
        }))
    return results


def get_nearby_restaurants(lat: float, lon: float, top_n: int = TOP_N) -> list[dict]:
    """
    Return the nearest restaurants to a given GPS coordinate, sorted by distance.
    Returns an empty list if the restaurants dataset is not available.

    Args:
        lat:   Reference latitude
        lon:   Reference longitude
        top_n: Number of results to return

    Returns:
        List of restaurant dictionaries with distance information
    """
    if not RESTAURANTS_AVAILABLE or restaurants_df is None:
        return []

    df = restaurants_df.copy()
    add_distance_column(df, lat, lon)
    nearest = df.sort_values("distance_km").head(top_n)

    results = []
    for _, row in nearest.iterrows():
        results.append(_clean_value({
            "name":          row.get("Restaurant_Name", "N/A"),
            "dining_rating": _safe_float(row.get("Dining_Rating")),
            "price_for_2":   _safe_float(row.get("Pricing_for_2")),
            "distance_km":   round(float(row["distance_km"]), 3),
        }))
    return results


def get_nearby_hospitals(lat: float, lon: float, top_n: int = TOP_N) -> list[dict]:
    """
    Return the nearest hospitals to a given GPS coordinate, sorted by distance.

    Args:
        lat:   Reference latitude
        lon:   Reference longitude
        top_n: Number of results to return

    Returns:
        List of hospital dictionaries with distance information
    """
    df = hospital_df.copy()
    add_distance_column(df, lat, lon, lat_col="LATITUDE", lon_col="LONGITUDE")
    nearest = df.sort_values("distance_km").head(top_n)

    results = []
    for _, row in nearest.iterrows():
        results.append(_clean_value({
            "name":         row.get("Hospital Name", "N/A"),
            "address":      row.get("Address", "N/A"),
            "distance_km":  round(float(row["distance_km"]), 3),
        }))
    return results


def get_nearby_shopping(lat: float, lon: float, top_n: int = TOP_N) -> list[dict]:
    """
    Return the nearest shopping markets to a given GPS coordinate, sorted by distance.

    Args:
        lat:   Reference latitude
        lon:   Reference longitude
        top_n: Number of results to return

    Returns:
        List of shopping market dictionaries with distance information
    """
    df = shopping_df.copy()
    add_distance_column(df, lat, lon)
    nearest = df.sort_values("distance_km").head(top_n)

    results = []
    for _, row in nearest.iterrows():
        results.append(_clean_value({
            "name":          row.get("Market", "N/A"),
            "category":      row.get("Category", "N/A"),
            "best_for":      row.get("Best_For", "N/A"),
            "opening_hours": row.get("Opening_Hours", "N/A"),
            "weekly_off":    str(row.get("Weekly_Off", "N/A")),
            "google_rating": _safe_float(row.get("Google_Rating")),
            "price_level":   _safe_float(row.get("Price_Level")),
            "distance_km":   round(float(row["distance_km"]), 3),
        }))
    return results


def get_nearby_tourist_places(lat: float, lon: float, top_n: int = TOP_N) -> list[dict]:
    """
    Return nearby tourist places (excluding the selected place itself).

    Args:
        lat:   Reference latitude
        lon:   Reference longitude
        top_n: Number of results to return

    Returns:
        List of tourist place dictionaries with distance information
    """
    df = tourist_df.copy()
    add_distance_column(df, lat, lon)
    nearby = df[df["distance_km"] > 0].sort_values("distance_km").head(top_n)

    results = []
    for _, row in nearby.iterrows():
        results.append(_clean_value({
            "name":         row["Name"].strip(),
            "type":         row.get("Type", "N/A"),
            "rating":       _safe_float(row.get("Google review rating")),
            "entrance_fee": row.get("Entrance Fee in INR", 0),
            "distance_km":  round(float(row["distance_km"]), 3),
        }))
    return results


# ============================================================
#  MASTER RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(place_name: str) -> dict:
    """
    Main entry point: given a tourist place name, return a structured
    dictionary with place details, live weather, and all recommendations.

    Args:
        place_name: Name of the tourist place (case-insensitive)

    Returns:
        Dictionary with keys:
            - place       : dict | None
            - weather     : dict | None
            - weather_tip : str
            - hotels      : list[dict]
            - restaurants : list[dict]
            - hospitals   : list[dict]
            - shopping    : list[dict]
            - nearby_places: list[dict]
            - error       : str | None
    """
    # ── 1. Lookup the place ──────────────────────────────────
    place = get_place_details(place_name)

    if place is None:
        available = list(tourist_df["Name"].str.strip().values)
        return {
            "place":        None,
            "weather":      None,
            "weather_tip":  "",
            "hotels":       [],
            "restaurants":  [],
            "hospitals":    [],
            "shopping":     [],
            "nearby_places": [],
            "error":        f"Place '{place_name}' not found. Available places: {available}",
        }

    lat = place["latitude"]
    lon = place["longitude"]

    # ── 2. Fetch live weather ────────────────────────────────
    weather     = get_weather(lat, lon)
    weather_tip = get_weather_recommendation(weather)

    # ── 3. Build all recommendations ────────────────────────
    return {
        "place":         place,
        "weather":       weather,
        "weather_tip":   weather_tip,
        "hotels":        get_nearby_hotels(lat, lon),
        "restaurants":   get_nearby_restaurants(lat, lon),
        "hospitals":     get_nearby_hospitals(lat, lon),
        "shopping":      get_nearby_shopping(lat, lon),
        "nearby_places": get_nearby_tourist_places(lat, lon),
        "error":         None,
    }


# ============================================================
#  SIMILARITY RECOMMENDATION FUNCTIONS  (bonus / optional use)
# ============================================================

def recommend_similar_hotels(hotel_name: str, top_n: int = 5) -> Optional[list[dict]]:
    """Return KNN-based similar hotels for a given hotel name."""
    matches = hotels_df[hotels_df["Hotel Name"] == hotel_name]
    if matches.empty:
        return None

    model = knn_models["hotels"]
    idx = matches.index[0]
    vec = model["X_scaled"][idx].reshape(1, -1)
    _, indices = model["knn"].kneighbors(vec)
    similar = hotels_df.iloc[indices[0][1:top_n + 1]]

    return similar[["Hotel Name", "Rating", "Price", "Star Rating"]].to_dict(orient="records")


def recommend_similar_markets(market_name: str, top_n: int = 5) -> Optional[list[dict]]:
    """Return KNN-based similar shopping markets for a given market name."""
    matches = shopping_df[shopping_df["Market"].str.lower() == market_name.lower()]
    if matches.empty:
        return None

    model = knn_models["shopping"]
    idx = matches.index[0]
    vec = model["X_scaled"][idx].reshape(1, -1)
    _, indices = model["knn"].kneighbors(vec)
    similar = shopping_df.iloc[indices[0][1:top_n + 1]]

    return similar[["Market", "Category", "Google_Rating", "Price_Level", "Best_For"]].to_dict(orient="records")


def recommend_similar_restaurants(restaurant_name: str, top_n: int = 5) -> Optional[list[dict]]:
    """Return KNN-based similar restaurants for a given restaurant name."""
    if not RESTAURANTS_AVAILABLE or "restaurants" not in knn_models:
        return None

    matches = restaurants_df[restaurants_df["Restaurant_Name"] == restaurant_name]
    if matches.empty:
        return None

    model = knn_models["restaurants"]
    idx = matches.index[0]
    vec = model["X_scaled"][idx].reshape(1, -1)
    _, indices = model["knn"].kneighbors(vec)
    similar = restaurants_df.iloc[indices[0][1:top_n + 1]]

    return similar[["Restaurant_Name", "Dining_Rating", "Pricing_for_2"]].to_dict(orient="records")


# ============================================================
#  HELPERS
# ============================================================

def _safe_float(value) -> Optional[float]:
    """Convert a value to native Python float safely, returning None on failure."""
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        f = float(value)
        return None if math.isnan(f) else f
    except (ValueError, TypeError):
        return None


def _safe_int(value) -> Optional[int]:
    """Convert a value to native Python int safely, returning None on failure."""
    try:
        return None if value is None else int(value)
    except (ValueError, TypeError):
        return None


def _clean_value(value):
    """
    Recursively convert numpy/pandas scalar types to native Python types
    so FastAPI's JSON encoder never chokes on them.
    """
    import numpy as np
    import math as _math

    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        f = float(value)
        return None if _math.isnan(f) else f
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and _math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean_value(v) for v in value]
    return value
