"""
main.py
-------
FastAPI application for the Delhi Tourism Recommendation System.

Run with:
    uvicorn main:app --reload

Endpoints:
    GET /                          → Health check
    GET /recommend/{place}         → Full recommendations for a tourist place
    GET /places                    → List all available tourist places
    GET /similar/hotels/{name}     → KNN-similar hotels
    GET /similar/restaurants/{name}→ KNN-similar restaurants
    GET /similar/markets/{name}    → KNN-similar shopping markets
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from recommender import (
    get_recommendations,
    get_place_details,
    recommend_similar_hotels,
    recommend_similar_restaurants,
    recommend_similar_markets,
    tourist_df,
)

# ── App Setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Delhi Tourism Recommendation API",
    description=(
        "A production-ready API that recommends hotels, restaurants, hospitals, "
        "and shopping markets near any Delhi tourist attraction. "
        "Also provides live weather data and KNN-based similarity recommendations."
    ),
    version="1.0.0",
)

# Allow all origins (adjust in production to specific frontend URLs)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
#  ROUTES
# ============================================================

@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "Delhi Tourism API is live 🏛️",
        "docs": "/docs",
    }


@app.get("/places", tags=["Places"])
def list_places():
    """
    Return a list of all tourist place names available in the dataset.
    Use these names with the /recommend/{place} endpoint.
    """
    places = sorted(tourist_df["Name"].str.strip().unique().tolist())
    return {"total": len(places), "places": places}


@app.get("/recommend/{place}", tags=["Recommendations"])
def recommend(place: str):
    """
    Get full travel recommendations for a Delhi tourist place.

    Returns:
    - Place details (rating, entrance fee, best visiting time, etc.)
    - Live weather at the location
    - Top 5 nearest hotels
    - Top 5 nearest restaurants (if dataset available)
    - Top 5 nearest hospitals
    - Top 5 nearest shopping markets
    - Top 5 nearby tourist places
    - A weather-based visit tip
    """
    result = get_recommendations(place)

    # If the place was not found, raise a 404 with helpful message
    if result["error"] is not None:
        raise HTTPException(status_code=404, detail=result["error"])

    return result


@app.get("/place/{place_name}", tags=["Places"])
def place_details(place_name: str):
    """
    Return details for a single tourist place without running full recommendations.
    Useful for quick lookups.
    """
    details = get_place_details(place_name)
    if not details:
        raise HTTPException(
            status_code=404,
            detail=f"Place '{place_name}' not found. Call /places to see all options."
        )
    return details


@app.get("/similar/hotels/{hotel_name}", tags=["Similarity"])
def similar_hotels(hotel_name: str, top_n: int = 5):
    """
    Return KNN-based similar hotels to the one specified.

    Uses features: Rating, Price, Star Rating, Distance to Landmark
    """
    result = recommend_similar_hotels(hotel_name, top_n=top_n)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Hotel '{hotel_name}' not found in the dataset."
        )
    return {"query": hotel_name, "similar_hotels": result}


@app.get("/similar/restaurants/{restaurant_name}", tags=["Similarity"])
def similar_restaurants(restaurant_name: str, top_n: int = 5):
    """
    Return KNN-based similar restaurants to the one specified.
    Requires the zom.csv dataset to be present.

    Uses features: Dining Rating, Price for 2
    """
    result = recommend_similar_restaurants(restaurant_name, top_n=top_n)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Restaurant '{restaurant_name}' not found, or "
                "the restaurants dataset (zom.csv) is not loaded."
            )
        )
    return {"query": restaurant_name, "similar_restaurants": result}


@app.get("/similar/markets/{market_name}", tags=["Similarity"])
def similar_markets(market_name: str, top_n: int = 5):
    """
    Return KNN-based similar shopping markets to the one specified.

    Uses features: Google Rating, Price Level
    """
    result = recommend_similar_markets(market_name, top_n=top_n)
    if result is None:
        raise HTTPException(
            status_code=404,
            detail=f"Market '{market_name}' not found in the dataset."
        )
    return {"query": market_name, "similar_markets": result}
