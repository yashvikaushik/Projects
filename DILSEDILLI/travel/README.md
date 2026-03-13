# 🏛️ Delhi Tourism Recommendation API

A production-ready FastAPI backend converted from the `delhi_tourism.ipynb` Jupyter notebook.

---

## 📁 Project Structure

```
delhi_tourism_api/
├── main.py              # FastAPI app — all routes defined here
├── recommender.py       # Core recommendation engine + KNN models
├── weather.py           # OpenWeatherMap API integration
├── distance_utils.py    # Haversine distance utilities
├── requirements.txt     # Python dependencies
│
├── delhi_tourist_places_100_fixed.csv
├── delhi_hotels_final_processed.csv
├── final_hospital.csv
├── delhi_unique_shopping_markets_dataset.csv
└── zom.csv              # Optional — enables restaurant recommendations
```

---

## ⚙️ Setup

### 1. Place your CSV files
Copy all your dataset CSVs into the same folder as `main.py`.

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set your Weather API key (optional — a default key is included)
```bash
# Windows
set WEATHER_API_KEY=your_key_here

# Mac/Linux
export WEATHER_API_KEY=your_key_here
```

### 4. Run the server
```bash
uvicorn main:app --reload
```

The API will be live at: **http://127.0.0.1:8000**

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/places` | List all available tourist places |
| GET | `/recommend/{place}` | **Full recommendations** for a place |
| GET | `/place/{place_name}` | Quick details for a single place |
| GET | `/similar/hotels/{hotel_name}` | KNN-similar hotels |
| GET | `/similar/restaurants/{name}` | KNN-similar restaurants |
| GET | `/similar/markets/{market_name}` | KNN-similar markets |

### Example Request
```
GET http://127.0.0.1:8000/recommend/India Gate
```

### Example Response
```json
{
  "place": {
    "name": "India Gate",
    "type": "Tourist Attraction",
    "rating": 4.6,
    "entrance_fee": 0,
    "best_time": "Oct-Mar",
    "latitude": 28.6129,
    "longitude": 77.2295
  },
  "weather": {
    "temperature": 36.3,
    "feels_like": 33.5,
    "humidity": 9,
    "wind_speed": 2.6,
    "rain_1h": 0,
    "description": "Broken clouds"
  },
  "weather_tip": "Weather looks great — perfect time to explore!",
  "hotels": [ ... ],
  "restaurants": [ ... ],
  "hospitals": [ ... ],
  "shopping": [ ... ],
  "nearby_places": [ ... ],
  "error": null
}
```

---

## 📖 Interactive API Docs
FastAPI auto-generates interactive documentation:
- **Swagger UI**: http://127.0.0.1:8000/docs
- **ReDoc**: http://127.0.0.1:8000/redoc
