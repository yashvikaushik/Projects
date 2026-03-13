"""
distance_utils.py
-----------------
Utility functions for calculating distances between GPS coordinates.
Used throughout the recommendation engine to rank nearby places.
"""

import math


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance (in km) between two GPS coordinates
    using the Haversine formula.

    Args:
        lat1: Latitude of point 1
        lon1: Longitude of point 1
        lat2: Latitude of point 2
        lon2: Longitude of point 2

    Returns:
        Distance in kilometres (float)
    """
    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def add_distance_column(df, ref_lat: float, ref_lon: float,
                        lat_col: str = "Latitude", lon_col: str = "Longitude") -> None:
    """
    Add a 'distance_km' column to a DataFrame, computed from a reference point.

    Args:
        df:       Pandas DataFrame to modify in place
        ref_lat:  Reference latitude
        ref_lon:  Reference longitude
        lat_col:  Name of the latitude column in df
        lon_col:  Name of the longitude column in df
    """
    df["distance_km"] = df.apply(
        lambda row: haversine(ref_lat, ref_lon, row[lat_col], row[lon_col]),
        axis=1
    )
