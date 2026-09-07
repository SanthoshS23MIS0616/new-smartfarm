"""
Reverse geocoding: lat/lng → Indian state name.
Uses Nominatim with a 1-hour in-memory cache + bundled fallback lookup.
"""
from __future__ import annotations

import time
import threading
from typing import Optional
import requests

# ── 1-hour in-memory cache ────────────────────────────────────────────────────
_cache: dict[tuple[float, float], tuple[str, float]] = {}
_lock = threading.Lock()
_TTL = 3600.0  # seconds

# ── Bundled India state name normaliser ───────────────────────────────────────
_STATE_ALIASES: dict[str, str] = {
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chhattisgarh": "Chhattisgarh",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "tamilnadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "west bengal": "West Bengal",
    "delhi": "Delhi",
    "jammu and kashmir": "Jammu and Kashmir",
    "ladakh": "Ladakh",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
}

_HEADERS = {"User-Agent": "CropIntelligencePlatform/2.0 (academic-project)"}

# Offline bounding boxes for Indian states and union territories. These are
# intentionally conservative: they are used only when online reverse geocoding
# fails, so a broad but correct state fallback is better than a dataset default.
_STATE_BOUNDS: tuple[tuple[str, float, float, float, float], ...] = (
    ("Kerala", 8.0, 12.9, 74.8, 77.6),
    ("Tamil Nadu", 8.0, 13.6, 76.0, 80.4),
    ("Karnataka", 11.5, 18.6, 74.0, 78.6),
    ("Goa", 14.8, 15.9, 73.6, 74.4),
    ("Maharashtra", 15.6, 22.1, 72.6, 80.9),
    ("Gujarat", 20.0, 24.8, 68.0, 74.7),
    ("Rajasthan", 23.0, 30.3, 69.3, 78.3),
    ("Punjab", 29.5, 32.6, 73.8, 76.9),
    ("Haryana", 27.6, 30.9, 74.4, 77.7),
    ("Delhi", 28.4, 28.9, 76.8, 77.4),
    ("Uttar Pradesh", 23.8, 30.5, 77.0, 84.7),
    ("Uttarakhand", 28.4, 31.5, 77.5, 81.1),
    ("Himachal Pradesh", 30.2, 33.3, 75.5, 79.1),
    ("Jammu and Kashmir", 32.2, 35.2, 73.5, 76.9),
    ("Ladakh", 32.2, 35.8, 76.0, 80.4),
    ("Madhya Pradesh", 21.0, 26.9, 74.0, 82.9),
    ("Chhattisgarh", 17.7, 24.2, 80.2, 84.4),
    ("Telangana", 15.8, 19.9, 77.1, 81.4),
    ("Andhra Pradesh", 12.6, 19.9, 76.7, 84.8),
    ("Odisha", 17.7, 22.6, 81.4, 87.6),
    ("West Bengal", 21.4, 27.3, 85.8, 89.9),
    ("Bihar", 24.2, 27.6, 83.3, 88.3),
    ("Jharkhand", 21.8, 25.4, 83.3, 87.9),
    ("Sikkim", 27.0, 28.2, 88.0, 89.0),
    ("Assam", 24.0, 28.3, 89.6, 96.1),
    ("Meghalaya", 25.0, 26.2, 89.8, 92.9),
    ("Tripura", 22.9, 24.6, 91.0, 92.5),
    ("Mizoram", 21.9, 24.6, 92.1, 93.7),
    ("Manipur", 23.8, 25.8, 93.0, 94.8),
    ("Nagaland", 25.1, 27.1, 93.2, 95.3),
    ("Arunachal Pradesh", 26.6, 29.5, 91.5, 97.5),
)


def _normalise_state(raw: str) -> str:
    return _STATE_ALIASES.get(raw.strip().lower(), raw.strip().title())


def _fallback_state_from_bounds(lat: float, lng: float) -> Optional[str]:
    for state, min_lat, max_lat, min_lng, max_lng in _STATE_BOUNDS:
        if min_lat <= lat <= max_lat and min_lng <= lng <= max_lng:
            return state
    return None


def reverse_geocode_state(lat: float, lng: float) -> Optional[str]:
    """
    Return the Indian state name for (lat, lng).
    Returns None if lookup fails (caller should use dataset default).
    Results are cached for 1 hour.
    """
    key = (round(lat, 3), round(lng, 3))  # 111 m resolution
    now = time.time()

    with _lock:
        if key in _cache:
            state, ts = _cache[key]
            if now - ts < _TTL:
                return state

    try:
        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lng}&format=json&zoom=5&addressdetails=1"
        )
        resp = requests.get(url, headers=_HEADERS, timeout=4)
        if resp.ok:
            addr = resp.json().get("address", {})
            raw_state = addr.get("state") or addr.get("state_district") or ""
            if raw_state:
                state = _normalise_state(raw_state)
                with _lock:
                    _cache[key] = (state, now)
                return state
    except Exception:
        pass

    fallback_state = _fallback_state_from_bounds(lat, lng)
    if fallback_state:
        with _lock:
            _cache[key] = (fallback_state, now)
        return fallback_state

    return None
