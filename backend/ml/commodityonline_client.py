"""
commodityonline_client.py
=========================
Fetches live mandi prices from commodityonline.com
URL pattern: https://www.commodityonline.com/mandiprices/{crop-slug}/{state-slug}

- No API key required
- 4-hour in-memory cache per crop+state
- Falls back to static CSV if scrape fails
- Converts Rs/Quintal -> Rs/kg automatically
"""
from __future__ import annotations

import logging
import re
import time
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

_BASE = "https://www.commodityonline.com/mandiprices"
_TIMEOUT  = 8
_CACHE_TTL = 4 * 3600  # 4 hours

# -- Crop name ? URL slug ----------------------------------------------------
_CROP_SLUG: dict[str, str] = {
    "Rice":        "rice",
    "Maize":       "maize",
    "Blackgram":   "blackgram",
    "Lentil":      "lentil",
    "Jute":        "jute",
    "Chickpea":    "chickpea",
    "Kidneybeans": "kidney-beans",
    "Pigeonpeas":  "arhar-tur",
    "Mothbeans":   "moth-beans",
    "Mungbean":    "moong",
    "Cotton":      "cotton",
    "Soybean":     "soybean",
    "Groundnut":   "groundnut",
    "Wheat":       "wheat",
    "Sorghum":     "jowar",
    "Sugarcane":   "sugarcane",
    "Tomato":      "tomato",
    "Onion":       "onion",
    "Chilli":      "dry-chilli",
    "Watermelon":  "watermelon",
    "Muskmelon":   "muskmelon",
    "Apple":       "apple",
    "Banana":      "banana",
    "Coconut":     "coconut",
    "Coffee":      "coffee",
    "Grapes":      "grapes",
    "Mango":       "mango",
    "Orange":      "orange",
    "Papaya":      "papaya",
    "Pomegranate": "pomegranate",
}

# -- State name ? URL slug ----------------------------------------------------
_STATE_SLUG: dict[str, str] = {
    "Andhra Pradesh":     "andhra-pradesh",
    "Arunachal Pradesh":  "arunachal-pradesh",
    "Assam":              "assam",
    "Bihar":              "bihar",
    "Chhattisgarh":       "chhattisgarh",
    "Goa":                "goa",
    "Gujarat":            "gujarat",
    "Haryana":            "haryana",
    "Himachal Pradesh":   "himachal-pradesh",
    "Jharkhand":          "jharkhand",
    "Karnataka":          "karnataka",
    "Kerala":             "kerala",
    "Madhya Pradesh":     "madhya-pradesh",
    "Maharashtra":        "maharashtra",
    "Manipur":            "manipur",
    "Meghalaya":          "meghalaya",
    "Mizoram":            "mizoram",
    "Nagaland":           "nagaland",
    "Odisha":             "odisha",
    "Punjab":             "punjab",
    "Rajasthan":          "rajasthan",
    "Sikkim":             "sikkim",
    "Tamil Nadu":         "tamil-nadu",
    "Telangana":          "telangana",
    "Tripura":            "tripura",
    "Uttar Pradesh":      "uttar-pradesh",
    "Uttarakhand":        "uttarakhand",
    "West Bengal":        "west-bengal",
}

_cache: dict[str, tuple[float, float | None]] = {}


def _cache_get(key: str) -> float | None:
    entry = _cache.get(key)
    if entry and time.time() - entry[0] < _CACHE_TTL:
        return entry[1]
    return None


def _cache_set(key: str, value: float | None) -> None:
    _cache[key] = (time.time(), value)


def _scrape(crop_slug: str, state_slug: str | None) -> float | None:
    """Scrape modal/average price from commodityonline.com. Returns Rs/kg or None."""
    url = f"{_BASE}/{crop_slug}"
    if state_slug:
        url += f"/{state_slug}"
    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html",
            },
        )
        html = urllib.request.urlopen(req, timeout=_TIMEOUT).read().decode("utf-8", "ignore")

        # Pattern 1: "average X price is Rs NNNN/Quintal"
        m = re.search(
            r"average\s+\w+\s+price\s+is\s+[?Rs\.]+\s*([\d,]+(?:\.\d+)?)/Quintal",
            html, re.IGNORECASE,
        )
        # Pattern 2: "Average Price ... ?NNNN/Quintal"  (in summary box)
        if not m:
            m = re.search(
                r"Average Price[\s\S]{0,200}?([\d,]{4,}(?:\.\d+)?)/Quintal",
                html, re.IGNORECASE,
            )
        # Pattern 3: first standalone Rs/Quintal number
        if not m:
            m = re.search(r"([\d,]{4,}(?:\.\d+)?)/Quintal", html)

        if m:
            quintal_price = float(m.group(1).replace(",", ""))
            if quintal_price > 0:
                return round(quintal_price / 100.0, 2)  # Rs/quintal -> Rs/kg
    except Exception as exc:
        logger.debug("CommodityOnline scrape failed %s/%s: %s", crop_slug, state_slug, exc)
    return None


def fetch_price_rs_per_kg(crop: str, state: str | None = None) -> tuple[float | None, str]:
    """
    Return (price_rs_per_kg, source_label).
    source_label: 'live_commodityonline' | 'cached_commodityonline' | 'static_csv'
    """
    crop_slug  = _CROP_SLUG.get(crop)
    state_slug = _STATE_SLUG.get(state or "") if state else None

    if not crop_slug:
        return None, "static_csv"

    cache_key = f"{crop_slug}::{state_slug or '*'}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached, "cached_commodityonline"

    # Try state-specific first, then all-India
    price = _scrape(crop_slug, state_slug) if state_slug else None
    if price is None:
        price = _scrape(crop_slug, None)

    _cache_set(cache_key, price)
    if price:
        logger.info("CommodityOnline live price %s (%s): Rs%.2f/kg", crop, state or "all-India", price)
        return price, "live_commodityonline"
    return None, "static_csv"


def clear_cache() -> None:
    _cache.clear()
