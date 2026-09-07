"""
agmarknet_client.py
===================
Fetches live mandi (wholesale) prices from India's Agmarknet API
(data.gov.in, Ministry of Agriculture & Farmers Welfare).

Usage
-----
Set the ``AGMARKNET_API_KEY`` environment variable to your data.gov.in API key.
Register free at: https://data.gov.in/user/register

If the environment variable is not set, or the API call fails (timeout,
rate-limit, etc.), the client silently returns None and the engine
falls back to the static market_prices_realistic.csv values.

API Reference
-------------
Resource ID : 9ef84268-d588-465a-a308-a864a43d0070
Base URL    : https://api.data.gov.in/resource/
Fields      : state, district, market, commodity, variety, grade,
              min_price, max_price, modal_price, arrival_date
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

_API_KEY_ENV = "AGMARKNET_API_KEY"
_BASE_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
_TIMEOUT_SECONDS = 5
_CACHE_TTL_SECONDS = 6 * 3600  # 6-hour cache per crop+state combo

# ── Commodity name mappings (our crop name → Agmarknet commodity name) ──────
_CROP_TO_COMMODITY: dict[str, list[str]] = {
    "Rice": ["Rice", "Paddy(Dhan)(Common)"],
    "Maize": ["Maize"],
    "Blackgram": ["Black Gram (Urd Beans)(Whole)", "Black Gram Dal (Urd Dal)"],
    "Lentil": ["Lentil (Whole)", "Masur Dal"],
    "Jute": ["Jute"],
    "Chickpea": ["Bengal Gram(Gram)(Whole)", "Gram Raw(Gram)"],
    "Kidneybeans": ["Rajgir", "Kidney Beans"],
    "Pigeonpeas": ["Arhar (Tur/Red Gram)(Whole)", "Arhar Dal(Tur Dal)"],
    "Mothbeans": ["Moth (Whole)", "Moth Dal"],
    "Mungbean": ["Moong (Whole)", "Moong Dal"],
    "Cotton": ["Cotton", "Cotton(Kapas)"],
    "Watermelon": ["Water Melon"],
    "Muskmelon": ["Musk Melon"],
    "Apple": ["Apple"],
    "Banana": ["Banana"],
    "Coconut": ["Coconut", "Coconut(Dry)"],
    "Coffee": ["Coffee(Robusta)", "Coffee(Arabica)"],
    "Grapes": ["Grapes"],
    "Mango": ["Mango"],
    "Orange": ["Orange"],
    "Papaya": ["Papaya"],
    "Pomegranate": ["Pomegranate"],
}

# ── Simple in-memory TTL cache ────────────────────────────────────────────────
_cache: dict[str, tuple[float, float | None]] = {}  # key → (timestamp, price_rs_per_kg)


def _cache_get(key: str) -> float | None:
    if key in _cache:
        ts, value = _cache[key]
        if time.time() - ts < _CACHE_TTL_SECONDS:
            return value
    return None


def _cache_set(key: str, value: float | None) -> None:
    _cache[key] = (time.time(), value)


def _api_key() -> str | None:
    return os.environ.get(_API_KEY_ENV)


def _fetch_raw(commodity: str, state: str | None, limit: int = 10) -> list[dict[str, Any]]:
    """Make a single API call. Returns empty list on any error."""
    try:
        import urllib.request
        import json

        key = _api_key()
        if not key:
            return []

        params = [
            f"api-key={key}",
            "format=json",
            f"limit={limit}",
            f"filters[commodity]={commodity.replace(' ', '%20')}",
        ]
        if state:
            params.append(f"filters[state]={state.replace(' ', '%20')}")

        url = f"{_BASE_URL}?{'&'.join(params)}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})

        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
            return data.get("records", [])

    except Exception as exc:  # noqa: BLE001
        logger.debug("Agmarknet API error for %s: %s", commodity, exc)
        return []


def _parse_modal_price(records: list[dict[str, Any]]) -> float | None:
    """Extract median modal price (Rs/quintal → Rs/kg)."""
    prices = []
    for rec in records:
        raw = rec.get("modal_price") or rec.get("min_price") or rec.get("max_price")
        try:
            p = float(str(raw).replace(",", ""))
            if p > 0:
                prices.append(p)
        except (ValueError, TypeError):
            continue
    if not prices:
        return None
    median_quintal = sorted(prices)[len(prices) // 2]
    # Agmarknet prices are in Rs/quintal (100 kg); convert to Rs/kg
    return round(median_quintal / 100.0, 2)


def fetch_price_rs_per_kg(crop: str, state: str | None = None) -> tuple[float | None, str]:
    """
    Return (price_rs_per_kg, source_label).

    ``source_label`` is one of:
      "live_agmarknet"   — freshly fetched from API
      "cached_agmarknet" — returned from 6-hour in-memory cache
      "static_csv"       — API not configured or failed (caller uses fallback)
    """
    if not _api_key():
        return None, "static_csv"

    cache_key = f"{crop}::{state or '*'}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached, "cached_agmarknet"

    commodities = _CROP_TO_COMMODITY.get(crop, [crop])
    for commodity in commodities:
        records = _fetch_raw(commodity, state)
        price = _parse_modal_price(records)
        if price:
            _cache_set(cache_key, price)
            logger.info(
                "Agmarknet live price for %s (%s): Rs %.2f/kg",
                crop, state or "all-India", price,
            )
            return price, "live_agmarknet"

    # All commodities failed — cache None to avoid hammering the API
    _cache_set(cache_key, None)
    return None, "static_csv"


def clear_cache() -> None:
    """Clear the in-memory price cache (useful for testing)."""
    _cache.clear()


def api_configured() -> bool:
    """Return True if an Agmarknet API key is configured."""
    return bool(_api_key())
