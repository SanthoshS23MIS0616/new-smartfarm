"""
soil_health_thresholds.py
=========================
State-specific soil fertility thresholds from ICAR / TNAU / State Agriculture
Universities for N, P, K (available nutrients) and pH.

Sources:
  - TNAU Agritech Portal (tnau.ac.in)
  - ICAR Handbook of Agriculture (8th edition)
  - State Agriculture Department norms

Units:
  N  : kg/ha (available N, Alkaline KMnO4 method)
  P  : kg/ha (available P2O5, Olsen or Bray method) 
  K  : kg/ha (available K2O, Ammonium acetate method)
  pH : dimensionless

Model units (Kaggle dataset) are mg/kg (ppm) approx:
  To convert model N (ppm) -> kg/ha: multiply by ~1.5 (15cm depth, 1.5g/cc bulk density)
  To convert model P (ppm) -> P2O5 kg/ha: multiply by ~2.3
  To convert model K (ppm) -> K2O kg/ha: multiply by ~1.8
"""
from __future__ import annotations

# Format: (low_max, medium_max) -> above medium_max = High
# N in kg/ha, P in kg/ha (P2O5), K in kg/ha (K2O)
_STATE_THRESHOLDS: dict[str, dict[str, tuple[float, float]]] = {
    "Tamil Nadu": {
        "nitrogen_kg_ha":   (280.0, 560.0),   # TNAU norms: Low<280, Medium 280-560, High>560
        "phosphorus_kg_ha": (11.2,  22.4),    # Olsen method: Low<11.2, Medium 11.2-22.4, High>22.4
        "potassium_kg_ha":  (117.0, 280.0),   # Low<117, Medium 117-280, High>280
        "ph_range":         (5.5,   8.0),
    },
    "Karnataka": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (5.5,   8.0),
    },
    "Andhra Pradesh": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (110.0, 270.0),
        "ph_range":         (5.5,   8.5),
    },
    "Telangana": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (115.0, 275.0),
        "ph_range":         (5.5,   8.5),
    },
    "Kerala": {
        "nitrogen_kg_ha":   (240.0, 480.0),
        "phosphorus_kg_ha": (10.0,  20.0),
        "potassium_kg_ha":  (100.0, 240.0),
        "ph_range":         (5.0,   7.5),
    },
    "Maharashtra": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (6.0,   8.5),
    },
    "Gujarat": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (6.5,   8.5),
    },
    "Punjab": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 300.0),
        "ph_range":         (6.5,   8.5),
    },
    "Haryana": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 300.0),
        "ph_range":         (6.5,   8.5),
    },
    "Uttar Pradesh": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (6.0,   8.5),
    },
    "West Bengal": {
        "nitrogen_kg_ha":   (240.0, 480.0),
        "phosphorus_kg_ha": (10.0,  20.0),
        "potassium_kg_ha":  (100.0, 240.0),
        "ph_range":         (5.0,   7.5),
    },
    "Rajasthan": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (7.0,   9.0),
    },
    "Madhya Pradesh": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (6.0,   8.5),
    },
    "Bihar": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (5.5,   8.0),
    },
    "Odisha": {
        "nitrogen_kg_ha":   (240.0, 480.0),
        "phosphorus_kg_ha": (10.0,  20.0),
        "potassium_kg_ha":  (100.0, 240.0),
        "ph_range":         (5.0,   7.5),
    },
    "Assam": {
        "nitrogen_kg_ha":   (240.0, 480.0),
        "phosphorus_kg_ha": (10.0,  20.0),
        "potassium_kg_ha":  (100.0, 240.0),
        "ph_range":         (5.0,   7.0),
    },
    # Default (national ICAR norms)
    "default": {
        "nitrogen_kg_ha":   (280.0, 560.0),
        "phosphorus_kg_ha": (11.0,  22.0),
        "potassium_kg_ha":  (120.0, 280.0),
        "ph_range":         (5.5,   8.5),
    },
}

# Conversion factors: model ppm units -> kg/ha (ICAR standard)
# Assumes 15cm depth, bulk density 1.5 g/cc
_PPM_TO_KG_HA = {
    "nitrogen":    1.5,    # N ppm -> N kg/ha
    "phosphorous": 2.29,   # P ppm -> P2O5 kg/ha
    "potassium":   1.80,   # K ppm -> K2O kg/ha
}


def get_soil_thresholds(state: str) -> dict:
    """Return ICAR/TNAU soil fertility thresholds for the given state."""
    return _STATE_THRESHOLDS.get(state, _STATE_THRESHOLDS["default"])


def rate_soil_nutrient(value_ppm: float, nutrient: str, state: str) -> dict:
    """
    Rate a soil nutrient value against state norms.
    
    Args:
        value_ppm : Model-scale value (Kaggle dataset units, approx mg/kg)
        nutrient  : 'nitrogen', 'phosphorous', or 'potassium'
        state     : Indian state name
    
    Returns:
        dict with keys: rating (Low/Medium/High), value_kg_ha, low_max, high_min, note
    """
    thresholds = get_soil_thresholds(state)
    conv = _PPM_TO_KG_HA.get(nutrient, 1.5)
    value_kg_ha = round(value_ppm * conv, 1)

    key_map = {
        "nitrogen":    "nitrogen_kg_ha",
        "phosphorous": "phosphorus_kg_ha",
        "potassium":   "potassium_kg_ha",
    }
    thresh_key = key_map.get(nutrient)
    if not thresh_key or thresh_key not in thresholds:
        return {"rating": "Unknown", "value_kg_ha": value_kg_ha}

    low_max, high_min = thresholds[thresh_key]
    if value_kg_ha < low_max:
        rating = "Low"
        color  = "red"
        advice = f"Apply additional {nutrient.upper()} fertilizer. Value {value_kg_ha} kg/ha is below threshold ({low_max} kg/ha)."
    elif value_kg_ha <= high_min:
        rating = "Medium"
        color  = "orange"
        advice = f"{nutrient.title()} is adequate but can be improved. Value: {value_kg_ha} kg/ha."
    else:
        rating = "High"
        color  = "green"
        advice = f"{nutrient.title()} is sufficient. Value: {value_kg_ha} kg/ha."

    return {
        "rating":       rating,
        "color":        color,
        "value_ppm":    round(value_ppm, 1),
        "value_kg_ha":  value_kg_ha,
        "low_max_kg_ha":  low_max,
        "high_min_kg_ha": high_min,
        "advice":       advice,
    }


def rate_ph(ph: float, state: str) -> dict:
    """Rate soil pH against state norms."""
    thresholds = get_soil_thresholds(state)
    lo, hi = thresholds.get("ph_range", (5.5, 8.5))
    if lo <= ph <= hi:
        return {"rating": "Acceptable", "color": "green", "range": f"{lo}–{hi}", "value": ph}
    elif ph < lo:
        return {"rating": "Too Acidic", "color": "red", "range": f"{lo}–{hi}", "value": ph,
                "advice": f"Apply lime to raise pH from {ph} toward {lo}+"}
    else:
        return {"rating": "Too Alkaline", "color": "red", "range": f"{lo}–{hi}", "value": ph,
                "advice": f"Apply gypsum or sulfur to lower pH from {ph} toward {hi}-"}


def full_soil_health_report(nitrogen: float, phosphorous: float, potassium: float,
                             ph: float, state: str) -> dict:
    """Generate a complete soil health report with ratings per nutrient."""
    return {
        "state":       state,
        "nitrogen":    rate_soil_nutrient(nitrogen,    "nitrogen",    state),
        "phosphorous": rate_soil_nutrient(phosphorous, "phosphorous", state),
        "potassium":   rate_soil_nutrient(potassium,   "potassium",   state),
        "ph":          rate_ph(ph, state),
        "source":      "ICAR / TNAU state norms",
    }
