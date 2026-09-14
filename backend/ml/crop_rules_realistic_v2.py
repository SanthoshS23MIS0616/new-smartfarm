from __future__ import annotations

from datetime import date


MONTH_NAMES = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

# crop_type: "annual" | "perennial"
# annual   → short-cycle seasonal crops (duration ≤ 6 months) → standard ranked list
# perennial → orchard/plantation crops  (duration > 6 months) → separate investment track
# irrigation_requirement: "rain_fed_ok" | "needs_irrigation"

CROP_DATABASE: dict[str, dict[str, object]] = {
    # ── ANNUALS ──────────────────────────────────────────────────────────────
    "rice": {
        "crop_type": "annual",
        "duration_months": 5,
        "yield_avg_t_ha": 3.6,
        "water_need": "high",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (20.0, 30.0),
        "base_price_rs_per_kg": 24.0,
        "base_cost_rs_per_ha": 65000.0,
        "sowing_months": [6, 7, 8, 11, 12],
        "peak_harvest_months": [10, 11, 12],
        "new_farmer_penalty": 0.26,
        "base_risk": 0.40,
        "base_sustainability": 0.44,
        "price_volatility": 0.16,
        "chemical_dependency": 0.24,
        "soil_sensitivity": 0.20,
        "mono_crop_penalty": 0.12,
    },
    "maize": {
        "crop_type": "annual",
        "duration_months": 4,
        "yield_avg_t_ha": 3.9,
        "water_need": "medium",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (18.0, 26.0),
        "base_price_rs_per_kg": 22.0,
        "base_cost_rs_per_ha": 52000.0,
        "sowing_months": [2, 3, 4, 6, 7],
        "peak_harvest_months": [8, 9, 10],
        "new_farmer_penalty": 0.24,
        "base_risk": 0.36,
        "base_sustainability": 0.64,
        "price_volatility": 0.17,
        "chemical_dependency": 0.20,
        "soil_sensitivity": 0.20,
        "mono_crop_penalty": 0.10,
    },
    "blackgram": {
        "crop_type": "annual",
        "duration_months": 3,
        "yield_avg_t_ha": 0.95,
        "water_need": "low",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (58.0, 76.0),
        "base_price_rs_per_kg": 66.0,
        "base_cost_rs_per_ha": 34000.0,
        "sowing_months": [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12],
        "peak_harvest_months": [3, 4, 8, 9, 11, 12],

        "new_farmer_penalty": 0.22,
        "base_risk": 0.31,
        "base_sustainability": 0.78,
        "price_volatility": 0.18,
        "chemical_dependency": 0.14,
        "soil_sensitivity": 0.18,
        "mono_crop_penalty": 0.08,
    },
    "jute": {
        "crop_type": "annual",
        "duration_months": 5,
        "yield_avg_t_ha": 2.3,
        "water_need": "medium",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (42.0, 58.0),
        "base_price_rs_per_kg": 50.0,
        "base_cost_rs_per_ha": 62000.0,
        "sowing_months": [3, 4, 5],
        "peak_harvest_months": [8, 9, 10],
        "new_farmer_penalty": 0.24,
        "base_risk": 0.38,
        "base_sustainability": 0.66,
        "price_volatility": 0.20,
        "chemical_dependency": 0.16,
        "soil_sensitivity": 0.20,
        "mono_crop_penalty": 0.08,
    },
    "lentil": {
        "crop_type": "annual",
        "duration_months": 5,
        "yield_avg_t_ha": 1.1,
        "water_need": "low",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (62.0, 84.0),
        "base_price_rs_per_kg": 72.0,
        "base_cost_rs_per_ha": 41000.0,
        "sowing_months": [10, 11],
        "peak_harvest_months": [2, 3],
        "new_farmer_penalty": 0.23,
        "base_risk": 0.33,
        "base_sustainability": 0.79,
        "price_volatility": 0.19,
        "chemical_dependency": 0.13,
        "soil_sensitivity": 0.18,
        "mono_crop_penalty": 0.07,
    },
    "chickpea": {
        "crop_type": "annual",
        "duration_months": 4,
        "yield_avg_t_ha": 1.4,
        "water_need": "low",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (48.0, 72.0),
        "base_price_rs_per_kg": 56.0,
        "base_cost_rs_per_ha": 45000.0,
        "sowing_months": [10, 11, 12],
        "peak_harvest_months": [1, 2, 3],
        "new_farmer_penalty": 0.22,
        "base_risk": 0.32,
        "base_sustainability": 0.82,
        "price_volatility": 0.18,
        "chemical_dependency": 0.14,
        "soil_sensitivity": 0.18,
        "mono_crop_penalty": 0.07,
    },
    "cotton": {
        "crop_type": "annual",
        "duration_months": 6,
        "yield_avg_t_ha": 2.2,
        "water_need": "medium",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (55.0, 82.0),
        "base_price_rs_per_kg": 65.0,
        "base_cost_rs_per_ha": 95000.0,
        "sowing_months": [5, 6, 7],
        "peak_harvest_months": [10, 11, 12, 1],
        "new_farmer_penalty": 0.30,
        "base_risk": 0.46,
        "base_sustainability": 0.48,
        "price_volatility": 0.24,
        "chemical_dependency": 0.35,
        "soil_sensitivity": 0.22,
        "mono_crop_penalty": 0.14,
    },
    "mothbeans": {
        "crop_type": "annual",
        "duration_months": 3,
        "yield_avg_t_ha": 0.8,
        "water_need": "low",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (50.0, 70.0),
        "base_price_rs_per_kg": 58.0,
        "base_cost_rs_per_ha": 30000.0,
        "sowing_months": [6, 7, 8, 9],
        "peak_harvest_months": [9, 10, 11],
        "new_farmer_penalty": 0.20,
        "base_risk": 0.28,
        "base_sustainability": 0.80,
        "price_volatility": 0.17,
        "chemical_dependency": 0.12,
        "soil_sensitivity": 0.15,
        "mono_crop_penalty": 0.06,
    },
    "mungbean": {
        "crop_type": "annual",
        "duration_months": 3,
        "yield_avg_t_ha": 1.0,
        "water_need": "low",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (58.0, 78.0),
        "base_price_rs_per_kg": 66.0,
        "base_cost_rs_per_ha": 38000.0,
        "sowing_months": [2, 3, 6, 7, 10, 11],
        "peak_harvest_months": [4, 5, 8, 9, 12, 1],
        "new_farmer_penalty": 0.22,
        "base_risk": 0.30,
        "base_sustainability": 0.80,
        "price_volatility": 0.18,
        "chemical_dependency": 0.14,
        "soil_sensitivity": 0.17,
        "mono_crop_penalty": 0.07,
    },
    "pigeonpeas": {
        "crop_type": "annual",
        "duration_months": 5,
        "yield_avg_t_ha": 1.5,
        "water_need": "low",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (55.0, 78.0),
        "base_price_rs_per_kg": 62.0,
        "base_cost_rs_per_ha": 50000.0,
        "sowing_months": [6, 7, 9],
        "peak_harvest_months": [11, 12, 1, 2],
        "new_farmer_penalty": 0.24,
        "base_risk": 0.34,
        "base_sustainability": 0.78,
        "price_volatility": 0.19,
        "chemical_dependency": 0.18,
        "soil_sensitivity": 0.18,
        "mono_crop_penalty": 0.08,
    },
    "kidneybeans": {
        "crop_type": "annual",
        "duration_months": 4,
        "yield_avg_t_ha": 1.6,
        "water_need": "medium",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (60.0, 88.0),
        "base_price_rs_per_kg": 70.0,
        "base_cost_rs_per_ha": 58000.0,
        "sowing_months": [10, 11, 1, 2],
        "peak_harvest_months": [1, 2, 4, 5],
        "new_farmer_penalty": 0.25,
        "base_risk": 0.36,
        "base_sustainability": 0.75,
        "price_volatility": 0.20,
        "chemical_dependency": 0.18,
        "soil_sensitivity": 0.20,
        "mono_crop_penalty": 0.08,
    },
    "watermelon": {
        "crop_type": "annual",
        "duration_months": 3,
        "yield_avg_t_ha": 25.0,
        "water_need": "medium",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (8.0, 18.0),
        "base_price_rs_per_kg": 12.0,
        "base_cost_rs_per_ha": 78000.0,
        "sowing_months": [1, 2, 3, 10, 11],
        "peak_harvest_months": [4, 5, 6, 1, 2],
        "new_farmer_penalty": 0.28,
        "base_risk": 0.44,
        "base_sustainability": 0.58,
        "price_volatility": 0.26,
        "chemical_dependency": 0.25,
        "soil_sensitivity": 0.22,
        "mono_crop_penalty": 0.12,
    },
    "muskmelon": {
        "crop_type": "annual",
        "duration_months": 3,
        "yield_avg_t_ha": 18.0,
        "water_need": "medium",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (10.0, 24.0),
        "base_price_rs_per_kg": 15.0,
        "base_cost_rs_per_ha": 72000.0,
        "sowing_months": [1, 2, 3, 10, 11],
        "peak_harvest_months": [4, 5, 6, 12, 1],
        "new_farmer_penalty": 0.28,
        "base_risk": 0.45,
        "base_sustainability": 0.57,
        "price_volatility": 0.27,
        "chemical_dependency": 0.26,
        "soil_sensitivity": 0.22,
        "mono_crop_penalty": 0.12,
    },
    # ── PERENNIALS / ORCHARDS ─────────────────────────────────────────────────

    "apple": {
        "crop_type": "perennial",
        "duration_months": 18,

        "yield_avg_t_ha": 11.0,
        "water_need": "medium",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (28.0, 42.0),
        "base_price_rs_per_kg": 34.0,
        "base_cost_rs_per_ha": 210000.0,
        "sowing_months": [7, 8],
        "peak_harvest_months": [8, 9, 10],
        "new_farmer_penalty": 0.38,
        "base_risk": 0.58,
        "base_sustainability": 0.54,
        "price_volatility": 0.32,
        "chemical_dependency": 0.42,
        "soil_sensitivity": 0.34,
        "mono_crop_penalty": 0.04,
        "payback_years": 4,
    },
    "banana": {
        "crop_type": "perennial",
        "duration_months": 11,
        "yield_avg_t_ha": 28.0,
        "water_need": "high",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (11.0, 18.0),
        "base_price_rs_per_kg": 14.0,
        "base_cost_rs_per_ha": 185000.0,
        "sowing_months": list(range(1, 13)),
        "peak_harvest_months": [9, 10, 11],
        "new_farmer_penalty": 0.34,
        "base_risk": 0.46,
        "base_sustainability": 0.48,
        "price_volatility": 0.24,
        "chemical_dependency": 0.36,
        "soil_sensitivity": 0.26,
        "mono_crop_penalty": 0.06,
        "payback_years": 2,
    },
    "coconut": {
        "crop_type": "perennial",
        "duration_months": 15,
        "yield_avg_t_ha": 8.0,
        "water_need": "high",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (45.0, 70.0),
        "base_price_rs_per_kg": 54.0,
        "base_cost_rs_per_ha": 170000.0,
        "sowing_months": list(range(1, 13)),
        "peak_harvest_months": [9, 10, 11],
        "new_farmer_penalty": 0.37,
        "base_risk": 0.55,
        "base_sustainability": 0.49,
        "price_volatility": 0.28,
        "chemical_dependency": 0.30,
        "soil_sensitivity": 0.32,
        "mono_crop_penalty": 0.04,
        "payback_years": 5,
    },
    "coffee": {
        "crop_type": "perennial",
        "duration_months": 18,
        "yield_avg_t_ha": 1.5,
        "water_need": "medium",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (180.0, 260.0),
        "base_price_rs_per_kg": 210.0,
        "base_cost_rs_per_ha": 155000.0,
        "sowing_months": [6, 7, 8],
        "peak_harvest_months": [11, 12, 1],
        "new_farmer_penalty": 0.39,
        "base_risk": 0.62,
        "base_sustainability": 0.58,
        "price_volatility": 0.30,
        "chemical_dependency": 0.34,
        "soil_sensitivity": 0.32,
        "mono_crop_penalty": 0.04,
        "payback_years": 3,
    },
    "grapes": {
        "crop_type": "perennial",
        "duration_months": 14,
        "yield_avg_t_ha": 16.0,
        "water_need": "medium",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (26.0, 42.0),
        "base_price_rs_per_kg": 31.0,
        "base_cost_rs_per_ha": 200000.0,
        "sowing_months": [10, 11, 12],
        "peak_harvest_months": [2, 3, 4],
        "new_farmer_penalty": 0.36,
        "base_risk": 0.57,
        "base_sustainability": 0.50,
        "price_volatility": 0.26,
        "chemical_dependency": 0.38,
        "soil_sensitivity": 0.28,
        "mono_crop_penalty": 0.04,
        "payback_years": 3,
    },
    "mango": {
        "crop_type": "perennial",
        "duration_months": 18,
        "yield_avg_t_ha": 9.0,
        "water_need": "medium",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (18.0, 32.0),
        "base_price_rs_per_kg": 24.0,
        "base_cost_rs_per_ha": 165000.0,
        "sowing_months": [6, 7, 8],
        "peak_harvest_months": [4, 5, 6],
        "new_farmer_penalty": 0.38,
        "base_risk": 0.55,
        "base_sustainability": 0.56,
        "price_volatility": 0.25,
        "chemical_dependency": 0.31,
        "soil_sensitivity": 0.28,
        "mono_crop_penalty": 0.04,
        "payback_years": 5,
    },
    "orange": {
        "crop_type": "perennial",
        "duration_months": 16,
        "yield_avg_t_ha": 11.0,
        "water_need": "medium",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (18.0, 30.0),
        "base_price_rs_per_kg": 24.0,
        "base_cost_rs_per_ha": 172000.0,
        "sowing_months": [6, 7, 8],
        "peak_harvest_months": [11, 12, 1],
        "new_farmer_penalty": 0.36,
        "base_risk": 0.52,
        "base_sustainability": 0.55,
        "price_volatility": 0.24,
        "chemical_dependency": 0.30,
        "soil_sensitivity": 0.27,
        "mono_crop_penalty": 0.04,
        "payback_years": 4,
    },
    "papaya": {
        "crop_type": "perennial",
        "duration_months": 10,
        "yield_avg_t_ha": 22.0,
        "water_need": "medium",
        "irrigation_requirement": "rain_fed_ok",
        "price_range_rs_per_kg": (11.0, 20.0),
        "base_price_rs_per_kg": 15.0,
        "base_cost_rs_per_ha": 145000.0,
        "sowing_months": list(range(1, 13)),
        "peak_harvest_months": [6, 7, 8],
        "new_farmer_penalty": 0.33,
        "base_risk": 0.43,
        "base_sustainability": 0.57,
        "price_volatility": 0.22,
        "chemical_dependency": 0.28,
        "soil_sensitivity": 0.24,
        "mono_crop_penalty": 0.06,
        "payback_years": 2,
    },
    "pomegranate": {
        "crop_type": "perennial",
        "duration_months": 12,
        "yield_avg_t_ha": 14.0,
        "water_need": "medium",
        "irrigation_requirement": "needs_irrigation",
        "price_range_rs_per_kg": (60.0, 105.0),
        "base_price_rs_per_kg": 78.0,
        "base_cost_rs_per_ha": 165000.0,
        "sowing_months": [1, 2, 6, 7, 9, 10],
        "peak_harvest_months": [7, 8, 12, 1, 3, 4],
        "new_farmer_penalty": 0.35,
        "base_risk": 0.48,
        "base_sustainability": 0.60,
        "price_volatility": 0.25,
        "chemical_dependency": 0.32,
        "soil_sensitivity": 0.25,
        "mono_crop_penalty": 0.05,
        "payback_years": 3,
    },
}

# ── Soil type presets ─────────────────────────────────────────────────────────
SOIL_TYPE_PRESETS: dict[str, dict[str, float]] = {
    "alluvial": {
        "nitrogen": 68.0, "phosphorous": 48.0, "potassium": 55.0,
        "ph": 7.0, "moisture": 30.0,
    },
    "black_cotton": {
        "nitrogen": 55.0, "phosphorous": 35.0, "potassium": 70.0,
        "ph": 7.8, "moisture": 35.0,
    },
    "red_laterite": {
        "nitrogen": 40.0, "phosphorous": 25.0, "potassium": 30.0,
        "ph": 5.8, "moisture": 22.0,
    },
    "sandy_loam": {
        "nitrogen": 45.0, "phosphorous": 30.0, "potassium": 35.0,
        "ph": 6.5, "moisture": 18.0,
    },
    "clay": {
        "nitrogen": 75.0, "phosphorous": 55.0, "potassium": 80.0,
        "ph": 6.8, "moisture": 40.0,
    },
}


def derive_season(month: int) -> str:
    if month in (6, 7, 8, 9, 10):
        return "Kharif"
    if month in (11, 12, 1, 2):
        return "Rabi"
    return "Summer"


def month_name(month: int) -> str:
    return MONTH_NAMES[int(((month - 1) % 12) + 1)]


def harvest_month(sowing_month: int, duration_months: int) -> int:
    return int(((sowing_month - 1 + duration_months - 1) % 12) + 1)


def late_sowing_penalty(month: int, sowing_months: list[int]) -> float:
    if len(sowing_months) <= 1:
        return 0.0
    if month == sowing_months[-1]:
        return 0.25
    if month != sowing_months[0]:
        return 0.12
    return 0.0


def water_need_factor(level: str) -> float:
    return {"low": 0.12, "medium": 0.22, "high": 0.34}.get(level, 0.22)


def mono_crop_penalty(crop_name: str, previous_crop: str | None) -> float:
    """Return an extra yield penalty if the same crop was grown last season."""
    if not previous_crop:
        return 0.0
    if crop_name.strip().lower() == previous_crop.strip().lower():
        rules = CROP_DATABASE.get(crop_name.strip().lower(), {})
        return float(rules.get("mono_crop_penalty", 0.10))
    return 0.0


def irrigation_penalty(crop_rules: dict, irrigation_source: str | None) -> float:
    """Penalise water-hungry crops on rain-fed farms."""
    if not irrigation_source or irrigation_source != "rain_fed":
        return 0.0
    if crop_rules.get("irrigation_requirement") == "needs_irrigation":
        return 0.18
    if crop_rules.get("water_need") == "high":
        return 0.10
    return 0.0


def current_analysis_context() -> dict[str, object]:
    today = date.today()
    return {
        "analysis_date": today.isoformat(),
        "current_month_number": today.month,
        "current_month": month_name(today.month),
        "season": derive_season(today.month),
        "year": today.year,
    }


# ── Preferred ranges per crop (for contradiction detection & rejected detail) ─
# Source: ICAR crop guidelines + Kaggle crop dataset statistics
_PREFERRED_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "rice":        {"nitrogen":(60,130),"phosphorous":(30,80),"potassium":(30,80),"ph":(5.5,7.0),"temperature_c":(20,38),"humidity":(60,95),"rainfall_mm":(100,300)},
    "maize":       {"nitrogen":(50,120),"phosphorous":(25,75),"potassium":(30,80),"ph":(5.5,7.5),"temperature_c":(18,35),"humidity":(50,80),"rainfall_mm":(50,200)},
    "blackgram":   {"nitrogen":(10,50), "phosphorous":(20,70),"potassium":(30,80),"ph":(6.0,7.5),"temperature_c":(22,36),"humidity":(50,80),"rainfall_mm":(50,200)},
    "chickpea":    {"nitrogen":(10,50), "phosphorous":(30,90),"potassium":(30,80),"ph":(5.5,7.5),"temperature_c":(16,30),"humidity":(40,70),"rainfall_mm":(30,120)},
    "lentil":      {"nitrogen":(10,40), "phosphorous":(20,60),"potassium":(20,70),"ph":(6.0,7.5),"temperature_c":(14,30),"humidity":(30,70),"rainfall_mm":(30,120)},
    "cotton":      {"nitrogen":(80,140),"phosphorous":(30,80),"potassium":(40,100),"ph":(6.0,8.0),"temperature_c":(25,40),"humidity":(50,80),"rainfall_mm":(60,200)},
    "jute":        {"nitrogen":(60,120),"phosphorous":(30,80),"potassium":(30,80),"ph":(6.0,7.5),"temperature_c":(24,38),"humidity":(70,95),"rainfall_mm":(150,300)},
    "kidneybeans": {"nitrogen":(10,50), "phosphorous":(25,70),"potassium":(20,70),"ph":(5.5,7.0),"temperature_c":(14,28),"humidity":(50,80),"rainfall_mm":(50,200)},
    "mothbeans":   {"nitrogen":(10,40), "phosphorous":(20,60),"potassium":(20,70),"ph":(5.5,7.5),"temperature_c":(24,40),"humidity":(30,65),"rainfall_mm":(30,120)},
    "mungbean":    {"nitrogen":(10,50), "phosphorous":(25,70),"potassium":(30,80),"ph":(6.0,7.5),"temperature_c":(22,36),"humidity":(50,80),"rainfall_mm":(50,200)},
    "pigeonpeas":  {"nitrogen":(10,50), "phosphorous":(25,70),"potassium":(20,70),"ph":(5.5,7.5),"temperature_c":(18,36),"humidity":(40,80),"rainfall_mm":(60,200)},
    "muskmelon":   {"nitrogen":(60,120),"phosphorous":(30,80),"potassium":(40,100),"ph":(6.0,7.5),"temperature_c":(25,40),"humidity":(30,65),"rainfall_mm":(30,120)},
    "watermelon":  {"nitrogen":(60,120),"phosphorous":(30,80),"potassium":(40,100),"ph":(6.0,7.5),"temperature_c":(25,40),"humidity":(30,65),"rainfall_mm":(30,120)},
    "wheat":       {"nitrogen":(50,120),"phosphorous":(30,80),"potassium":(30,80),"ph":(6.0,7.5),"temperature_c":(12,25),"humidity":(40,70),"rainfall_mm":(30,120)},
    "soybean":     {"nitrogen":(20,80), "phosphorous":(30,80),"potassium":(30,80),"ph":(5.5,7.0),"temperature_c":(20,34),"humidity":(50,80),"rainfall_mm":(60,180)},
    "groundnut":   {"nitrogen":(10,50), "phosphorous":(30,80),"potassium":(30,80),"ph":(5.5,7.0),"temperature_c":(22,36),"humidity":(40,75),"rainfall_mm":(40,160)},
    "sugarcane":   {"nitrogen":(80,140),"phosphorous":(30,80),"potassium":(40,120),"ph":(6.0,7.5),"temperature_c":(24,38),"humidity":(60,90),"rainfall_mm":(80,250)},
    "sunflower":   {"nitrogen":(40,100),"phosphorous":(30,80),"potassium":(30,80),"ph":(6.0,7.5),"temperature_c":(18,35),"humidity":(30,65),"rainfall_mm":(30,120)},
    "sorghum":     {"nitrogen":(40,100),"phosphorous":(25,70),"potassium":(25,70),"ph":(5.5,7.5),"temperature_c":(22,38),"humidity":(30,70),"rainfall_mm":(30,150)},
    "onion":       {"nitrogen":(60,120),"phosphorous":(40,90),"potassium":(40,100),"ph":(6.0,7.5),"temperature_c":(15,30),"humidity":(50,80),"rainfall_mm":(40,120)},
    "tomato":      {"nitrogen":(60,130),"phosphorous":(40,90),"potassium":(40,110),"ph":(5.5,7.5),"temperature_c":(18,32),"humidity":(50,80),"rainfall_mm":(40,150)},
    "chilli":      {"nitrogen":(60,120),"phosphorous":(30,80),"potassium":(30,90),"ph":(5.5,7.5),"temperature_c":(18,35),"humidity":(50,80),"rainfall_mm":(40,150)},
    # Perennials
    "apple":       {"nitrogen":(50,120),"phosphorous":(30,80),"potassium":(40,100),"ph":(5.5,7.0),"temperature_c":(5,20),"humidity":(50,80),"rainfall_mm":(60,200)},
    "mango":       {"nitrogen":(60,120),"phosphorous":(30,80),"potassium":(40,100),"ph":(5.5,7.5),"temperature_c":(24,38),"humidity":(50,85),"rainfall_mm":(60,200)},
    "banana":      {"nitrogen":(80,140),"phosphorous":(40,90),"potassium":(60,140),"ph":(5.5,7.0),"temperature_c":(22,38),"humidity":(60,90),"rainfall_mm":(80,250)},
    "coconut":     {"nitrogen":(40,100),"phosphorous":(20,60),"potassium":(40,120),"ph":(5.5,7.5),"temperature_c":(22,38),"humidity":(60,95),"rainfall_mm":(80,250)},
    "coffee":      {"nitrogen":(50,120),"phosphorous":(30,80),"potassium":(30,80),"ph":(5.5,6.5),"temperature_c":(15,28),"humidity":(60,90),"rainfall_mm":(80,250)},
    "grapes":      {"nitrogen":(40,100),"phosphorous":(30,80),"potassium":(40,120),"ph":(5.5,7.5),"temperature_c":(15,36),"humidity":(30,70),"rainfall_mm":(30,120)},
    "orange":      {"nitrogen":(50,120),"phosphorous":(30,80),"potassium":(40,100),"ph":(5.5,7.5),"temperature_c":(16,34),"humidity":(50,85),"rainfall_mm":(60,200)},
    "papaya":      {"nitrogen":(60,120),"phosphorous":(30,80),"potassium":(40,110),"ph":(5.5,7.5),"temperature_c":(22,38),"humidity":(50,85),"rainfall_mm":(60,200)},
    "pomegranate": {"nitrogen":(30,80), "phosphorous":(20,60),"potassium":(30,80),"ph":(5.5,7.5),"temperature_c":(18,38),"humidity":(30,70),"rainfall_mm":(30,120)},
}

# Inject preferred_ranges into every crop entry
for _crop_key, _crop_data in CROP_DATABASE.items():
    if _crop_key in _PREFERRED_RANGES:
        _crop_data["preferred_ranges"] = _PREFERRED_RANGES[_crop_key]

# ── Data source transparency ──────────────────────────────────────────────────
DATA_SOURCES = {
    "weather":       "Open-Meteo Forecast + Archive API (real-time, 1km resolution)",
    "market_price":  "Agmarknet live API → 6h cache → historical CSV fallback (2018-2024)",
    "yield_data":    "ICRISAT District-Level Database + Kaggle Crop Recommendation Dataset",
    "soil_presets":  "ICAR Soil Health Card norms (alluvial/black/laterite/sandy/clay)",
    "crop_rules":    "ICAR crop production guidelines + Indian MSP notifications 2024",
    "ml_model":      "LightGBM + CatBoost + Stacking Ensemble — trained on 22-crop NPK dataset",
    "profit_type":   "NET profit (revenue minus effective cost). Effective cost = fixed + variable×yield_ratio.",
    "price_year":    "Market prices: 2024 season. Historical CSV: 2018-2024 average.",
}
