from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from backend.ml.config_realistic_v2 import (
    CLASSIFICATION_CATEGORICAL_FEATURES,
    CLASSIFICATION_NUMERIC_FEATURES,
    CROP_PROFILES_DATASET_PATH,
    MARKET_PRICES_DATASET_PATH,
    MODEL_DIR,
    PROJECT_MASTER_DATASET_PATH,
    REPORT_DIR,
    REGRESSION_CATEGORICAL_FEATURES,
    REGRESSION_NUMERIC_FEATURES,
)
from backend.ml.crop_rules_realistic_v2 import (
    CROP_DATABASE,
    SOIL_TYPE_PRESETS,
    current_analysis_context,
    derive_season,
    harvest_month,
    irrigation_penalty,
    late_sowing_penalty,
    mono_crop_penalty,
    month_name,
    water_need_factor,
)
from backend.ml.geocoder import reverse_geocode_state


TOP_K_DEFAULT = 5
RANKING_POOL_SIZE = 31  # 22 Kaggle + 9 additional crops
SUITABILITY_THRESHOLD = 0.65

# ── Crop rotation: legumes fix nitrogen → boost next crop ────────────────────
_NITROGEN_FIXERS = {"blackgram", "mungbean", "chickpea", "lentil", "pigeonpeas",
                    "kidneybeans", "mothbeans", "soybean", "groundnut"}

_ROTATION_BOOST: dict[str, set[str]] = {
    # previous_crop → crops that benefit
    "blackgram":  {"maize", "rice", "wheat", "cotton", "tomato", "chilli"},
    "mungbean":   {"maize", "rice", "wheat", "cotton", "tomato", "chilli"},
    "chickpea":   {"wheat", "maize", "sorghum", "cotton"},
    "lentil":     {"wheat", "rice", "maize"},
    "pigeonpeas": {"wheat", "sorghum", "maize", "cotton"},
    "kidneybeans":{"maize", "rice", "tomato"},
    "mothbeans":  {"sorghum", "maize", "wheat"},
    "soybean":    {"maize", "wheat", "cotton", "sunflower"},
    "groundnut":  {"cotton", "maize", "wheat", "sorghum", "sugarcane"},
}


def rotation_bonus(previous_crop: str | None, crop_name: str) -> float:
    """Return a yield bonus (0–0.08) when previous crop fixes nitrogen for crop_name."""
    if not previous_crop:
        return 0.0
    prev = previous_crop.strip().lower()
    crop = crop_name.strip().lower()
    if crop in _ROTATION_BOOST.get(prev, set()):
        return 0.06  # 6% yield bonus from nitrogen carry-over
    return 0.0


def rotation_note(previous_crop: str | None, crop_name: str) -> str | None:
    """Human-readable note about rotation benefit."""
    if not previous_crop:
        return None
    prev = previous_crop.strip().lower()
    crop = crop_name.strip().lower()
    if crop in _ROTATION_BOOST.get(prev, set()):
        return (f"Previous crop ({previous_crop.title()}) is a nitrogen-fixer — "
                f"+6% yield boost applied for {crop_name.title()}.")
    return None


def irrigation_bonus(crop_rules: dict, irrigation_source: str | None) -> float:
    """
    Offset rain-fed penalty for crops when proper irrigation is present.
    Drip/canal irrigation fully negates rain-fed penalty for high-water crops.
    """
    if not irrigation_source or irrigation_source == "rain_fed":
        return 0.0
    water_need = str(crop_rules.get("water_need", "low"))
    irr_req    = str(crop_rules.get("irrigation_requirement", "rain_fed_ok"))
    if irr_req == "needs_irrigation":
        if irrigation_source in ("drip", "canal", "borewell", "sprinkler"):
            return 0.18 if water_need == "high" else 0.10
    return 0.0


def yield_range(expected: float, base: float, penalty: float) -> tuple[float, float, float]:
    """Return (low, expected, high) yield range for display."""
    spread = max(expected * 0.15, 0.05)
    low  = max(round(expected - spread, 2), 0.05)
    high = round(expected + spread, 2)
    return low, round(expected, 2), high


def reconciliation_note(
    crop_name: str,
    rule_concerns: list[str],
    ml_prob: float,
    top_driver: str,
) -> str | None:
    """
    Bridge between rule-based concerns and ML recommendation.
    Returns a plain-English explanation when concerns exist.
    """
    if not rule_concerns:
        return None
    concerns_text = ", ".join(rule_concerns[:3])
    return (
        f"Although {concerns_text.lower()} for {crop_name.title()}, "
        f"the ML model (confidence {ml_prob*100:.0f}%) found that "
        f"{top_driver.replace('_',' ')} has a stronger positive influence "
        f"on {crop_name.title()} suitability under your current conditions."
    )


def format_rejected_detail(
    crop_name: str,
    reason: str,
    prepared: dict,
    crop_rules: dict | None,
) -> dict:
    """Build a structured rejected-crop entry with current vs required values."""
    detail: dict[str, object] = {
        "crop": crop_name,
        "reason": reason,
        "current_vs_required": [],
    }
    if not crop_rules:
        return detail

    preferred = crop_rules.get("preferred_ranges", {})
    checks = {
        "rainfall_mm":    ("Rainfall",     prepared.get("rainfall_mm"),    "mm"),
        "temperature_c":  ("Temperature",  prepared.get("temperature_c"),  "°C"),
        "humidity":       ("Humidity",     prepared.get("humidity"),       "%"),
        "ph":             ("pH",           prepared.get("ph"),             ""),
        "nitrogen":       ("Nitrogen",     prepared.get("nitrogen"),       ""),
        "phosphorous":    ("Phosphorous",  prepared.get("phosphorous"),    ""),
        "potassium":      ("Potassium",    prepared.get("potassium"),      ""),
    }
    rows = []
    for key, (label, current, unit) in checks.items():
        rng = preferred.get(key)
        if rng and current is not None:
            lo, hi = rng
            if not (lo <= float(current) <= hi):
                delta = float(current) - lo if float(current) < lo else float(current) - hi
                rows.append({
                    "parameter": label,
                    "current":   f"{current}{unit}",
                    "required":  f"{lo}–{hi}{unit}",
                    "delta":     f"{delta:+.1f}{unit}",
                    "verdict":   "too low" if float(current) < lo else "too high",
                })
    detail["current_vs_required"] = rows
    return detail


@dataclass
class ModelBundle:
    classification_preprocessor: Any
    regression_preprocessor: Any
    label_encoder: Any
    lgbm_model: Any
    catboost_model: Any
    yield_model: Any
    stacking_model: Any
    metadata: dict[str, Any]


def _to_float32_array(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _candidate_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _clip(value: float, minimum: float, maximum: float) -> float:
    return float(min(max(value, minimum), maximum))


def _format_range(lo: float, hi: float, unit: str = "") -> str:
    return f"{lo:g}-{hi:g}{unit}"


def _report_url(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    try:
        relative = path.relative_to(REPORT_DIR.parent)
    except ValueError:
        relative = Path("realistic_v2") / path.name
    return "/reports/" + relative.as_posix()


def _extract_class_shap(values: Any, class_index: int) -> np.ndarray:
    if isinstance(values, list):
        return np.asarray(values[class_index], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0, :, class_index].reshape(1, -1)
        if arr.shape[1] == 1:
            return arr[class_index, 0, :].reshape(1, -1)
        if arr.shape[0] > class_index:
            return arr[class_index]
        return arr[:, :, class_index]
    raise ValueError(f"Unsupported SHAP class output shape: {arr.shape}")


def _build_metadata(runtime_metadata: dict[str, Any], master_df: pd.DataFrame, profiles_df: pd.DataFrame, market_df: pd.DataFrame) -> dict[str, Any]:
    defaults = {
        "nitrogen": float(master_df["nitrogen"].median()),
        "phosphorous": float(master_df["phosphorous"].median()),
        "potassium": float(master_df["potassium"].median()),
        "temperature_c": float(master_df["temperature_c"].median()),
        "humidity": float(master_df["humidity"].median()),
        "ph": float(master_df["ph"].median()),
        "rainfall_mm": float(master_df["rainfall_mm"].median()),
        "moisture": float(master_df["moisture"].median()),
        "area": float(master_df["area_ha"].median()),
        "season": str(master_df["season"].mode().iloc[0]),
        "state_name": str(master_df["state_name"].mode().iloc[0]),
        "district_name": str(master_df["district_name"].mode().iloc[0]),
        "crop_year": int(master_df["crop_year"].median()),
        "price_per_ton": float(master_df["price_per_ton"].median()),
    }

    climate_ranges = {
        "temperature_c": {"min": float(master_df["temperature_c"].min()), "max": float(master_df["temperature_c"].max())},
        "rainfall_mm": {"min": float(master_df["rainfall_mm"].min()), "max": float(master_df["rainfall_mm"].max())},
        "humidity": {"min": float(master_df["humidity"].min()), "max": float(master_df["humidity"].max())},
        "ph": {"min": float(master_df["ph"].min()), "max": float(master_df["ph"].max())},
    }

    crop_profiles: dict[str, dict[str, Any]] = {}
    for crop_name, frame in profiles_df.groupby("crop", sort=False):
        typical = frame.loc[frame["profile_variant"] == "typical"]
        row = typical.iloc[0] if not typical.empty else frame.iloc[0]
        crop_market = market_df.loc[market_df["crop"] == crop_name, "price_per_ton"]
        crop_yield = master_df.loc[master_df["crop"] == crop_name, "target_yield_t_ha"]
        crop_key = str(crop_name).strip().lower()
        crop_rules = CROP_DATABASE.get(crop_key, {})
        crop_profiles[str(crop_name)] = {
            "min_nitrogen": float(row["min_nitrogen"]),
            "max_nitrogen": float(row["max_nitrogen"]),
            "min_phosphorous": float(row["min_phosphorous"]),
            "max_phosphorous": float(row["max_phosphorous"]),
            "min_potassium": float(row["min_potassium"]),
            "max_potassium": float(row["max_potassium"]),
            "min_temp_c": float(row["min_temp_c"]),
            "max_temp_c": float(row["max_temp_c"]),
            "min_humidity": float(row["min_humidity"]),
            "max_humidity": float(row["max_humidity"]),
            "min_ph": float(row["min_ph"]),
            "max_ph": float(row["max_ph"]),
            "min_rainfall_mm": float(row["min_rainfall_mm"]),
            "max_rainfall_mm": float(row["max_rainfall_mm"]),
            "price_per_ton": float(crop_market.median()) if not crop_market.empty else defaults["price_per_ton"],
            "market_source": "market_prices_realistic.csv",
            "market_type": "state_season_realistic_market",
            "historical_median_yield": float(crop_yield.median()) if not crop_yield.empty else None,
            "duration_months": int(crop_rules.get("duration_months", 6)),
            "sowing_months": list(crop_rules.get("sowing_months", [6, 7, 8])),
            "peak_harvest_months": list(crop_rules.get("peak_harvest_months", [10, 11])),
        }

    shap_summary = runtime_metadata["training_report"].get("shap_summary", {})

    return {
        "dataset_summary": {
            "recommendation_rows": int(runtime_metadata["training_report"]["dataset_summary"]["classification_rows"]),
            "production_rows": int(runtime_metadata["training_report"]["dataset_summary"]["regression_rows"]),
            "crop_count": int(master_df["crop"].nunique()),
            "default_inputs": defaults,
        },
        "training_report": runtime_metadata["training_report"],
        "crop_profiles": crop_profiles,
        "climate_ranges": climate_ranges,
        "ranking_pool_size": RANKING_POOL_SIZE,
        "market_prices": market_df.to_dict(orient="records"),
        "xai_assets": {
            "lightgbm_summary_plot": _report_url(shap_summary.get("lightgbm_summary_plot")),
            "catboost_summary_plot": _report_url(shap_summary.get("catboost_summary_plot")),
            "top_features": shap_summary.get("top_features", []),
        },
    }


def load_model_bundle(model_dir: Path | None = None) -> ModelBundle:
    model_dir = Path(model_dir or MODEL_DIR)
    runtime_metadata = json.loads((model_dir / "runtime_metadata.json").read_text(encoding="utf-8"))
    master_df = pd.read_csv(PROJECT_MASTER_DATASET_PATH)
    profiles_df = pd.read_csv(CROP_PROFILES_DATASET_PATH)
    market_df = pd.read_csv(MARKET_PRICES_DATASET_PATH)
    metadata = _build_metadata(runtime_metadata, master_df, profiles_df, market_df)

    return ModelBundle(
        classification_preprocessor=joblib.load(model_dir / "classification_preprocessor.joblib"),
        regression_preprocessor=joblib.load(model_dir / "regression_preprocessor.joblib"),
        label_encoder=joblib.load(model_dir / "crop_label_encoder.joblib"),
        lgbm_model=joblib.load(model_dir / "lgbm_model.joblib"),
        catboost_model=joblib.load(model_dir / "catboost_model.joblib"),
        yield_model=joblib.load(model_dir / "yield_model.joblib"),
        stacking_model=joblib.load(model_dir / "stacking_model.joblib"),
        metadata=metadata,
    )


class InferenceEngine:
    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self._explainers: dict[str, Any] = {}

    @classmethod
    def from_artifacts(cls, model_dir: Path | None = None) -> "InferenceEngine":
        return cls(load_model_bundle(model_dir))

    def _fill_defaults(self, payload: dict[str, Any]) -> dict[str, Any]:
        defaults = self.bundle.metadata["dataset_summary"]["default_inputs"]
        prepared = defaults.copy()

        # Apply soil type preset BEFORE user overrides so explicit values win
        soil_type = payload.get("soil_type")
        if soil_type and soil_type in SOIL_TYPE_PRESETS:
            prepared.update(SOIL_TYPE_PRESETS[soil_type])

        prepared.update({k: v for k, v in payload.items() if v is not None})
        prepared["crop_year"] = date.today().year
        prepared["_state_source"] = "user provided" if payload.get("state_name") else "dataset default"

        # ── Reverse-geocode lat/lng → real state_name ──────────────────────
        lat = payload.get("latitude")
        lng = payload.get("longitude")
        if lat is not None and lng is not None:
            geocoded_state = reverse_geocode_state(float(lat), float(lng))
            if geocoded_state:
                prepared["state_name"] = geocoded_state
                prepared["_state_source"] = "GPS coordinates"

        # ── IMD: Region-aware season detection (North ≠ South India) ───────
        # South India Kharif extends to November due to NE Monsoon
        try:
            from backend.ml.imd_state_rainfall import derive_season_for_state, get_state_monthly_rainfall
            state_name = str(prepared.get("state_name", ""))
            current_month = date.today().month
            prepared["season"] = derive_season_for_state(current_month, state_name)

            # ── IMD: Historical rainfall normal for this state+month ─────────
            # Stored for advisory context; does NOT override Open-Meteo live value
            imd_normal_mm = get_state_monthly_rainfall(state_name, current_month)
            prepared["_imd_rainfall_normal_mm"] = round(imd_normal_mm, 1)

            # If no live rainfall provided, use IMD normal as sensible default
            if payload.get("rainfall_mm") is None:
                prepared["rainfall_mm"] = imd_normal_mm
                prepared["_rainfall_source"] = "IMD historical normal"
            else:
                prepared["_rainfall_source"] = "user/Open-Meteo"
        except Exception:
            prepared["season"] = derive_season(date.today().month)  # original fallback
            prepared["_imd_rainfall_normal_mm"] = None
            prepared["_rainfall_source"] = "default"

        # ── NPK Soil Adjustment from Previous Crop (from reference project) ─
        # Science: heavy feeders (rice, cotton) deplete soil; legumes restore N
        _PREV_CROP_NPK_IMPACT: dict[str, dict[str, float]] = {
            "rice":      {"n": -15, "p": -3,  "k": -10},
            "maize":     {"n": -20, "p": -8,  "k": -12},
            "wheat":     {"n": -10, "p": -5,  "k": -8},
            "cotton":    {"n": -25, "p": -10, "k": -15},
            "sugarcane": {"n": -30, "p": -12, "k": -20},
            "potato":    {"n": -18, "p": -8,  "k": -15},
            "tomato":    {"n": -12, "p": -6,  "k": -10},
            "onion":     {"n": -8,  "p": -4,  "k": -6},
            "jute":      {"n": -8,  "p": -4,  "k": -8},
            # Nitrogen fixers (legumes) — restore nitrogen
            "blackgram": {"n": +12, "p": -2,  "k": -3},
            "lentil":    {"n": +10, "p": -2,  "k": -3},
            "soybean":   {"n": +15, "p": -5,  "k": -5},
            "groundnut": {"n": +10, "p": -3,  "k": -4},
            "chickpea":  {"n": +12, "p": -2,  "k": -3},
            "mungbean":  {"n": +8,  "p": -2,  "k": -2},
        }
        previous_crop = str(payload.get("previous_crop", "") or "").strip().lower()
        impact = _PREV_CROP_NPK_IMPACT.get(previous_crop)
        prepared["_npk_adjusted"] = False
        prepared["_npk_adjustment_note"] = None
        if impact:
            orig_n = float(prepared.get("nitrogen", 0))
            orig_p = float(prepared.get("phosphorous", 0))
            orig_k = float(prepared.get("potassium", 0))
            prepared["nitrogen"]    = round(max(0.0, orig_n + impact["n"]), 1)
            prepared["phosphorous"] = round(max(0.0, orig_p + impact["p"]), 1)
            prepared["potassium"]   = round(max(0.0, orig_k + impact["k"]), 1)
            prepared["_npk_adjusted"] = True
            direction = "nitrogen-fixing — boosts N" if impact["n"] > 0 else "heavy feeder — depletes nutrients"
            prepared["_npk_adjustment_note"] = (
                f"Previous crop ({previous_crop.title()}) is a {direction}. "
                f"Soil N adjusted {orig_n:g}→{prepared['nitrogen']:g}, "
                f"P {orig_p:g}→{prepared['phosphorous']:g}, "
                f"K {orig_k:g}→{prepared['potassium']:g}."
            )

        return prepared


    def _feature_name(self, raw_name: str) -> str:
        cleaned = raw_name.replace("num__", "").replace("cat__", "")
        cleaned = cleaned.replace("_", " ")
        return cleaned.title()

    def _build_soil_health_report(self, prepared: dict) -> dict:
        """
        Generate ICAR/TNAU state-specific soil health ratings for N, P, K, pH.
        Converts model ppm units to kg/ha for comparison against state norms.
        """
        try:
            from backend.ml.soil_health_thresholds import full_soil_health_report
            state = str(prepared.get("state_name", "default"))
            n   = float(prepared.get("nitrogen",    0))
            p   = float(prepared.get("phosphorous", 0))
            k   = float(prepared.get("potassium",   0))
            ph  = float(prepared.get("ph",          7.0))
            return full_soil_health_report(n, p, k, ph, state)
        except Exception as exc:
            return {"error": str(exc), "source": "ICAR/TNAU norms unavailable"}

    def _get_explainer(self, key: str, model: Any) -> Any | None:
        if key in self._explainers:
            return self._explainers[key]
        try:
            import shap  # type: ignore
        except ImportError:
            self._explainers[key] = None
            return None
        explainer = shap.TreeExplainer(model)
        self._explainers[key] = explainer
        return explainer

    def _top_contributions(self, values: np.ndarray, feature_names: list[str], transformed_row: np.ndarray, top_n: int = 5) -> dict[str, list[dict[str, float | str]]]:
        flat_values = np.asarray(values, dtype=float).reshape(-1)
        flat_row = np.asarray(transformed_row, dtype=float).reshape(-1)
        pairs = [
            {
                "feature": self._feature_name(feature_names[idx]),
                "raw_feature": feature_names[idx],
                "contribution": float(flat_values[idx]),
                "magnitude": float(abs(flat_values[idx])),
                "feature_value": float(flat_row[idx]),
            }
            for idx in range(len(feature_names))
        ]
        pairs = [
            item
            for item in pairs
            if not (str(item["raw_feature"]).startswith("cat__") and abs(float(item["feature_value"])) < 1e-9)
        ]
        positive = sorted((item for item in pairs if item["contribution"] > 0), key=lambda item: item["magnitude"], reverse=True)[:top_n]
        negative = sorted((item for item in pairs if item["contribution"] < 0), key=lambda item: item["magnitude"], reverse=True)[:top_n]
        return {"positive": positive, "negative": negative}

    def _local_shap_explanations(
        self,
        X_class: np.ndarray,
        X_reg: np.ndarray,
        crop_name: str,
    ) -> dict[str, Any]:
        feature_names_cls = list(self.bundle.classification_preprocessor.get_feature_names_out())
        feature_names_reg = list(self.bundle.regression_preprocessor.get_feature_names_out())
        classification = {"available": False, "positive": [], "negative": []}
        regression = {"available": False, "positive": [], "negative": []}

        class_index = int(self.bundle.label_encoder.transform([crop_name])[0])
        cls_explainer = self._get_explainer("lgbm_classifier", self.bundle.lgbm_model)
        if cls_explainer is not None:
            try:
                shap_values = cls_explainer.shap_values(X_class)
                selected = _extract_class_shap(shap_values, class_index)
                contributions = self._top_contributions(selected, feature_names_cls, X_class[0])
                classification = {"available": True, **contributions}
            except Exception:
                classification = {"available": False, "positive": [], "negative": []}

        reg_explainer = self._get_explainer("yield_regressor", self.bundle.yield_model)
        if reg_explainer is not None:
            try:
                shap_values = np.asarray(reg_explainer.shap_values(X_reg), dtype=float)
                contributions = self._top_contributions(shap_values, feature_names_reg, X_reg[0])
                regression = {"available": True, **contributions}
            except Exception:
                regression = {"available": False, "positive": [], "negative": []}

        return {"classification": classification, "yield": regression}

    def _rule_based_explanation(self, prepared: dict[str, Any], profile: dict[str, Any], crop_name: str) -> dict[str, list[str]]:
        positives: list[str] = []
        concerns: list[str] = []

        checks = [
            ("nitrogen", "min_nitrogen", "max_nitrogen", "Nitrogen"),
            ("phosphorous", "min_phosphorous", "max_phosphorous", "Phosphorous"),
            ("potassium", "min_potassium", "max_potassium", "Potassium"),
            ("temperature_c", "min_temp_c", "max_temp_c", "Temperature"),
            ("humidity", "min_humidity", "max_humidity", "Humidity"),
            ("ph", "min_ph", "max_ph", "Soil pH"),
            ("rainfall_mm", "min_rainfall_mm", "max_rainfall_mm", "Rainfall"),
        ]
        for feature, min_key, max_key, label in checks:
            value = float(prepared[feature])
            min_val = float(profile[min_key])
            max_val = float(profile[max_key])
            if min_val <= value <= max_val:
                positives.append(f"{label} is within the suitable range for {crop_name}.")
            else:
                concerns.append(f"{label} is outside the preferred range for {crop_name}.")
        return {"positives": positives[:4], "concerns": concerns[:4]}

    def _range_for(
        self,
        crop_rules: dict[str, Any],
        profile: dict[str, Any],
        feature: str,
        min_key: str,
        max_key: str,
    ) -> tuple[float, float]:
        preferred = crop_rules.get("preferred_ranges", {})
        if isinstance(preferred, dict) and feature in preferred:
            lo, hi = preferred[feature]
            return float(lo), float(hi)
        return float(profile[min_key]), float(profile[max_key])

    def _suitability_assessment(
        self,
        prepared: dict[str, Any],
        profile: dict[str, Any],
        crop_rules: dict[str, Any],
    ) -> dict[str, Any]:
        checks = [
            ("temperature_c", "min_temp_c", "max_temp_c", "Temperature", "C", 0.24, True),
            ("rainfall_mm", "min_rainfall_mm", "max_rainfall_mm", "Rainfall", "mm", 0.20, True),
            ("ph", "min_ph", "max_ph", "Soil pH", "", 0.16, True),
            ("humidity", "min_humidity", "max_humidity", "Humidity", "%", 0.10, False),
            ("nitrogen", "min_nitrogen", "max_nitrogen", "Nitrogen", "", 0.10, False),
            ("phosphorous", "min_phosphorous", "max_phosphorous", "Phosphorous", "", 0.10, False),
            ("potassium", "min_potassium", "max_potassium", "Potassium", "", 0.10, False),
        ]

        weighted_score = 0.0
        total_weight = 0.0
        hard_reasons: list[str] = []
        soft_reasons: list[str] = []
        details: list[dict[str, str]] = []

        for feature, min_key, max_key, label, unit, weight, hard_check in checks:
            value = float(prepared[feature])
            lower, upper = self._range_for(crop_rules, profile, feature, min_key, max_key)
            span = max(upper - lower, 1.0)
            total_weight += weight

            if lower <= value <= upper:
                component = 1.0
                verdict = "within range"
            else:
                gap = lower - value if value < lower else value - upper
                component = max(0.0, 1.0 - (gap / span))
                verdict = "too low" if value < lower else "too high"
                reason = (
                    f"{label} {value:g}{unit} is {verdict}; "
                    f"ideal {_format_range(lower, upper, unit)}"
                )
                if hard_check and gap > (span * 0.35):
                    hard_reasons.append(reason)
                else:
                    soft_reasons.append(reason)

            weighted_score += component * weight
            details.append(
                {
                    "parameter": label,
                    "current": f"{value:g}{unit}",
                    "required": _format_range(lower, upper, unit),
                    "verdict": verdict,
                }
            )

        suitability_score = weighted_score / max(total_weight, 1e-9)

        irrigation_source = prepared.get("irrigation_source")
        if crop_rules.get("irrigation_requirement") == "needs_irrigation" and irrigation_source in (None, "rain_fed"):
            suitability_score = min(suitability_score, 0.74)
            soft_reasons.append("needs assured irrigation but current source is rain-fed or not provided")

        reject = bool(hard_reasons) or suitability_score < SUITABILITY_THRESHOLD
        if reject and not hard_reasons:
            hard_reasons.append(
                f"overall suitability {suitability_score * 100:.0f}% is below "
                f"{SUITABILITY_THRESHOLD * 100:.0f}% threshold"
            )

        return {
            "score": float(_clip(suitability_score, 0.0, 1.0)),
            "reject": reject,
            "reasons": hard_reasons + soft_reasons,
            "details": details,
        }

    def _agronomic_explanation(
        self,
        prepared: dict[str, Any],
        profile: dict[str, Any],
        crop_rules: dict[str, Any],
        crop_name: str,
    ) -> dict[str, list[str]]:
        positives: list[str] = []
        advice: list[str] = []
        checks = [
            ("nitrogen", "min_nitrogen", "max_nitrogen", "Nitrogen", "Apply urea or compost in split doses."),
            ("phosphorous", "min_phosphorous", "max_phosphorous", "Phosphorous", "Apply SSP or DAP before sowing."),
            ("potassium", "min_potassium", "max_potassium", "Potassium", "Apply MOP before sowing."),
            ("ph", "min_ph", "max_ph", "Soil pH", "Correct pH before sowing using lime for acidic soil or organic matter for alkaline soil."),
            ("rainfall_mm", "min_rainfall_mm", "max_rainfall_mm", "Rainfall", "Plan supplemental irrigation or drainage based on the mismatch."),
            ("temperature_c", "min_temp_c", "max_temp_c", "Temperature", "Avoid sowing outside the crop temperature window."),
        ]
        for feature, min_key, max_key, label, recommendation in checks:
            value = float(prepared[feature])
            lower, upper = self._range_for(crop_rules, profile, feature, min_key, max_key)
            if lower <= value <= upper:
                positives.append(f"{label} is suitable for {crop_name}.")
            elif value < lower:
                advice.append(
                    f"{label} is below the recommended range "
                    f"({value:g} vs {_format_range(lower, upper)}). {recommendation}"
                )
            else:
                advice.append(
                    f"{label} is above the recommended range "
                    f"({value:g} vs {_format_range(lower, upper)}). {recommendation}"
                )
        return {"positives": positives[:4], "advice": advice[:5]}

    def _market_price_rs_per_kg(self, crop_name: str, state_name: str, season: str) -> tuple[float, str]:
        """Return (price_rs_per_kg, source_label).

        Priority:
          1. CommodityOnline live scrape (state-specific, then all-India)
          2. Agmarknet API (if AGMARKNET_API_KEY set)
          3. Static market_prices CSV blended with crop-rules baseline
        """
        from backend.ml.commodityonline_client import fetch_price_rs_per_kg as co_fetch
        from backend.ml.agmarknet_client import fetch_price_rs_per_kg as ag_fetch

        crop_key = crop_name.strip().lower()
        crop_rules = CROP_DATABASE.get(crop_key, {})
        minimum, maximum = crop_rules.get("price_range_rs_per_kg", (12.0, 40.0))
        baseline = float(crop_rules.get("base_price_rs_per_kg", (float(minimum) + float(maximum)) / 2.0))

        # ── 1. CommodityOnline (no key needed, always available) ─────────────
        live_price, source = co_fetch(crop_name, state_name)

        # ── 2. Agmarknet fallback (requires API key) ─────────────────────────
        if live_price is None:
            live_price, source = ag_fetch(crop_name, state_name)

        if live_price is not None:
            # Blend 65% live + 35% baseline to smooth extreme outliers
            blended = (0.65 * live_price) + (0.35 * baseline)
            return _clip(blended, float(minimum), float(maximum)), source

        # ── 3. Static CSV fallback ───────────────────────────────────────────
        market_df = pd.DataFrame(self.bundle.metadata["market_prices"])
        exact = market_df[
            (market_df["crop"] == crop_name)
            & (market_df["state_name"] == state_name)
            & (market_df["season"].astype(str).str.strip() == season)
        ]
        if not exact.empty:
            observed = float(exact["price_per_ton"].median()) / 1000.0
        else:
            crop_only = market_df[market_df["crop"] == crop_name]
            observed = float(crop_only["price_per_ton"].median()) / 1000.0 if not crop_only.empty else baseline
        return _clip((0.55 * baseline) + (0.45 * observed), float(minimum), float(maximum)), "static_csv"


    def _build_regression_frame(self, prepared: dict[str, Any], crop_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "area_ha": float(prepared["area"]),
                    "nitrogen": float(prepared["nitrogen"]),
                    "phosphorous": float(prepared["phosphorous"]),
                    "potassium": float(prepared["potassium"]),
                    "temperature_c": float(prepared["temperature_c"]),
                    "humidity": float(prepared["humidity"]),
                    "ph": float(prepared["ph"]),
                    "rainfall_mm": float(prepared["rainfall_mm"]),
                    "moisture": float(prepared["moisture"]),
                    "crop": crop_name,
                    "state_name": prepared["state_name"],
                    "district_name": prepared["district_name"],
                    "season": prepared["season"],
                }
            ]
        )

    def _baseline_yield(self, crop_name: str, predicted_yield: float, profile: dict[str, Any], crop_rules: dict[str, Any]) -> float:
        yield_candidates = [float(predicted_yield), float(crop_rules["yield_avg_t_ha"])]
        if profile.get("historical_median_yield") is not None:
            yield_candidates.append(float(profile["historical_median_yield"]))
        return max(float(np.median(yield_candidates)), 0.15)

    def _climate_adjustments(self, prepared: dict[str, Any], profile: dict[str, Any], crop_rules: dict[str, Any]) -> tuple[float, bool, list[str]]:
        penalties: list[float] = []
        reasons: list[str] = []

        rainfall = float(prepared["rainfall_mm"])
        rain_min, rain_max = self._range_for(
            crop_rules, profile, "rainfall_mm", "min_rainfall_mm", "max_rainfall_mm"
        )
        if rainfall < rain_min:
            deficit = (rain_min - rainfall) / max(rain_min, 1.0)
            penalties.append(0.10 if deficit <= 0.30 else 0.22)
            reasons.append("rainfall deficit")
            if deficit > 0.55:
                return 1.0, True, reasons + ["severe rainfall mismatch"]
        elif rainfall > rain_max:
            excess = (rainfall - rain_max) / max(rain_max, 1.0)
            penalties.append(0.08 if excess <= 0.30 else 0.16)
            reasons.append("rainfall excess")
            if excess > 0.70 and crop_rules["water_need"] == "low":
                return 1.0, True, reasons + ["severe rainfall excess"]

        temperature = float(prepared["temperature_c"])
        temp_min, temp_max = self._range_for(
            crop_rules, profile, "temperature_c", "min_temp_c", "max_temp_c"
        )
        temp_span = max(temp_max - temp_min, 1.0)
        if temperature < temp_min:
            mismatch = (temp_min - temperature) / temp_span
            penalties.append(0.08 if mismatch <= 0.30 else 0.18)
            reasons.append("temperature below range")
            if mismatch > 0.55:
                return 1.0, True, reasons + ["severe temperature mismatch"]
        elif temperature > temp_max:
            mismatch = (temperature - temp_max) / temp_span
            penalties.append(0.08 if mismatch <= 0.30 else 0.18)
            reasons.append("temperature above range")
            if mismatch > 0.55:
                return 1.0, True, reasons + ["severe temperature mismatch"]

        humidity = float(prepared["humidity"])
        humidity_min, humidity_max = self._range_for(
            crop_rules, profile, "humidity", "min_humidity", "max_humidity"
        )
        if humidity < humidity_min or humidity > humidity_max:
            penalties.append(0.03)
            reasons.append("humidity mismatch")

        ph_value = float(prepared["ph"])
        ph_min, ph_max = self._range_for(crop_rules, profile, "ph", "min_ph", "max_ph")
        if ph_value < ph_min or ph_value > ph_max:
            ph_gap = min(abs(ph_value - ph_min), abs(ph_value - ph_max))
            penalties.append(0.05 if ph_gap <= 0.6 else 0.10)
            reasons.append("soil pH mismatch")
            if ph_gap > 1.2:
                return 1.0, True, reasons + ["severe soil pH mismatch"]

        nutrient_penalty = 0.0
        for key, min_key, max_key in [
            ("nitrogen", "min_nitrogen", "max_nitrogen"),
            ("phosphorous", "min_phosphorous", "max_phosphorous"),
            ("potassium", "min_potassium", "max_potassium"),
        ]:
            value = float(prepared[key])
            lower, upper = self._range_for(crop_rules, profile, key, min_key, max_key)
            if value < lower or value > upper:
                nutrient_penalty += 0.02
        if nutrient_penalty:
            penalties.append(min(nutrient_penalty, 0.06))
            reasons.append("nutrient mismatch")

        moisture = float(prepared["moisture"])
        if crop_rules["water_need"] == "high" and moisture < 28:
            penalties.append(0.06)
            reasons.append("low field moisture")
        elif crop_rules["water_need"] == "low" and moisture > 55:
            penalties.append(0.03)
            reasons.append("excess field moisture")

        return min(sum(penalties), 0.70), False, reasons

    def _price_adjustment(self, crop_name: str, sowing_month: int, crop_rules: dict[str, Any], prepared: dict[str, Any]) -> tuple[float, str, str]:
        base_price, market_source = self._market_price_rs_per_kg(crop_name, str(prepared["state_name"]), str(prepared["season"]))
        harvest_month_number = harvest_month(sowing_month, int(crop_rules["duration_months"]))
        peak_months = set(crop_rules["peak_harvest_months"])
        if harvest_month_number in peak_months:
            factor = 0.82 if float(crop_rules["price_volatility"]) >= 0.25 else 0.88
            reason = "peak harvest supply"
        elif float(crop_rules["price_volatility"]) >= 0.26:
            factor = 1.12
            reason = "off-season support"
        else:
            factor = 1.03
            reason = "stable market"
        adjusted = base_price * factor
        minimum, maximum = crop_rules["price_range_rs_per_kg"]
        return _clip(adjusted, float(minimum), float(maximum)), reason, market_source

    def _cost_model(self, crop_rules: dict[str, Any], area_ha: float) -> dict[str, float]:
        base_cost = float(crop_rules["base_cost_rs_per_ha"]) * 0.75
        fixed_cost = base_cost * 0.22
        fertilizers = base_cost * 0.18
        pesticides = base_cost * float(crop_rules["chemical_dependency"]) * 0.18
        irrigation = base_cost * water_need_factor(str(crop_rules["water_need"]))
        labour = base_cost * 0.20
        machinery = base_cost * 0.08
        post_harvest = base_cost * 0.09
        subtotal = fixed_cost + fertilizers + pesticides + irrigation + labour + machinery + post_harvest
        buffer = subtotal * 0.12
        total_cost_rs_per_ha = subtotal + buffer
        return {
            "fixed_cost_rs_per_ha": fixed_cost,
            "fertilizer_cost_rs_per_ha": fertilizers,
            "pesticide_cost_rs_per_ha": pesticides,
            "irrigation_cost_rs_per_ha": irrigation,
            "labour_cost_rs_per_ha": labour,
            "machinery_cost_rs_per_ha": machinery,
            "post_harvest_cost_rs_per_ha": post_harvest,
            "buffer_cost_rs_per_ha": buffer,
            "total_cost_rs_per_ha": total_cost_rs_per_ha,
            "total_cost": total_cost_rs_per_ha * area_ha,
        }

    def _stable_price_rs_per_kg(self, crop_name: str, prepared: dict[str, Any], crop_rules: dict[str, Any]) -> float:
        base_price, _source = self._market_price_rs_per_kg(crop_name, str(prepared["state_name"]), str(prepared["season"]))
        minimum, maximum = crop_rules["price_range_rs_per_kg"]
        baseline = float(crop_rules["base_price_rs_per_kg"])
        stable_price = (0.70 * base_price) + (0.30 * baseline)
        return _clip(stable_price, float(minimum), float(maximum))

    def predict(self, payload: dict[str, Any], top_k: int = TOP_K_DEFAULT) -> dict[str, Any]:
        analysis_context = current_analysis_context()
        current_month = int(analysis_context["current_month_number"])
        current_season = str(analysis_context["season"])
        analysis_date = str(analysis_context["analysis_date"])
        prepared = self._fill_defaults(payload)
        class_frame = pd.DataFrame(
            [
                {
                    **{feature: prepared.get(feature) for feature in CLASSIFICATION_NUMERIC_FEATURES},
                    **{feature: prepared.get(feature) for feature in CLASSIFICATION_CATEGORICAL_FEATURES},
                }
            ]
        )
        X_class = _to_float32_array(self.bundle.classification_preprocessor.transform(class_frame))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            lgbm_probs = np.asarray(self.bundle.lgbm_model.predict_proba(X_class))[0]
            stacked_probs = np.asarray(self.bundle.stacking_model.predict_proba(X_class))[0]

        labels = self.bundle.label_encoder.inverse_transform(np.arange(len(stacked_probs)))
        ranked_indices = np.argsort(stacked_probs)[::-1][: self.bundle.metadata["ranking_pool_size"]]
        crop_profiles = self.bundle.metadata["crop_profiles"]
        area_ha = float(prepared["area"])

        previous_crop: str | None = payload.get("previous_crop")
        irrigation_source: str | None = payload.get("irrigation_source")

        # ── Parallel prefetch live mandi prices for all candidates ────────────
        try:
            from backend.ml.commodityonline_client import bulk_prefetch
            candidate_crops = [str(labels[idx]) for idx in ranked_indices]
            bulk_prefetch(candidate_crops, str(prepared.get("state_name")))
        except Exception:
            pass

        annual_candidates: list[dict[str, Any]] = []
        perennial_candidates: list[dict[str, Any]] = []
        rejected: list[dict[str, str]] = []

        for idx in ranked_indices:
            crop_name = str(labels[idx])
            crop_key = crop_name.strip().lower()
            crop_rules = CROP_DATABASE.get(crop_key)
            profile = crop_profiles.get(crop_name, {})
            if not crop_rules or not profile:
                rejected.append({"crop": crop_name, "reason": "missing crop profile or crop rule"})
                continue

            is_perennial = str(crop_rules.get("crop_type", "annual")) == "perennial"
            sowing_months = list(crop_rules["sowing_months"])
            suitability = self._suitability_assessment(prepared, profile, crop_rules)
            if suitability["reject"]:
                rejected.append({
                    "crop": crop_name,
                    "reason": "; ".join(suitability["reasons"][:3]),
                })
                continue

            # ── Perennial crops → separate investment track, skip sowing-window filter ──
            if is_perennial:
                reg_frame = self._build_regression_frame(prepared, crop_name)
                X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    ml_yield = float(self.bundle.yield_model.predict(X_reg)[0])

                base_yield_t_ha = self._baseline_yield(crop_name, ml_yield, profile, crop_rules)
                climate_pen, should_reject, climate_notes = self._climate_adjustments(prepared, profile, crop_rules)
                if should_reject:
                    rejected.append({"crop": crop_name, "reason": ", ".join(climate_notes)})
                    continue

                irr_pen = irrigation_penalty(crop_rules, irrigation_source)
                total_pen = min(float(crop_rules["new_farmer_penalty"]) + climate_pen + irr_pen, 0.58)
                expected_yield = max(base_yield_t_ha * (1.0 - total_pen), 0.15)
                stable_price = self._stable_price_rs_per_kg(crop_name, prepared, crop_rules)
                cost_model = self._cost_model(crop_rules, area_ha)
                revenue_ha = expected_yield * stable_price * 1000.0
                # Amortize setup cost over payback years for realistic annual profit
                payback_years = int(crop_rules.get("payback_years", 3))
                annual_cost_ha = cost_model["total_cost_rs_per_ha"] / payback_years
                profit_ha = revenue_ha - annual_cost_ha
                irr_note = (" Irrigation needed — rain-fed penalty applied." if irr_pen > 0 else "")
                perennial_candidates.append({
                    "crop": crop_name,
                    "crop_type": "perennial",
                    "payback_years": payback_years,
                    "expected_yield_t_ha": float(expected_yield),
                    "stable_price_rs_per_kg": float(stable_price),
                    "total_cost_rs_per_ha": float(cost_model["total_cost_rs_per_ha"]),
                    "revenue_rs_per_ha": float(revenue_ha),
                    "profit_rs_per_ha": float(profit_ha),  # annual profit after amortization
                    "duration_months": int(crop_rules["duration_months"]),
                    "risk": float(_clip(
                        float(crop_rules["base_risk"]) * 0.34 + water_need_factor(str(crop_rules["water_need"])) * 0.22
                        + float(crop_rules["price_volatility"]) * 0.14 + float(crop_rules["new_farmer_penalty"]) * 0.16
                        + climate_pen * 0.14, 0.12, 0.88)),
                    "sustainability_score": float(_clip(
                        float(crop_rules["base_sustainability"])
                        - (0.12 if crop_rules["water_need"] == "high" else 0.04 if crop_rules["water_need"] == "medium" else 0.0)
                        - float(crop_rules["chemical_dependency"]) * 0.18
                        - float(crop_rules["soil_sensitivity"]) * 0.10, 0.18, 0.92)),
                    "investment_note": (
                        f"Profit shown is annual avg after {payback_years}-yr amortisation. "
                        f"Total setup cost: ₹{cost_model['total_cost_rs_per_ha']:,.0f}/ha "
                        f"(≈₹{annual_cost_ha:,.0f}/ha/yr). First harvest in ~{payback_years} yrs."
                        + irr_note
                    ),
                    "agronomic_suitability": float(suitability["score"]),
                    "agronomic_suitability_pct": round(float(suitability["score"]) * 100, 1),
                    "climate_notes": climate_notes or ["good climate fit"],
                })
                continue

            # ── Annual crops → standard sowing-window ranked list ──
            if current_month not in sowing_months:
                rejected.append({"crop": crop_name, "reason": f"outside sowing window for {month_name(current_month)}"})
                continue

            reg_frame = self._build_regression_frame(prepared, crop_name)
            X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                ml_yield = float(self.bundle.yield_model.predict(X_reg)[0])

            base_yield_t_ha = self._baseline_yield(crop_name, ml_yield, profile, crop_rules)
            climate_penalty, should_reject, climate_notes = self._climate_adjustments(prepared, profile, crop_rules)
            if should_reject:
                rejected.append({"crop": crop_name, "reason": ", ".join(climate_notes)})
                continue

            sowing_penalty    = late_sowing_penalty(current_month, sowing_months)
            new_farmer_penalty = float(crop_rules["new_farmer_penalty"])
            mono_pen          = mono_crop_penalty(crop_name, previous_crop)
            irr_pen           = irrigation_penalty(crop_rules, irrigation_source)
            irr_bon           = irrigation_bonus(crop_rules, irrigation_source)   # Fix #8
            rot_bon           = rotation_bonus(previous_crop, crop_name)           # Fix #10
            rot_note_text     = rotation_note(previous_crop, crop_name)

            # Net penalty after bonuses (capped 0–0.55)
            net_penalty = max(0.0, min(
                new_farmer_penalty + climate_penalty + sowing_penalty + mono_pen + irr_pen
                - irr_bon - rot_bon,
                0.55
            ))
            # Floor = 30% of base yield
            expected_yield_t_ha = max(
                base_yield_t_ha * (1.0 - net_penalty),
                base_yield_t_ha * 0.30
            )
            # Yield confidence range for display
            y_low, y_exp, y_high = yield_range(expected_yield_t_ha, base_yield_t_ha, net_penalty)

            adjusted_price_rs_per_kg, price_reason, market_source = self._price_adjustment(crop_name, current_month, crop_rules, prepared)
            harvest_month_number = harvest_month(current_month, int(crop_rules["duration_months"]))
            cost_model = self._cost_model(crop_rules, area_ha)
            base_total_cost = cost_model["total_cost_rs_per_ha"]
            yield_ratio = expected_yield_t_ha / max(base_yield_t_ha, 0.01)
            effective_cost_rs_per_ha = base_total_cost * (0.45 + 0.55 * yield_ratio)
            revenue_rs_per_ha = expected_yield_t_ha * adjusted_price_rs_per_kg * 1000.0
            profit_rs_per_ha  = revenue_rs_per_ha - effective_cost_rs_per_ha
            revenue = revenue_rs_per_ha * area_ha
            profit  = profit_rs_per_ha  * area_ha


            risk = _clip(
                (float(crop_rules["base_risk"]) * 0.30)
                + (water_need_factor(str(crop_rules["water_need"])) * 0.25)
                + (float(crop_rules["price_volatility"]) * 0.20)
                + (float(crop_rules["new_farmer_penalty"]) * 0.15)
                + (climate_penalty * 0.10),
                0.12,
                0.88,
            )
            sustainability = _clip(
                float(crop_rules["base_sustainability"])
                - (0.12 if crop_rules["water_need"] == "high" else 0.04 if crop_rules["water_need"] == "medium" else 0.0)
                - (float(crop_rules["chemical_dependency"]) * 0.18)
                - (float(crop_rules["soil_sensitivity"]) * 0.10),
                0.18,
                0.92,
            )

            advisory: list[str] = [
                f"Current month is {month_name(current_month)} and season is {current_season}.",
                f"Price adjusted for {price_reason}.",
                "Yield is reduced for new farmer conditions, sowing timing, and climate fit.",
            ]
            if mono_pen > 0:
                advisory.append(f"Mono-cropping penalty applied — {crop_name} was grown last season.")
            if irr_pen > 0:
                advisory.append("Irrigation penalty applied — this crop needs irrigation on a rain-fed farm.")

            annual_candidates.append(
                {
                    "crop": crop_name,
                    "crop_type": "annual",
                    # ML probabilities
                    "classification_probability": float(stacked_probs[idx]),
                    "lightgbm_probability":       float(lgbm_probs[idx]),
                    "suitability_pct":            round(float(suitability["score"]) * 100, 1),
                    "agronomic_suitability":      float(suitability["score"]),
                    "ml_suitability_pct":         round(float(stacked_probs[idx]) * 100, 1),
                    # Calendar
                    "sowing_month":   month_name(current_month),
                    "harvest_month":  month_name(harvest_month_number),
                    "duration_months": int(crop_rules["duration_months"]),
                    # Yield — expected + confidence range
                    "expected_yield_t_ha":   float(y_exp),
                    "yield_range_low_t_ha":  float(y_low),
                    "yield_range_high_t_ha": float(y_high),
                    "yield_range_label":     f"{y_low}–{y_high} t/ha (expected {y_exp})",
                    "base_yield_t_ha":       float(base_yield_t_ha),
                    "ml_yield_signal_t_ha":  float(ml_yield),
                    # Economics (clearly labeled)
                    "adjusted_price_rs_per_kg": float(adjusted_price_rs_per_kg),
                    "market_source":            market_source,
                    "price_year":               "2024",
                    "total_cost_rs_per_ha":     float(effective_cost_rs_per_ha),
                    "revenue_rs_per_ha":        float(revenue_rs_per_ha),
                    "profit_rs_per_ha":         float(profit_rs_per_ha),
                    "profit_type":              "Net profit (revenue − effective cost)",
                    "initial_spend_rs_per_ha":  float(cost_model["total_cost_rs_per_ha"]),
                    # Risk / sustainability
                    "risk":               float(risk),
                    "sustainability_score": sustainability,
                    # Penalty breakdown
                    "yield_penalty":       float(net_penalty),
                    "irrigation_penalty":  float(irr_pen),
                    "irrigation_bonus":    float(irr_bon),
                    "rotation_bonus":      float(rot_bon),
                    "rotation_note":       rot_note_text,
                    "climate_penalty":     float(climate_penalty),
                    "mono_crop_penalty":   float(mono_pen),
                    "new_farmer_penalty":  float(new_farmer_penalty),
                    "sowing_penalty":      float(sowing_penalty),
                    # Explainability
                    "market_type":     price_reason,
                    "advisory_notes":  advisory,
                    "rejection_checks_passed": climate_notes or ["suitability filter and seasonal screening passed"],
                    # Data sources
                    "data_sources": {
                        "weather":      "Open-Meteo Forecast + Archive API",
                        "price":        f"{'CommodityOnline Live' if 'commodityonline' in market_source else 'Agmarknet' if 'agmarknet' in market_source else 'Static CSV'} ({market_source})",
                        "yield":        "ICRISAT + Kaggle crop dataset",
                        "profit_basis": "Net profit: revenue minus variable+fixed costs",
                    },
                }
            )


        # Use annual_candidates for the main ranked list
        candidates = annual_candidates

        # ── Graceful fallback: ALWAYS show results, never block with error ──────
        # If no annual crop passes the sowing-window filter this month,
        # do a second relaxed pass ignoring the sowing window constraint.
        # Each crop gets an "off_season" advisory tag so the user knows.
        if not candidates:
            off_season_advisory = (
                f"No annual crop has {month_name(current_month)} in its primary sowing window. "
                f"Showing the best-fit crops for your soil/climate — "
                f"sowing now is suboptimal but viable with proper management."
            )
            for idx in ranked_indices:
                crop_name = str(labels[idx])
                crop_key = crop_name.strip().lower()
                crop_rules = CROP_DATABASE.get(crop_key)
                profile = crop_profiles.get(crop_name, {})
                if not crop_rules or not profile:
                    continue
                is_perennial = str(crop_rules.get("crop_type", "annual")) == "perennial"
                if is_perennial:
                    continue

                suitability = self._suitability_assessment(prepared, profile, crop_rules)
                if suitability["reject"]:
                    continue

                reg_frame = self._build_regression_frame(prepared, crop_name)
                X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="X does not have valid feature names")
                    ml_yield = float(self.bundle.yield_model.predict(X_reg)[0])

                base_yield_t_ha = self._baseline_yield(crop_name, ml_yield, profile, crop_rules)
                climate_penalty, should_reject, climate_notes = self._climate_adjustments(prepared, profile, crop_rules)
                if should_reject:
                    continue

                # Extra 20% penalty for off-season sowing
                off_season_penalty = 0.20
                irr_pen = irrigation_penalty(crop_rules, irrigation_source)
                irr_bon = irrigation_bonus(crop_rules, irrigation_source)
                rot_bon = rotation_bonus(previous_crop, crop_name)
                mono_pen = mono_crop_penalty(crop_name, previous_crop)
                rot_note_text = rotation_note(previous_crop, crop_name)

                net_penalty = max(0.0, min(
                    float(crop_rules["new_farmer_penalty"]) + climate_penalty + off_season_penalty
                    + mono_pen + irr_pen - irr_bon - rot_bon, 0.60
                ))
                expected_yield_t_ha = max(base_yield_t_ha * (1.0 - net_penalty), base_yield_t_ha * 0.30)
                y_low, y_exp, y_high = yield_range(expected_yield_t_ha, base_yield_t_ha, net_penalty)

                # Use nearest sowing month for harvest calculation
                sowing_months = list(crop_rules["sowing_months"])
                nearest_sow_month = min(sowing_months, key=lambda m: min(abs(m - current_month), 12 - abs(m - current_month)))
                adjusted_price_rs_per_kg, price_reason, market_source = self._price_adjustment(
                    crop_name, nearest_sow_month, crop_rules, prepared
                )
                harvest_month_number = harvest_month(nearest_sow_month, int(crop_rules["duration_months"]))
                cost_model = self._cost_model(crop_rules, area_ha)
                base_total_cost = cost_model["total_cost_rs_per_ha"]
                yield_ratio = expected_yield_t_ha / max(base_yield_t_ha, 0.01)
                effective_cost_rs_per_ha = base_total_cost * (0.45 + 0.55 * yield_ratio)
                revenue_rs_per_ha = expected_yield_t_ha * adjusted_price_rs_per_kg * 1000.0
                profit_rs_per_ha  = revenue_rs_per_ha - effective_cost_rs_per_ha

                risk = _clip(
                    (float(crop_rules["base_risk"]) * 0.30)
                    + (water_need_factor(str(crop_rules["water_need"])) * 0.25)
                    + (float(crop_rules["price_volatility"]) * 0.20)
                    + (float(crop_rules["new_farmer_penalty"]) * 0.15)
                    + (climate_penalty * 0.10),
                    0.12, 0.88,
                )
                sustainability = _clip(
                    float(crop_rules["base_sustainability"])
                    - (0.12 if crop_rules["water_need"] == "high" else 0.04 if crop_rules["water_need"] == "medium" else 0.0)
                    - (float(crop_rules["chemical_dependency"]) * 0.18)
                    - (float(crop_rules["soil_sensitivity"]) * 0.10),
                    0.18, 0.92,
                )
                candidates.append({
                    "crop": crop_name,
                    "crop_type": "annual",
                    "off_season": True,
                    "off_season_advisory": off_season_advisory,
                    "sowing_timing": "Off-Season",
                    "classification_probability": float(stacked_probs[idx]),
                    "lightgbm_probability":       float(lgbm_probs[idx]),
                    "suitability_pct":            round(float(suitability["score"]) * 100, 1),
                    "agronomic_suitability":      float(suitability["score"]),
                    "ml_suitability_pct":         round(float(stacked_probs[idx]) * 100, 1),
                    "sowing_month":   month_name(nearest_sow_month),
                    "harvest_month":  month_name(harvest_month_number),
                    "duration_months": int(crop_rules["duration_months"]),
                    "expected_yield_t_ha":   float(y_exp),
                    "yield_range_low_t_ha":  float(y_low),
                    "yield_range_high_t_ha": float(y_high),
                    "yield_range_label":     f"{y_low}–{y_high} t/ha (expected {y_exp})",
                    "base_yield_t_ha":       float(base_yield_t_ha),
                    "ml_yield_signal_t_ha":  float(ml_yield),
                    "adjusted_price_rs_per_kg": float(adjusted_price_rs_per_kg),
                    "market_source":            market_source,
                    "price_year":               "2024",
                    "total_cost_rs_per_ha":     float(effective_cost_rs_per_ha),
                    "revenue_rs_per_ha":        float(revenue_rs_per_ha),
                    "profit_rs_per_ha":         float(profit_rs_per_ha),
                    "profit_type":              "Net profit (off-season sowing — 20% yield penalty applied)",
                    "initial_spend_rs_per_ha":  float(cost_model["total_cost_rs_per_ha"]),
                    "risk":               float(risk),
                    "sustainability_score": sustainability,
                    "yield_penalty":       float(net_penalty),
                    "irrigation_penalty":  float(irr_pen),
                    "irrigation_bonus":    float(irr_bon),
                    "rotation_bonus":      float(rot_bon),
                    "rotation_note":       rot_note_text,
                    "climate_penalty":     float(climate_penalty),
                    "mono_crop_penalty":   float(mono_pen),
                    "new_farmer_penalty":  float(crop_rules["new_farmer_penalty"]),
                    "sowing_penalty":      float(off_season_penalty),
                    "market_type":     price_reason,
                    "advisory_notes":  [
                        off_season_advisory,
                        f"Nearest optimal sowing month: {month_name(nearest_sow_month)}.",
                        f"Price adjusted for {price_reason}.",
                        "20% yield penalty applied for off-season sowing.",
                    ],
                    "rejection_checks_passed": climate_notes or ["climate and soil suitability passed"],
                    "data_sources": {
                        "weather":      "Open-Meteo Forecast + Archive API",
                        "price":        f"{'CommodityOnline Live' if 'commodityonline' in market_source else 'Agmarknet' if 'agmarknet' in market_source else 'Static CSV'} ({market_source})",
                        "yield":        "ICRISAT + Kaggle crop dataset",
                        "profit_basis": "Net profit: revenue minus variable+fixed costs",
                    },
                })

            if not candidates:
                # Absolute last resort — return perennials promoted to main list
                perennial_summary = ", ".join(p["crop"] for p in perennial_candidates[:3]) if perennial_candidates else "Banana, Coconut, Papaya"
                # Create synthetic entries for top perennials so UI always has something
                for p in perennial_candidates[:3]:
                    p["off_season"] = False
                    p["sowing_timing"] = "Perennial"
                    p["off_season_advisory"] = "No annual crops suitable — showing perennial options."
                    p["crop_type"] = "perennial_promoted"
                    p["classification_probability"] = 0.5
                    p["lightgbm_probability"] = 0.5
                    p["suitability_pct"] = round(float(p.get("agronomic_suitability_pct", 50)), 1)
                    p["ml_suitability_pct"] = 50.0
                    p["sowing_month"] = month_name(current_month)
                    p["harvest_month"] = "Multi-year"
                    p["adjusted_price_rs_per_kg"] = float(p.get("stable_price_rs_per_kg", 20.0))
                    p["market_source"] = "static_csv"
                    p["price_year"] = "2024"
                    p["total_cost_rs_per_ha"] = float(p.get("total_cost_rs_per_ha", 0.0))
                    p["revenue_rs_per_ha"] = float(p.get("revenue_rs_per_ha", 0.0))
                    p["profit_type"] = "Annual avg after amortization"
                    p["initial_spend_rs_per_ha"] = float(p.get("total_cost_rs_per_ha", 0.0))
                    p["yield_penalty"] = 0.0
                    p["irrigation_penalty"] = 0.0
                    p["irrigation_bonus"] = 0.0
                    p["rotation_bonus"] = 0.0
                    p["rotation_note"] = None
                    p["climate_penalty"] = 0.0
                    p["mono_crop_penalty"] = 0.0
                    p["new_farmer_penalty"] = 0.35
                    p["sowing_penalty"] = 0.0
                    p["market_type"] = "stable market"
                    p["advisory_notes"] = ["Perennial crop — multi-year investment. No annual crop is suitable right now."]
                    p["rejection_checks_passed"] = ["perennial — no seasonal restriction"]
                    p["yield_range_low_t_ha"] = float(p.get("expected_yield_t_ha", 0.5)) * 0.8
                    p["yield_range_high_t_ha"] = float(p.get("expected_yield_t_ha", 0.5)) * 1.2
                    p["yield_range_label"] = f"{p['yield_range_low_t_ha']:.2f}–{p['yield_range_high_t_ha']:.2f} t/ha"
                    p["base_yield_t_ha"] = float(p.get("expected_yield_t_ha", 0.5))
                    p["ml_yield_signal_t_ha"] = float(p.get("expected_yield_t_ha", 0.5))
                    p["data_sources"] = {"price": "static_csv", "yield": "ICRISAT", "weather": "Open-Meteo", "profit_basis": "Net profit"}
                    candidates.append(p)



        # Move loss-making crops to rejected rather than showing negative profit in results
        profitable = [c for c in candidates if c["profit_rs_per_ha"] > 0]
        unprofitable = [c for c in candidates if c["profit_rs_per_ha"] <= 0]
        for u in unprofitable:
            rejected.append({
                "crop": u["crop"],
                "reason": f"not profitable under current conditions (profit ₹{u['profit_rs_per_ha']:,.0f}/ha after penalties)",
            })
        if profitable:
            candidates = profitable

        profit_scores        = _candidate_scores([item["profit_rs_per_ha"] for item in candidates])
        low_cost_scores      = _candidate_scores([-item["initial_spend_rs_per_ha"] for item in candidates])
        sustainability_scores = _candidate_scores([item.get("sustainability_score", 0.5) for item in candidates])
        safety_scores        = _candidate_scores([1.0 - item.get("risk", 0.5) for item in candidates])

        for item, p_sc, lc_sc, sus_sc, saf_sc in zip(
            candidates, profit_scores, low_cost_scores, sustainability_scores, safety_scores
        ):
            item["profit_score"]   = float(p_sc)
            item["low_cost_score"] = float(lc_sc)

            # Use RAW (absolute) ML probability — not normalized — so a crop with
            # ml_prob=0.001 actually scores near-zero, not artificially elevated.
            raw_ml = float(item.get("classification_probability", 0.0))
            raw_ag = float(item.get("agronomic_suitability", 0.5))

            # Historical prominence boost: reward crops with high average yield
            # (iconic/significant crops like Rice, Cotton, Maize, Sugarcane) over
            # minor pulses when ML probability is otherwise similar.
            crop_key = item["crop"].strip().lower()
            cr = CROP_DATABASE.get(crop_key, {})
            yield_avg = float(cr.get("yield_avg_t_ha", 1.0))
            # Normalize prominence: max yield in our DB is 70t/ha (sugarcane), min 0.7
            prominence = float(np.clip((yield_avg - 0.7) / (70.0 - 0.7), 0.0, 1.0))

            # ── Rebalanced final_score ─────────────────────────────────────────
            # Raw ML prob  (0.20): actual model confidence, not relative
            # Agronomic fit(0.15): soil+climate suitability, raw
            # Profit       (0.25): relative profit among candidates
            # Prominence   (0.10): historical crop significance
            # Affordability(0.15): relative low-cost among candidates
            # Safety       (0.15): relative safety among candidates
            item["final_score"] = float(
                (raw_ml   * 0.20)
                + (raw_ag * 0.15)
                + (p_sc   * 0.25)
                + (prominence * 0.10)
                + (lc_sc  * 0.15)
                + (saf_sc * 0.15)
            )
            item["high_investment_profit_score"] = float(
                (p_sc    * 0.50)
                + (raw_ml * 0.20)
                + ((1.0 - lc_sc) * 0.15)
                + (saf_sc * 0.10)
                + (sus_sc * 0.05)
            )


        standard_candidates = sorted(candidates, key=lambda item: item["final_score"], reverse=True)
        preferred_slots = max(min(top_k, 5) - 1, 1)
        spend_values = [item["initial_spend_rs_per_ha"] for item in candidates]
        premium_threshold = float(np.quantile(spend_values, 0.65)) if len(spend_values) > 1 else spend_values[0]

        low_medium_pool = [item for item in standard_candidates if item["initial_spend_rs_per_ha"] <= premium_threshold]
        high_spend_pool = [item for item in candidates if item["initial_spend_rs_per_ha"] > premium_threshold]

        top_candidates = low_medium_pool[:preferred_slots]
        selected_crops = {item["crop"] for item in top_candidates}

        if len(top_candidates) < preferred_slots:
            low_medium_fallback = [item for item in standard_candidates if item["crop"] not in selected_crops]
            while len(top_candidates) < preferred_slots and low_medium_fallback:
                candidate = low_medium_fallback.pop(0)
                if candidate["crop"] not in selected_crops:
                    top_candidates.append(candidate)
                    selected_crops.add(candidate["crop"])

        premium_pool = [item for item in high_spend_pool if item["crop"] not in selected_crops]
        if premium_pool:
            premium_candidate = max(premium_pool, key=lambda item: item["high_investment_profit_score"])
            top_candidates.append(premium_candidate)
            selected_crops.add(premium_candidate["crop"])

        remaining_pool = [item for item in standard_candidates if item["crop"] not in selected_crops]
        while len(top_candidates) < top_k and remaining_pool:
            top_candidates.append(remaining_pool.pop(0))

        ideal_candidates: list[dict[str, Any]] = []
        for idx in ranked_indices:
            crop_name = str(labels[idx])
            crop_key = crop_name.strip().lower()
            crop_rules = CROP_DATABASE.get(crop_key)
            profile = crop_profiles.get(crop_name, {})
            if not crop_rules or not profile:
                continue
            suitability = self._suitability_assessment(prepared, profile, crop_rules)
            if suitability["reject"]:
                continue

            reg_frame = self._build_regression_frame(prepared, crop_name)
            X_reg = _to_float32_array(self.bundle.regression_preprocessor.transform(reg_frame))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="X does not have valid feature names")
                ml_yield = float(self.bundle.yield_model.predict(X_reg)[0])

            base_yield_t_ha = self._baseline_yield(crop_name, ml_yield, profile, crop_rules)
            climate_penalty, should_reject, climate_notes = self._climate_adjustments(prepared, profile, crop_rules)
            if should_reject:
                continue

            conservative_new_farmer_penalty = min(float(crop_rules["new_farmer_penalty"]) + 0.05, 0.42)
            long_term_yield_penalty = min(conservative_new_farmer_penalty + climate_penalty, 0.55)
            expected_yield_t_ha = max(base_yield_t_ha * (1.0 - long_term_yield_penalty), 0.15)

            stable_price_rs_per_kg = self._stable_price_rs_per_kg(crop_name, prepared, crop_rules)
            cost_model = self._cost_model(crop_rules, area_ha)
            revenue_rs_per_ha = expected_yield_t_ha * stable_price_rs_per_kg * 1000.0
            profit_rs_per_ha = revenue_rs_per_ha - cost_model["total_cost_rs_per_ha"]

            risk = _clip(
                (float(crop_rules["base_risk"]) * 0.36)
                + (water_need_factor(str(crop_rules["water_need"])) * 0.22)
                + (float(crop_rules["price_volatility"]) * 0.10)
                + (float(crop_rules["new_farmer_penalty"]) * 0.16)
                + (climate_penalty * 0.16),
                0.10,
                0.88,
            )
            sustainability = _clip(
                float(crop_rules["base_sustainability"])
                - (0.12 if crop_rules["water_need"] == "high" else 0.04 if crop_rules["water_need"] == "medium" else 0.0)
                - (float(crop_rules["chemical_dependency"]) * 0.18)
                - (float(crop_rules["soil_sensitivity"]) * 0.10),
                0.18,
                0.92,
            )

            land_suitability = float(suitability["score"])
            ideal_candidates.append(
                {
                    "crop": crop_name,
                    "expected_yield_t_ha": float(expected_yield_t_ha),
                    "stable_price_rs_per_kg": float(stable_price_rs_per_kg),
                    "total_cost_rs_per_ha": float(cost_model["total_cost_rs_per_ha"]),
                    "revenue_rs_per_ha": float(revenue_rs_per_ha),
                    "profit_rs_per_ha": float(profit_rs_per_ha),
                    "risk": float(risk),
                    "sustainability_score": float(sustainability),
                    "land_suitability": float(land_suitability),
                    "climate_penalty": float(climate_penalty),
                    "notes": climate_notes or ["strong general soil and climate fit"],
                    "duration_months": int(crop_rules["duration_months"]),
                }
            )

        if ideal_candidates:
            ideal_profit_scores = _candidate_scores([item["profit_rs_per_ha"] for item in ideal_candidates])
            ideal_low_cost_scores = _candidate_scores([-item["total_cost_rs_per_ha"] for item in ideal_candidates])
            for item, profit_score, low_cost_score in zip(ideal_candidates, ideal_profit_scores, ideal_low_cost_scores):
                item["profit_score"] = float(profit_score)
                item["low_cost_score"] = float(low_cost_score)
                item["ideal_ground_score"] = float(
                    (item["land_suitability"] * 0.35)
                    + ((1.0 - item["risk"]) * 0.25)
                    + (low_cost_score * 0.20)
                    + (profit_score * 0.20)
                )

        ideal_candidates.sort(key=lambda item: item["ideal_ground_score"], reverse=True)
        ideal_ground = ideal_candidates[0] if ideal_candidates else None

        best_crop_name = top_candidates[0]["crop"]
        best_profile = crop_profiles[best_crop_name]
        best_reg_frame = self._build_regression_frame(prepared, best_crop_name)
        best_reg_matrix = _to_float32_array(self.bundle.regression_preprocessor.transform(best_reg_frame))
        local_explanations = self._local_shap_explanations(X_class, best_reg_matrix, best_crop_name)
        rule_explanations = self._rule_based_explanation(prepared, best_profile, best_crop_name)
        best_rules = CROP_DATABASE.get(best_crop_name.strip().lower(), {})
        agronomic_explanations = self._agronomic_explanation(
            prepared, best_profile, best_rules, best_crop_name
        )

        # Model overview (for academic section only)
        classification_metrics = self.bundle.metadata["training_report"].get("classification_metrics", {})
        model_overview = sorted([
            {
                "model":        name.replace("_", " ").title(),
                "accuracy":     float(m.get("accuracy", 0.0)),
                "f1_weighted":  float(m.get("f1_weighted", 0.0)),
                "top3_accuracy":float(m.get("top3_accuracy", 0.0)),
            }
            for name, m in classification_metrics.items()
            if isinstance(m, dict) and "accuracy" in m
        ], key=lambda x: x["accuracy"], reverse=True)

        # Sort perennial options
        perennial_candidates.sort(key=lambda x: (x["sustainability_score"], x["profit_rs_per_ha"]), reverse=True)

        # Build reconciliation note for best crop
        best_concerns = rule_explanations.get("concerns", [])
        best_prob = top_candidates[0].get("classification_probability", 0.0)
        top_shap_feature = ""
        cls_shap = local_explanations.get("classification", {})
        if cls_shap.get("positive"):
            top_shap_feature = cls_shap["positive"][0].get("feature", "")

        recon_note = reconciliation_note(best_crop_name, best_concerns, best_prob, top_shap_feature)

        # Enrich rejected crops with current-vs-required detail
        enriched_rejected = []
        for rej in rejected:
            rej_key = rej["crop"].strip().lower()
            enriched_rejected.append(
                format_rejected_detail(
                    rej["crop"], rej["reason"], prepared, CROP_DATABASE.get(rej_key)
                )
            )

        from backend.ml.crop_rules_realistic_v2 import DATA_SOURCES

        return {
            "best_crop": best_crop_name,
            "top_crops": top_candidates,
            "perennial_options": perennial_candidates[:5],
            "training_summary": self.bundle.metadata["training_report"],

            "analysis_context": {
                "analysis_date":    analysis_date,
                "current_month":    month_name(current_month),
                "season":           current_season,
                "area_hectares":    area_ha,
                "state_name":       str(prepared.get("state_name", "Unknown")),
                "state_source":     str(prepared.get("_state_source", "dataset default")),
                "imd_rainfall_normal_mm": prepared.get("_imd_rainfall_normal_mm"),
                "rainfall_source":  prepared.get("_rainfall_source", "default"),
                "npk_adjusted":     prepared.get("_npk_adjusted", False),
                "npk_adjustment_note": prepared.get("_npk_adjustment_note"),
                "decision_rule": (
                    "Crops must pass agronomic suitability and hard rejection checks before economics. "
                    "Suitable crops are ranked by combined profit, cost, risk, and sustainability score."
                ),
            },
            "soil_health_report": self._build_soil_health_report(prepared),
            "rejected_crops": enriched_rejected,
            "reconciliation_note": recon_note,
            "used_defaults": {
                "season":       current_season,
                "state_name":   prepared["state_name"],
                "district_name": prepared.get("district_name"),
                "crop_year":    date.today().year,
            },
            "data_sources": DATA_SOURCES,
            "ideal_ground_recommendation": (
                {
                    "crop":               ideal_ground["crop"],
                    "land_suitability":   ideal_ground["land_suitability"],
                    "expected_yield_t_ha": ideal_ground["expected_yield_t_ha"],
                    "stable_price_rs_per_kg": ideal_ground["stable_price_rs_per_kg"],
                    "total_cost_rs_per_ha":   ideal_ground["total_cost_rs_per_ha"],
                    "revenue_rs_per_ha":  ideal_ground["revenue_rs_per_ha"],
                    "profit_rs_per_ha":   ideal_ground["profit_rs_per_ha"],
                    "risk":               ideal_ground["risk"],
                    "sustainability_score": ideal_ground["sustainability_score"],
                    "duration_months":    ideal_ground["duration_months"],
                    "ideal_ground_score": ideal_ground["ideal_ground_score"],
                    "why": [
                        "Long-term land suitability recommendation — ignores current sowing window.",
                        "Prioritises strong soil/climate fit, lower risk, lower cost, and stable profit.",
                    ] + ideal_ground["notes"][:3],
                }
                if ideal_ground else None
            ),
            "explainability": {
                # LOCAL: explains THIS specific prediction
                "local_shap_label": "Why this prediction — SHAP values for this specific field input",
                "best_crop_local_explanation": {
                    "crop":                 best_crop_name,
                    "model_explanation": {
                        "classification_shap": local_explanations["classification"],
                        "yield_shap": local_explanations["yield"],
                        "note": (
                            "SHAP explains which input features influenced the ML classification "
                            "and yield models for this specific prediction."
                        ),
                    },
                    "agronomic_explanation": {
                        "positives": agronomic_explanations["positives"],
                        "advice": agronomic_explanations["advice"],
                        "note": (
                            "Agronomic advice compares current field values against crop-specific "
                            "soil and climate ranges."
                        ),
                    },
                    "classification_shap":  local_explanations["classification"],
                    "yield_shap":           local_explanations["yield"],
                    "rule_based_positives": rule_explanations["positives"],
                    "rule_based_concerns":  rule_explanations["concerns"],
                    "agronomic_positives":  agronomic_explanations["positives"],
                    "agronomic_advice":     agronomic_explanations["advice"],
                    "reconciliation_note":  recon_note,
                    "shap_note": (
                        "Classification SHAP: features driving the crop classification score. "
                        "Yield SHAP: features driving the yield regression estimate. "
                        "Both are LOCAL to this prediction."
                    ),
                },
                # GLOBAL: explains overall model behaviour (not this prediction)
                "global_label": "Overall model behaviour — feature importance across all training data",
                "global_top_features": self.bundle.metadata.get("xai_assets", {}).get("top_features", [])[:8],
                "summary_plot_urls": {
                    "lightgbm": self.bundle.metadata.get("xai_assets", {}).get("lightgbm_summary_plot"),
                    "catboost": self.bundle.metadata.get("xai_assets", {}).get("catboost_summary_plot"),
                },
                # Model accuracy — for advanced/academic use only
                "model_overview_note": "Model accuracy metrics — for academic review only, not for farming decisions.",
                "model_overview": model_overview,
            },
        }
