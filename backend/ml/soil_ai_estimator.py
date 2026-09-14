"""
soil_ai_estimator.py
====================
AI-driven agricultural topsoil parameter estimator for SmartFarm:
- Uses OpenAI API (model: gpt-5.6-luna / gpt-4o-mini) to predict typical N, P, K, and pH
  based on geographic coordinates, district, state, and ICAR Soil Health Card benchmarks.
- Robust deterministic fallback using ICAR agro-climatic zone baselines if API is offline.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from backend.ml.geocoder import reverse_geocode_state

logger = logging.getLogger(__name__)

# ICAR Agro-Climatic State Soil Baselines (used as deterministic calibration fallback)
_ICAR_STATE_SOIL_BASELINES: dict[str, dict[str, Any]] = {
    "Tamil Nadu": {
        "nitrogen": 48.0,
        "phosphorus": 52.0,
        "potassium": 28.0,
        "ph": 6.8,
        "soil_type": "red_laterite",
        "description": "Red laterite / coastal alluvial soil. Moderate N, balanced P, moderate K.",
    },
    "Karnataka": {
        "nitrogen": 52.0,
        "phosphorus": 46.0,
        "potassium": 34.0,
        "ph": 6.9,
        "soil_type": "red_laterite",
        "description": "Red sandy loam with medium phosphorus and moderate potash.",
    },
    "Andhra Pradesh": {
        "nitrogen": 55.0,
        "phosphorus": 58.0,
        "potassium": 32.0,
        "ph": 7.2,
        "soil_type": "black_cotton",
        "description": "Black cotton / coastal delta soil with good phosphorus availability.",
    },
    "Telangana": {
        "nitrogen": 45.0,
        "phosphorus": 48.0,
        "potassium": 30.0,
        "ph": 7.1,
        "soil_type": "black_cotton",
        "description": "Deccan plateau black clay loam, moderate nitrogen and potash.",
    },
    "Kerala": {
        "nitrogen": 42.0,
        "phosphorus": 38.0,
        "potassium": 30.0,
        "ph": 5.8,
        "soil_type": "red_laterite",
        "description": "Acidic humid coastal laterite soil, high organic matter, moderate P.",
    },
    "Maharashtra": {
        "nitrogen": 50.0,
        "phosphorus": 42.0,
        "potassium": 55.0,
        "ph": 7.8,
        "soil_type": "black_cotton",
        "description": "Deep black vertisol with high potassium and slightly alkaline pH.",
    },
    "Gujarat": {
        "nitrogen": 48.0,
        "phosphorus": 50.0,
        "potassium": 45.0,
        "ph": 7.6,
        "soil_type": "sandy_loam",
        "description": "Alluvial / medium black sandy loam, rich in potash and phosphorus.",
    },
    "Punjab": {
        "nitrogen": 85.0,
        "phosphorus": 62.0,
        "potassium": 42.0,
        "ph": 7.4,
        "soil_type": "alluvial",
        "description": "Indo-Gangetic fertile alluvial plains with high N and P from intensive cropping.",
    },
    "Haryana": {
        "nitrogen": 78.0,
        "phosphorus": 58.0,
        "potassium": 40.0,
        "ph": 7.5,
        "soil_type": "alluvial",
        "description": "Alluvial loam with high nutrient availability and neutral-alkaline pH.",
    },
    "Uttar Pradesh": {
        "nitrogen": 70.0,
        "phosphorus": 52.0,
        "potassium": 36.0,
        "ph": 7.2,
        "soil_type": "alluvial",
        "description": "Gangetic alluvial belt with balanced nitrogen and phosphorus.",
    },
    "Madhya Pradesh": {
        "nitrogen": 46.0,
        "phosphorus": 44.0,
        "potassium": 48.0,
        "ph": 7.3,
        "soil_type": "black_cotton",
        "description": "Central Indian black soil with good potassium reserves.",
    },
    "West Bengal": {
        "nitrogen": 65.0,
        "phosphorus": 48.0,
        "potassium": 32.0,
        "ph": 6.4,
        "soil_type": "alluvial",
        "description": "Deltaic alluvial soil with high moisture retention and slightly acidic pH.",
    },
    "Rajasthan": {
        "nitrogen": 35.0,
        "phosphorus": 38.0,
        "potassium": 36.0,
        "ph": 7.9,
        "soil_type": "sandy_loam",
        "description": "Arid sandy loam, alkaline pH, lower organic nitrogen, moderate potash.",
    },
}

_DEFAULT_INDIA_BASELINE: dict[str, Any] = {
    "nitrogen": 50.0,
    "phosphorus": 48.0,
    "potassium": 35.0,
    "ph": 6.8,
    "soil_type": "alluvial",
    "description": "National agricultural Soil Health Card median baseline.",
}


def estimate_soil_parameters(
    latitude: float,
    longitude: float,
    district: str | None = None,
    state: str | None = None,
) -> dict[str, Any]:
    """
    Estimate topsoil N, P, K, pH and Soil Type using OpenAI API if available,
    falling back to ICAR state agro-climatic database.
    """
    detected_state = state or reverse_geocode_state(latitude, longitude) or "Tamil Nadu"
    baseline = _ICAR_STATE_SOIL_BASELINES.get(detected_state, _DEFAULT_INDIA_BASELINE)

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            system_prompt = (
                "You are an expert soil scientist and ICAR agronomist in India. "
                "Predict realistic agricultural topsoil chemical parameters for coordinates in India. "
                "Output STRICT raw JSON only, with keys: "
                "'nitrogen' (float, kg/ha, typical 20-120), "
                "'phosphorus' (float, kg/ha, typical 15-90), "
                "'potassium' (float, kg/ha, typical 15-80), "
                "'ph' (float, 5.0-8.5), "
                "'soil_type' (one of 'alluvial', 'black_cotton', 'red_laterite', 'sandy_loam', 'clay'), "
                "'confidence' ('high' or 'medium'), "
                "'description' (short 1-sentence explanation)."
            )
            user_prompt = (
                f"Coordinates: Latitude {latitude:.4f}, Longitude {longitude:.4f}\n"
                f"Region: {district or 'Near'}, {detected_state}, India.\n"
                "Provide the typical soil N, P, K, pH and soil type based on Soil Health Card benchmarks."
            )

            for model_name in ["gpt-5.6-luna", "gpt-4o-mini", "gpt-4o"]:
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=0.2,
                        max_tokens=300,
                    )
                    content = response.choices[0].message.content or ""
                    match = re.search(r"\{.*\}", content, re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                        return {
                            "nitrogen": round(float(parsed.get("nitrogen", baseline["nitrogen"])), 1),
                            "phosphorus": round(float(parsed.get("phosphorus", baseline["phosphorus"])), 1),
                            "potassium": round(float(parsed.get("potassium", baseline["potassium"])), 1),
                            "ph": round(float(parsed.get("ph", baseline["ph"])), 1),
                            "soil_type": parsed.get("soil_type", baseline["soil_type"]),
                            "confidence": parsed.get("confidence", "high"),
                            "source": f"OpenAI ({model_name}) + ICAR Benchmarks",
                            "state": detected_state,
                            "description": parsed.get("description", baseline["description"]),
                        }
                except Exception as model_err:
                    logger.debug("OpenAI model %s failed for soil estimate: %s", model_name, model_err)
                    continue
        except Exception as exc:
            logger.warning("OpenAI soil estimation failed, falling back to ICAR baseline: %s", exc)

    # Deterministic fallback using ICAR state baseline
    return {
        "nitrogen": baseline["nitrogen"],
        "phosphorus": baseline["phosphorus"],
        "potassium": baseline["potassium"],
        "ph": baseline["ph"],
        "soil_type": baseline["soil_type"],
        "confidence": "high",
        "source": f"ICAR Agro-Climatic Zone Baselines ({detected_state})",
        "state": detected_state,
        "description": baseline["description"],
    }
