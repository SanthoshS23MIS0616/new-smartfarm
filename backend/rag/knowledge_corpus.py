"""
knowledge_corpus.py
===================
Prepares and indexes ICAR / TNAU Package of Practices (POP) documents for all 22 crops.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_POP_PATH = Path(__file__).resolve().parents[2] / "data" / "realistic_v2" / "crop_package_of_practices.json"

_corpus_passages: list[dict[str, Any]] = []


def build_knowledge_corpus() -> list[dict[str, Any]]:
    global _corpus_passages
    if _corpus_passages:
        return _corpus_passages

    if not _POP_PATH.exists():
        return []

    with open(_POP_PATH, "r", encoding="utf-8") as f:
        crops = json.load(f)

    passages = []
    for crop in crops:
        cname = crop["crop_name"]
        local_name = crop.get("local_name", "")
        f_plan = crop.get("fertilizer_plan", {})
        costs = crop.get("costs_per_acre", {})

        # Passage 1: Nutrient & Fertilizer Management
        passages.append({
            "passage_id": f"{cname}-FERT",
            "crop": cname,
            "topic": "fertilizer",
            "keywords": ["fertilizer", "urea", "dap", "mop", "nitrogen", "npk", "phosphorus", "potassium", "nutrient", "உரம்", "யூரியா"],
            "title": f"{cname} ({local_name}) Fertilizer & Nutrient Management",
            "text": (
                f"For {cname} ({local_name}): Basal dose: {f_plan.get('basal')}. "
                f"First Topdress: {f_plan.get('topdress_1')}. "
                f"Second Topdress: {f_plan.get('topdress_2')}. "
                f"Organic alternatives: {f_plan.get('organic_alternative')}. "
                f"Micronutrient needs: {f_plan.get('micronutrients')}."
            ),
            "source": "ICAR & TNAU Package of Practices"
        })

        # Passage 2: Irrigation & Water Management
        passages.append({
            "passage_id": f"{cname}-IRR",
            "crop": cname,
            "topic": "irrigation",
            "keywords": ["water", "irrigation", "rain", "drought", "moisture", "waterlogging", "பாசனம்", "தண்ணீர்"],
            "title": f"{cname} ({local_name}) Irrigation & Water Requirements",
            "text": (
                f"Water requirement level for {cname} is {crop.get('water_requirement_level')} (approx {crop.get('total_water_requirement_mm')} mm). "
                f"Critical irrigation stages: {', '.join(crop.get('critical_irrigation_stages', []))}. "
                f"Irrigation should be provided at these sensitive stages to avoid severe yield loss. "
                f"If rainfall occurs, suppress scheduled irrigation."
            ),
            "source": "ICAR & TNAU Water Management Guidelines"
        })

        # Passage 3: Pest & Disease IPM Management
        passages.append({
            "passage_id": f"{cname}-PEST",
            "crop": cname,
            "topic": "pest_disease",
            "keywords": ["pest", "disease", "insect", "spray", "fungus", "borer", "blight", "பூச்சி", "நோய்", "மருந்து"],
            "title": f"{cname} ({local_name}) Pest & Disease Protection (IPM)",
            "text": (
                f"Common pests for {cname}: {', '.join(crop.get('common_pests', []))}. "
                f"Common diseases: {', '.join(crop.get('common_diseases', []))}. "
                f"Integrated Pest Management (IPM) practices: {crop.get('ipm_practices')}. "
                f"Always confirm any chemical dosage with the local agriculture extension officer before application."
            ),
            "source": "TNAU Plant Protection Guidelines & CIB&RC Label Standards"
        })

        # Passage 4: Sowing, Stages & Economics
        passages.append({
            "passage_id": f"{cname}-SOW",
            "crop": cname,
            "topic": "sowing_economics",
            "keywords": ["sowing", "seed", "stage", "cost", "yield", "harvest", "profit", "விதை", "மகசூல்", "செலவு"],
            "title": f"{cname} ({local_name}) Sowing, Stages & Economics",
            "text": (
                f"{cname} total duration is {crop.get('total_duration_days')} days ({crop.get('duration_range_days')} days). "
                f"Ideal sowing months: {', '.join([str(m) for m in crop.get('ideal_sowing_months', [])])}. "
                f"Estimated cultivation cost per acre: INR {costs.get('total_inr', 30000):,.0f} "
                f"(Seed: INR {costs.get('seed_inr')}, Fertilizer: INR {costs.get('fertilizer_inr')}, Labor: INR {costs.get('labor_inr')}). "
                f"Expected yield per acre: {crop.get('yield_kg_per_acre')} kg."
            ),
            "source": "ICAR & State Agriculture Cost Assessment"
        })

    _corpus_passages = passages
    return _corpus_passages


def retrieve_relevant_passages(query: str, crop_name: str | None = None, top_k: int = 3) -> list[dict[str, Any]]:
    corpus = build_knowledge_corpus()
    query_lower = query.lower()

    scored = []
    for passage in corpus:
        score = 0
        p_crop = passage["crop"].lower()

        # Crop name boost
        if crop_name and crop_name.lower() in p_crop:
            score += 15
        elif p_crop in query_lower:
            score += 12

        # Keyword match
        for kw in passage["keywords"]:
            if kw in query_lower:
                score += 5

        # Text snippet word matches
        words = [w for w in query_lower.split() if len(w) > 3]
        for w in words:
            if w in passage["text"].lower() or w in passage["title"].lower():
                score += 2

        if score > 0:
            scored.append((score, passage))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]