"""
plan_generator.py
=================
Generates dynamic, dated sowing-to-harvest crop management plans with:
- Task calendars with confirmation checkboxes
- Budget & safety margin calculations
- Stage-by-stage agronomic advisories from verified POP dataset
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_POP_PATH = Path(__file__).resolve().parents[2] / "data" / "realistic_v2" / "crop_package_of_practices.json"

_pop_cache: dict[str, dict[str, Any]] = {}


def load_pop_database() -> dict[str, dict[str, Any]]:
    global _pop_cache
    if _pop_cache:
        return _pop_cache

    if not _POP_PATH.exists():
        logger.warning("POP database file not found at %s", _POP_PATH)
        return {}

    with open(_POP_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        key = item["crop_name"].strip().lower()
        _pop_cache[key] = item
    return _pop_cache


def get_crop_pop(crop_name: str) -> dict[str, Any] | None:
    db = load_pop_database()
    return db.get(crop_name.strip().lower())


def check_budget_feasibility(crop_name: str, area_acres: float, farmer_budget_inr: float) -> dict[str, Any]:
    pop = get_crop_pop(crop_name)
    if not pop:
        return {"error": f"Crop {crop_name} not found in Package of Practices database"}

    costs_per_acre = pop["costs_per_acre"]
    base_cost = float(costs_per_acre["total_inr"]) * area_acres
    emergency_reserve = round(base_cost * 0.15, 2)
    min_recommended_budget = round(base_cost + emergency_reserve, 2)

    is_affordable = farmer_budget_inr >= base_cost
    buffer_margin = farmer_budget_inr - base_cost

    itemized_costs = {
        "seed_inr": round(float(costs_per_acre["seed_inr"]) * area_acres, 2),
        "fertilizer_inr": round(float(costs_per_acre["fertilizer_inr"]) * area_acres, 2),
        "pesticide_inr": round(float(costs_per_acre["pesticide_inr"]) * area_acres, 2),
        "labor_inr": round(float(costs_per_acre["labor_inr"]) * area_acres, 2),
        "irrigation_inr": round(float(costs_per_acre["irrigation_inr"]) * area_acres, 2),
        "machinery_inr": round(float(costs_per_acre["machinery_inr"]) * area_acres, 2),
        "operational_subtotal_inr": round(base_cost, 2),
        "emergency_reserve_15pct_inr": emergency_reserve,
        "total_recommended_budget_inr": min_recommended_budget,
    }

    status = "adequate" if buffer_margin >= emergency_reserve else ("tight" if is_affordable else "insufficient")
    guidance = (
        f"Budget is sufficient with a safe emergency buffer of INR {buffer_margin:,.0f}."
        if status == "adequate"
        else (
            f"Budget covers baseline cultivation (INR {base_cost:,.0f}) but leaves little reserve for unexpected pests/weather."
            if status == "tight"
            else f"Budget of INR {farmer_budget_inr:,.0f} is below estimated cultivation cost of INR {base_cost:,.0f}."
        )
    )

    return {
        "crop_name": pop["crop_name"],
        "area_acres": area_acres,
        "farmer_budget_inr": farmer_budget_inr,
        "is_affordable": is_affordable,
        "status": status,
        "guidance": guidance,
        "cost_breakdown": itemized_costs,
    }


def generate_sowing_plan(
    crop_name: str,
    sowing_date_str: str,
    area_acres: float = 1.0,
    farmer_budget_inr: float = 50000.0,
    irrigation_source: str = "Borewell",
) -> dict[str, Any]:
    pop = get_crop_pop(crop_name)
    if not pop:
        raise ValueError(f"Crop {crop_name} not recognized in agricultural database.")

    try:
        sow_dt = datetime.strptime(sowing_date_str, "%Y-%m-%d").date()
    except ValueError:
        sow_dt = date.today()

    duration = int(pop.get("total_duration_days", 100))
    harvest_dt = sow_dt + timedelta(days=duration)
    budget_check = check_budget_feasibility(crop_name, area_acres, farmer_budget_inr)

    # ── Milestone Task Generator ──────────────────────────────────────────────
    f_plan = pop.get("fertilizer_plan", {})
    crit_irr = pop.get("critical_irrigation_stages", [])

    tasks: list[dict[str, Any]] = [
        {
            "task_id": "TSK-001",
            "day_offset": -5,
            "due_date": (sow_dt - timedelta(days=5)).isoformat(),
            "task_type": "land_prep",
            "title": "Land Preparation & Basal Manure",
            "description": f"Plough field thoroughly. Incorporate {f_plan.get('basal', 'Basal manure')} across {area_acres:.1f} acre(s).",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["fertilizer_inr"] * 0.40, 2),
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-002",
            "day_offset": 0,
            "due_date": sow_dt.isoformat(),
            "task_type": "sowing",
            "title": "Seed Treatment & Sowing",
            "description": f"Treat seeds with bio-priming culture ({pop.get('fertilizer_plan', {}).get('organic_alternative', 'Bio-inoculants')}) and complete sowing.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["seed_inr"], 2),
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-003",
            "day_offset": max(15, int(duration * 0.15)),
            "due_date": (sow_dt + timedelta(days=max(15, int(duration * 0.15)))).isoformat(),
            "task_type": "weeding",
            "title": "First Weeding & Stand Inspection",
            "description": "Manual weeding / hoeing to eliminate early crop-weed competition and gap-filling.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["labor_inr"] * 0.30, 2),
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-004",
            "day_offset": max(25, int(duration * 0.28)),
            "due_date": (sow_dt + timedelta(days=max(25, int(duration * 0.28)))).isoformat(),
            "task_type": "fertilizer",
            "title": "Topdress Round 1 + Vegetative Irrigation",
            "description": f"Apply {f_plan.get('topdress_1', 'Vegetative N split')} followed immediately by irrigation.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["fertilizer_inr"] * 0.35, 2),
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-005",
            "day_offset": max(45, int(duration * 0.45)),
            "due_date": (sow_dt + timedelta(days=max(45, int(duration * 0.45)))).isoformat(),
            "task_type": "irrigation",
            "title": f"Critical Stage Irrigation ({crit_irr[0] if crit_irr else 'Flowering'})",
            "description": f"Provide adequate irrigation at critical stage: {crit_irr[0] if crit_irr else 'Flowering'}. Check field moisture.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["irrigation_inr"] * 0.40, 2),
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-006",
            "day_offset": max(55, int(duration * 0.55)),
            "due_date": (sow_dt + timedelta(days=max(55, int(duration * 0.55)))).isoformat(),
            "task_type": "pest_scout",
            "title": "Pest & Disease Scouting",
            "description": f"Scout for: {', '.join(pop.get('common_pests', [])[:2])}. Preventive measure: {pop.get('ipm_practices', 'IPM traps')}.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["pesticide_inr"] * 0.50, 2),
            "is_critical": False,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-007",
            "day_offset": max(70, int(duration * 0.70)),
            "due_date": (sow_dt + timedelta(days=max(70, int(duration * 0.70)))).isoformat(),
            "task_type": "fertilizer",
            "title": "Topdress Round 2 / Pod-Fruit Setting",
            "description": f"Apply {f_plan.get('topdress_2', 'Secondary topdress')} to enhance grain/fruit weight.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["fertilizer_inr"] * 0.25, 2),
            "is_critical": False,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-008",
            "day_offset": duration - 5,
            "due_date": (sow_dt + timedelta(days=duration - 5)).isoformat(),
            "task_type": "harvest",
            "title": "Pre-Harvest Inspection & Drainage",
            "description": "Withhold irrigation 5-7 days prior to harvest. Ensure threshing and storage equipment readiness.",
            "estimated_cost_inr": 0.0,
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
        {
            "task_id": "TSK-009",
            "day_offset": duration,
            "due_date": harvest_dt.isoformat(),
            "task_type": "harvest",
            "title": "Harvest & Post-Harvest Storage",
            "description": f"Harvest at physiological maturity. Expected yield: {pop.get('yield_kg_per_acre', 'standard')} per acre.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["labor_inr"] * 0.40, 2),
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
    ]

    plan_id = f"PLAN-{crop_name[:3].upper()}-{sow_dt.strftime('%Y%m%d')}-{int(area_acres*100)}"

    return {
        "plan_id": plan_id,
        "crop_name": pop["crop_name"],
        "local_name": pop.get("local_name", ""),
        "scientific_name": pop.get("scientific_name", ""),
        "category": pop.get("category", "Cereal"),
        "sowing_date": sow_dt.isoformat(),
        "expected_harvest_date": harvest_dt.isoformat(),
        "duration_days": duration,
        "area_acres": area_acres,
        "farmer_budget_inr": farmer_budget_inr,
        "irrigation_source": irrigation_source,
        "growth_stages": pop.get("growth_stages", []),
        "budget_analysis": budget_check,
        "tasks": tasks,
        "total_tasks_count": len(tasks),
        "completed_tasks_count": 0,
        "plan_status": "active",
        "created_at": datetime.now().isoformat(),
    }