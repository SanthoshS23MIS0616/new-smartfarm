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
    is_legume = "pulse" in pop.get("category", "").lower() or crop_name.lower() in (
        "blackgram", "chickpea", "lentil", "mungbean", "pigeonpeas", "mothbeans"
    )

    t2_desc = (
        f"Bio-prime seeds with Rhizobium culture (30g/kg) + Phosphobacteria (30g/kg) + Trichoderma viride (4g/kg) "
        f"to stimulate biological nitrogen fixation and prevent root rot. Sow at 30x10 cm spacing."
        if is_legume
        else f"Treat seeds with bio-priming culture ({f_plan.get('organic_alternative', 'Bio-inoculants')}) and complete sowing."
    )

    t4_title = "Foliar Nutrition Round 1 (2% DAP Spray) & Stand Inspection" if is_legume else "Topdress Round 1 + Vegetative Irrigation"
    t4_desc = (
        "CRITICAL: Avoid granular urea soil topdressing! (Soil mineral nitrogen inhibits Rhizobium nodulation and nitrogenase). "
        "Apply foliar 2% DAP spray (2 kg DAP soaked overnight in 10 L water, supernatant diluted to 100 L) at flower initiation."
        if is_legume
        else f"Apply {f_plan.get('topdress_1', 'Vegetative N split')} followed immediately by irrigation."
    )

    t7_title = "Pod Setting Foliar Booster (TNAU Pulse Wonder)" if is_legume else "Topdress Round 2 / Pod-Fruit Setting"
    t7_desc = (
        "Foliar spray TNAU Pulse Wonder @ 2 kg/acre in 200 L water at 45 DAS to arrest flower shedding, "
        "boost pod elongation, and enhance test grain weight by 15-20%."
        if is_legume
        else f"Apply {f_plan.get('topdress_2', 'Secondary topdress')} to enhance grain/fruit weight."
    )

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
            "title": "Rhizobium Bio-Priming Seed Treatment & Sowing" if is_legume else "Seed Treatment & Sowing",
            "description": t2_desc,
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
            "title": t4_title,
            "description": t4_desc,
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
            "day_offset": max(60, int(duration * 0.65)),
            "due_date": (sow_dt + timedelta(days=max(60, int(duration * 0.65)))).isoformat(),
            "task_type": "fertilizer",
            "title": t7_title,
            "description": t7_desc,
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
            "description": f"Harvest at physiological maturity (pods dark brown/black). Expected yield: {pop.get('yield_kg_per_acre', 'standard')} per acre.",
            "estimated_cost_inr": round(budget_check["cost_breakdown"]["labor_inr"] * 0.40, 2),
            "is_critical": True,
            "is_completed": False,
            "completed_at": None,
        },
    ]

    plan_id = f"PLAN-{crop_name[:3].upper()}-{sow_dt.strftime('%Y%m%d')}-{int(area_acres*100)}"

    # ── 1-Year Multi-Crop Rotation Plan (365-Day Cycle with 20-Day Soil Rest Gaps) ──
    annual_rotation = generate_annual_crop_cycle(
        primary_crop=pop["crop_name"],
        start_date=sow_dt,
        area_acres=area_acres,
        farmer_budget_inr=farmer_budget_inr,
    )

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
        "annual_rotation_cycle": annual_rotation,
        "total_tasks_count": len(tasks),
        "completed_tasks_count": 0,
        "plan_status": "active",
        "created_at": datetime.now().isoformat(),
    }


def generate_annual_crop_cycle(
    primary_crop: str,
    start_date: date,
    area_acres: float = 1.0,
    farmer_budget_inr: float = 50000.0,
) -> dict[str, Any]:
    """
    Generates a realistic 1-Year (365-Day) Multi-Crop Rotation Plan
    with mandatory 20-day safety rest & recuperation gaps between successive crops.
    """
    db = load_pop_database()
    p1 = db.get(primary_crop.lower()) or get_crop_pop(primary_crop) or list(db.values())[0]

    # Crop rotation partner selection based on ICAR ecological principles:
    cat1 = p1.get("category", "Cereal")
    
    # Partner 2 (Rabi / Succession):
    if "Pulse" in cat1 or primary_crop.lower() in ("blackgram", "chickpea", "lentil", "mungbean", "pigeonpeas", "mothbeans"):
        # Legume primary -> rotate with Cereal or Oilseed to use fixed nitrogen
        c2_name = "Maize" if "maize" in db else "Rice"
    elif "Cereal" in cat1 or primary_crop.lower() in ("rice", "maize"):
        # Cereal primary -> rotate with nitrogen-fixing Legume
        c2_name = "Chickpea" if "chickpea" in db else "Lentil"
    elif "Fiber" in cat1 or "Commercial" in cat1 or primary_crop.lower() in ("cotton", "jute"):
        c2_name = "Chickpea" if "chickpea" in db else "Blackgram"
    else:
        c2_name = "Blackgram" if "blackgram" in db else "Maize"

    p2 = db.get(c2_name.lower()) or list(db.values())[0]

    # Partner 3 (Summer / Zaid / Catch crop):
    if primary_crop.lower() == "blackgram":
        c3_name = "Mungbean" if "mungbean" in db else "Lentil"
    elif c2_name.lower() == "chickpea":
        c3_name = "Blackgram" if "blackgram" in db else "Maize"
    else:
        c3_name = "Mungbean" if "mungbean" in db else "Chickpea"
    p3 = db.get(c3_name.lower()) or list(db.values())[0]

    # Cycle 1:
    dur1 = int(p1.get("total_duration_days", 95))
    sow1 = start_date
    harv1 = sow1 + timedelta(days=dur1)

    gap1_start = harv1
    gap1_end = gap1_start + timedelta(days=20)

    # Cycle 2:
    dur2 = int(p2.get("total_duration_days", 100))
    sow2 = gap1_end
    harv2 = sow2 + timedelta(days=dur2)

    gap2_start = harv2
    gap2_end = gap2_start + timedelta(days=20)

    # Cycle 3 (Remainder of 365 days):
    days_used = dur1 + 20 + dur2 + 20
    remaining_days = max(60, 365 - days_used)
    dur3 = min(int(p3.get("total_duration_days", 70)), remaining_days)
    sow3 = gap2_end
    harv3 = sow3 + timedelta(days=dur3)

    # Financial estimates per cycle (scaled to area)
    profit1 = round(float(p1.get("costs_per_acre", {}).get("total_inr", 15000)) * 1.65 * area_acres, 2)
    profit2 = round(float(p2.get("costs_per_acre", {}).get("total_inr", 14000)) * 1.55 * area_acres, 2)
    profit3 = round(float(p3.get("costs_per_acre", {}).get("total_inr", 12000)) * 1.45 * area_acres, 2)
    total_annual_profit = round(profit1 + profit2 + profit3, 2)

    cycles = [
        {
            "cycle_number": 1,
            "crop_name": p1["crop_name"],
            "local_name": p1.get("local_name", ""),
            "category": p1.get("category", "Field Crop"),
            "season_label": "Kharif / Main Season",
            "sowing_date": sow1.isoformat(),
            "expected_harvest_date": harv1.isoformat(),
            "duration_days": dur1,
            "expected_yield_per_acre": str(p1.get("yield_kg_per_acre", "800-1200 kg")),
            "estimated_net_profit_inr": profit1,
            "role": "Primary high-value seasonal harvest",
            "safety_gap_after": {
                "duration_days": 20,
                "start_date": gap1_start.isoformat(),
                "end_date": gap1_end.isoformat(),
                "activity": "Deep summer/post-harvest ploughing, soil solarization, sunnhemp/dhaincha green manuring.",
                "soil_benefit": "Replenishes +15 kg/ha organic nitrogen, suppresses fungal spores and weed seedbank.",
            },
        },
        {
            "cycle_number": 2,
            "crop_name": p2["crop_name"],
            "local_name": p2.get("local_name", ""),
            "category": p2.get("category", "Field Crop"),
            "season_label": "Rabi / Winter Succession",
            "sowing_date": sow2.isoformat(),
            "expected_harvest_date": harv2.isoformat(),
            "duration_days": dur2,
            "expected_yield_per_acre": str(p2.get("yield_kg_per_acre", "1000-1500 kg")),
            "estimated_net_profit_inr": profit2,
            "role": "Rotational nutrient-balancing and pest cycle break",
            "safety_gap_after": {
                "duration_days": 20,
                "start_date": gap2_start.isoformat(),
                "end_date": gap2_end.isoformat(),
                "activity": "Stubble incorporation, FYM (Farm Yard Manure 4-5 t/acre) basal application, harrowing.",
                "soil_benefit": "Increases microbial biomass carbon and soil water-holding capacity.",
            },
        },
        {
            "cycle_number": 3,
            "crop_name": p3["crop_name"],
            "local_name": p3.get("local_name", ""),
            "category": p3.get("category", "Field Crop"),
            "season_label": "Summer / Zaid Catch Crop",
            "sowing_date": sow3.isoformat(),
            "expected_harvest_date": harv3.isoformat(),
            "duration_days": dur3,
            "expected_yield_per_acre": str(p3.get("yield_kg_per_acre", "600-900 kg")),
            "estimated_net_profit_inr": profit3,
            "role": "Short-duration catch crop maximizing annual farm land productivity",
            "safety_gap_after": None,
        },
    ]

    return {
        "cycle_duration_total_days": (harv3 - sow1).days,
        "start_date": sow1.isoformat(),
        "end_date": harv3.isoformat(),
        "total_crops": 3,
        "total_rest_gap_days": 40,
        "rest_gap_per_interval_days": 20,
        "estimated_annual_profit_inr": total_annual_profit,
        "soil_health_rating": "Optimal (Legume-Cereal Synergistic Cycle)",
        "cycles": cycles,
    }