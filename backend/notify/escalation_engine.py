"""
escalation_engine.py
====================
Closed-Loop Graded Escalation & Environmental Recalibration Engine for SmartFarm:
- Dynamic task recalibration based on real-time weather & schedule adherence
- Graded notification ladder:
    Tier 0: Normal (Silent Log)
    Tier 1: Watch (In-App notification with checkbox confirmation)
    Tier 2: Warning (WhatsApp alert with explicit cost-benefit calculation)
    Tier 3: Critical (WhatsApp + SMS + Automated Voice Call in English/Tamil)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


def evaluate_task_escalation(
    task: dict[str, Any],
    current_date: date | None = None,
    current_weather: dict[str, Any] | None = None,
    crop_price_rs_per_kg: float = 30.0,
) -> dict[str, Any]:
    if current_date is None:
        current_date = date.today()

    due_dt = datetime.strptime(task["due_date"], "%Y-%m-%d").date()
    days_overdue = (current_date - due_dt).days
    is_completed = bool(task.get("is_completed", False))

    if is_completed:
        return {
            "tier": 0,
            "tier_label": "Normal",
            "channel": "silent_log",
            "message": f"Task '{task['title']}' completed on time.",
            "is_action_required": False,
            "profit_at_risk_inr": 0.0,
        }

    # Recalibration 1: Rain suppresses scheduled irrigation
    if current_weather and task.get("task_type") == "irrigation":
        rain_mm = float(current_weather.get("recent_rainfall_mm", 0.0) or current_weather.get("rainfall_mm", 0.0))
        if rain_mm >= 20.0 and abs(days_overdue) <= 2:
            return {
                "tier": 1,
                "tier_label": "Recalibrated",
                "channel": "in_app",
                "message": (
                    f"Rainfall of {rain_mm:.1f} mm recorded. "
                    f"Scheduled irrigation '{task['title']}' is auto-suppressed to conserve water and prevent root rot."
                ),
                "is_action_required": False,
                "auto_recalibrated": True,
                "profit_at_risk_inr": 0.0,
            }

    # Recalibration 2: Rain postpones foliar spraying
    if current_weather and task.get("task_type") in ("fertilizer", "pest_scout"):
        rain_mm = float(current_weather.get("rainfall_mm", 0.0))
        if rain_mm >= 25.0 and days_overdue == 0:
            return {
                "tier": 1,
                "tier_label": "Postponed",
                "channel": "in_app",
                "message": (
                    f"Rain forecast of {rain_mm:.1f} mm detected. "
                    f"Postpone '{task['title']}' by 24-48 hours to avoid wash-off loss."
                ),
                "is_action_required": True,
                "auto_recalibrated": True,
                "profit_at_risk_inr": round(float(task.get("estimated_cost_inr", 0.0)), 2),
            }

    # Escalation Logic based on days overdue and task criticality
    cost = float(task.get("estimated_cost_inr", 1000.0))
    # Risk factor: unperformed tasks jeopardize ~3-5x the task cost in yield loss
    profit_at_risk = round(cost * 3.8, 2)

    if days_overdue < 0:
        # Task is in the future
        days_remaining = abs(days_overdue)
        if days_remaining <= 1:
            return {
                "tier": 1,
                "tier_label": "Watch",
                "channel": "in_app",
                "message": f"Upcoming task tomorrow: '{task['title']}'. Prepare inputs.",
                "is_action_required": True,
                "profit_at_risk_inr": 0.0,
            }
        return {
            "tier": 0,
            "tier_label": "Scheduled",
            "channel": "silent_log",
            "message": f"Task due in {days_remaining} days.",
            "is_action_required": False,
            "profit_at_risk_inr": 0.0,
        }

    if days_overdue == 0:
        # Due today
        return {
            "tier": 1,
            "tier_label": "Watch",
            "channel": "in_app",
            "message": f"Today's Task: '{task['title']}'. Tap checkbox once completed.",
            "is_action_required": True,
            "profit_at_risk_inr": profit_at_risk,
        }

    if 1 <= days_overdue <= 2:
        # Overdue by 1-2 days -> Tier 2 (WhatsApp)
        return {
            "tier": 2,
            "tier_label": "Warning",
            "channel": "whatsapp",
            "message": (
                f"Reminder: '{task['title']}' was due {days_overdue} day(s) ago. "
                f"Action required (est. cost INR {cost:,.0f}) to protect approx INR {profit_at_risk:,.0f} in crop yield."
            ),
            "tamil_message": (
                f"நினைவூட்டல்: '{task['title']}' {days_overdue} நாட்களுக்கு முன் செய்ய வேண்டியது. "
                f"மகசூல் இழப்பைத் தவிர்க்க உடனே செய்யவும்."
            ),
            "is_action_required": True,
            "profit_at_risk_inr": profit_at_risk,
        }

    # Overdue by 3+ days -> Tier 3 (Critical - WhatsApp + SMS + Voice Call)
    return {
        "tier": 3,
        "tier_label": "Critical",
        "channel": "whatsapp_sms_call",
        "message": (
            f"URGENT: Critical task '{task['title']}' is {days_overdue} days overdue! "
            f"Immediate intervention required to prevent severe yield loss of approx INR {profit_at_risk:,.0f}."
        ),
        "tamil_message": (
            f"அவசரம்: '{task['title']}' {days_overdue} நாட்களாக செய்யப்படவில்லை! "
            f"பயிர் இழப்பைத் தவிர்க்க உடனடியாக கவனம் செலுத்தவும்."
        ),
        "is_action_required": True,
        "profit_at_risk_inr": profit_at_risk,
    }


def recalibrate_entire_plan(
    plan: dict[str, Any],
    current_weather: dict[str, Any] | None = None,
    current_date: date | None = None,
) -> dict[str, Any]:
    if current_date is None:
        current_date = date.today()

    recalibrated_tasks = []
    total_profit_at_risk = 0.0
    highest_tier = 0
    active_alerts = []

    for task in plan.get("tasks", []):
        eval_result = evaluate_task_escalation(
            task,
            current_date=current_date,
            current_weather=current_weather,
        )
        task_copy = dict(task)
        task_copy["escalation_tier"] = eval_result["tier"]
        task_copy["escalation_label"] = eval_result["tier_label"]
        task_copy["escalation_channel"] = eval_result["channel"]
        task_copy["alert_message"] = eval_result["message"]
        task_copy["profit_at_risk_inr"] = eval_result["profit_at_risk_inr"]

        if eval_result.get("auto_recalibrated") and eval_result["tier_label"] == "Recalibrated":
            task_copy["is_completed"] = True
            task_copy["completed_at"] = current_date.isoformat()
            task_copy["recalibration_note"] = eval_result["message"]

        if eval_result["is_action_required"] and eval_result["tier"] > 0:
            active_alerts.append({
                "task_id": task["task_id"],
                "title": task["title"],
                "tier": eval_result["tier"],
                "tier_label": eval_result["tier_label"],
                "channel": eval_result["channel"],
                "message": eval_result["message"],
                "tamil_message": eval_result.get("tamil_message", ""),
                "profit_at_risk_inr": eval_result["profit_at_risk_inr"],
            })

        total_profit_at_risk += eval_result["profit_at_risk_inr"]
        highest_tier = max(highest_tier, eval_result["tier"])
        recalibrated_tasks.append(task_copy)

    completed_count = sum(1 for t in recalibrated_tasks if t.get("is_completed"))

    return {
        "plan_id": plan.get("plan_id"),
        "crop_name": plan.get("crop_name"),
        "rechecked_at": datetime.now().isoformat(),
        "highest_escalation_tier": highest_tier,
        "highest_escalation_label": {0: "Normal", 1: "Watch", 2: "Warning", 3: "Critical"}.get(highest_tier, "Normal"),
        "total_profit_at_risk_inr": round(total_profit_at_risk, 2),
        "total_tasks": len(recalibrated_tasks),
        "completed_tasks": completed_count,
        "completion_percentage": round((completed_count / max(1, len(recalibrated_tasks))) * 100, 1),
        "active_alerts": active_alerts,
        "tasks": recalibrated_tasks,
    }

def dispatch_escalation_alert(
    alert: dict[str, Any],
    farmer_phone: str = "+919876543210",
    farmer_name: str = "Farmer",
) -> dict[str, Any]:
    """
    Physical cellular dispatch via Twilio (if credentials provided) or structured log simulation.
    Handles WhatsApp, SMS, and IVR Voice Call in English or Tamil.
    """
    import os
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    from_phone = os.environ.get("TWILIO_PHONE_NUMBER")
    if not farmer_phone or farmer_phone == "+919876543210":
        farmer_phone = os.environ.get("FARMER_PHONE_NUMBER", farmer_phone)

    tier = alert.get("tier", 1)
    channel = alert.get("channel", "in_app")
    msg = alert.get("tamil_message") or alert.get("message")

    dispatch_result = {
        "tier": tier,
        "channel": channel,
        "recipient": farmer_phone,
        "message": msg,
        "status": "simulated",
        "timestamp": datetime.now().isoformat(),
    }

    if not (account_sid and auth_token and from_phone):
        logger.info(
            "[ALERT DISPATCH SIMULATION] Tier %d (%s) -> %s: %s",
            tier, channel, farmer_phone, alert.get("message")
        )
        return dispatch_result

    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)

        if channel == "whatsapp":
            message = client.messages.create(
                from_=f"whatsapp:{from_phone}",
                to=f"whatsapp:{farmer_phone}",
                body=f"🌾 *SmartFarm Alert (Tier {tier})*\n\n{msg}"
            )
            dispatch_result["status"] = "dispatched"
            dispatch_result["sid"] = message.sid

        elif channel in ("sms", "whatsapp_sms_call"):
            message = client.messages.create(
                from_=from_phone,
                to=farmer_phone,
                body=f"SmartFarm Tier {tier} Alert: {alert.get('message')}"
            )
            dispatch_result["status"] = "dispatched"
            dispatch_result["sid"] = message.sid

            # Also place automated IVR Voice Call in Tamil or English
            try:
                is_ta = bool(alert.get("tamil_message"))
                speech_text = alert.get("tamil_message") or alert.get("message")
                lang_code = "ta-IN" if is_ta else "en-IN"
                twiml_str = f'<Response><Say language="{lang_code}">{speech_text}</Say></Response>'
                call = client.calls.create(
                    twiml=twiml_str,
                    to=farmer_phone,
                    from_=from_phone
                )
                dispatch_result["call_sid"] = call.sid
            except Exception as call_err:
                logger.warning("Twilio voice call dispatch note: %s", call_err)

        return dispatch_result
    except Exception as exc:
        logger.warning("Twilio physical dispatch failed: %s", exc)
        dispatch_result["status"] = f"failed: {exc}"
        return dispatch_result
