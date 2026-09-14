from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Header, HTTPException

from backend.app.schemas import (
    AssistantAskRequest,
    GoogleAuthRequest,
    PlanGenerateRequest,
    PlanRecheckRequest,
    PredictionInput,
    SendOTPRequest,
    SoilEstimateRequest,
    TaskConfirmRequest,
    TrainRequest,
    VerifyOTPRequest,
)
from backend.app.services.predictor import get_engine, model_artifact_status, train_models

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["crop-intelligence"])

_ADMIN_KEY_ENV = "ADMIN_API_KEY"
_ADMIN_KEY_DEFAULT = "crop-admin-secret-2026"  # override via env var in production


def _require_admin_key(x_admin_key: str | None) -> None:
    """Raise 403 if the X-Admin-Key header is missing or incorrect."""
    expected = os.environ.get(_ADMIN_KEY_ENV, _ADMIN_KEY_DEFAULT)
    if not x_admin_key or x_admin_key != expected:
        raise HTTPException(
            status_code=403,
            detail="Admin key required. Set X-Admin-Key header with the correct value.",
        )


@router.get("/health")
def health() -> dict[str, object]:
    status = model_artifact_status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail={"status": "models-not-ready", **status})
    return {"status": "ok", **status}


@router.get("/metadata")
def metadata() -> dict:
    try:
        engine = get_engine()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=503, detail=f"Models are not ready: {exc}") from exc
    return engine.bundle.metadata["dataset_summary"] | {
        "training_report": engine.bundle.metadata["training_report"],
        "xai_assets": engine.bundle.metadata.get("xai_assets", {}),
    }


@router.post("/predict")
def predict(request: PredictionInput) -> dict:
    try:
        engine = get_engine()
        payload = request.model_dump()
        top_k = payload.pop("top_k", 3)
        return engine.predict(payload, top_k=top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Trained models not found: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/train")
def train(
    request: TrainRequest,
    x_admin_key: str | None = Header(default=None),
) -> dict:
    _require_admin_key(x_admin_key)
    try:
        return train_models(data_dir=request.data_dir)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Farmer Authentication & OTP Endpoints ────────────────────────────────────

@router.post("/auth/otp/send")
def send_otp(request: SendOTPRequest) -> dict:
    from backend.storage.plan_db import generate_otp_for_phone
    try:
        phone = request.phone_number.strip().replace(" ", "").replace("-", "")
        if len(phone) < 8:
            raise HTTPException(status_code=400, detail="Invalid phone number format. Please enter a valid mobile number.")

        target_phone = phone if phone.startswith("+") else f"+91{phone}" if len(phone) == 10 else f"+{phone}"
        otp_code = generate_otp_for_phone(target_phone)

        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        verify_service_sid = os.environ.get("TWILIO_VERIFY_SERVICE_SID", "VA195dd6ecc329a4cb3cd06b6ca3fd4228")
        from_phone = os.environ.get("TWILIO_PHONE_NUMBER")

        dispatch_msg = "OTP verification code ready."
        real_sms_sent = False
        whatsapp_sent = False

        if account_sid and auth_token:
            try:
                from twilio.rest import Client
                client = Client(account_sid, auth_token)

                # Primary: Twilio Verify Service SMS
                try:
                    verif = client.verify.v2.services(verify_service_sid).verifications.create(
                        to=target_phone,
                        channel="sms"
                    )
                    if verif.status in ("pending", "approved"):
                        dispatch_msg = f"Official SMS OTP sent to {target_phone}."
                        real_sms_sent = True
                except Exception as verif_err:
                    logger.info("Twilio Verify SMS note: %s", verif_err)

                # Secondary fallback: Standard SMS
                if not real_sms_sent and from_phone:
                    try:
                        client.messages.create(
                            from_=from_phone,
                            to=target_phone,
                            body=f"SmartFarm: Your verification code is {otp_code}. Valid for 10 minutes."
                        )
                        dispatch_msg = f"SMS OTP sent to {target_phone}."
                        real_sms_sent = True
                    except Exception as sms_err:
                        logger.debug("Twilio standard SMS note: %s", sms_err)

                # WhatsApp OTP — same code sent via WhatsApp
                wa_body = f"\U0001f33e SmartFarm OTP: Your verification code is *{otp_code}*. Valid 10 minutes."
                try:
                    # Try WhatsApp sandbox (user must send 'join <keyword>' to +14155238886 first)
                    client.messages.create(
                        from_="whatsapp:+14155238886",
                        to=f"whatsapp:{target_phone}",
                        body=wa_body
                    )
                    whatsapp_sent = True
                    logger.info("WhatsApp OTP sent via sandbox to %s", target_phone)
                except Exception as wa_sandbox_err:
                    logger.debug("WhatsApp sandbox OTP note: %s", wa_sandbox_err)
                    if from_phone:
                        try:
                            client.messages.create(
                                from_=f"whatsapp:{from_phone}",
                                to=f"whatsapp:{target_phone}",
                                body=wa_body
                            )
                            whatsapp_sent = True
                        except Exception as wa_err:
                            logger.debug("WhatsApp direct OTP note: %s", wa_err)
            except Exception as tw_err:
                logger.warning("Twilio client error: %s", tw_err)

        channels = []
        if real_sms_sent:
            channels.append("SMS")
        if whatsapp_sent:
            channels.append("WhatsApp")
        if channels:
            dispatch_msg = f"OTP sent via {' & '.join(channels)} to {target_phone}."

        return {
            "status": "success",
            "message": f"{dispatch_msg} (Code: {otp_code} or master code 1234)",
            "phone_number": target_phone,
            "otp_demo": otp_code,
            "sms_dispatched": real_sms_sent,
            "whatsapp_dispatched": whatsapp_sent,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("send_otp error: %s", exc)
        raise HTTPException(status_code=500, detail=f"OTP service error: {exc}") from exc


@router.post("/soil/estimate")
def get_soil_estimate(request: SoilEstimateRequest) -> dict:
    from backend.ml.soil_ai_estimator import estimate_soil_parameters
    try:
        data = estimate_soil_parameters(
            latitude=request.latitude,
            longitude=request.longitude,
            district=request.district,
            state=request.state,
        )
        return {"status": "success", "data": data}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/auth/otp/verify")
def verify_otp(request: VerifyOTPRequest) -> dict:
    from backend.storage.plan_db import verify_otp_for_phone, save_or_update_user
    phone = request.phone_number.strip().replace(" ", "").replace("-", "")
    target_phone = phone if phone.startswith("+") else f"+91{phone}" if len(phone) == 10 else f"+{phone}"
    
    is_valid = False
    
    # 1. Verify via Twilio Verify Service if available
    account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
    auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
    verify_service_sid = os.environ.get("TWILIO_VERIFY_SERVICE_SID", "VA195dd6ecc329a4cb3cd06b6ca3fd4228")
    if account_sid and auth_token:
        try:
            from twilio.rest import Client
            client = Client(account_sid, auth_token)
            check = client.verify.v2.services(verify_service_sid).verification_checks.create(
                to=target_phone,
                code=request.otp_code.strip()
            )
            if check.status == "approved":
                is_valid = True
        except Exception as check_err:
            logger.debug("Twilio Verify check note: %s", check_err)

    # 2. Verify via local database OTP or master demo code "1234"
    if not is_valid:
        is_valid = verify_otp_for_phone(target_phone, request.otp_code) or verify_otp_for_phone(phone, request.otp_code)

    if not is_valid:
        raise HTTPException(status_code=400, detail="Invalid or expired OTP. Please try again.")

    user = save_or_update_user(
        phone_number=target_phone,
        full_name=request.full_name or "Farmer",
        auth_provider="phone_otp"
    )
    return {
        "status": "success",
        "message": "Authentication successful.",
        "user": user,
    }


@router.post("/auth/google")
def google_auth(request: GoogleAuthRequest) -> dict:
    from backend.storage.plan_db import save_or_update_user
    google_client_id = os.environ.get("GOOGLE_CLIENT_ID")
    google_client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")
    
    email = request.email or ""
    full_name = request.full_name or "Google Farmer User"
    phone = request.phone_number or "google_user_" + os.urandom(4).hex()

    if getattr(request, "token", None) and google_client_id:
        try:
            from google.oauth2 import id_token
            from google.auth.transport import requests as google_requests
            id_info = id_token.verify_oauth2_token(request.token, google_requests.Request(), google_client_id)
            email = id_info.get("email", email)
            full_name = id_info.get("name", full_name)
        except Exception as g_err:
            logger.warning("Google token verification note: %s", g_err)

    user = save_or_update_user(
        phone_number=phone,
        full_name=full_name,
        email=email,
        auth_provider="google_oauth"
    )
    return {
        "status": "success",
        "message": f"Google authentication successful for {full_name}.",
        "user": user,
        "google_client_id": google_client_id,
    }


@router.get("/farmer/{phone_number}/plans")
def get_farmer_plans(phone_number: str) -> dict:
    from backend.storage.plan_db import get_plans_by_farmer_phone, get_user_by_phone
    phone = phone_number.strip().replace(" ", "").replace("-", "")
    user = get_user_by_phone(phone)
    plans = get_plans_by_farmer_phone(phone)
    return {
        "phone_number": phone,
        "user": user,
        "plans_count": len(plans),
        "plans": plans,
    }


# ── Active Sowing-to-Harvest Plans & Execution ──────────────────────────────
_ACTIVE_PLANS: dict[str, dict] = {}


@router.post("/plan/generate")
def generate_plan(request: PlanGenerateRequest) -> dict:
    from backend.ml.plan_generator import generate_sowing_plan
    from datetime import date
    sowing_date = request.sowing_date or date.today().isoformat()
    try:
        plan = generate_sowing_plan(
            crop_name=request.crop_name,
            sowing_date_str=sowing_date,
            area_acres=request.area_acres,
            farmer_budget_inr=request.farmer_budget_inr,
            irrigation_source=request.irrigation_source,
        )
        if request.farmer_phone:
            plan["farmer_phone"] = request.farmer_phone.strip().replace(" ", "").replace("-", "")
            from backend.storage.plan_db import save_or_update_user
            save_or_update_user(phone_number=plan["farmer_phone"], full_name=request.farmer_name or "Farmer")

        from backend.storage.plan_db import save_plan
        save_plan(plan, farmer_phone=request.farmer_phone)
        _ACTIVE_PLANS[plan["plan_id"]] = plan

        # WhatsApp: Send 1st task details to farmer right after plan generation
        if request.farmer_phone:
            try:
                from twilio.rest import Client
                account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
                auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
                from_phone = os.environ.get("TWILIO_PHONE_NUMBER")
                farmer_phone = plan["farmer_phone"]
                tasks = plan.get("tasks", [])
                if account_sid and auth_token and tasks:
                    first_task = tasks[0]
                    wa_msg = (
                        f"\U0001f33e *SmartFarm Plan Created!*\n"
                        f"Crop: *{plan.get('crop_name')}* | Area: {plan.get('area_acres', 1):.1f} acres\n"
                        f"Sowing: {plan.get('sowing_date')} → Harvest: {plan.get('expected_harvest_date')}\n\n"
                        f"\U0001f4cb *Task 1 (Start Today):*\n"
                        f"*{first_task.get('title')}*\n"
                        f"{first_task.get('description', '')}\n"
                        f"\U0001f4c5 Due: {first_task.get('due_date')} | Est. Cost: \u20b9{int(first_task.get('estimated_cost_inr', 0)):,}\n\n"
                        f"Check SmartFarm dashboard for full schedule \u2705"
                    )
                    client = Client(account_sid, auth_token)
                    try:
                        client.messages.create(
                            from_="whatsapp:+14155238886",
                            to=f"whatsapp:{farmer_phone}",
                            body=wa_msg
                        )
                    except Exception:
                        if from_phone:
                            try:
                                client.messages.create(
                                    from_=f"whatsapp:{from_phone}",
                                    to=f"whatsapp:{farmer_phone}",
                                    body=wa_msg
                                )
                            except Exception:
                                pass
            except Exception as wa_plan_err:
                logger.debug("WhatsApp plan notify note: %s", wa_plan_err)

        return plan
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to generate plan: {exc}") from exc


@router.get("/plan/{plan_id}")
def get_plan(plan_id: str) -> dict:
    from backend.storage.plan_db import get_plan_by_id
    plan = get_plan_by_id(plan_id) or _ACTIVE_PLANS.get(plan_id)
    if not plan:
        raise HTTPException(status_code=404, detail=f"Plan '{plan_id}' not found.")
    return plan


@router.post("/plan/task/confirm")
def confirm_task(request: TaskConfirmRequest) -> dict:
    import os
    from datetime import datetime
    from backend.storage.plan_db import update_task_completion, get_plan_by_id
    from backend.notify.escalation_engine import dispatch_escalation_alert

    updated_plan = update_task_completion(
        plan_id=request.plan_id,
        task_id=request.task_id,
        is_completed=request.is_completed,
        completed_at=datetime.now().isoformat() if request.is_completed else None,
    )

    if not updated_plan:
        plan = _ACTIVE_PLANS.get(request.plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail=f"Plan '{request.plan_id}' not found.")
        found = False
        for task in plan.get("tasks", []):
            if task.get("task_id") == request.task_id:
                task["is_completed"] = request.is_completed
                task["completed_at"] = datetime.now().isoformat() if request.is_completed else None
                found = True
                break
        if not found:
            raise HTTPException(status_code=404, detail=f"Task '{request.task_id}' not found in plan.")
        plan["completed_tasks_count"] = sum(1 for t in plan["tasks"] if t.get("is_completed"))
        updated_plan = plan

    was_voice_escalated = _ACTIVE_PLANS.get(request.plan_id, {}).get("escalation_voice_call_sent", False)
    _ACTIVE_PLANS[request.plan_id] = updated_plan
    if was_voice_escalated:
        _ACTIVE_PLANS[request.plan_id]["escalation_voice_call_sent"] = True

    # ── Checkbox Escalation Logic ─────────────────────────────────────────────

    # Intentional Scenario:
    # 1. Box 1 (TSK-001) completed
    # 2. Box 2 (TSK-002) and Box 3 (TSK-003) skipped
    # 3. Box 4 (TSK-004) selected -> TRIGGERS TWILIO LIVE VOICE CALL!
    # 4. Following that, when any other checklist is selected -> TRIGGERS WHATSAPP ALERT!

    escalation_info = None
    farmer_phone = updated_plan.get("farmer_phone") or os.environ.get("FARMER_PHONE_NUMBER", "+919994525549")
    task_map = {t["task_id"]: t for t in updated_plan.get("tasks", [])}

    if request.is_completed:
        # Scenario A: Task 4 checked while Task 1 is done and Tasks 2 & 3 skipped
        if request.task_id == "TSK-004":
            t1_done = task_map.get("TSK-001", {}).get("is_completed", False)
            t2_done = task_map.get("TSK-002", {}).get("is_completed", False)
            t3_done = task_map.get("TSK-003", {}).get("is_completed", False)

            if t1_done and (not t2_done or not t3_done):
                # Trigger critical Twilio Voice Call
                alert = {
                    "tier": 3,
                    "channel": "whatsapp_sms_call",
                    "title": "Critical Sequence Alert: Sowing & Weeding Tasks Skipped!",
                    "message": (
                        "SmartFarm Urgent Call: Task 4 (Topdress Round 1) was checked, but Task 2 (Seed Treatment & Sowing) "
                        "and Task 3 (First Weeding) were SKIPPED! Applying fertilizer to unsown or weed-heavy land causes "
                        "severe financial loss. Immediate on-field inspection is required."
                    ),
                    "tamil_message": (
                        "ஸ்மார்ட்ஃபார்ம் அவசர எச்சரிக்கை அழைப்பு: நீங்கள் பணி 4-ஐ (மேலுரமிடுதல்) தேர்ந்தெடுத்துள்ளீர்கள், "
                        "ஆனால் விதை நேர்த்தி மற்றும் முதல் களையெடுத்தல் பணிகள் விடுபட்டுள்ளன! "
                        "பயிர் இழப்பைத் தவிர்க்க உடனடியாக நிலத்தைப் பார்வையிடவும்."
                    ),
                }
                dispatch_res = dispatch_escalation_alert(alert, farmer_phone=farmer_phone)
                updated_plan["escalation_voice_call_sent"] = True
                _ACTIVE_PLANS[request.plan_id]["escalation_voice_call_sent"] = True
                escalation_info = {
                    "triggered": True,
                    "type": "voice_call",
                    "title": "🚨 Live Voice Call Dispatched!",
                    "message": alert["message"],
                    "tamil_message": alert["tamil_message"],
                    "skipped_tasks": ["TSK-002: Seed Treatment & Sowing", "TSK-003: First Weeding & Stand Inspection"],
                    "dispatch": dispatch_res,
                }

        # Scenario B: Task checked after voice call escalation occurred -> triggers WhatsApp alert
        elif _ACTIVE_PLANS.get(request.plan_id, {}).get("escalation_voice_call_sent"):
            task_title = task_map.get(request.task_id, {}).get("title", request.task_id)
            alert = {
                "tier": 2,
                "channel": "whatsapp",
                "title": f"SmartFarm Update: {task_title} Confirmed",
                "message": (
                    f"🌾 SmartFarm WhatsApp Notice: Task '{task_title}' was confirmed on your schedule. "
                    "Ensure prior skipped tasks are verified to safeguard expected crop harvest yield."
                ),
                "tamil_message": (
                    f"🌾 ஸ்மார்ட்ஃபார்ம் வாட்ஸ்அப் அறிவிப்பு: '{task_title}' பணி உங்கள் அட்டவணையில் உறுதிப்படுத்தப்பட்டது. "
                    "முந்தைய விடுபட்ட பணிகளை சரிபார்த்து மகசூலைப் பாதுகாக்கவும்."
                ),
            }
            dispatch_res = dispatch_escalation_alert(alert, farmer_phone=farmer_phone)
            escalation_info = {
                "triggered": True,
                "type": "whatsapp",
                "title": "📱 WhatsApp Alert Dispatched!",
                "message": alert["message"],
                "tamil_message": alert["tamil_message"],
                "dispatch": dispatch_res,
            }

    # -- WhatsApp done confirmation for EVERY task completion --
    if request.is_completed:
        try:
            from twilio.rest import Client as _TwCl
            _acct = os.environ.get("TWILIO_ACCOUNT_SID")
            _auth = os.environ.get("TWILIO_AUTH_TOKEN")
            _from = os.environ.get("TWILIO_PHONE_NUMBER")
            _task_title = task_map.get(request.task_id, {}).get("title", request.task_id)
            _done_count = updated_plan.get("completed_tasks_count", 0)
            _total = len(updated_plan.get("tasks", []))
            _crop = updated_plan.get("crop_name", "Crop")
            _wa_msg = (
                f"\u2705 *SmartFarm Task Done!*\n"
                f"Crop: *{_crop}*\n"
                f"\u2714\ufe0f Completed: *{_task_title}*\n"
                f"Progress: {_done_count}/{_total} tasks done\n\n"
                f"Great job! Keep up the schedule \U0001f33e"
            )
            if _acct and _auth:
                _twcl = _TwCl(_acct, _auth)
                try:
                    _twcl.messages.create(
                        from_="whatsapp:+14155238886",
                        to=f"whatsapp:{farmer_phone}",
                        body=_wa_msg
                    )
                except Exception:
                    if _from:
                        try:
                            _twcl.messages.create(
                                from_=f"whatsapp:{_from}",
                                to=f"whatsapp:{farmer_phone}",
                                body=_wa_msg
                            )
                        except Exception:
                            pass
        except Exception as _wa_done_err:
            logger.debug("WhatsApp done notify: %s", _wa_done_err)

    return {
        "status": "success",
        "plan_id": request.plan_id,
        "task_id": request.task_id,
        "is_completed": request.is_completed,
        "completed_tasks_count": updated_plan["completed_tasks_count"],
        "total_tasks_count": len(updated_plan["tasks"]),
        "escalation": escalation_info,
    }


@router.post("/plan/recheck")
def recheck_plan(request: PlanRecheckRequest) -> dict:
    from backend.notify.escalation_engine import recalibrate_entire_plan, dispatch_escalation_alert
    from backend.storage.plan_db import get_plan_by_id, update_plan_tasks

    plan = request.plan
    if not plan and request.plan_id:
        plan = get_plan_by_id(request.plan_id) or _ACTIVE_PLANS.get(request.plan_id)

    if not plan:
        raise HTTPException(status_code=400, detail="Either plan_id or plan object must be provided.")

    recalibrated = recalibrate_entire_plan(plan, current_weather=request.weather)
    
    # If farmer_phone is attached to the plan, trigger dispatch logic
    farmer_phone = plan.get("farmer_phone")
    if farmer_phone:
        for alert in recalibrated.get("active_alerts", []):
            dispatch_escalation_alert(alert, farmer_phone=farmer_phone)

    if recalibrated.get("plan_id"):
        update_plan_tasks(recalibrated["plan_id"], recalibrated.get("tasks", []))
        _ACTIVE_PLANS[recalibrated["plan_id"]] = recalibrated
    return recalibrated


@router.post("/assistant/ask")
def ask_assistant(request: AssistantAskRequest) -> dict:
    from backend.rag.assistant import answer_farmer_query
    return answer_farmer_query(
        query=request.query,
        crop_name=request.crop_name,
        farmer_context=request.farmer_context,
        language=request.language,
    )
