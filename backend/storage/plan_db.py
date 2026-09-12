"""
plan_db.py
==========
Persistent SQLite storage for SmartFarm Farmer Accounts, Auth & Sowing-to-Harvest Plans:
- Lifelong persistence of farmer profiles (Phone + OTP, Google Auth)
- Per-farmer sowing plans, tasks, and task completion state
- Zero external dependencies (uses standard python sqlite3)
"""
from __future__ import annotations

import json
import sqlite3
import os
import random
from datetime import datetime, timedelta
from typing import Any

DB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "smartfarm_plans.db")
)


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()
        
        # 1. Farmers / Users Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                phone_number TEXT PRIMARY KEY,
                full_name TEXT,
                email TEXT,
                auth_provider TEXT DEFAULT 'phone_otp',
                created_at TEXT NOT NULL
            )
        """)

        # 2. Temporary OTPs Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS otps (
                phone_number TEXT PRIMARY KEY,
                otp_code TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)

        # 3. Plans Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plans (
                plan_id TEXT PRIMARY KEY,
                farmer_phone TEXT,
                crop_name TEXT NOT NULL,
                area_acres REAL NOT NULL,
                sowing_date TEXT NOT NULL,
                expected_harvest_date TEXT NOT NULL,
                duration_days INTEGER NOT NULL,
                farmer_budget_inr REAL NOT NULL,
                irrigation_source TEXT,
                budget_analysis_json TEXT,
                plan_status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                FOREIGN KEY (farmer_phone) REFERENCES users(phone_number)
            )
        """)

        # Migration: Check if farmer_phone column exists in plans table
        cursor.execute("PRAGMA table_info(plans)")
        columns = [col[1] for col in cursor.fetchall()]
        if "farmer_phone" not in columns:
            cursor.execute("ALTER TABLE plans ADD COLUMN farmer_phone TEXT")

        # 4. Plan Tasks Table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS plan_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                day_offset INTEGER NOT NULL,
                due_date TEXT NOT NULL,
                task_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT,
                estimated_cost_inr REAL DEFAULT 0,
                is_critical INTEGER DEFAULT 0,
                is_completed INTEGER DEFAULT 0,
                completed_at TEXT,
                escalation_tier INTEGER DEFAULT 0,
                escalation_label TEXT,
                escalation_channel TEXT,
                alert_message TEXT,
                recalibration_note TEXT,
                UNIQUE(plan_id, task_id),
                FOREIGN KEY (plan_id) REFERENCES plans (plan_id) ON DELETE CASCADE
            )
        """)
        conn.commit()


# ── Farmer Authentication Storage Functions ─────────────────────────────────

def save_or_update_user(phone_number: str, full_name: str = "Farmer", email: str = "", auth_provider: str = "phone_otp") -> dict[str, Any]:
    init_db()
    phone_clean = phone_number.strip().replace(" ", "").replace("-", "")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (phone_number, full_name, email, auth_provider, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(phone_number) DO UPDATE SET
                full_name = COALESCE(NULLIF(excluded.full_name, ''), users.full_name),
                email = COALESCE(NULLIF(excluded.email, ''), users.email),
                auth_provider = excluded.auth_provider
        """, (phone_clean, full_name, email, auth_provider, datetime.now().isoformat()))
        conn.commit()

    return get_user_by_phone(phone_clean)


def get_user_by_phone(phone_number: str) -> dict[str, Any] | None:
    init_db()
    phone_clean = phone_number.strip().replace(" ", "").replace("-", "")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE phone_number = ?", (phone_clean,))
        row = cursor.fetchone()
        return dict(row) if row else None


def generate_otp_for_phone(phone_number: str) -> str:
    init_db()
    phone_clean = phone_number.strip().replace(" ", "").replace("-", "")
    # Standard 4-digit OTP
    otp_code = f"{random.randint(1000, 9999)}"
    expires_at = (datetime.now() + timedelta(minutes=10)).isoformat()

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO otps (phone_number, otp_code, expires_at)
            VALUES (?, ?, ?)
            ON CONFLICT(phone_number) DO UPDATE SET
                otp_code = excluded.otp_code,
                expires_at = excluded.expires_at
        """, (phone_clean, otp_code, expires_at))
        conn.commit()

    return otp_code


def verify_otp_for_phone(phone_number: str, otp_code: str) -> bool:
    init_db()
    phone_clean = phone_number.strip().replace(" ", "").replace("-", "")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM otps WHERE phone_number = ?", (phone_clean,))
        row = cursor.fetchone()
        if not row:
            return False

        stored_otp = row["otp_code"]
        expires_at = datetime.fromisoformat(row["expires_at"])
        
        if datetime.now() > expires_at:
            return False

        # Allow 4-digit code match or master demo code "1234"
        if otp_code.strip() == stored_otp or otp_code.strip() == "1234":
            cursor.execute("DELETE FROM otps WHERE phone_number = ?", (phone_clean,))
            conn.commit()
            return True

    return False


# ── Plan Storage Functions ──────────────────────────────────────────────────

def save_plan(plan: dict[str, Any], farmer_phone: str | None = None) -> None:
    init_db()
    phone = farmer_phone or plan.get("farmer_phone", "")
    if phone:
        phone = phone.strip().replace(" ", "").replace("-", "")

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO plans (
                plan_id, farmer_phone, crop_name, area_acres, sowing_date, expected_harvest_date,
                duration_days, farmer_budget_inr, irrigation_source,
                budget_analysis_json, plan_status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            plan["plan_id"],
            phone,
            plan.get("crop_name", "Crop"),
            float(plan.get("area_acres", 1.0)),
            plan.get("sowing_date", datetime.now().strftime("%Y-%m-%d")),
            plan.get("expected_harvest_date", ""),
            int(plan.get("duration_days", 90)),
            float(plan.get("farmer_budget_inr", 0.0)),
            plan.get("irrigation_source", "Borewell"),
            json.dumps(plan.get("budget_analysis", {})),
            plan.get("plan_status", "active"),
            plan.get("created_at", datetime.now().isoformat()),
        ))

        for t in plan.get("tasks", []):
            cursor.execute("""
                INSERT OR REPLACE INTO plan_tasks (
                    plan_id, task_id, day_offset, due_date, task_type,
                    title, description, estimated_cost_inr, is_critical,
                    is_completed, completed_at, escalation_tier,
                    escalation_label, escalation_channel, alert_message, recalibration_note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                plan["plan_id"],
                t["task_id"],
                int(t.get("day_offset", 0)),
                t["due_date"],
                t.get("task_type", "task"),
                t["title"],
                t.get("description", ""),
                float(t.get("estimated_cost_inr", 0.0)),
                1 if t.get("is_critical") else 0,
                1 if t.get("is_completed") else 0,
                t.get("completed_at"),
                int(t.get("escalation_tier", 0)),
                t.get("escalation_label"),
                t.get("escalation_channel"),
                t.get("alert_message"),
                t.get("recalibration_note"),
            ))
        conn.commit()


def update_plan_tasks(plan_id: str, tasks: list[dict[str, Any]]) -> None:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        for t in tasks:
            cursor.execute("""
                UPDATE plan_tasks
                SET is_completed = ?, completed_at = ?,
                    escalation_tier = ?, escalation_label = ?,
                    escalation_channel = ?, alert_message = ?, recalibration_note = ?
                WHERE plan_id = ? AND task_id = ?
            """, (
                1 if t.get("is_completed") else 0,
                t.get("completed_at"),
                int(t.get("escalation_tier", 0)),
                t.get("escalation_label"),
                t.get("escalation_channel"),
                t.get("alert_message"),
                t.get("recalibration_note"),
                plan_id,
                t["task_id"]
            ))
        conn.commit()


def get_plan_by_id(plan_id: str) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM plans WHERE plan_id = ?", (plan_id,))
        row = cursor.fetchone()
        if not row:
            return None

        plan = dict(row)
        if plan.get("budget_analysis_json"):
            plan["budget_analysis"] = json.loads(plan["budget_analysis_json"])
            del plan["budget_analysis_json"]

        cursor.execute("SELECT * FROM plan_tasks WHERE plan_id = ? ORDER BY day_offset ASC", (plan_id,))
        task_rows = cursor.fetchall()
        tasks = []
        for tr in task_rows:
            td = dict(tr)
            td["is_critical"] = bool(td["is_critical"])
            td["is_completed"] = bool(td["is_completed"])
            del td["id"]
            tasks.append(td)

        plan["tasks"] = tasks
        plan["total_tasks_count"] = len(tasks)
        plan["completed_tasks_count"] = sum(1 for t in tasks if t["is_completed"])
        return plan


def update_task_completion(plan_id: str, task_id: str, is_completed: bool, completed_at: str | None = None) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE plan_tasks
            SET is_completed = ?, completed_at = ?
            WHERE plan_id = ? AND task_id = ?
        """, (1 if is_completed else 0, completed_at if is_completed else None, plan_id, task_id))
        conn.commit()

    return get_plan_by_id(plan_id)


def get_plans_by_farmer_phone(phone_number: str) -> list[dict[str, Any]]:
    init_db()
    phone_clean = phone_number.strip().replace(" ", "").replace("-", "")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT plan_id FROM plans WHERE farmer_phone = ? ORDER BY created_at DESC", (phone_clean,))
        rows = cursor.fetchall()

    plans = []
    for r in rows:
        p = get_plan_by_id(r["plan_id"])
        if p:
            plans.append(p)
    return plans


def get_all_active_plans() -> list[dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT plan_id FROM plans WHERE plan_status = 'active' ORDER BY created_at DESC")
        rows = cursor.fetchall()
    
    plans = []
    for r in rows:
        p = get_plan_by_id(r["plan_id"])
        if p:
            plans.append(p)
    return plans
