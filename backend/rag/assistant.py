"""
assistant.py
============
Voice-First Grounded RAG Assistant ("Ask SmartFarm"):
- Uses verified ICAR / TNAU POP knowledge corpus
- Protects against chemical dosage hallucinations
- Responds in English or Tamil
- Primary LLM: OpenAI API (model: gpt-5.6-luna) with rate-limiting & quota protection
- Fallbacks: Groq (Llama-3.3-70B), Google Gemini, or direct verified passage
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
import urllib.error
from typing import Any
from backend.rag.knowledge_corpus import retrieve_relevant_passages

logger = logging.getLogger(__name__)

# Global rate limiting & LRU response cache for Free Tier Quota Protection
_last_openai_call_time: float = 0.0
_llm_response_cache: dict[str, str] = {}


def _load_dotenv() -> None:
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'").strip('"')
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_dotenv()

SAFETY_DISCLAIMER_EN = (
    "Important: Confirm any pesticide or fertilizer dose with the current product label "
    "and your local Krishi Vigyan Kendra (KVK) / Agriculture Extension Officer before applying."
)

SAFETY_DISCLAIMER_TA = (
    "முக்கிய குறிப்பு: பூச்சிக்கொல்லி அல்லது உர அளவை உங்கள் பகுதி வேளாண் அலுவலரிடம் (KVK) "
    "உறுதிப்படுத்திய பின் மட்டுமே பயன்படுத்தவும்."
)


def _is_tamil(text: str) -> bool:
    for char in text:
        if 0x0B80 <= ord(char) <= 0x0BFF:
            return True
    return False


def _call_openai_llm(system_prompt: str, user_prompt: str) -> str | None:
    """Primary OpenAI LLM engine using gpt-5.6-luna with strict rate limiting & quota protection."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    global _last_openai_call_time, _llm_response_cache

    # 1. Rate Limiting Guardrail (1.2s delay between calls for free-tier quota protection)
    now = time.time()
    time_since_last = now - _last_openai_call_time
    if time_since_last < 1.2:
        time.sleep(1.2 - time_since_last)
    _last_openai_call_time = time.time()

    # 2. In-memory caching for identical queries
    cache_key = f"{user_prompt.strip().lower()}:{system_prompt[-30:]}"
    if cache_key in _llm_response_cache:
        return _llm_response_cache[cache_key]

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-5.6-luna",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        content = response.choices[0].message.content
        if content:
            _llm_response_cache[cache_key] = content
            # Limit cache size to 100 entries
            if len(_llm_response_cache) > 100:
                _llm_response_cache.pop(next(iter(_llm_response_cache)))
        return content
    except Exception as exc:
        logger.debug("OpenAI model gpt-5.6-luna call failed: %s", exc)
        return None


def _call_groq_llm(system_prompt: str, user_prompt: str) -> str | None:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.debug("Groq LLM call failed: %s", exc)
        return None


def _call_gemini_llm(system_prompt: str, user_prompt: str) -> str | None:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    models_to_try = ["gemini-3.6-flash", "gemini-1.5-flash-latest", "gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    prompt_combined = system_prompt + "\n\n" + user_prompt
    payload = {
        "contents": [{"parts": [{"text": prompt_combined}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 500}
    }

    for model_name in models_to_try:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as exc:
            logger.debug("Gemini model %s failed: %s", model_name, exc)
            continue
    return None


def answer_farmer_query(
    query: str,
    crop_name: str | None = None,
    farmer_context: dict[str, Any] | None = None,
    language: str | None = None,
) -> dict[str, Any]:
    _load_dotenv()
    if not query or not query.strip():
        return {
            "answer": "Please ask a question about your crop, fertilizers, irrigation, or pests.",
            "language": "en",
            "sources": [],
            "disclaimer": SAFETY_DISCLAIMER_EN,
        }

    is_ta = language == "ta" or _is_tamil(query)
    passages = retrieve_relevant_passages(query, crop_name=crop_name, top_k=3)

    if not passages:
        fallback_msg = (
            "மன்னிக்கவும், உங்கள் கேள்விக்கான துல்லியமான வேளாண் வழிகாட்டுதல் தரவுத்தளத்தில் கிடைக்கவில்லை. தயவுசெய்து உங்கள் வட்டார வேளாண் அலுவலரை அணுகவும்."
            if is_ta
            else "I could not find a verified advisory for this specific question in the Package of Practices database. Please verify with your local agricultural extension officer."
        )
        return {
            "answer": fallback_msg,
            "language": "ta" if is_ta else "en",
            "sources": [],
            "disclaimer": SAFETY_DISCLAIMER_TA if is_ta else SAFETY_DISCLAIMER_EN,
        }

    context_parts = []
    for p in passages:
        context_parts.append(f"[{p['title']} ({p['source']})]\n{p['text']}")
    context_text = "\n\n".join(context_parts)

    farmer_notes = ""
    if farmer_context:
        farmer_notes = (
            f"Farmer Context: Committed Crop = {farmer_context.get('crop_name', crop_name)}, "
            f"Area = {farmer_context.get('area_acres', '1')} acres, "
            f"Water Source = {farmer_context.get('irrigation_source', 'Borewell')}."
        )

    system_prompt = (
        "You are 'SmartFarm Assistant', an agronomic advisory system for Indian farmers. "
        "Strictly answer the farmer's question using ONLY the provided verified agricultural context below. "
        "Do NOT hallucinate chemical dosages or invent unverified practices. "
        "CRITICAL FORMAT REQUIREMENT: Always begin your response with a clear 1-sentence direct summary answer on the very first line! "
        "Then follow with short bullet points for details if needed. "
        "Do NOT add redundant disclaimers or manual source footnotes at the end, as the system appends citations automatically. "
        + ("Respond in clear Tamil language." if is_ta else "Respond in clear, professional English.")
    )

    user_prompt = f"Verified Agricultural Context:\n{context_text}\n\n{farmer_notes}\n\nFarmer Question: {query}"

    # 1. Primary Model: OpenAI gpt-5.6-luna (with Rate Limiting & Quota Control)
    llm_answer = _call_openai_llm(system_prompt, user_prompt)

    # 2. Secondary Fallback: Groq (Llama-3.3-70B)
    if not llm_answer:
        llm_answer = _call_groq_llm(system_prompt, user_prompt)

    # 3. Tertiary Fallback: Google Gemini
    if not llm_answer:
        llm_answer = _call_gemini_llm(system_prompt, user_prompt)

    # 4. Quaternary Grounded Passage Fallback
    if not llm_answer:
        primary = passages[0]
        llm_answer = f"**{primary['title']}**\n\n{primary['text']}"

    disclaimer = SAFETY_DISCLAIMER_TA if is_ta else SAFETY_DISCLAIMER_EN

    return {
        "answer": llm_answer,
        "language": "ta" if is_ta else "en",
        "sources": [{"title": p["title"], "source": p["source"]} for p in passages],
        "disclaimer": disclaimer,
    }
