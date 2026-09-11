"""
assistant.py
============
Voice-First Grounded RAG Assistant ("Ask SmartFarm"):
- Uses verified ICAR / TNAU POP knowledge corpus
- Protects against chemical dosage hallucinations
- Responds in English or Tamil
- Supports optional Groq (Llama-3.3-70B) or Google Gemini LLM with automatic fallback
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
import urllib.error
from typing import Any
from backend.rag.knowledge_corpus import retrieve_relevant_passages

logger = logging.getLogger(__name__)

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
            max_tokens=600,
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
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 600}
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
        "Explain clearly, practically, and respectfully. "
        + ("Respond in clear Tamil language." if is_ta else "Respond in clear, professional English.")
    )

    user_prompt = f"Verified Agricultural Context:\n{context_text}\n\n{farmer_notes}\n\nFarmer Question: {query}"

    llm_answer = _call_groq_llm(system_prompt, user_prompt)
    if not llm_answer:
        llm_answer = _call_gemini_llm(system_prompt, user_prompt)

    if not llm_answer:
        primary = passages[0]
        if is_ta:
            llm_answer = f"**{primary['title']}**\n\n{primary['text']}\n\n*(ஆதாரம்: {primary['source']})*"
        else:
            llm_answer = f"**{primary['title']}**\n\n{primary['text']}\n\n*(Source: {primary['source']})*"

    disclaimer = SAFETY_DISCLAIMER_TA if is_ta else SAFETY_DISCLAIMER_EN

    return {
        "answer": f"{llm_answer}\n\n{disclaimer}",
        "language": "ta" if is_ta else "en",
        "sources": [{"title": p["title"], "source": p["source"]} for p in passages],
        "disclaimer": disclaimer,
    }
