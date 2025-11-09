# backend/llm_client.py
import os
import json
import requests
from typing import Dict, Any

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    response_format_json: bool = True,
    temperature: float = 0.1,
) -> Dict[str, Any] | str:
    """
    Simple helper to call OpenRouter's chat/completions endpoint.
    Uses the OPENROUTER_API_KEY environment variable.
    """
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY environment variable is not set")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "RegDoc Classifier",
    }

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }

    if response_format_json:
        payload["response_format"] = {"type": "json_object"}

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=90)
    resp.raise_for_status()
    data = resp.json()

    content = data["choices"][0]["message"]["content"]

    if response_format_json:
        return json.loads(content)
    return content
