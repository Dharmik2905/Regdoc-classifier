import json
import os
import re
from functools import lru_cache
from typing import Dict, Any, List, Optional

from .pii_detection import find_pii
from .safety import naive_unsafe_check, profanity_pages, sensitive_equipment_pages
from .llm_client import call_openrouter_chat

# === Prompt library paths ===
PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "prompts")
PROMPT_CONFIG_PATH = os.path.join(PROMPTS_DIR, "prompt_config.json")

PRIMARY_MODEL = "meta-llama/llama-3.1-8b-instruct"
VALIDATOR_MODEL = "meta-llama/llama-3.1-70b-instruct"
VALIDATION_THRESHOLD = 0.6


@lru_cache(maxsize=1)
def load_prompt_config() -> Dict[str, Any]:
    try:
        with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"default": ["base_classification.txt"]}


def load_prompt_template(filename: str) -> str:
    path = os.path.join(PROMPTS_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_system_prompt(context_flags: Dict[str, bool]) -> str:
    cfg = load_prompt_config()
    if context_flags.get("unsafe_keyword_flag"):
        key = "unsafe"
    elif context_flags.get("has_ssn") or context_flags.get("has_pii"):
        key = "sensitive"
    else:
        key = "public"

    template_list = cfg.get(key, cfg.get("default", ["base_classification.txt"]))
    pieces: List[str] = []
    for fname in template_list:
        try:
            pieces.append(load_prompt_template(fname))
        except FileNotFoundError:
            continue
    if not pieces:
        pieces.append(load_prompt_template("base_classification.txt"))
    return "\n\n".join(pieces)


def run_llm_classification(payload: Dict[str, Any], model: str, system_prompt: str) -> Dict[str, Any]:
    user_prompt = json.dumps(payload, ensure_ascii=False)
    print(f"[LLM] Calling OpenRouter model={model}")

    llm_raw = call_openrouter_chat(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format_json=True,
        temperature=0.1,
    )

    category = llm_raw.get("category", "Public")
    unsafe = bool(llm_raw.get("unsafe", False))
    kid_safe = bool(llm_raw.get("kid_safe", not unsafe))
    try:
        confidence = float(llm_raw.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    confidence = max(0.0, min(1.0, confidence))
    reasoning = llm_raw.get("reasoning", "No reasoning provided.")
    citations = llm_raw.get("citations", []) or []

    return {
        "category": category,
        "unsafe": unsafe,
        "kid_safe": kid_safe,
        "confidence": confidence,
        "reasoning": reasoning,
        "citations": citations,
    }


# ---------------------------------------------------------------------------
# Helper: detect internal/template/equipment keywords (policy rules)
# ---------------------------------------------------------------------------
def detect_policy_keywords(text: str) -> Dict[str, bool]:
    low = text.lower()
    return {
        "internal": bool(re.search(r"\b(internal use|restricted circulation|proposal|confidential memo|research)\b", low)),
        "template": bool(re.search(r"\b(template|editable|shared)\b", low)),
        "equipment": bool(re.search(r"(fighter|aircraft|drone|missile|f-\d+|serial\s?(no\.|number|#))", low)),
    }


# ---------------------------------------------------------------------------
# Main classification orchestrator
# ---------------------------------------------------------------------------
def classify_document(doc_info: Dict[str, Any]) -> Dict[str, Any]:
    pages = doc_info["pages"]
    num_pages = doc_info["num_pages"]
    num_images = doc_info["num_images"]

    # === Heuristic extraction ===
    pii_raw = find_pii(pages)
    pii = [f for f in pii_raw if not (f.get("type") == "email" and f.get("is_business"))]

    unsafe_flag_heuristic = naive_unsafe_check(pages)
    prof_pages = profanity_pages(pages)
    has_ssn = any(f["type"] == "ssn" for f in pii)
    has_pii = bool(pii)
    equipment_pages = sensitive_equipment_pages(pages)
    has_sensitive_equipment = bool(equipment_pages) and num_images > 0

    # === Dynamic system prompt ===
    context_flags = {
        "unsafe_keyword_flag": unsafe_flag_heuristic,
        "has_ssn": has_ssn,
        "has_pii": has_pii,
    }
    system_prompt = build_system_prompt(context_flags)

    # === Summaries ===
    page_summaries: List[Dict[str, Any]] = []
    for p in pages:
        text = (p.get("text") or "").strip()
        if len(text) > 800:
            text = text[:800] + "..."
        page_summaries.append({"page": p["page_num"], "text": text})

    user_payload = {
        "num_pages": num_pages,
        "num_images": num_images,
        "pii_findings": pii,
        "unsafe_keyword_flag": unsafe_flag_heuristic,
        "profanity_pages": prof_pages,
        "page_summaries": page_summaries,
    }

    # === 1) Primary LLM ===
    primary = run_llm_classification(user_payload, PRIMARY_MODEL, system_prompt)

    # === 2) Optional validator ===
    validator: Optional[Dict[str, Any]] = None
    if primary["confidence"] < VALIDATION_THRESHOLD:
        validator = run_llm_classification(user_payload, VALIDATOR_MODEL, system_prompt)

    category = primary["category"]
    unsafe_flag_llm = primary["unsafe"]
    kid_safe = primary["kid_safe"]
    confidence = primary["confidence"]
    reasoning = primary["reasoning"]
    citations: List[Dict[str, Any]] = list(primary["citations"] or [])

    if validator is not None:
        disagreement = (
            validator["category"] != primary["category"] or validator["unsafe"] != primary["unsafe"]
        )
        if disagreement:
            print("[Validator] 70B disagreed with 8B.")
            category = validator["category"]
            unsafe_flag_llm = validator["unsafe"]
            kid_safe = validator["kid_safe"]
            confidence = min(primary["confidence"], validator["confidence"], 0.7)
            reasoning = (
                "Validator cross-check: 70B disagreed with 8B.\n"
                f"Primary said '{primary['category']}', validator said '{validator['category']}'.\n"
                "Validator chosen for higher precision.\n\n"
                f"Primary reasoning: {primary['reasoning']}\n\n"
                f"Validator reasoning: {validator['reasoning']}"
            )
            citations += (validator.get("citations") or [])

        # === Merge with deterministic / policy rules ===
    unsafe_flag = unsafe_flag_llm or unsafe_flag_heuristic

    # 1. Unsafe override
    if unsafe_flag and "Unsafe" not in category:
        category = f"{category} and Unsafe" if category else "Unsafe"

    # 2. SSN override
    if has_ssn and "Highly Sensitive" not in category:
        category = "Highly Sensitive" if "Unsafe" not in category else "Highly Sensitive and Unsafe"
        confidence = max(confidence, 0.9)

    # === Policy keyword detection (TC3–TC5) ===
    full_text = " ".join(p.get("text", "") for p in pages)
    policies = detect_policy_keywords(full_text)
    text_lower = full_text.lower()

    # --- marketing/public-safe guard ---
    is_marketing_or_public = any(
        kw in text_lower
        for kw in [
            "brochure", "marketing", "customer story", "press release",
            "product portfolio", "case study", "advertisement"
        ]
    )

    # --- 1️⃣ Internal / restricted (TC3) ---
    if policies["internal"]:
        category = "Confidential"
        confidence = max(confidence, 0.85)
        reasoning += (
            "\nPolicy rule: Detected internal or restricted-use document "
            "(e.g., memo/proposal). Classified as Confidential."
        )
        citations.append({"page": 1, "reason": "Detected internal or restricted-use wording"})

    # --- 2️⃣ Equipment (TC4) ---
    elif (has_sensitive_equipment or policies["equipment"]) and not is_marketing_or_public:
        category = "Confidential"
        confidence = max(confidence, 0.9)
        reasoning += (
            "\nPolicy rule: Contains identifiable aircraft or sensitive equipment serials. "
            "Classified as Confidential even if text appears public."
        )
        citations.append({"page": 1, "reason": "Detected aircraft/serial references"})

    # --- 3️⃣ Template / shared editable (TC5) ---
    elif policies["template"] and not is_marketing_or_public:
        if any(k in text_lower for k in ["flight", "manual", "operations", "safety"]):
            category = "Confidential"
            confidence = max(confidence, 0.85)
            reasoning += (
                "\nPolicy rule: Editable or shared operational template detected; "
                "classified as Confidential due to potential misuse."
            )
            citations.append({"page": 1, "reason": "Detected shared editable operational template"})

    # --- Equipment fallback from safety module
    elif has_sensitive_equipment:
        if "Confidential" not in category and "Highly Sensitive" not in category:
            category = "Confidential"
        confidence = max(confidence, 0.8)
        reasoning += (
            "\nPolicy rule: Detected diagrams of military aircraft or sensitive equipment; "
            "classified as Confidential even if surrounding text is public-facing."
        )

    # 4. Add PII / profanity citations
    for f in pii:
        citations.append({"page": f["page"], "reason": f"Detected {f['type'].upper()}: {f['value']}"})
    for pnum in prof_pages:
        citations.append({"page": pnum, "reason": "Strong profanity detected (not kid-safe)."})


    # === Kid-safe normalization ===
    kid_safe_final = (not unsafe_flag) and (not prof_pages)
    if has_sensitive_equipment and kid_safe_final:
        kid_safe_final = True

    # === Optional note for public business contact info ===
    if not unsafe_flag and any(f.get("is_business") for f in pii_raw):
        reasoning += "\nNote: Detected only public business contact info — does not increase sensitivity."

    return {
        "category": category,
        "unsafe": unsafe_flag,
        "kid_safe": kid_safe_final,
        "confidence": confidence,
        "reasoning": reasoning.strip(),
        "citations": citations,
    }
