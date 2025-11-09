# backend/classification.py

import json
import os
from typing import Dict, Any, List, Optional

from .pii_detection import find_pii
from .safety import naive_unsafe_check
from .llm_client import call_openrouter_chat

# Path to the base prompt in backend/prompts/base_classification.txt
PROMPT_PATH = os.path.join(
    os.path.dirname(__file__),
    "prompts",
    "base_classification.txt",
)

# OpenRouter models
PRIMARY_MODEL = "meta-llama/llama-3.1-8b-instruct"
VALIDATOR_MODEL = "meta-llama/llama-3.1-70b-instruct"  # used only when confidence is low

# If primary confidence is below this, run validator
VALIDATION_THRESHOLD = 0.6


def load_base_prompt() -> str:
    """Load the base classification prompt from disk."""
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read()


def run_llm_classification(payload: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
    Single LLM call to a given model on OpenRouter.
    Expects JSON-only response with category, unsafe, kid_safe, etc.
    """
    system_prompt = load_base_prompt()
    user_prompt = json.dumps(payload, ensure_ascii=False)

    print(f"[LLM] Calling OpenRouter model={model}")  # debug in terminal

    llm_raw = call_openrouter_chat(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        response_format_json=True,
        temperature=0.1,
    )

    # Normalize and provide safe defaults
    category = llm_raw.get("category", "Public")
    unsafe = bool(llm_raw.get("unsafe", False))
    kid_safe = bool(llm_raw.get("kid_safe", not unsafe))
    try:
        confidence = float(llm_raw.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    # clamp confidence between 0 and 1
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


def classify_document(doc_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main orchestration:
    - Extract features (PII, unsafe heuristics, page summaries)
    - Call primary LLaMA 8B
    - Optionally call validator LLaMA 70B on low confidence
    - Merge LLM outputs with hard rules (SSN, unsafe keywords)
    - Return final classification + reasoning + citations
    """
    pages = doc_info["pages"]
    num_pages = doc_info["num_pages"]
    num_images = doc_info["num_images"]

    # Heuristic feature extraction
    pii = find_pii(pages)
    unsafe_flag_heuristic = naive_unsafe_check(pages)
    has_ssn = any(f["type"] == "ssn" for f in pii)

    # Compact page summaries (truncate to keep token usage sane)
    page_summaries: List[Dict[str, Any]] = []
    for p in pages:
        text = (p.get("text") or "").strip()
        if len(text) > 800:
            text = text[:800] + "..."
        page_summaries.append(
            {
                "page": p["page_num"],
                "text": text,
            }
        )

    # Payload sent to the LLM(s)
    user_payload: Dict[str, Any] = {
        "num_pages": num_pages,
        "num_images": num_images,
        "pii_findings": pii,                 # regex-based PII hints
        "unsafe_keyword_flag": unsafe_flag_heuristic,
        "page_summaries": page_summaries,    # per-page text excerpts
    }

    # 1) Primary: LLaMA 3.1 8B
    primary = run_llm_classification(user_payload, PRIMARY_MODEL)

    # 2) Optional validator: LLaMA 3.1 70B (only when low confidence)
    validator: Optional[Dict[str, Any]] = None
    if primary["confidence"] < VALIDATION_THRESHOLD:
        validator = run_llm_classification(user_payload, VALIDATOR_MODEL)

    # Start with primary outputs
    category = primary["category"]
    unsafe_flag_llm = primary["unsafe"]
    kid_safe = primary["kid_safe"]
    confidence = primary["confidence"]
    reasoning = primary["reasoning"]
    citations: List[Dict[str, Any]] = list(primary["citations"] or [])

    # If validator exists and disagrees, prefer its output and explain why
    if validator is not None:
        validator_cat = validator["category"]
        validator_unsafe = validator["unsafe"]

        disagreement = (
            validator_cat != primary["category"]
            or validator_unsafe != primary["unsafe"]
        )

        if disagreement:
            print("[Validator] 70B validator disagreed with 8B primary.")

            category = validator_cat
            unsafe_flag_llm = validator_unsafe
            kid_safe = validator["kid_safe"]

            # Confidence is conservative: min of both, capped at 0.7
            confidence = min(primary["confidence"], validator["confidence"], 0.7)

            reasoning = (
                "Cross-validation triggered: the 70B validator model disagreed with "
                "the 8B primary model. "
                f"Primary model proposed '{primary['category']}' "
                f"(confidence {primary['confidence']*100:.0f}%), "
                f"validator proposed '{validator['category']}' "
                f"(confidence {validator['confidence']*100:.0f}%). "
                "The final category favors the validator for higher precision.\n\n"
                f"Primary reasoning: {primary['reasoning']}\n\n"
                f"Validator reasoning: {validator['reasoning']}"
            )

            # Merge citations from both models
            citations = (primary.get("citations") or []) + (validator.get("citations") or [])

    # 3) Merge with deterministic rules (heuristics)

    # Unsafe: LLM OR heuristic keywords
    unsafe_flag = unsafe_flag_llm or unsafe_flag_heuristic

    # Ensure "Unsafe" is reflected in the category if the flag is true
    if unsafe_flag and "Unsafe" not in category:
        if category and category != "Unsafe":
            category = f"{category} and Unsafe"
        else:
            category = "Unsafe"

    # Hard rule: SSN means Highly Sensitive (possibly plus Unsafe)
    if has_ssn and "Highly Sensitive" not in category:
        if "Unsafe" in category:
            category = "Highly Sensitive and Unsafe"
        else:
            category = "Highly Sensitive"
        # SSN-based classification should be high confidence
        confidence = max(confidence, 0.9)

    # Always add PII-based citations for audit
    for f in pii:
        citations.append(
            {
                "page": f["page"],
                "reason": f"Detected {f['type'].upper()}: {f['value']}",
            }
        )

    # Normalize kid_safe: if unsafe, not kid-safe
    kid_safe_final = kid_safe and not unsafe_flag

    # Final payload consumed by the UI
    return {
        "category": category,
        "unsafe": unsafe_flag,
        "kid_safe": kid_safe_final,
        "confidence": confidence,
        "reasoning": reasoning,
        "citations": citations,
    }
