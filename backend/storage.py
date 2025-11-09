# backend/storage.py
import json
import os
from datetime import datetime
from typing import Any, Dict, List

# history.json in project root
HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "history.json")


def load_history() -> List[Dict[str, Any]]:
    """Load the full audit trail from disk."""
    path = os.path.abspath(HISTORY_PATH)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # if file is corrupted, start fresh
        return []


def save_result(
    filename: str,
    doc_info: Dict[str, Any],
    ai_result: Dict[str, Any],
    final_category: str,
    reviewer_comment: str = "",
) -> None:
    """Append a single classification + review entry to the audit log."""
    history = load_history()

    entry = {
        "filename": filename,
        "pages": doc_info.get("num_pages"),
        "images": doc_info.get("num_images"),
        "ai_category": ai_result.get("category"),
        "final_category": final_category,
        "unsafe": ai_result.get("unsafe"),
        "kid_safe": ai_result.get("kid_safe"),
        "confidence": ai_result.get("confidence"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "reviewer_comment": reviewer_comment,
    }

    history.append(entry)

    path = os.path.abspath(HISTORY_PATH)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
