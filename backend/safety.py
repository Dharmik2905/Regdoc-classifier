# backend/safety.py

from typing import List, Dict, Any

# Only very explicit, truly harmful phrases for "unsafe"
UNSAFE_KEYWORDS = [
    "child porn",
    "child pornography",
    "csam",
    "kill myself",
    "kill him",
    "kill her",
    "kill them",
    "shoot up",
    "school shooting",
    "mass shooting",
    "bomb recipe",
    "how to make a bomb",
    "how to build a bomb",
    "suicide tutorial",
    "suicide instructions",
    "rape",
    "lynch",
    "hang yourself",
    "behead",
]

# Strong profanity that should make content "not kid-safe"
PROFANITY_WORDS = [
    "fuck",
    "fucking",
    "fucked",
    "motherfucker",
    "shit",
    "bullshit",
    "bitch",
    "bitches",
    "asshole",
    "dickhead",
    "bastard",
    "Cunt"
]


def naive_unsafe_check(pages: List[Dict[str, Any]]) -> bool:
    """
    Return True only for clearly dangerous/illegal content,
    NOT just swearing. This feeds the 'unsafe' flag and
    can upgrade the category to include 'Unsafe'.
    """
    text = " ".join((p.get("text") or "") for p in pages).lower()
    return any(kw in text for kw in UNSAFE_KEYWORDS)


def profanity_pages(pages: List[Dict[str, Any]]) -> List[int]:
    """
    Return list of page numbers that contain strong profanity.
    We use this to mark kid_safe = False, but we do NOT automatically
    mark the document as 'Unsafe' just because of profanity.
    """
    prof_pages: List[int] = []
    for p in pages:
        text = (p.get("text") or "").lower()
        if any(word in text for word in PROFANITY_WORDS):
            prof_pages.append(p["page_num"])
    return prof_pages
