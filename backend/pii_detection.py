# backend/pii_detection.py
import re
from typing import List, Dict, Any

SSN_REGEX = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
EMAIL_REGEX = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_REGEX = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\d{3}[-.\s]?){2}\d{4}\b")


def find_pii(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    for page in pages:
        pnum = page["page_num"]
        text = page["text"] or ""

        for m in SSN_REGEX.finditer(text):
            findings.append(
                {
                    "type": "ssn",
                    "page": pnum,
                    "value": m.group(0),
                    "snippet": text[max(0, m.start() - 20) : m.end() + 20],
                }
            )

        for m in EMAIL_REGEX.finditer(text):
            findings.append(
                {
                    "type": "email",
                    "page": pnum,
                    "value": m.group(0),
                    "snippet": text[max(0, m.start() - 20) : m.end() + 20],
                }
            )

        for m in PHONE_REGEX.finditer(text):
            findings.append(
                {
                    "type": "phone",
                    "page": pnum,
                    "value": m.group(0),
                    "snippet": text[max(0, m.start() - 20) : m.end() + 20],
                }
            )

    return findings
