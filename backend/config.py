# backend/config.py

CATEGORIES = ["Public", "Confidential", "Highly Sensitive", "Unsafe"]

# Threshold above which we skip HITL in theory (for now just metadata)
CONFIDENCE_HITL_THRESHOLD = 0.85

# Placeholder for model name / provider
LLM_MODEL_NAME = "llama-vision-or-text-model"
