"""
Consensus indicator computation.

Pure Python — zero LLM calls.
Computes evidence_balance_ratio and evidence_balance_label from pros/cons evidence lists.
"""
from typing import Dict, Any, List, Optional

from app.constants.analysis import (
    CONSENSUS_STRONG_THRESHOLD,
    CONSENSUS_MODERATE_THRESHOLD,
    CONSENSUS_CONTESTED_THRESHOLD,
)

# ============================================================================
# LABELS
# ============================================================================

LABEL_STRONG       = "Mostly supporting evidence found"
LABEL_MODERATE     = "More supporting than contradicting evidence found"
LABEL_CONTESTED    = "Mixed evidence found"
LABEL_MINORITY     = "Mostly contradicting evidence found"
LABEL_INSUFFICIENT = "Insufficient sources found"

# ============================================================================
# LOGIC
# ============================================================================


def compute_consensus(pros: List[Any], cons: List[Any]) -> Dict[str, Optional[Any]]:
    """
    Compute evidence balance ratio and label from pros/cons evidence lists.

    No LLM calls — deterministic post-processing only.

    evidence_balance_ratio = len(pros) / (len(pros) + len(cons))

    Args:
        pros: List of supporting evidence items
        cons: List of contradicting evidence items

    Returns:
        {"ratio": float | None, "label": str,
         "evidence_balance_ratio": float | None, "evidence_balance_label": str}
    """
    total = len(pros) + len(cons)

    if total == 0:
        return {
            "ratio": None,
            "label": LABEL_INSUFFICIENT,
            "evidence_balance_ratio": None,
            "evidence_balance_label": LABEL_INSUFFICIENT,
        }

    ratio = len(pros) / total

    if ratio >= CONSENSUS_STRONG_THRESHOLD:
        label = LABEL_STRONG
    elif ratio >= CONSENSUS_MODERATE_THRESHOLD:
        label = LABEL_MODERATE
    elif ratio >= CONSENSUS_CONTESTED_THRESHOLD:
        label = LABEL_CONTESTED
    else:
        label = LABEL_MINORITY

    rounded = round(ratio, 3)
    return {
        "ratio": rounded,
        "label": label,
        "evidence_balance_ratio": rounded,
        "evidence_balance_label": label,
    }
