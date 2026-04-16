"""
Consensus indicator computation.

Pure Python — zero LLM calls.
Computes consensus_ratio and consensus_label from pros/cons evidence lists.
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

LABEL_STRONG      = "Strong consensus"
LABEL_MODERATE    = "Moderate consensus"
LABEL_CONTESTED   = "Contested"
LABEL_MINORITY    = "Minority position"
LABEL_INSUFFICIENT = "Insufficient data"

# ============================================================================
# LOGIC
# ============================================================================


def compute_consensus(pros: List[Any], cons: List[Any]) -> Dict[str, Optional[Any]]:
    """
    Compute consensus ratio and label from pros/cons evidence lists.

    No LLM calls — deterministic post-processing only.

    consensus_ratio = len(pros) / (len(pros) + len(cons))

    Args:
        pros: List of supporting evidence items
        cons: List of contradicting evidence items

    Returns:
        {"ratio": float | None, "label": str}
        or {"consensus_ratio": float | None, "consensus_label": str}
        (both keys provided for compatibility)
    """
    total = len(pros) + len(cons)

    if total == 0:
        return {
            "ratio": None,
            "label": LABEL_INSUFFICIENT,
            "consensus_ratio": None,
            "consensus_label": LABEL_INSUFFICIENT,
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
        "consensus_ratio": rounded,
        "consensus_label": label,
    }
