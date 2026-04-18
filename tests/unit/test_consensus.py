"""
Unit tests for consensus.py — pure Python, no LLM, no mocking needed.
"""
import pytest
from app.agents.analysis.consensus import (
    compute_consensus,
    LABEL_STRONG,
    LABEL_MODERATE,
    LABEL_CONTESTED,
    LABEL_MINORITY,
    LABEL_INSUFFICIENT,
)


def test_empty_pros_and_cons_returns_insufficient():
    result = compute_consensus([], [])
    assert result["ratio"] is None
    assert result["label"] == LABEL_INSUFFICIENT


def test_all_pros_returns_strong_consensus():
    pros = [{"claim": "c", "source": "s"}] * 5
    result = compute_consensus(pros, [])
    assert result["ratio"] == 1.0
    assert result["label"] == LABEL_STRONG


def test_all_cons_returns_minority_position():
    cons = [{"claim": "c", "source": "s"}] * 5
    result = compute_consensus([], cons)
    assert result["ratio"] == 0.0
    assert result["label"] == LABEL_MINORITY


def test_strong_consensus_threshold():
    # 3 pros, 1 con → ratio = 0.75 → Mostly supporting
    pros = [{}] * 3
    cons = [{}] * 1
    result = compute_consensus(pros, cons)
    assert result["ratio"] == 0.75
    assert result["label"] == LABEL_STRONG


def test_moderate_consensus_threshold():
    # 2 pros, 2 cons → ratio = 0.5 → between 0.35 and 0.55 → Mixed
    pros = [{}] * 2
    cons = [{}] * 2
    result = compute_consensus(pros, cons)
    assert result["ratio"] == 0.5
    assert result["label"] == LABEL_CONTESTED


def test_moderate_consensus_label():
    # 3 pros, 2 cons → ratio = 0.6 → More supporting than contradicting
    pros = [{}] * 3
    cons = [{}] * 2
    result = compute_consensus(pros, cons)
    assert result["ratio"] == 0.6
    assert result["label"] == LABEL_MODERATE


def test_ratio_rounded_to_three_decimals():
    # 1 pros, 3 cons → ratio = 0.25
    result = compute_consensus([{}], [{}, {}, {}])
    assert result["ratio"] == 0.25


def test_evidence_balance_keys_present():
    """Both ratio/label and evidence_balance_ratio/evidence_balance_label keys are available."""
    result = compute_consensus([{}], [{}])
    assert "ratio" in result
    assert "label" in result
    assert "evidence_balance_ratio" in result
    assert "evidence_balance_label" in result
    assert result["ratio"] == result["evidence_balance_ratio"]
    assert result["label"] == result["evidence_balance_label"]


def test_label_values_are_descriptive_not_scientific():
    """Labels describe evidence found, not scientific consensus claims."""
    assert LABEL_STRONG      == "Mostly supporting evidence found"
    assert LABEL_MODERATE    == "More supporting than contradicting evidence found"
    assert LABEL_CONTESTED   == "Mixed evidence found"
    assert LABEL_MINORITY    == "Mostly contradicting evidence found"
    assert LABEL_INSUFFICIENT == "Insufficient sources found"
