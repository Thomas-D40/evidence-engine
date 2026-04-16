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
    # 3 pros, 1 con → ratio = 0.75 → Strong consensus
    pros = [{}] * 3
    cons = [{}] * 1
    result = compute_consensus(pros, cons)
    assert result["ratio"] == 0.75
    assert result["label"] == LABEL_STRONG


def test_moderate_consensus_threshold():
    # 2 pros, 2 cons → ratio = 0.5 → between 0.35 and 0.55 → Contested
    pros = [{}] * 2
    cons = [{}] * 2
    result = compute_consensus(pros, cons)
    assert result["ratio"] == 0.5
    assert result["label"] == LABEL_CONTESTED


def test_moderate_consensus_label():
    # 3 pros, 2 cons → ratio = 0.6 → Moderate consensus
    pros = [{}] * 3
    cons = [{}] * 2
    result = compute_consensus(pros, cons)
    assert result["ratio"] == 0.6
    assert result["label"] == LABEL_MODERATE


def test_ratio_rounded_to_three_decimals():
    # 1 pros, 3 cons → ratio = 0.25
    result = compute_consensus([{}], [{}, {}, {}])
    assert result["ratio"] == 0.25


def test_consensus_ratio_and_label_keys_both_present():
    """Both key naming conventions are available for compatibility."""
    result = compute_consensus([{}], [{}])
    assert "ratio" in result
    assert "label" in result
    assert "consensus_ratio" in result
    assert "consensus_label" in result
    assert result["ratio"] == result["consensus_ratio"]
    assert result["label"] == result["consensus_label"]
