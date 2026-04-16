"""
Unit tests for relevance_filter.py — no external calls.
"""
import pytest
from app.utils.relevance_filter import (
    extract_keywords,
    calculate_relevance_score,
    filter_relevant_results,
)


class TestExtractKeywords:
    def test_extracts_basic_words(self):
        kw = extract_keywords("coffee cancer risk")
        assert "coffee" in kw
        assert "cancer" in kw
        assert "risk" in kw

    def test_filters_french_stop_words(self):
        kw = extract_keywords("le café réduit les risques")
        assert "le" not in kw
        assert "les" not in kw

    def test_filters_short_words(self):
        kw = extract_keywords("a be do coffee", min_length=3)
        assert "a" not in kw
        assert "be" not in kw
        assert "coffee" in kw

    def test_empty_string_returns_empty_set(self):
        assert extract_keywords("") == set()


class TestCalculateRelevanceScore:
    def test_identical_texts_score_one(self):
        score = calculate_relevance_score("coffee cancer risk", "coffee cancer risk")
        assert score == 1.0

    def test_no_overlap_score_zero(self):
        score = calculate_relevance_score("coffee cancer", "quantum physics")
        assert score == 0.0

    def test_partial_overlap(self):
        score = calculate_relevance_score("coffee cancer risk", "coffee and diabetes")
        assert 0.0 < score < 1.0

    def test_empty_argument_returns_zero(self):
        assert calculate_relevance_score("", "coffee cancer") == 0.0

    def test_empty_snippet_returns_zero(self):
        assert calculate_relevance_score("coffee cancer", "") == 0.0


class TestFilterRelevantResults:
    def test_returns_empty_for_no_results(self):
        assert filter_relevant_results("argument", []) == []

    def test_filters_below_min_score(self):
        results = [
            {"title": "X", "snippet": "quantum physics neutrons"},
        ]
        filtered = filter_relevant_results("coffee cancer", results, min_score=0.5)
        assert filtered == []

    def test_respects_max_results(self):
        results = [
            {"title": str(i), "snippet": "coffee cancer risk epidemiology"}
            for i in range(10)
        ]
        filtered = filter_relevant_results("coffee cancer risk", results, max_results=3)
        assert len(filtered) <= 3

    def test_sorted_by_score_descending(self):
        results = [
            {"snippet": "coffee"},
            {"snippet": "coffee cancer risk epidemiology study"},
        ]
        filtered = filter_relevant_results("coffee cancer risk", results, min_score=0.0, max_results=10)
        if len(filtered) >= 2:
            assert filtered[0]["relevance_score"] >= filtered[1]["relevance_score"]
