"""
Unit tests for the relevance screening agent.
Pure-Python helpers tested directly; LLM path uses module-level mocks.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.agents.enrichment.screening import (
    _parse_screening_response,
    _select_top_sources,
    _attach_scores_to_sources,
    screen_sources_by_relevance,
    get_screening_stats,
)
from app.constants import RELEVANCE_THRESHOLD_HIGH, RELEVANCE_THRESHOLD_MEDIUM_MIN


def _make_llm_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


def _make_source(title: str = "Study", url: str = "https://example.com") -> dict:
    return {"title": title, "url": url, "snippet": "Some abstract."}


class TestParseScreeningResponse:
    def test_parse_valid_response(self):
        content = json.dumps({
            "scores": [
                {"source_id": 1, "score": 0.85, "reason": "Directly relevant."},
            ]
        })
        result = _parse_screening_response(content, num_sources=2)
        assert result == {0: {"score": 0.85, "reason": "Directly relevant."}}

    def test_parse_invalid_json_returns_empty(self):
        result = _parse_screening_response("not json at all", num_sources=2)
        assert result == {}

    def test_parse_out_of_range_source_id_skipped(self):
        content = json.dumps({
            "scores": [{"source_id": 99, "score": 0.9, "reason": "Out of range."}]
        })
        result = _parse_screening_response(content, num_sources=2)
        assert result == {}

    def test_parse_score_clamped_above_1(self):
        content = json.dumps({
            "scores": [{"source_id": 1, "score": 1.5, "reason": "High."}]
        })
        result = _parse_screening_response(content, num_sources=2)
        assert result[0]["score"] == 1.0


class TestSelectTopSources:
    def _make_scored(self, score: float) -> dict:
        return {**_make_source(), "relevance_score": score}

    def test_respects_min_score(self):
        sources = [
            self._make_scored(0.9),
            self._make_scored(0.7),
            self._make_scored(0.4),
        ]
        selected, rejected = _select_top_sources(sources, top_n=5, min_score=0.6)
        assert len(selected) == 2
        assert len(rejected) == 1

    def test_respects_top_n(self):
        sources = [self._make_scored(0.9)] * 4
        selected, rejected = _select_top_sources(sources, top_n=2, min_score=0.0)
        assert len(selected) == 2
        assert len(rejected) == 2


class TestScreenSourcesByRelevance:
    def test_disabled_returns_top_n(self, mock_settings):
        mock_settings.fulltext_screening_enabled = False
        sources = [_make_source(f"Study {i}") for i in range(5)]
        mock_cls = MagicMock()
        with patch("app.agents.enrichment.screening.OpenAI", mock_cls):
            selected, rejected = screen_sources_by_relevance("test arg", sources, top_n=3)
        assert len(selected) == 3
        assert len(rejected) == 2
        mock_cls.assert_not_called()

    def test_empty_sources_returns_empty_tuples(self):
        selected, rejected = screen_sources_by_relevance("test arg", [])
        assert selected == []
        assert rejected == []

    def test_sources_leq_top_n_selects_all(self):
        sources = [_make_source("Study A"), _make_source("Study B")]
        mock_cls = MagicMock()
        with patch("app.agents.enrichment.screening.OpenAI", mock_cls):
            selected, rejected = screen_sources_by_relevance("test arg", sources, top_n=3)
        assert selected == sources
        assert rejected == []
        mock_cls.assert_not_called()

    def test_screen_llm_happy_path(self):
        sources = [_make_source(f"Study {i}") for i in range(4)]
        llm_resp = json.dumps({
            "scores": [
                {"source_id": 1, "score": 0.9, "reason": "Highly relevant."},
                {"source_id": 2, "score": 0.8, "reason": "Relevant."},
                {"source_id": 3, "score": 0.4, "reason": "Somewhat."},
                {"source_id": 4, "score": 0.2, "reason": "Not relevant."},
            ]
        })
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_resp)

        with patch("app.agents.enrichment.screening.OpenAI", mock_cls):
            selected, rejected = screen_sources_by_relevance(
                "test arg", sources, top_n=2, min_score=0.6
            )
        assert len(selected) == 2
        assert len(rejected) == 2

    def test_screen_llm_parse_failure_fallback(self):
        sources = [_make_source(f"Study {i}") for i in range(4)]
        # LLM returns valid JSON but empty scores — triggers fallback
        llm_resp = json.dumps({"scores": []})
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(llm_resp)

        with patch("app.agents.enrichment.screening.OpenAI", mock_cls):
            selected, rejected = screen_sources_by_relevance(
                "test arg", sources, top_n=2, min_score=0.6
            )
        # Falls back to top-N slice
        assert len(selected) == 2

    def test_screen_llm_exception_fallback(self):
        sources = [_make_source(f"Study {i}") for i in range(4)]
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("LLM down")

        with patch("app.agents.enrichment.screening.OpenAI", mock_cls):
            selected, rejected = screen_sources_by_relevance(
                "test arg", sources, top_n=2, min_score=0.6
            )
        # Falls back to top-N slice
        assert len(selected) == 2


class TestGetScreeningStats:
    def test_empty_returns_zeros(self):
        result = get_screening_stats([])
        assert result["total"] == 0
        assert result["avg_score"] == 0.0

    def test_counts_by_threshold(self):
        # RELEVANCE_THRESHOLD_HIGH = 0.7, RELEVANCE_THRESHOLD_MEDIUM_MIN = 0.4
        sources = [
            {"relevance_score": 0.9},  # >= 0.7 → high
            {"relevance_score": 0.6},  # 0.4 <= 0.6 < 0.7 → medium
            {"relevance_score": 0.3},  # < 0.4 → low
        ]
        result = get_screening_stats(sources)
        assert result["high_relevance"] == 1
        assert result["medium_relevance"] == 1
        assert result["low_relevance"] == 1
        assert result["total"] == 3
