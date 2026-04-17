"""
Unit tests for the aggregation agent.
All OpenAI calls are mocked at module level.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.agents.analysis.aggregate import aggregate_results, _fallback_aggregation
from app.constants import (
    RELIABILITY_NO_SOURCES,
    RELIABILITY_BASE_SCORE,
    RELIABILITY_PER_SOURCE_INCREMENT,
    RELIABILITY_MAX_FALLBACK,
)


def _make_llm_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


def _make_openai_mock(content: str) -> tuple:
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_llm_response(content)
    return mock_cls, mock_client.chat.completions.create


_SAMPLE_ITEM = {
    "argument": "Coffee reduces liver cancer risk.",
    "pros": [{"claim": "Reduces risk", "source": "https://example.com/1"}],
    "cons": [],
    "stance": "affirmatif",
}


class TestAggregateResults:
    def test_valid_response_returns_reliability(self):
        resp = json.dumps({
            "arguments": [{
                "argument": "Coffee reduces liver cancer risk.",
                "pros": [],
                "cons": [],
                "reliability": 0.8,
                "stance": "affirmatif",
            }]
        })
        mock_cls, _ = _make_openai_mock(resp)
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([_SAMPLE_ITEM])
        assert result["arguments"][0]["reliability"] == 0.8

    def test_reliability_clamped_above_1(self):
        resp = json.dumps({
            "arguments": [{
                "argument": "Coffee reduces liver cancer risk.",
                "pros": [], "cons": [],
                "reliability": 1.5,
                "stance": "affirmatif",
            }]
        })
        mock_cls, _ = _make_openai_mock(resp)
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([_SAMPLE_ITEM])
        assert result["arguments"][0]["reliability"] == 1.0

    def test_reliability_clamped_below_0(self):
        resp = json.dumps({
            "arguments": [{
                "argument": "Coffee reduces liver cancer risk.",
                "pros": [], "cons": [],
                "reliability": -0.3,
                "stance": "affirmatif",
            }]
        })
        mock_cls, _ = _make_openai_mock(resp)
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([_SAMPLE_ITEM])
        assert result["arguments"][0]["reliability"] == 0.0

    def test_invalid_reliability_type_defaults(self):
        resp = json.dumps({
            "arguments": [{
                "argument": "Coffee reduces liver cancer risk.",
                "pros": [], "cons": [],
                "reliability": "high",
                "stance": "affirmatif",
            }]
        })
        mock_cls, _ = _make_openai_mock(resp)
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([_SAMPLE_ITEM])
        assert result["arguments"][0]["reliability"] == 0.5

    def test_invalid_stance_defaults_to_affirmatif(self):
        resp = json.dumps({
            "arguments": [{
                "argument": "Coffee reduces liver cancer risk.",
                "pros": [], "cons": [],
                "reliability": 0.5,
                "stance": "neutral",
            }]
        })
        mock_cls, _ = _make_openai_mock(resp)
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([_SAMPLE_ITEM])
        assert result["arguments"][0]["stance"] == "affirmatif"

    def test_empty_items_returns_empty(self):
        mock_cls, mock_create = _make_openai_mock("{}")
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([])
        assert result == {"arguments": []}
        mock_create.assert_not_called()

    def test_llm_exception_triggers_fallback(self):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("LLM down")
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([_SAMPLE_ITEM])
        # Fallback produces a non-empty result
        assert len(result["arguments"]) == 1
        assert "reliability" in result["arguments"][0]

    def test_json_error_triggers_fallback(self):
        mock_cls, _ = _make_openai_mock("not valid json")
        with patch("app.agents.analysis.aggregate.OpenAI", mock_cls):
            result = aggregate_results([_SAMPLE_ITEM])
        assert len(result["arguments"]) == 1


class TestFallbackAggregation:
    def test_fallback_no_sources_returns_zero_score(self):
        item = {"argument": "Test.", "pros": [], "cons": [], "stance": "affirmatif"}
        result = _fallback_aggregation([item])
        assert result["arguments"][0]["reliability"] == RELIABILITY_NO_SOURCES

    def test_fallback_increments_per_source(self):
        item = {
            "argument": "Test.",
            "pros": [{"claim": "p1", "source": "u1"}, {"claim": "p2", "source": "u2"}],
            "cons": [{"claim": "c1", "source": "u3"}],
            "stance": "affirmatif",
        }
        result = _fallback_aggregation([item])
        num_sources = 3  # 2 pros + 1 con
        expected = min(
            RELIABILITY_MAX_FALLBACK,
            RELIABILITY_BASE_SCORE + num_sources * RELIABILITY_PER_SOURCE_INCREMENT,
        )
        assert result["arguments"][0]["reliability"] == expected
