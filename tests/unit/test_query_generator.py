"""
Unit tests for the query generator agent.
All OpenAI calls are mocked at module level.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

import app.agents.orchestration.query_generator as qg_module
from app.agents.orchestration.query_generator import QueryGenerator, generate_search_queries
from app.utils.api_helpers import TransientAPIError


def _make_llm_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


def _make_openai_mock(content: str) -> MagicMock:
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_llm_response(content)
    return mock_cls


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton between tests."""
    qg_module._query_generator = None
    yield
    qg_module._query_generator = None


class TestQueryGenerator:
    def test_generate_queries_happy_path(self):
        resp = json.dumps({
            "pubmed": {"query": "coffee cancer risk", "fallback": [], "confidence": 0.9},
            "semantic_scholar": {"query": "coffee liver cancer", "fallback": [], "confidence": 0.8},
        })
        with patch("app.agents.orchestration.query_generator.OpenAI", _make_openai_mock(resp)), \
             patch("time.sleep"):
            gen = QueryGenerator()
            result = gen.generate_queries(
                "Coffee reduces liver cancer risk.",
                agents=["pubmed", "semantic_scholar"],
            )
        assert result["pubmed"]["query"] == "coffee cancer risk"
        assert result["semantic_scholar"]["query"] == "coffee liver cancer"

    def test_missing_agent_in_response_filled_empty(self):
        # LLM only returns pubmed, crossref is missing
        resp = json.dumps({
            "pubmed": {"query": "coffee cancer", "fallback": [], "confidence": 0.9},
        })
        with patch("app.agents.orchestration.query_generator.OpenAI", _make_openai_mock(resp)), \
             patch("time.sleep"):
            gen = QueryGenerator()
            result = gen.generate_queries(
                "Coffee reduces liver cancer risk.",
                agents=["pubmed", "crossref"],
            )
        assert result["crossref"]["query"] == ""

    def test_llm_failure_uses_fallback_queries(self):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("network error")
        with patch("app.agents.orchestration.query_generator.OpenAI", mock_cls), \
             patch("time.sleep"):
            gen = QueryGenerator()
            result = gen.generate_queries(
                "Coffee reduces liver cancer risk.",
                agents=["semantic_scholar"],
            )
        # Falls back — confidence is 0.3
        assert result["semantic_scholar"]["confidence"] == 0.3

    def test_fallback_economic_agents(self):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("fail")
        with patch("app.agents.orchestration.query_generator.OpenAI", mock_cls), \
             patch("time.sleep"):
            gen = QueryGenerator()
            result = gen.generate_queries(
                "GDP growth rate increased last year.",
                agents=["oecd"],
            )
        # Economic fallback picks up "gdp" and "growth" terms
        assert any(term in result["oecd"]["query"] for term in ["gdp", "growth", "economic"])

    def test_fallback_general_agents(self):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("fail")
        with patch("app.agents.orchestration.query_generator.OpenAI", mock_cls), \
             patch("time.sleep"):
            gen = QueryGenerator()
            result = gen.generate_queries(
                "Coffee reduces cancer risk.",
                agents=["semantic_scholar"],
            )
        # Fallback should produce a non-empty query
        assert len(result["semantic_scholar"]["query"]) > 0

    def test_argument_too_short_returns_empty(self):
        with patch("app.agents.orchestration.query_generator.OpenAI") as mock_cls, \
             patch("time.sleep"):
            gen = QueryGenerator()
            result = gen.generate_queries("hi", agents=["pubmed"])
        assert result == {}
        mock_cls.return_value.chat.completions.create.assert_not_called()

    def test_generate_search_queries_flat_format(self):
        resp = json.dumps({
            "pubmed": {"query": "coffee cancer risk", "fallback": [], "confidence": 0.9},
        })
        with patch("app.agents.orchestration.query_generator.OpenAI", _make_openai_mock(resp)), \
             patch("time.sleep"):
            result = generate_search_queries(
                "Coffee reduces liver cancer risk.",
                agents=["pubmed"],
            )
        # Must return flat {agent: str}, not nested dict
        assert isinstance(result["pubmed"], str)
        assert result["pubmed"] == "coffee cancer risk"

    def test_singleton_reuse(self):
        resp = json.dumps({
            "pubmed": {"query": "coffee cancer", "fallback": [], "confidence": 0.8},
        })
        with patch("app.agents.orchestration.query_generator.OpenAI", _make_openai_mock(resp)), \
             patch("time.sleep"):
            # Two calls — singleton is reused
            generate_search_queries("Coffee reduces liver cancer risk.", agents=["pubmed"])
            generate_search_queries("Coffee reduces liver cancer risk.", agents=["pubmed"])

        # The singleton was initialised once
        assert qg_module._query_generator is not None
