"""
Unit tests for the adversarial query generation agent.
All OpenAI calls are mocked at module level.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

import app.agents.orchestration.adversarial_query as aq_module
from app.agents.orchestration.adversarial_query import AdversarialQueryGenerator


def _make_llm_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton between tests."""
    aq_module._adversarial_generator = None
    yield
    aq_module._adversarial_generator = None


class TestAdversarialQueryGenerator:
    def test_generates_refutation_queries(self, make_llm_response):
        llm_json = json.dumps({
            "semantic_scholar": {
                "adversarial_query": "coffee cancer refutation",
                "confidence": 0.8,
            }
        })
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_llm_response(llm_json)

        with patch("app.agents.orchestration.adversarial_query.OpenAI", mock_cls), \
             patch("time.sleep"):
            gen = AdversarialQueryGenerator()
            result = gen.generate("Coffee reduces cancer risk.", ["semantic_scholar"])

        assert result == {"semantic_scholar": "coffee cancer refutation"}

    def test_llm_exception_returns_empty_dict(self):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("timeout")

        with patch("app.agents.orchestration.adversarial_query.OpenAI", mock_cls), \
             patch("time.sleep"):
            gen = AdversarialQueryGenerator()
            result = gen.generate("Coffee reduces cancer risk.", ["semantic_scholar"])

        assert result == {}

    def test_no_openai_key_returns_empty(self, mock_settings):
        mock_settings.openai_api_key = ""
        mock_cls = MagicMock()

        with patch("app.agents.orchestration.adversarial_query.OpenAI", mock_cls):
            gen = AdversarialQueryGenerator()
            result = gen.generate("Coffee reduces cancer risk.", ["semantic_scholar"])

        assert result == {}
