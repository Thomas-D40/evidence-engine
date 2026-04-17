"""
Unit tests for the adversarial query generation agent.
All OpenAI calls are mocked at module level.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

import app.agents.orchestration.adversarial_query as aq_module
from app.agents.orchestration.adversarial_query import AdversarialQueryGenerator
from app.pipeline import _is_genuinely_adversarial


def _make_llm_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the module-level singleton between tests."""
    aq_module._adversarial_generator = None
    yield
    aq_module._adversarial_generator = None


# ============================================================================
# AdversarialQueryGenerator — parsing from new "queries" key
# ============================================================================

class TestAdversarialQueryGenerator:
    def test_adversarial_parsed_from_queries_key(self, make_llm_response):
        """Generator extracts adversarial_query from nested 'queries' key."""
        llm_json = json.dumps({
            "reasoning": {
                "a": "Confounders: sedentary lifestyle correlates with both.",
                "b": "N/A",
                "c": "Observational studies cannot establish causality.",
                "d": "N/A",
                "e": "Meta-analyses show heterogeneous results.",
            },
            "queries": {
                "semantic_scholar": {
                    "adversarial_query": "observational bias confounders cohort cancer",
                    "angle": "c",
                    "confidence": 0.8,
                }
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

        assert result == {"semantic_scholar": "observational bias confounders cohort cancer"}

    def test_reasoning_key_present_in_raw_response(self, make_llm_response):
        """Raw LLM response includes a 'reasoning' dict with 5 entries."""
        reasoning = {"a": "...", "b": "N/A", "c": "...", "d": "N/A", "e": "..."}
        llm_json = json.dumps({
            "reasoning": reasoning,
            "queries": {
                "pubmed": {"adversarial_query": "selection bias cohort study", "angle": "c", "confidence": 0.7},
            }
        })
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_llm_response(llm_json)

        captured_raw = {}
        original_call = None

        with patch("app.agents.orchestration.adversarial_query.OpenAI", mock_cls), \
             patch("time.sleep"):
            gen = AdversarialQueryGenerator()
            # Wrap _call_llm to capture raw output
            original_call = gen._call_llm

            def capture_call(argument, agents):
                result = original_call(argument, agents)
                captured_raw.update(result)
                return result

            gen._call_llm = capture_call
            gen.generate("Coffee reduces cancer risk.", ["pubmed"])

        assert "reasoning" in captured_raw
        assert len(captured_raw["reasoning"]) == 5

    def test_missing_queries_key_returns_empty_strings(self, make_llm_response):
        """If LLM returns no 'queries' key, all agents get empty string."""
        llm_json = json.dumps({"reasoning": {"a": "x", "b": "x", "c": "x", "d": "x", "e": "x"}})
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = make_llm_response(llm_json)

        with patch("app.agents.orchestration.adversarial_query.OpenAI", mock_cls), \
             patch("time.sleep"):
            gen = AdversarialQueryGenerator()
            result = gen.generate("Coffee reduces cancer risk.", ["pubmed", "semantic_scholar"])

        assert result["pubmed"] == ""
        assert result["semantic_scholar"] == ""

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


# ============================================================================
# _is_genuinely_adversarial quality gate
# ============================================================================

class TestIsGenuinelyAdversarial:
    def test_dissimilar_query_passes(self):
        support = "coffee liver cancer risk"
        adversarial = "observational bias cohort confounders methodology"
        assert _is_genuinely_adversarial(support, adversarial) is True

    def test_too_similar_query_fails(self):
        support = "coffee liver cancer risk"
        adversarial = "coffee liver cancer null risk"
        assert _is_genuinely_adversarial(support, adversarial) is False

    def test_empty_adversarial_query_fails(self):
        support = "coffee liver cancer risk"
        assert _is_genuinely_adversarial(support, "") is False

    def test_empty_support_query_fails(self):
        adversarial = "confounders selection bias cohort"
        assert _is_genuinely_adversarial("", adversarial) is False

    def test_both_empty_fails(self):
        assert _is_genuinely_adversarial("", "") is False

    def test_custom_threshold_permissive(self):
        # With threshold=0.99, even highly similar queries pass
        support = "coffee liver cancer risk study"
        adversarial = "coffee liver cancer risk"
        assert _is_genuinely_adversarial(support, adversarial, threshold=0.99) is True

    def test_stop_words_excluded_from_token_set(self):
        # Short words (<=3 chars) and stop words are excluded — both sets become empty → False
        support = "the risk of cancer"
        adversarial = "the risk of cancer"
        assert _is_genuinely_adversarial(support, adversarial) is False
