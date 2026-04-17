"""
Unit tests for the topic classifier agent.
All OpenAI calls are mocked at module level.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.agents.orchestration.topic_classifier import (
    classify_argument_topic,
    get_agents_for_argument,
    get_research_strategy,
    CATEGORY_AGENTS_MAP,
    PRIORITY_AGENT_MAP,
)


def _make_llm_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


class TestClassifyArgumentTopic:
    def _mock_openai(self, content: str):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(content)
        return mock_cls

    def test_valid_single_category(self):
        resp = json.dumps({"categories": ["medicine"]})
        with patch("app.agents.orchestration.topic_classifier.OpenAI", self._mock_openai(resp)):
            result = classify_argument_topic("Coffee reduces liver cancer risk.")
        assert result == ["medicine"]

    def test_multiple_valid_categories(self):
        resp = json.dumps({"categories": ["medicine", "biology"]})
        with patch("app.agents.orchestration.topic_classifier.OpenAI", self._mock_openai(resp)):
            result = classify_argument_topic("Coffee reduces liver cancer risk.")
        assert result == ["medicine", "biology"]

    def test_unknown_category_falls_back_to_general(self):
        resp = json.dumps({"categories": ["unknown_domain"]})
        with patch("app.agents.orchestration.topic_classifier.OpenAI", self._mock_openai(resp)):
            result = classify_argument_topic("Coffee reduces liver cancer risk.")
        assert result == ["general"]

    def test_empty_categories_falls_back_to_general(self):
        resp = json.dumps({"categories": []})
        with patch("app.agents.orchestration.topic_classifier.OpenAI", self._mock_openai(resp)):
            result = classify_argument_topic("Coffee reduces liver cancer risk.")
        assert result == ["general"]

    def test_llm_exception_falls_back_to_general(self):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("timeout")
        with patch("app.agents.orchestration.topic_classifier.OpenAI", mock_cls):
            result = classify_argument_topic("Coffee reduces liver cancer risk.")
        assert result == ["general"]

    def test_no_openai_key_skips_llm(self, mock_settings):
        mock_settings.openai_api_key = ""
        mock_cls = MagicMock()
        with patch("app.agents.orchestration.topic_classifier.OpenAI", mock_cls):
            result = classify_argument_topic("Coffee reduces liver cancer risk.")
        assert result == ["general"]
        mock_cls.assert_not_called()


class TestGetAgentsForArgument:
    def test_deduplication(self):
        # medicine and biology both include pubmed and semantic_scholar
        resp = json.dumps({"categories": ["medicine", "biology"]})
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(resp)
        with patch("app.agents.orchestration.topic_classifier.OpenAI", mock_cls):
            agents = get_agents_for_argument("Coffee reduces liver cancer risk.")
        # No duplicates
        assert len(agents) == len(set(agents))
        assert "pubmed" in agents
        assert "semantic_scholar" in agents


class TestGetResearchStrategy:
    def test_structure_and_priority(self):
        resp = json.dumps({"categories": ["economics"]})
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.return_value = _make_llm_response(resp)
        with patch("app.agents.orchestration.topic_classifier.OpenAI", mock_cls):
            strategy = get_research_strategy("French GDP is rising.")
        assert "categories" in strategy
        assert "agents" in strategy
        assert "priority" in strategy
        assert strategy["priority"] == "oecd"
