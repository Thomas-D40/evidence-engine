"""
Unit tests for the pros/cons extraction agent.
All OpenAI calls are mocked at module level.
"""
import json
import pytest
from unittest.mock import patch, MagicMock

from app.agents.analysis.pros_cons import extract_pros_cons
from app.constants import PROS_CONS_MAX_CONTENT_LENGTH, PROS_CONS_MIN_PARTIAL_CONTENT


def _make_llm_response(content: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=content))])


def _make_openai_mock(content: str) -> tuple:
    """Return (mock_cls, mock_create) so callers can inspect calls."""
    mock_cls = MagicMock()
    mock_client = MagicMock()
    mock_cls.return_value = mock_client
    mock_client.chat.completions.create.return_value = _make_llm_response(content)
    return mock_cls, mock_client.chat.completions.create


class TestExtractProsCons:
    def test_returns_pros_and_cons(self):
        resp = json.dumps({
            "pros": [{"claim": "Reduces risk", "source": "https://pubmed.ncbi.nlm.nih.gov/1"}],
            "cons": [],
        })
        mock_cls, _ = _make_openai_mock(resp)
        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            result = extract_pros_cons(
                "Coffee reduces liver cancer risk.",
                [{"title": "Study A", "url": "https://pubmed.ncbi.nlm.nih.gov/1", "snippet": "Reduces risk."}],
            )
        assert result["pros"] == [{"claim": "Reduces risk", "source": "https://pubmed.ncbi.nlm.nih.gov/1"}]
        assert result["cons"] == []

    def test_empty_articles_returns_empty(self):
        mock_cls, mock_create = _make_openai_mock("{}")
        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            result = extract_pros_cons("Coffee reduces liver cancer risk.", [])
        assert result == {"pros": [], "cons": []}
        mock_create.assert_not_called()

    def test_empty_argument_returns_empty(self):
        mock_cls, mock_create = _make_openai_mock("{}")
        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            result = extract_pros_cons(
                "",
                [{"title": "Study A", "url": "https://example.com", "snippet": "Some text."}],
            )
        assert result == {"pros": [], "cons": []}
        mock_create.assert_not_called()

    def test_json_decode_error_returns_empty(self):
        mock_cls, _ = _make_openai_mock("not valid json {{{{")
        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            result = extract_pros_cons(
                "Coffee reduces liver cancer risk.",
                [{"title": "Study A", "url": "https://example.com", "snippet": "Some text."}],
            )
        assert result == {"pros": [], "cons": []}

    def test_llm_exception_returns_empty(self):
        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("LLM timeout")
        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            result = extract_pros_cons(
                "Coffee reduces liver cancer risk.",
                [{"title": "Study A", "url": "https://example.com", "snippet": "Some text."}],
            )
        assert result == {"pros": [], "cons": []}

    def test_prefers_fulltext_over_snippet(self):
        captured_prompts = []

        def capture_create(**kwargs):
            captured_prompts.append(kwargs["messages"][-1]["content"])
            return _make_llm_response(json.dumps({"pros": [], "cons": []}))

        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = capture_create

        article = {
            "title": "Study A",
            "url": "https://example.com",
            "snippet": "This is the snippet.",
            "fulltext": "This is the full text content.",
        }
        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            extract_pros_cons("Coffee reduces liver cancer risk.", [article])

        assert len(captured_prompts) == 1
        assert "Full Text:" in captured_prompts[0]
        assert "Summary:" not in captured_prompts[0]

    def test_content_truncated_at_max_length(self):
        # Single article whose snippet exceeds max length — triggers fallback block
        long_snippet = "x" * (PROS_CONS_MAX_CONTENT_LENGTH + 500)
        article = {"title": "Study A", "url": "https://example.com", "snippet": long_snippet}

        captured_prompts = []

        def capture_create(**kwargs):
            captured_prompts.append(kwargs["messages"][-1]["content"])
            return _make_llm_response(json.dumps({"pros": [], "cons": []}))

        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = capture_create

        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            extract_pros_cons("Coffee reduces liver cancer risk.", [article])

        # Prompt must have been built — content capped at max
        assert len(captured_prompts) == 1
        assert len(captured_prompts[0]) <= PROS_CONS_MAX_CONTENT_LENGTH + 2000  # some overhead for prompt template

    def test_partial_article_skipped_below_min(self):
        # First article consumes almost all budget; second article would leave < MIN remaining
        first_content = "A" * (PROS_CONS_MAX_CONTENT_LENGTH - PROS_CONS_MIN_PARTIAL_CONTENT // 2)
        second_content = "B" * (PROS_CONS_MIN_PARTIAL_CONTENT * 2)

        articles = [
            {"title": "First", "url": "https://first.com", "snippet": first_content},
            {"title": "Second", "url": "https://second.com", "snippet": second_content},
        ]

        captured_prompts = []

        def capture_create(**kwargs):
            captured_prompts.append(kwargs["messages"][-1]["content"])
            return _make_llm_response(json.dumps({"pros": [], "cons": []}))

        mock_cls = MagicMock()
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = capture_create

        with patch("app.agents.analysis.pros_cons.OpenAI", mock_cls):
            extract_pros_cons("Coffee reduces liver cancer risk.", articles)

        assert len(captured_prompts) == 1
        # Second article URL should not be present (it was skipped)
        assert "https://second.com" not in captured_prompts[0]
