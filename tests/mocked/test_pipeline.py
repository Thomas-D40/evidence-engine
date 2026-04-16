"""
Mocked tests for the analysis pipeline.
All LLM and research service calls are mocked.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.models.request import AnalyzeRequest, AnalysisMode
from app.agents.analysis.consensus import compute_consensus


class TestPipelineMocked:
    @pytest.mark.asyncio
    async def test_analyze_argument_returns_result(self, sample_argument):
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["medicine"], "priority": "semantic_scholar"}
        mock_queries = {"semantic_scholar": "coffee liver cancer"}
        mock_sources = [
            {"title": "Study A", "url": "https://example.com/1", "snippet": "Coffee reduces risk.", "source": "Semantic Scholar", "retrieved_for": "support"},
        ]
        mock_analysis = {
            "pros": [{"claim": "Coffee reduces liver cancer risk.", "source": "https://example.com/1"}],
            "cons": [],
        }
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": mock_analysis["pros"], "cons": [], "reliability": 0.75, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value=mock_queries), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=mock_sources), \
             patch("app.pipeline.filter_relevant_results", return_value=mock_sources), \
             patch("app.pipeline.screen_sources_by_relevance", return_value=(mock_sources, [])), \
             patch("app.pipeline.extract_pros_cons", return_value=mock_analysis), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.argument == sample_argument
        assert result.reliability_score == 0.75
        assert len(result.pros) == 1
        assert len(result.cons) == 0

    @pytest.mark.asyncio
    async def test_adversarial_failure_continues_pipeline(self, sample_argument):
        """When adversarial query generation fails, pipeline proceeds with support-only."""
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_queries = {"semantic_scholar": "coffee liver cancer"}
        mock_sources = []
        mock_analysis = {"pros": [], "cons": []}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value=mock_queries), \
             patch("app.pipeline.generate_adversarial_queries", side_effect=Exception("LLM timeout")), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=mock_sources), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", return_value=mock_analysis), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        # Should complete without raising, even with no sources
        assert result.argument == sample_argument
        assert result.consensus_label == "Insufficient data"

    @pytest.mark.asyncio
    async def test_retrieved_for_tag_stripped_before_llm(self, sample_argument):
        """Verify retrieved_for tag is not passed to extract_pros_cons."""
        captured_articles = []

        def capture_pros_cons(argument, articles, **kwargs):
            captured_articles.extend(articles)
            return {"pros": [], "cons": []}

        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        source_with_tag = {"title": "X", "url": "https://x.com", "snippet": "Test.", "source": "Semantic Scholar", "retrieved_for": "support"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "test"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[source_with_tag]), \
             patch("app.pipeline.filter_relevant_results", return_value=[source_with_tag]), \
             patch("app.pipeline.extract_pros_cons", side_effect=capture_pros_cons), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            await analyze_argument(request)

        # Verify no article passed to LLM has the retrieved_for tag
        for article in captured_articles:
            assert "retrieved_for" not in article, "retrieved_for tag was NOT stripped before LLM call"
