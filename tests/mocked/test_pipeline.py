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
        assert result.estimated_reliability == 0.75
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
        assert result.evidence_balance_label == "Insufficient sources found"

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


class TestPipelineModes:
    @pytest.mark.asyncio
    async def test_simple_mode_skips_screening(self, sample_argument):
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "test query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.screen_sources_by_relevance") as mock_screen, \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            await analyze_argument(request)

        mock_screen.assert_not_called()

    @pytest.mark.asyncio
    async def test_simple_mode_skips_fulltext(self, sample_argument):
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "test query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.fetch_fulltext_for_sources") as mock_fulltext, \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            await analyze_argument(request)

        mock_fulltext.assert_not_called()

    @pytest.mark.asyncio
    async def test_medium_mode_calls_screening(self, sample_argument):
        mock_source = {"title": "Study A", "url": "https://example.com/1", "snippet": "Test.", "source": "SS"}
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.5, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "test query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[mock_source]), \
             patch("app.pipeline.filter_relevant_results", return_value=[mock_source, mock_source, mock_source, mock_source]), \
             patch("app.pipeline.screen_sources_by_relevance", return_value=([mock_source], [])) as mock_screen, \
             patch("app.pipeline.fetch_fulltext_for_sources", new_callable=AsyncMock, return_value=[mock_source]), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.MEDIUM)
            await analyze_argument(request)

        mock_screen.assert_called_once()


class TestPipelineErrorHandling:
    @pytest.mark.asyncio
    async def test_research_strategy_failure_uses_fallback_agents(self, sample_argument):
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", side_effect=Exception("classifier down")), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "query", "crossref": "query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.search_crossref", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.argument == sample_argument

    @pytest.mark.asyncio
    async def test_all_research_agents_return_empty(self, sample_argument):
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.sources.total == 0
        assert result.estimated_reliability == 0.0

    @pytest.mark.asyncio
    async def test_pros_cons_failure_returns_empty(self, sample_argument):
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", side_effect=Exception("LLM error")), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.pros == []
        assert result.cons == []

    @pytest.mark.asyncio
    async def test_aggregate_failure_returns_no_sources_score(self, sample_argument):
        from app.constants import RELIABILITY_NO_SOURCES
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", side_effect=Exception("aggregation down")):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.estimated_reliability == RELIABILITY_NO_SOURCES

    @pytest.mark.asyncio
    async def test_support_and_refutation_counts(self, sample_argument):
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        # Two search calls: first for support, second for refutation
        mock_search = AsyncMock(side_effect=[
            [{"title": "Support", "url": "https://s.com", "snippet": "Supports.", "source": "Semantic Scholar"}],
            [{"title": "Refutation", "url": "https://r.com", "snippet": "Refutes.", "source": "Semantic Scholar"}],
        ])

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "support query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={"semantic_scholar": "confounders cohort bias methodology"}), \
             patch("app.pipeline.search_semantic_scholar", mock_search), \
             patch("app.pipeline.filter_relevant_results", side_effect=lambda arg, sources, **kw: sources), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.support_sources == 1
        assert result.refutation_sources == 1

    @pytest.mark.asyncio
    async def test_adversarial_disabled_skips_generation(self, sample_argument, mock_settings):
        mock_settings.adversarial_queries_enabled = False
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "query"}), \
             patch("app.pipeline.generate_adversarial_queries") as mock_adv, \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            await analyze_argument(request)

        mock_adv.assert_not_called()


class TestPipelineAdversarialGate:
    @pytest.mark.asyncio
    async def test_used_adversarial_queries_true_when_valid(self, sample_argument):
        """Valid adversarial queries (dissimilar enough) set used_adversarial_queries=True."""
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "coffee liver cancer risk"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={"semantic_scholar": "observational bias cohort confounders methodology"}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.used_adversarial_queries is True

    @pytest.mark.asyncio
    async def test_used_adversarial_queries_false_when_all_filtered(self, sample_argument):
        """Adversarial queries too similar to support queries are filtered out."""
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "coffee liver cancer risk"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={"semantic_scholar": "coffee liver cancer null risk"}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=[]), \
             patch("app.pipeline.filter_relevant_results", return_value=[]), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.used_adversarial_queries is False

    @pytest.mark.asyncio
    async def test_source_breakdown_populated(self, sample_argument):
        """SourceBreakdown reflects the mix of sources in final_sources."""
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.0, "stance": "affirmatif"}]}
        mock_sources = [
            {"title": "A", "url": "https://a.com", "snippet": "x", "source": "Semantic Scholar", "retrieved_for": "support"},
            {"title": "B", "url": "https://b.com", "snippet": "x", "source": "PubMed", "retrieved_for": "support"},
        ]

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=mock_sources), \
             patch("app.pipeline.filter_relevant_results", return_value=mock_sources), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        assert result.sources.total == 2
        assert result.sources.academic >= 1

    @pytest.mark.asyncio
    async def test_evidence_item_has_source_type_and_depth(self, sample_argument):
        """Each EvidenceItem has non-empty source_type and content_depth."""
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["medicine"], "priority": "semantic_scholar"}
        mock_sources = [
            {"title": "Study A", "url": "https://example.com/1", "snippet": "Coffee reduces risk.", "source": "Semantic Scholar", "retrieved_for": "support"},
        ]
        mock_analysis = {
            "pros": [{"claim": "Coffee reduces liver cancer risk.", "source": "https://example.com/1"}],
            "cons": [],
        }
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": mock_analysis["pros"], "cons": [], "reliability": 0.75, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "coffee liver cancer"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=mock_sources), \
             patch("app.pipeline.filter_relevant_results", return_value=mock_sources), \
             patch("app.pipeline.screen_sources_by_relevance", return_value=(mock_sources, [])), \
             patch("app.pipeline.extract_pros_cons", return_value=mock_analysis), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        for item in result.pros + result.cons:
            assert item.source_type != ""
            assert item.content_depth != ""

    @pytest.mark.asyncio
    async def test_reliability_basis_contains_source_count(self, sample_argument):
        """reliability_basis string includes the source count as a digit."""
        mock_strategy = {"agents": ["semantic_scholar"], "categories": ["general"], "priority": "semantic_scholar"}
        mock_sources = [
            {"title": "A", "url": "https://a.com", "snippet": "x", "source": "Semantic Scholar", "retrieved_for": "support"},
        ]
        mock_agg = {"arguments": [{"argument": sample_argument, "pros": [], "cons": [], "reliability": 0.5, "stance": "affirmatif"}]}

        with patch("app.pipeline.get_research_strategy", return_value=mock_strategy), \
             patch("app.pipeline.generate_search_queries", return_value={"semantic_scholar": "query"}), \
             patch("app.pipeline.generate_adversarial_queries", return_value={}), \
             patch("app.pipeline.search_semantic_scholar", new_callable=AsyncMock, return_value=mock_sources), \
             patch("app.pipeline.filter_relevant_results", return_value=mock_sources), \
             patch("app.pipeline.extract_pros_cons", return_value={"pros": [], "cons": []}), \
             patch("app.pipeline.aggregate_results", return_value=mock_agg):

            from app.pipeline import analyze_argument
            request = AnalyzeRequest(argument=sample_argument, mode=AnalysisMode.SIMPLE)
            result = await analyze_argument(request)

        import re
        assert re.search(r"\d+", result.reliability_basis), "reliability_basis must contain a source count digit"
        assert "Not a verified fact-check" in result.reliability_basis
