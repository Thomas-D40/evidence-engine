"""
Mocked integration tests for the /analyze and /health endpoints.
No live API calls — OpenAI and research services are fully mocked.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.models.response import AnalysisResult, EvidenceItem, SourceBreakdown


def _make_evidence_item(claim: str, source: str) -> EvidenceItem:
    return EvidenceItem(
        claim=claim,
        source=source,
        source_type="academic",
        content_depth="snippet",
    )


def _make_source_breakdown(total: int = 5, support: int = 3, refutation: int = 2) -> SourceBreakdown:
    return SourceBreakdown(
        total=total,
        academic=total,
        statistical=0,
        news=0,
        fact_check=0,
        full_text=0,
        abstract_only=total,
    )


# ============================================================================
# /health
# ============================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_health_no_auth_required(self, client):
        response = client.get("/health")
        assert response.status_code == 200


# ============================================================================
# /analyze — auth enforcement
# ============================================================================

class TestAnalyzeAuth:
    def test_missing_api_key_returns_401(self, client):
        response = client.post(
            "/analyze",
            json={"argument": "Coffee reduces liver cancer risk."}
        )
        assert response.status_code == 401

    def test_invalid_api_key_returns_403(self, client):
        response = client.post(
            "/analyze",
            headers={"X-API-Key": "wrong-key"},
            json={"argument": "Coffee reduces liver cancer risk."}
        )
        assert response.status_code == 403

    def test_valid_api_key_accepted(self, client):
        mock_result = AnalysisResult(
            argument="Coffee reduces liver cancer risk.",
            argument_en="Coffee reduces liver cancer risk.",
            estimated_reliability=0.72,
            reliability_basis="AI estimate based on 5 sources (0 full text, 5 abstract only). Not a verified fact-check.",
            evidence_balance_ratio=0.667,
            evidence_balance_label="More supporting than contradicting evidence found",
            pros=[_make_evidence_item("Studies show reduced risk.", "https://pubmed.ncbi.nlm.nih.gov/1")],
            cons=[_make_evidence_item("Cohort study found no effect.", "https://pubmed.ncbi.nlm.nih.gov/2")],
            sources=_make_source_breakdown(),
            support_sources=3,
            refutation_sources=2,
            used_adversarial_queries=True,
        )

        with patch("app.api.analyze_argument", new_callable=AsyncMock, return_value=mock_result):
            response = client.post(
                "/analyze",
                headers={"X-API-Key": "test-key-valid"},
                json={"argument": "Coffee reduces liver cancer risk."}
            )
        assert response.status_code == 200


# ============================================================================
# /analyze — payload validation
# ============================================================================

class TestAnalyzePayload:
    def test_argument_too_short_returns_422(self, client):
        response = client.post(
            "/analyze",
            headers={"X-API-Key": "test-key-valid"},
            json={"argument": "Too short"}
        )
        assert response.status_code == 422

    def test_argument_missing_returns_422(self, client):
        response = client.post(
            "/analyze",
            headers={"X-API-Key": "test-key-valid"},
            json={"mode": "simple"}
        )
        assert response.status_code == 422

    def test_payload_too_large_returns_413(self, client):
        large_body = "x" * 15_000
        response = client.post(
            "/analyze",
            headers={
                "X-API-Key": "test-key-valid",
                "Content-Length": str(len(large_body)),
            },
            content=large_body,
        )
        assert response.status_code == 413

    def test_valid_modes_accepted(self, client):
        mock_result = AnalysisResult(
            argument="Coffee reduces liver cancer risk.",
            argument_en="Coffee reduces liver cancer risk.",
            estimated_reliability=0.5,
            reliability_basis="AI estimate based on 0 sources (0 full text, 0 abstract only). Not a verified fact-check.",
            evidence_balance_ratio=None,
            evidence_balance_label="Insufficient sources found",
            pros=[], cons=[],
            sources=SourceBreakdown(total=0, academic=0, statistical=0, news=0, fact_check=0, full_text=0, abstract_only=0),
            support_sources=0,
            refutation_sources=0,
            used_adversarial_queries=False,
        )
        for mode in ["simple", "medium", "hard"]:
            with patch("app.api.analyze_argument", new_callable=AsyncMock, return_value=mock_result):
                response = client.post(
                    "/analyze",
                    headers={"X-API-Key": "test-key-valid"},
                    json={"argument": "Coffee reduces liver cancer risk.", "mode": mode}
                )
            assert response.status_code == 200, f"Mode {mode!r} failed"

    def test_invalid_mode_returns_422(self, client):
        response = client.post(
            "/analyze",
            headers={"X-API-Key": "test-key-valid"},
            json={"argument": "Coffee reduces liver cancer risk.", "mode": "ultra"}
        )
        assert response.status_code == 422


# ============================================================================
# /analyze — response shape
# ============================================================================

class TestAnalyzeResponseShape:
    def test_response_contains_required_fields(self, client):
        mock_result = AnalysisResult(
            argument="Coffee reduces liver cancer risk.",
            argument_en="Coffee reduces liver cancer risk.",
            estimated_reliability=0.72,
            reliability_basis="AI estimate based on 3 sources (0 full text, 3 abstract only). Not a verified fact-check.",
            evidence_balance_ratio=0.667,
            evidence_balance_label="More supporting than contradicting evidence found",
            pros=[_make_evidence_item("Evidence.", "https://example.com")],
            cons=[],
            sources=_make_source_breakdown(total=3, support=2, refutation=1),
            support_sources=2,
            refutation_sources=1,
            used_adversarial_queries=True,
        )

        with patch("app.api.analyze_argument", new_callable=AsyncMock, return_value=mock_result):
            response = client.post(
                "/analyze",
                headers={"X-API-Key": "test-key-valid"},
                json={"argument": "Coffee reduces liver cancer risk."}
            )

        assert response.status_code == 200
        data = response.json()

        required_fields = [
            "argument", "argument_en",
            "estimated_reliability", "reliability_basis",
            "evidence_balance_ratio", "evidence_balance_label",
            "pros", "cons",
            "sources", "support_sources", "refutation_sources",
            "used_adversarial_queries",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_sources_breakdown_shape(self, client):
        """sources field is a breakdown object, not a flat count."""
        mock_result = AnalysisResult(
            argument="Coffee reduces liver cancer risk.",
            argument_en="Coffee reduces liver cancer risk.",
            estimated_reliability=0.5,
            reliability_basis="AI estimate based on 2 sources (0 full text, 2 abstract only). Not a verified fact-check.",
            evidence_balance_ratio=None,
            evidence_balance_label="Insufficient sources found",
            pros=[], cons=[],
            sources=SourceBreakdown(total=2, academic=2, statistical=0, news=0, fact_check=0, full_text=0, abstract_only=2),
            support_sources=2,
            refutation_sources=0,
            used_adversarial_queries=False,
        )

        with patch("app.api.analyze_argument", new_callable=AsyncMock, return_value=mock_result):
            response = client.post(
                "/analyze",
                headers={"X-API-Key": "test-key-valid"},
                json={"argument": "Coffee reduces liver cancer risk."}
            )

        data = response.json()
        sources = data["sources"]
        for key in ["total", "academic", "statistical", "news", "fact_check", "full_text", "abstract_only"]:
            assert key in sources, f"Missing sources breakdown key: {key}"

    def test_security_headers_present(self, client):
        response = client.get("/health")
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
