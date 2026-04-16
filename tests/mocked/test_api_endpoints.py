"""
Mocked integration tests for the /analyze and /health endpoints.
No live API calls — OpenAI and research services are fully mocked.
"""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from app.models.response import AnalysisResult, EvidenceItem


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
            reliability_score=0.72,
            consensus_ratio=0.667,
            consensus_label="Moderate consensus",
            pros=[EvidenceItem(claim="Studies show reduced risk.", source="https://pubmed.ncbi.nlm.nih.gov/1")],
            cons=[EvidenceItem(claim="Cohort study found no effect.", source="https://pubmed.ncbi.nlm.nih.gov/2")],
            sources_count=5,
            support_sources=3,
            refutation_sources=2,
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
            reliability_score=0.5,
            consensus_ratio=None,
            consensus_label="Insufficient data",
            pros=[], cons=[],
            sources_count=0, support_sources=0, refutation_sources=0,
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
            reliability_score=0.72,
            consensus_ratio=0.667,
            consensus_label="Moderate consensus",
            pros=[EvidenceItem(claim="Evidence.", source="https://example.com")],
            cons=[],
            sources_count=3,
            support_sources=2,
            refutation_sources=1,
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
            "argument", "argument_en", "reliability_score",
            "consensus_ratio", "consensus_label",
            "pros", "cons", "sources_count",
            "support_sources", "refutation_sources",
        ]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"

    def test_security_headers_present(self, client):
        response = client.get("/health")
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
