"""
End-to-end tests for the verify_api_key FastAPI dependency.
Uses the test client so the full request/response cycle is exercised.
"""
import time
import pytest
from unittest.mock import patch, AsyncMock

from app.models.response import AnalysisResult, EvidenceItem, SourceBreakdown


# Minimal valid AnalysisResult for mocking the pipeline
_MOCK_RESULT = AnalysisResult(
    argument="Coffee reduces liver cancer risk.",
    argument_en="Coffee reduces liver cancer risk.",
    estimated_reliability=0.7,
    reliability_basis="AI estimate based on 0 sources (0 full text, 0 abstract only). Not a verified fact-check.",
    evidence_balance_ratio=None,
    evidence_balance_label="Insufficient sources found",
    pros=[],
    cons=[],
    sources=SourceBreakdown(total=0, academic=0, statistical=0, news=0, fact_check=0, full_text=0, abstract_only=0),
    support_sources=0,
    refutation_sources=0,
    used_adversarial_queries=False,
)

_VALID_PAYLOAD = {"argument": "Coffee reduces liver cancer risk."}


class TestVerifyApiKey:
    def test_missing_key_returns_401(self, client, clean_ip_states):
        response = client.post("/analyze", json=_VALID_PAYLOAD)
        assert response.status_code == 401

    def test_invalid_key_returns_403(self, client, clean_ip_states):
        response = client.post(
            "/analyze",
            headers={"X-API-Key": "bad-key"},
            json=_VALID_PAYLOAD,
        )
        assert response.status_code == 403

        from app.core import auth
        ip = "testclient"
        assert auth._ip_states.get(ip, auth.IPState()).failure_count == 1

    def test_valid_key_resets_failures(self, client, clean_ip_states):
        from app.core import auth
        ip = "testclient"
        auth._ip_states[ip] = auth.IPState(failure_count=3)

        with patch("app.api.analyze_argument", new_callable=AsyncMock, return_value=_MOCK_RESULT):
            client.post(
                "/analyze",
                headers={"X-API-Key": "test-key-valid"},
                json=_VALID_PAYLOAD,
            )

        assert auth._ip_states[ip].failure_count == 0

    def test_blocked_ip_returns_429(self, client, clean_ip_states):
        from app.core import auth
        ip = "testclient"
        auth._ip_states[ip] = auth.IPState(blocked_until=time.time() + 999)

        response = client.post(
            "/analyze",
            headers={"X-API-Key": "test-key-valid"},
            json=_VALID_PAYLOAD,
        )
        assert response.status_code == 429
        assert "blocked" in response.json()["detail"].lower()

    def test_valid_key_accepted(self, client, clean_ip_states):
        with patch("app.api.analyze_argument", new_callable=AsyncMock, return_value=_MOCK_RESULT):
            response = client.post(
                "/analyze",
                headers={"X-API-Key": "test-key-valid"},
                json=_VALID_PAYLOAD,
            )
        assert response.status_code == 200
