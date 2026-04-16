"""
Shared pytest fixtures for evidence-engine tests.

No live API calls — all external services are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


# ============================================================================
# SETTINGS MOCK
# ============================================================================

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Patch get_settings() so tests never need a real .env file.
    Applied automatically to all tests.
    """
    mock = MagicMock()
    mock.openai_api_key = "test-openai-key"
    mock.openai_model = "gpt-4o-mini"
    mock.openai_smart_model = "gpt-4o"
    mock.allowed_api_keys = "test-key-valid"
    mock.api_keys_set = {"test-key-valid"}
    mock.fulltext_screening_enabled = True
    mock.fulltext_top_n = 3
    mock.fulltext_min_score = 0.6
    mock.adversarial_queries_enabled = True
    mock.env = "test"
    mock.host = "0.0.0.0"
    mock.port = 8001

    with patch("app.config.get_settings", return_value=mock):
        with patch("app.core.auth.get_settings", return_value=mock):
            yield mock


# ============================================================================
# TEST CLIENT
# ============================================================================

@pytest.fixture
def client(mock_settings):
    """FastAPI test client with mocked settings."""
    from app.api import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ============================================================================
# SAMPLE DATA
# ============================================================================

@pytest.fixture
def sample_argument():
    return "Coffee reduces liver cancer risk."


@pytest.fixture
def sample_sources():
    return [
        {
            "title": "Coffee and liver cancer: a meta-analysis",
            "url": "https://pubmed.ncbi.nlm.nih.gov/12345",
            "snippet": "Regular coffee consumption is associated with reduced liver cancer risk.",
            "source": "PubMed",
        },
        {
            "title": "Caffeinated beverages and hepatocellular carcinoma",
            "url": "https://pubmed.ncbi.nlm.nih.gov/67890",
            "snippet": "No significant association found between coffee and HCC in this cohort.",
            "source": "PubMed",
        },
    ]


@pytest.fixture
def sample_pros():
    return [
        {"claim": "Coffee significantly reduces liver cancer risk.", "source": "https://pubmed.ncbi.nlm.nih.gov/12345"},
        {"claim": "Two cups per day lower risk by 40%.", "source": "https://pubmed.ncbi.nlm.nih.gov/11111"},
    ]


@pytest.fixture
def sample_cons():
    return [
        {"claim": "No statistically significant reduction found in cohort study.", "source": "https://pubmed.ncbi.nlm.nih.gov/67890"},
    ]
