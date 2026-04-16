"""
OECD research agent — keyword-based dataset lookup.

Returns metadata pointers to OECD statistical datasets based on topic
matching. No external HTTP calls are made; all data is pre-defined.

API Documentation: https://data.oecd.org/api/sdmx-json-documentation/
"""
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class OECDAgent:
    """
    OECD research agent using keyword matching against known datasets.

    Returns dataset metadata (name, description, URL) without making
    live HTTP calls to the OECD API.
    """

    # Common OECD datasets with their dataflow IDs and metadata
    COMMON_DATASETS = {
        "gdp": {
            "dataflow_id": "QNA",
            "name": "Quarterly National Accounts - GDP",
            "description": "GDP and main aggregates from national accounts",
            "keywords": ["gdp", "gross domestic product", "economic output", "national accounts", "growth"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=13"
        },
        "unemployment": {
            "dataflow_id": "LFS_SEXAGE_I_R",
            "name": "Labour Force Statistics - Unemployment",
            "description": "Unemployment rates by sex and age group",
            "keywords": ["unemployment", "jobless", "labour force", "employment", "jobs"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=7"
        },
        "inflation": {
            "dataflow_id": "PRICES_CPI",
            "name": "Consumer Price Indices - Inflation",
            "description": "CPI and inflation rates",
            "keywords": ["inflation", "cpi", "consumer prices", "price index", "prices"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=17"
        },
        "education": {
            "dataflow_id": "EAG",
            "name": "Education at a Glance",
            "description": "Educational finance and spending indicators",
            "keywords": ["education", "school", "university", "educational spending", "students"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=4"
        },
        "health": {
            "dataflow_id": "HEALTH_STAT",
            "name": "Health Statistics",
            "description": "Health expenditure and health indicators",
            "keywords": ["health", "healthcare", "medical", "life expectancy", "hospital"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=8"
        },
        "environment": {
            "dataflow_id": "AIR_GHG",
            "name": "Air and Climate - GHG Emissions",
            "description": "Greenhouse gas emissions and air pollutants",
            "keywords": ["environment", "emissions", "co2", "climate", "pollution", "greenhouse"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=5"
        },
        "trade": {
            "dataflow_id": "MEI_TRADE",
            "name": "Main Economic Indicators - International Trade",
            "description": "Trade balance and international trade indicators",
            "keywords": ["trade", "exports", "imports", "balance of trade", "commerce"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=16"
        },
        "productivity": {
            "dataflow_id": "PDB_LV",
            "name": "Productivity - Level",
            "description": "Labour productivity indicators",
            "keywords": ["productivity", "efficiency", "labour productivity", "output per worker"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=14"
        },
        "poverty": {
            "dataflow_id": "IDD",
            "name": "Income Distribution Database",
            "description": "Income inequality and poverty indicators",
            "keywords": ["poverty", "inequality", "income", "gini", "wealth distribution"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=9"
        },
        "taxation": {
            "dataflow_id": "REV",
            "name": "Revenue Statistics",
            "description": "Tax revenue by type of tax",
            "keywords": ["tax", "taxation", "revenue", "fiscal", "government revenue"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=15"
        },
        "innovation": {
            "dataflow_id": "MSTI_PUB",
            "name": "Main Science and Technology Indicators",
            "description": "R&D and innovation indicators",
            "keywords": ["innovation", "research", "development", "r&d", "technology", "science"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=11"
        },
        "energy": {
            "dataflow_id": "IEA_ENERGY",
            "name": "Energy Statistics",
            "description": "Energy production and consumption",
            "keywords": ["energy", "electricity", "renewable", "power", "fuel"],
            "url": "https://data-explorer.oecd.org/?lc=en&pg=0&fc=Subject&snb=6"
        }
    }

    def _search_datasets(self, query: str) -> List[str]:
        """
        Search for relevant datasets by matching keywords.

        Args:
            query: Search query

        Returns:
            List of dataset keys that match the query
        """
        query_lower = query.lower()
        scores = []

        for key, dataset in self.COMMON_DATASETS.items():
            score = 0

            for keyword in dataset["keywords"]:
                if keyword in query_lower:
                    score += 2

            query_words = set(query_lower.split())
            name_words = set(dataset["name"].lower().split())
            desc_words = set(dataset["description"].lower().split())

            score += len(query_words & name_words) * 1.5
            score += len(query_words & desc_words) * 1

            if score > 0:
                scores.append((key, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        matched_keys = [key for key, _ in scores[:3]]

        return matched_keys if matched_keys else ["gdp", "unemployment", "inflation"]

    async def search_oecd_data(self, query: str, max_results: int = 3) -> List[Dict[str, str]]:
        """
        Search for statistical indicators in the OECD database.

        Uses keyword matching to find relevant OECD datasets and returns
        metadata about available indicators. No HTTP calls are made.

        Args:
            query: Search keywords (e.g., "GDP growth", "unemployment rate")
            max_results: Maximum number of results (default: 3)

        Returns:
            List of dictionaries containing:
            - title: Indicator name
            - url: URL to the data explorer
            - snippet: Indicator description
            - source: "OECD"
            - indicator_code: Dataflow ID
            - dataset: Dataset name
        """
        if not query or len(query.strip()) < 2:
            logger.warning("Query too short for OECD search")
            return []

        matched_keys = self._search_datasets(query)

        results = []
        for key in matched_keys[:max_results]:
            dataset = self.COMMON_DATASETS[key]
            results.append({
                "title": dataset["name"],
                "url": dataset["url"],
                "snippet": dataset["description"],
                "source": "OECD",
                "indicator_code": dataset["dataflow_id"],
                "dataset": "OECD Statistics",
                "access_type": "full_data",
                "has_full_text": True,
                "access_note": "Complete statistical data freely available"
            })

        logger.info(f"[OECD] Found {len(results)} indicators for: {query}")
        return results


# Module-level instance and async wrapper
_oecd_agent = None


async def search_oecd_data(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    Search for statistical indicators in the OECD database.

    Args:
        query: Search keywords (e.g., "GDP growth", "unemployment rate")
        max_results: Maximum number of results (default: 3)

    Returns:
        List of dictionaries containing:
        - title: Indicator name
        - url: URL to the data
        - snippet: Indicator description
        - source: "OECD"
        - indicator_code: Indicator code
        - dataset: Dataset name
    """
    global _oecd_agent

    if _oecd_agent is None:
        _oecd_agent = OECDAgent()

    return await _oecd_agent.search_oecd_data(query, max_results)
