"""
Enhanced World Bank research agent with improved search and error handling.

This agent uses the wbgapi library to access World Bank data.
wbgapi is synchronous — all calls are wrapped with asyncio.to_thread()
to avoid blocking the event loop.
"""
import asyncio
import logging
import re
from typing import List, Dict, Any, Optional

try:
    import wbgapi as wb
    WBGAPI_AVAILABLE = True
except ImportError:
    WBGAPI_AVAILABLE = False
    logging.warning("wbgapi not installed. Install with: pip install wbgapi")

logger = logging.getLogger(__name__)


class WorldBankAgent:
    """
    World Bank research agent.

    Features:
    - Multi-strategy indicator search
    - Context-aware country selection
    - Time series data retrieval
    - Robust error handling
    """

    # Common economic terms mapping (French → English)
    TERM_MAPPING = {
        "impôt": "tax revenue",
        "taxe": "tax",
        "pib": "gdp",
        "richesse": "wealth",
        "inégalité": "inequality gini",
        "revenu": "income",
        "chômage": "unemployment",
        "dette": "debt",
        "croissance": "growth gdp",
        "pauvreté": "poverty",
        "éducation": "education",
        "santé": "health",
        "mortalité": "mortality",
        "espérance de vie": "life expectancy",
        "inflation": "inflation",
        "commerce": "trade",
        "exportation": "exports",
        "importation": "imports"
    }

    # Country groups for different contexts
    COUNTRY_GROUPS = {
        "developed": ["USA", "DEU", "FRA", "GBR", "JPN", "CAN", "ITA"],
        "europe": ["FRA", "DEU", "GBR", "ITA", "ESP", "NLD", "BEL", "SWE"],
        "emerging": ["CHN", "IND", "BRA", "RUS", "ZAF", "MEX", "IDN"],
        "default": ["FRA", "WLD"],
        "major": ["USA", "CHN", "JPN", "DEU", "GBR", "FRA", "IND", "BRA", "WLD"]
    }

    # Fallback indicators for common topics
    FALLBACK_INDICATORS = {
        "gdp": ["NY.GDP.MKTP.CD", "NY.GDP.MKTP.KD.ZG"],
        "unemployment": ["SL.UEM.TOTL.ZS"],
        "poverty": ["SI.POV.DDAY", "SI.POV.NAHC"],
        "education": ["SE.XPD.TOTL.GD.ZS", "SE.ADT.LITR.ZS"],
        "health": ["SH.XPD.CHEX.GD.ZS", "SP.DYN.LE00.IN"],
        "inequality": ["SI.POV.GINI"],
        "trade": ["NE.TRD.GNFS.ZS"],
        "debt": ["GC.DOD.TOTL.GD.ZS"]
    }

    def __init__(self):
        """Initialize World Bank agent."""
        if not WBGAPI_AVAILABLE:
            logger.error("wbgapi not available. World Bank agent will not function.")
            self.available = False
        else:
            self.available = True
            logger.info("World Bank agent initialized successfully")

    def _extract_keywords(self, query: str) -> List[str]:
        """
        Extract search keywords from query.

        Args:
            query: Search query

        Returns:
            List of English keywords for search
        """
        query_lower = query.lower()
        keywords = []

        for french_term, english_term in self.TERM_MAPPING.items():
            if french_term in query_lower:
                keywords.append(english_term)

        words = re.findall(r'\w+', query_lower)
        stop_words = {'le', 'la', 'les', 'un', 'une', 'des', 'de', 'du', 'et', 'ou',
                      'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for'}
        for word in words:
            if len(word) > 3 and word not in stop_words:
                if word not in ' '.join(keywords):
                    keywords.append(word)

        return keywords[:5]

    def _detect_countries_from_query(self, query: str) -> List[str]:
        """
        Detect relevant countries from query.

        Args:
            query: Search query

        Returns:
            List of ISO country codes
        """
        query_lower = query.lower()

        country_map = {
            "france": "FRA", "français": "FRA", "french": "FRA",
            "germany": "DEU", "allemagne": "DEU", "german": "DEU",
            "usa": "USA", "united states": "USA", "america": "USA", "états-unis": "USA",
            "uk": "GBR", "britain": "GBR", "royaume-uni": "GBR",
            "china": "CHN", "chine": "CHN",
            "japan": "JPN", "japon": "JPN",
            "india": "IND", "inde": "IND",
            "world": "WLD", "monde": "WLD", "global": "WLD"
        }

        detected = []
        for name, code in country_map.items():
            if name in query_lower:
                detected.append(code)

        if not detected:
            detected = ["FRA", "WLD"]

        if "WLD" not in detected and len(detected) < 3:
            detected.append("WLD")

        return detected

    def _search_indicators_sync(self, query: str, max_indicators: int = 3) -> List[str]:
        """
        Search for World Bank indicators matching query (synchronous).

        Args:
            query: Search query
            max_indicators: Maximum number of indicators to return

        Returns:
            List of indicator codes
        """
        if not self.available:
            return []

        try:
            indicators = wb.series.info(q=query)
            found_codes = []

            for row in indicators:
                found_codes.append(row['id'])
                if len(found_codes) >= max_indicators:
                    break

            if found_codes:
                return found_codes

            # Try with extracted keywords
            keywords = self._extract_keywords(query)
            for keyword in keywords:
                try:
                    indicators = wb.series.info(q=keyword)
                    for row in indicators:
                        code = row['id']
                        if code not in found_codes:
                            found_codes.append(code)
                        if len(found_codes) >= max_indicators:
                            break
                    if len(found_codes) >= max_indicators:
                        break
                except Exception:
                    continue

            if found_codes:
                return found_codes

            # Fallback to common indicators based on topic
            query_lower = query.lower()
            for topic, indicator_codes in self.FALLBACK_INDICATORS.items():
                if topic in query_lower:
                    return indicator_codes[:max_indicators]

            return ["NY.GDP.MKTP.CD", "SP.POP.TOTL", "NY.GDP.PCAP.CD"][:max_indicators]

        except Exception as e:
            logger.error(f"World Bank indicator search failed: {e}")
            return []

    def _fetch_indicator_data_sync(
        self,
        indicator_codes: List[str],
        countries: List[str],
        years: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Fetch data for indicators (synchronous).

        Args:
            indicator_codes: List of indicator codes
            countries: List of country codes
            years: Number of recent years to fetch

        Returns:
            List of data points
        """
        if not self.available:
            return []

        try:
            data = wb.data.DataFrame(indicator_codes, countries, mrv=years)

            if data.empty:
                return []

            data_points = []
            df = data.reset_index()

            indicator_names = {}
            for code in indicator_codes:
                try:
                    info = wb.series.get(code)
                    indicator_names[code] = info.get('value', code)
                except Exception:
                    indicator_names[code] = code

            economy_name_map = {"FRA": "France", "WLD": "World", "USA": "United States", "DEU": "Germany"}

            for _, row in df.iterrows():
                economy = row.get('economy', 'Unknown')
                economy_name = economy_name_map.get(economy, economy)

                for code in indicator_codes:
                    if code in row and row[code] is not None:
                        year = row.get('time', None)
                        data_points.append({
                            "indicator": indicator_names.get(code, code),
                            "indicator_code": code,
                            "region": economy_name,
                            "country_code": economy,
                            "value": float(row[code]),
                            "year": year,
                            "source": "World Bank",
                            "access_type": "full_data",
                            "has_full_text": True,
                            "access_note": "Complete statistical data freely available"
                        })

            return data_points

        except Exception as e:
            logger.error(f"Failed to fetch World Bank data: {e}")
            return []

    def _search_world_bank_data_sync(
        self,
        query: str,
        countries: Optional[List[str]],
        years: int,
        max_indicators: int
    ) -> List[Dict[str, Any]]:
        """
        Synchronous implementation of World Bank search (called via asyncio.to_thread).

        Args:
            query: Search query
            countries: Country codes or None
            years: Number of years
            max_indicators: Max indicators

        Returns:
            List of data points
        """
        if not self.available:
            return []

        if not query or len(query.strip()) < 3:
            return []

        if countries is None:
            countries = self._detect_countries_from_query(query)

        if len(countries) > 5:
            countries = countries[:5]

        indicator_codes = self._search_indicators_sync(query, max_indicators)
        if not indicator_codes:
            logger.warning(f"No indicators found for query: {query}")
            return []

        data_points = self._fetch_indicator_data_sync(indicator_codes, countries, years)
        logger.info(f"[World Bank] Retrieved {len(data_points)} data points for: {query}")
        return data_points


# Module-level instance and async wrapper
_wb_agent = None


async def search_world_bank_data(
    query: str,
    countries: Optional[List[str]] = None,
    years: int = 1,
    max_indicators: int = 3
) -> List[Dict[str, Any]]:
    """
    Search World Bank data (async wrapper around synchronous wbgapi calls).

    Args:
        query: Search query (can be argument text or optimized query)
        countries: List of country codes (auto-detected if None)
        years: Number of recent years (default: 1 for compatibility)
        max_indicators: Maximum indicators (default: 3)

    Returns:
        List of data points
    """
    global _wb_agent

    if _wb_agent is None:
        _wb_agent = WorldBankAgent()

    try:
        return await asyncio.to_thread(
            _wb_agent._search_world_bank_data_sync,
            query,
            countries,
            years,
            max_indicators
        )
    except Exception as e:
        logger.error(f"World Bank search failed: {e}")
        return []
