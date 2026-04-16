"""
Text Processing Constants.

Keyword extraction parameters and stop words.
"""

# ============================================================================
# KEYWORD EXTRACTION PARAMETERS
# ============================================================================

KEYWORD_MIN_LENGTH = 3
"""Minimum word length for keyword extraction."""

QUERY_GEN_MIN_WORD_LENGTH = 3
"""Minimum word length for query generation."""

QUERY_GEN_MAX_KEYWORDS = 5
"""Maximum keywords to extract for query generation."""

ARXIV_MIN_WORD_LENGTH_FALLBACK = 4
"""Minimum word length for ArXiv fallback keyword extraction."""

ARXIV_MAX_KEYWORDS_FALLBACK = 4
"""Maximum keywords for ArXiv fallback search."""

# ============================================================================
# STOP WORDS
# ============================================================================

FRENCH_STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou",
    "mais", "donc", "car", "ni", "que", "qui", "quoi", "dont", "où",
    "ce", "cet", "cette", "ces", "mon", "ton", "son", "notre", "votre",
    "leur", "mes", "tes", "ses", "nos", "vos", "leurs", "je", "tu",
    "il", "elle", "on", "nous", "vous", "ils", "elles", "être", "avoir",
    "faire", "dire", "aller", "voir", "savoir", "pouvoir", "vouloir",
    "falloir", "devoir", "croire", "prendre", "donner", "tenir", "venir",
    "trouver", "mettre", "passer", "tout", "tous", "toute", "toutes",
    "pour", "dans", "par", "sur", "avec", "sans", "sous", "vers", "chez",
    "entre", "depuis", "pendant", "comme", "si", "plus", "moins", "très",
    "bien", "peu", "beaucoup", "trop", "assez", "encore", "déjà", "jamais",
    "toujours", "souvent", "parfois", "aussi", "ainsi", "alors", "après",
    "avant", "maintenant", "ici", "là", "partout", "ailleurs", "aujourd'hui",
    "hier", "demain", "ne", "pas", "point", "rien", "aucun", "personne",
    "quelque", "plusieurs", "autre", "même", "tel", "quel", "quelle", "quels"
}

ENGLISH_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "when",
    "at", "from", "by", "to", "of", "in", "on", "for", "with", "about",
    "as", "is", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "will", "would", "should", "could", "may",
    "might", "must", "can", "this", "that", "these", "those", "i", "you",
    "he", "she", "it", "we", "they", "what", "which", "who", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "s", "t", "just", "now",
}

COMMON_STOP_WORDS_EN_FR = ENGLISH_STOP_WORDS | FRENCH_STOP_WORDS
