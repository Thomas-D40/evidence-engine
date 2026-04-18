from pydantic import BaseModel


class EvidenceItem(BaseModel):
    claim:         str
    source:        str
    source_type:   str  # "academic" | "news" | "fact_check" | "statistical" | "unknown"
    content_depth: str  # "full_text" | "abstract" | "snippet"


class SourceBreakdown(BaseModel):
    total:         int
    academic:      int  # pubmed, semantic_scholar, arxiv, crossref, europepmc, core, doaj
    statistical:   int  # oecd, world_bank
    news:          int  # newsapi, gnews
    fact_check:    int  # google_factcheck, claimbuster
    full_text:     int  # sources where full text was fetched and used by LLM
    abstract_only: int  # sources where only snippet/abstract was available


class AnalysisResult(BaseModel):
    argument:     str
    argument_en:  str

    estimated_reliability: float
    reliability_basis:     str

    evidence_balance_ratio: float | None  # pros / (pros + cons), 0.0–1.0
    evidence_balance_label: str

    pros: list[EvidenceItem]
    cons: list[EvidenceItem]

    sources:             SourceBreakdown
    support_sources:     int  # sources retrieved via support query
    refutation_sources:  int  # sources retrieved via refutation query
    used_adversarial_queries: bool
