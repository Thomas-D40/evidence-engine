from pydantic import BaseModel


class EvidenceItem(BaseModel):
    claim:  str
    source: str


class AnalysisResult(BaseModel):
    argument:            str
    argument_en:         str
    reliability_score:   float
    consensus_ratio:     float | None   # pros / (pros + cons), 0.0–1.0
    consensus_label:     str            # "Strong consensus" | "Moderate consensus" | "Contested" | "Minority position"
    pros:                list[EvidenceItem]
    cons:                list[EvidenceItem]
    sources_count:       int
    support_sources:     int            # sources retrieved via support query
    refutation_sources:  int            # sources retrieved via refutation query
