from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class InputKind(str, Enum):
    pdf = "pdf"
    url = "url"
    patent_number = "patent_number"
    text = "text"


class AnalysisMode(str, Enum):
    single = "single"
    batch = "batch"


class PatentInput(BaseModel):
    kind: InputKind
    value: str
    label: str


class PatentDocument(BaseModel):
    source: PatentInput
    raw_text: str = ""
    title: str | None = None
    abstract: str | None = None
    claims: list[str] = Field(default_factory=list)
    description: str | None = None
    publication_number: str | None = None
    assignee: str | None = None
    representative: str | None = None


class ClaimElement(BaseModel):
    label: str
    text: str
    plain_explanation: str


class PatentAnalysis(BaseModel):
    title: str
    one_line_summary: str
    simple_explanation: str
    problem_to_solve: str
    why_patentable: str
    differentiators: list[str]
    key_claim_elements: list[ClaimElement]
    applications: list[str]
    design_around_points: list[str]
    confidence: Literal["low", "medium", "high"] = "low"
    notes: list[str] = Field(default_factory=list)


class PairwiseComparison(BaseModel):
    patent_a: str
    patent_b: str
    similarity_score: float
    similarity_level: Literal["very_low", "low", "medium", "high", "very_high"]
    score_reason: str
    common_points: list[str]
    key_differences: list[str]
    practical_interpretation: str
    recommended_review_points: list[str]
    confidence: Literal["low", "medium", "high"] = "medium"
    notes: list[str] = Field(default_factory=list)


class BatchComparisonSummary(BaseModel):
    overview: str
    major_groups: list[str]
    notable_relationships: list[str]
    interpretation_notes: list[str]
    confidence: Literal["low", "medium", "high"] = "medium"


class PatentPosition(BaseModel):
    label: str
    technology_focus: str
    distinctive_points: list[str]


class MultiPatentComparison(BaseModel):
    overview: str
    common_points: list[str]
    patent_positions: list[PatentPosition]
    key_relationships: list[str]
    design_around_insights: list[str]
    confidence: Literal["low", "medium", "high"] = "medium"
