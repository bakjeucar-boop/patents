from __future__ import annotations

import re

from src.models import ClaimElement, PatentAnalysis, PatentDocument


def _sentences(text: str, limit: int = 4) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?。])\s+", compact)
    return [part.strip() for part in parts if len(part.strip()) > 20][:limit]


def _split_claim_elements(claim: str) -> list[ClaimElement]:
    body = re.sub(r"^\d+\.\s*", "", claim).strip()
    raw_parts = re.split(r";|\n|,\s+(?=(?:a|an|the)\s)", body, flags=re.IGNORECASE)
    parts = [re.sub(r"\s+", " ", part).strip(" ,.;") for part in raw_parts if len(part.strip()) > 20]
    if len(parts) < 2:
        parts = _sentences(body, limit=6)
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    elements = []
    for idx, part in enumerate(parts[:8]):
        elements.append(
            ClaimElement(
                label=labels[idx],
                text=part,
                plain_explanation="청구항의 필수 구성요소 후보입니다. LLM API 연결 후 더 자연스럽게 해석합니다.",
            )
        )
    return elements


def analyze_without_llm(document: PatentDocument) -> PatentAnalysis:
    title = document.title or document.source.label
    basis = document.abstract or document.description or document.raw_text
    summary_sentences = _sentences(basis, limit=2)
    claim = document.claims[0] if document.claims else ""
    elements = _split_claim_elements(claim) if claim else []

    return PatentAnalysis(
        title=title,
        one_line_summary=summary_sentences[0] if summary_sentences else "원문에서 핵심 요약을 자동 추출할 정보가 부족합니다.",
        simple_explanation=(
            "현재는 API 키 없이 동작하는 기본 분석입니다. 원문 초록과 독립항 후보를 바탕으로 "
            "핵심 내용을 먼저 정리하고, API 연결 후 특허성/차별점 설명을 고도화합니다."
        ),
        problem_to_solve=summary_sentences[1] if len(summary_sentences) > 1 else "명세서의 배경기술/해결과제 섹션 분석이 필요합니다.",
        why_patentable=(
            "독립항의 구성요소 조합과 기존기술 대비 효과를 확인해야 합니다. "
            "LLM 분석 단계에서 '왜 특허가 되었는지'를 별도 항목으로 추론합니다."
        ),
        differentiators=[
            "청구항의 필수 구성요소 조합",
            "기존 냉각 구조/제어 방식 대비 달라진 부분",
            "명세서에 기재된 효과 또는 성능 개선 지점",
        ],
        key_claim_elements=elements,
        applications=["제품 적용 분야는 상세 분석 단계에서 추출합니다."],
        design_around_points=["필수 구성요소 중 대체 가능한 구조/조건을 Claim Chart 단계에서 검토합니다."],
        confidence="low",
        notes=["LLM API가 연결되지 않은 로컬 기본 분석 결과입니다."],
    )
