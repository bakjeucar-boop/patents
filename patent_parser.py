from __future__ import annotations

import re

from src.models import PatentDocument


SECTION_PATTERNS = {
    "abstract": re.compile(r"\b(Abstract|요약|초록|要約|摘要)\b", re.IGNORECASE),
    "claim_start": re.compile(r"\b(What is claimed is:?|청구범위|청구항|Claims?|請求の範囲|請求項|权利要求)\b", re.IGNORECASE),
    "description": re.compile(
        r"\b(Description|Detailed Description|발명의 설명|상세한 설명|発明の詳細な説明|詳細な説明|说明书)\b",
        re.IGNORECASE,
    ),
}


def guess_title(text: str) -> str | None:
    google_title_match = re.match(
        r"^\s*[A-Z]{0,2}\d[A-Z0-9]*\s*-\s*(.+?)\s*-\s*Google Patents",
        text[:1000],
        flags=re.IGNORECASE | re.DOTALL,
    )
    if google_title_match:
        title = re.sub(r"\s+", " ", google_title_match.group(1)).strip(" -")
        if title:
            return title

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    skip_prefixes = (
        "abstract",
        "claims",
        "description",
        "요약",
        "초록",
        "청구항",
        "발명의 설명",
        "要約",
        "請求",
        "発明",
    )
    for line in lines[:35]:
        if 8 <= len(line) <= 180 and not line.lower().startswith(skip_prefixes):
            return line
    return None


def extract_claims(text: str) -> list[str]:
    preferred_start = re.search(r"\bWhat is claimed is:?", text, re.IGNORECASE)
    claim_start = preferred_start or SECTION_PATTERNS["claim_start"].search(text)
    if not claim_start:
        return []
    claim_text = text[claim_start.end() :]
    description_match = SECTION_PATTERNS["description"].search(claim_text)
    if description_match:
        claim_text = claim_text[: description_match.start()]
    claim_text = re.sub(r"^\s*\(\s*\d+\s*\)\s*", "", claim_text)

    localized_parts = re.split(
        r"\s*(?=(?:청구항\s*\d+\s*에\s*있어서|제\s*\d+\s*항|【\s*請求項\s*\d+\s*】|請求項\s*\d+))",
        claim_text,
    )
    localized_claims = []
    for idx, part in enumerate(localized_parts):
        cleaned = re.sub(r"\s+", " ", part).strip()
        if not cleaned or cleaned in {"삭제", "Deleted"}:
            continue
        if idx == 0 and len(cleaned) > 40:
            localized_claims.append(f"1. {cleaned}")
        elif re.match(r"^(청구항\s*\d+\s*에\s*있어서|제\s*\d+\s*항|【\s*請求項\s*\d+\s*】|請求項\s*\d+)", cleaned):
            localized_claims.append(cleaned)
    if localized_claims:
        return localized_claims[:50]

    inline_claims = re.findall(
        r"(?:^|\s)(\d+\.\s+.*?)(?=\s+\d+\.\s+[A-Z가-힣ぁ-んァ-ン一-龥]|\s+\*+\s+\*+\s+\*+|$)",
        claim_text,
        flags=re.DOTALL,
    )
    if inline_claims:
        return [re.sub(r"\s+", " ", claim).strip() for claim in inline_claims if len(claim.strip()) > 30][:50]

    chunks = re.split(r"\n\s*(?=\d+\.\s+)", claim_text)
    claims = []
    for chunk in chunks:
        cleaned = re.sub(r"\s+", " ", chunk).strip()
        if re.match(r"^\d+\.\s+", cleaned) and len(cleaned) > 30:
            claims.append(cleaned)
    return claims[:50]


def extract_abstract(text: str) -> str | None:
    match = SECTION_PATTERNS["abstract"].search(text)
    if not match:
        return None
    tail = text[match.end() :]
    next_section_positions = [
        found.start()
        for key, pattern in SECTION_PATTERNS.items()
        if key != "abstract"
        for found in [pattern.search(tail)]
        if found
    ]
    end = min(next_section_positions) if next_section_positions else min(len(tail), 2500)
    abstract = re.sub(r"\s+", " ", tail[:end]).strip()
    return abstract[:2000] if abstract else None


def extract_description(text: str) -> str | None:
    match = SECTION_PATTERNS["description"].search(text)
    if not match:
        return None
    tail = text[match.end() :]
    claim_match = SECTION_PATTERNS["claim_start"].search(tail)
    if claim_match:
        tail = tail[: claim_match.start()]
    description = re.sub(r"\s+", " ", tail).strip()
    return description[:12000] if description else None


def _clean_party_name(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" -:;,")
    stop_words = [
        "Priority",
        "Filing",
        "Publication",
        "Application",
        "Inventor",
        "Classifications",
        "Links",
        "Images",
        "Abstract",
        "Claims",
        "Original Assignee",
        "Current Assignee",
        "발명자",
        "출원",
        "공개",
        "등록",
        "분류",
        "요약",
        "発明者",
        "出願",
        "公開",
        "登録",
        "分類",
        "要約",
    ]
    for word in stop_words:
        idx = value.find(word)
        if idx > 0:
            value = value[:idx].strip(" -:;,")
    return value[:160]


def extract_assignee_or_representative(text: str) -> tuple[str | None, str | None]:
    compact = re.sub(r"\s+", " ", text)
    assignee_patterns = [
        r"(?:Original Assignee|Current Assignee)\s+(.+?)(?=\s+(?:Priority|Filing|Publication|Application|Inventor|Classifications|Links|Images|Abstract|Claims)\b)",
        r"(?:Applicant|Assignee)\s+(.+?)(?=\s+(?:Inventor|Priority|Filing|Publication|Application|Classifications|Abstract|Claims)\b)",
        r"(?:출원인|권리자)\s+(.+?)(?=\s+(?:발명자|우선권|출원|공개|등록|분류|요약|청구항)\b)",
        r"(?:出願人|権利者)\s+(.+?)(?=\s+(?:発明者|優先日|出願|公開|登録|分類|要約|請求)\b)",
    ]
    for pattern in assignee_patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            candidate = _clean_party_name(match.group(1))
            if candidate:
                return candidate, None

    representative_patterns = [
        r"Inventor\s+(.+?)(?=\s+(?:Original Assignee|Current Assignee|Priority|Filing|Publication|Application|Classifications)\b)",
        r"발명자\s+(.+?)(?=\s+(?:출원인|권리자|우선권|출원|공개|등록|분류)\b)",
        r"発明者\s+(.+?)(?=\s+(?:出願人|権利者|優先日|出願|公開|登録|分類)\b)",
    ]
    for pattern in representative_patterns:
        match = re.search(pattern, compact, re.IGNORECASE)
        if match:
            candidate = _clean_party_name(match.group(1)).split(",")[0].strip()
            if candidate:
                return None, candidate
    return None, None


def structure_document(document: PatentDocument) -> PatentDocument:
    text = document.raw_text
    document.title = document.title or guess_title(text)
    document.abstract = document.abstract or extract_abstract(text)
    document.claims = document.claims or extract_claims(text)
    document.description = document.description or extract_description(text)
    assignee, representative = extract_assignee_or_representative(text)
    document.assignee = document.assignee or assignee
    document.representative = document.representative or representative
    return document
