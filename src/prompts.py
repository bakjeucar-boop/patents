from __future__ import annotations

import json
import re

from src.models import PatentAnalysis, PatentDocument


def _clip(text: str | None, limit: int) -> str:
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", text).strip()
    return compact[:limit]


def build_single_patent_prompt(document: PatentDocument, max_chars: int) -> str:
    claims_text = "\n".join(document.claims[:8])
    source_text = "\n\n".join(
        [
            f"ORIGINAL PATENT TITLE: {document.title or ''}",
            f"ABSTRACT: {_clip(document.abstract, 5000)}",
            f"CLAIMS: {_clip(claims_text, 18000)}",
            f"DESCRIPTION EXCERPT: {_clip(document.raw_text, max_chars)}",
        ]
    )
    return f"""
You are a patent technology analysis assistant for R&D engineers.
Analyze the patent technically, not as a legal opinion.

Important goals:
- Explain why this invention may have been patentable.
- Focus on the independent claim and essential technical elements.
- Explain in plain Korean for engineers who are not patent attorneys.
- Do not decide infringement. Provide only technical risk and design-around review points.
- Avoid generic summaries. Identify the invention's "protected technical idea".
- Explain the gap between the prior/common approach and this invention.
- Separate claim-limited elements from broader engineering advantages.
- If the text is incomplete or uncertain, say so in notes instead of guessing.
- The "title" field must be a concise Korean technical title only.
- Do not include patent numbers such as US10257963B2 in the "title" field.
- Translate English, Chinese, Japanese, German, or other foreign titles into Korean.
- The title must be a faithful, literal Korean translation of ORIGINAL PATENT TITLE.
- Never infer or rewrite the title from the abstract, claims, cooling method, or other body content.
- Do not add technical terms, mechanisms, or advantages that are absent from ORIGINAL PATENT TITLE.
- Do not use LaTeX commands or backslash escapes. Write mathematical relations as plain text such as T2 >= T1.

Return ONLY valid JSON with this exact shape:
{{
  "title": "concise Korean technical title only, without patent number",
  "one_line_summary": "string",
  "simple_explanation": "Explain the invention in easy Korean in 3-5 sentences",
  "problem_to_solve": "What technical problem this patent tries to solve",
  "why_patentable": "Why this may have become a patent: prior limitation + new combination + technical effect",
  "differentiators": ["specific technical differentiator, not a generic benefit"],
  "key_claim_elements": [
    {{
      "label": "A",
      "text": "claim element text",
      "plain_explanation": "easy Korean explanation"
    }}
  ],
  "applications": ["realistic product/system/process applications"],
  "design_around_points": ["technical points a competitor may inspect for design-around, without legal conclusion"],
  "confidence": "low|medium|high",
  "notes": ["string"]
}}

Patent source:
{source_text}
""".strip()


def _analysis_digest(item: PatentAnalysis) -> str:
    claim_elements = "\n".join(
        f"- {element.label}: {element.text} / {element.plain_explanation}"
        for element in item.key_claim_elements
    )
    return "\n".join(
        [
            f"TITLE: {item.title}",
            f"SUMMARY: {item.one_line_summary}",
            f"SIMPLE EXPLANATION: {item.simple_explanation}",
            f"PROBLEM: {item.problem_to_solve}",
            f"WHY PATENTABLE: {item.why_patentable}",
            f"DIFFERENTIATORS: {'; '.join(item.differentiators)}",
            f"CLAIM ELEMENTS:\n{claim_elements}",
            f"APPLICATIONS: {'; '.join(item.applications)}",
            f"DESIGN-AROUND POINTS: {'; '.join(item.design_around_points)}",
        ]
    )


def build_pairwise_comparison_prompt(
    patent_a: PatentAnalysis,
    patent_b: PatentAnalysis,
    similarity_score: float,
) -> str:
    return f"""
You are comparing two patent technology analyses for R&D engineers.
The numeric similarity score was calculated by text similarity over summaries, problems, and differentiators.
Explain what the score means in practical engineering terms.

Rules:
- Answer in Korean.
- Do not provide legal infringement conclusions.
- Explain why the score is high/medium/low.
- Distinguish "same technical field" from "same protected claim idea".
- Focus on common technical problem, claim elements, operating principle, and differentiators.
- If the score is low despite both being cooling patents, explain the axis of difference.

Similarity score: {similarity_score:.3f}

Patent A:
{_analysis_digest(patent_a)}

Patent B:
{_analysis_digest(patent_b)}

Return ONLY valid JSON with this exact shape:
{{
  "patent_a": "{patent_a.title}",
  "patent_b": "{patent_b.title}",
  "similarity_score": {similarity_score:.3f},
  "similarity_level": "very_low|low|medium|high|very_high",
  "score_reason": "why the numeric score is interpreted this way",
  "common_points": ["specific common technical point"],
  "key_differences": ["specific technical or claim-scope difference"],
  "practical_interpretation": "what an engineer should understand from the comparison",
  "recommended_review_points": ["what to review next"],
  "confidence": "low|medium|high",
  "notes": ["uncertainties or caveats"]
}}
""".strip()


def build_batch_summary_prompt(analyses: list[PatentAnalysis], matrix: list[list[float]]) -> str:
    patent_blocks = []
    for idx, analysis in enumerate(analyses, start=1):
        patent_blocks.append(
            "\n".join(
                [
                    f"P{idx}: {analysis.title}",
                    f"SUMMARY: {analysis.one_line_summary}",
                    f"PROBLEM: {analysis.problem_to_solve}",
                    f"DIFFERENTIATORS: {'; '.join(analysis.differentiators[:5])}",
                ]
            )
        )
    matrix_rows = [", ".join(f"{score:.3f}" for score in row) for row in matrix]
    return f"""
You are summarizing a multi-patent technology landscape for R&D engineers.
Explain the overall relationship once, without performing a detailed pair-by-pair analysis.
Use Korean. Treat numeric similarity as a reference signal, not a legal conclusion.

Patents:
{chr(10).join(patent_blocks)}

Similarity matrix:
{chr(10).join(matrix_rows)}

Return ONLY valid JSON:
{{
  "overview": "overall technology landscape",
  "major_groups": ["major cluster or category"],
  "notable_relationships": ["important high or low relationship using P1/P2 labels"],
  "interpretation_notes": ["how engineers should read the matrix"],
  "confidence": "low|medium|high"
}}
""".strip()


def build_multi_patent_comparison_prompt(
    labels: list[str],
    analyses: list[PatentAnalysis],
) -> str:
    blocks = [f"{label}\n{_analysis_digest(analysis)}" for label, analysis in zip(labels, analyses)]
    return f"""
You are performing a focused comparison of {len(analyses)} selected patents for R&D engineers.
Compare all selected patents together in one analysis, not as a series of pairwise comparisons.

Rules:
- Answer in Korean.
- Always refer to patents using the supplied labels such as P1, P3, P5.
- Identify the common technical foundation first.
- Then explain each patent's technical focus and distinctive claim idea.
- Explain relationships, overlaps, and important differences across the whole group.
- Include technical design-around insights, but do not conclude legal infringement.
- Avoid repeating the same sentence for each patent.

Selected patents:
{chr(10).join(blocks)}

Return ONLY valid JSON:
{{
  "overview": "overall comparison of the selected patents",
  "common_points": ["shared technical problem, mechanism, or component"],
  "patent_positions": [
    {{
      "label": "P1",
      "technology_focus": "main technical position of this patent",
      "distinctive_points": ["specific differentiator"]
    }}
  ],
  "key_relationships": ["relationship or contrast using P labels"],
  "design_around_insights": ["technical design-around insight across the selected group"],
  "confidence": "low|medium|high"
}}
""".strip()


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    parsed = _load_json_with_repairs(cleaned)
    if parsed is not None:
        return parsed

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response did not contain a JSON object.")
    candidate = cleaned[start : end + 1]
    parsed = _load_json_with_repairs(candidate)
    if parsed is None:
        raise ValueError("LLM response contained JSON that could not be repaired.")
    return parsed


def _load_json_with_repairs(candidate: str) -> dict | None:
    attempts = [candidate]
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', candidate)
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    attempts.append(repaired)
    for attempt in attempts:
        try:
            value = json.loads(attempt, strict=False)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    return None
