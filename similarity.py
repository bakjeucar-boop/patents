from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.models import PairwiseComparison, PatentAnalysis


def build_similarity_matrix(analyses: list[PatentAnalysis]) -> list[list[float]]:
    if len(analyses) < 2:
        return [[1.0]]
    corpus = [
        " ".join(
            [
                item.title,
                item.one_line_summary,
                item.simple_explanation,
                item.problem_to_solve,
                " ".join(item.differentiators),
            ]
        )
        for item in analyses
    ]
    vectors = TfidfVectorizer(max_features=3000).fit_transform(corpus)
    matrix = cosine_similarity(vectors)
    return matrix.round(3).tolist()


def similarity_level(score: float) -> str:
    if score >= 0.75:
        return "very_high"
    if score >= 0.55:
        return "high"
    if score >= 0.35:
        return "medium"
    if score >= 0.18:
        return "low"
    return "very_low"


def similarity_level_ko(score: float) -> str:
    return {
        "very_high": "매우 높음",
        "high": "높음",
        "medium": "보통",
        "low": "낮음",
        "very_low": "매우 낮음",
    }[similarity_level(score)]


def fallback_pairwise_comparison(
    patent_a: PatentAnalysis,
    patent_b: PatentAnalysis,
    similarity_score: float,
) -> PairwiseComparison:
    level = similarity_level(similarity_score)
    if level in {"high", "very_high"}:
        reason = "핵심 요약, 해결 문제, 차별점 표현이 많이 겹쳐 기술 방향성이 유사하게 계산되었습니다."
    elif level == "medium":
        reason = "일부 기술 목적이나 구성요소는 겹치지만, 보호하려는 핵심 구성 또는 구현 방식에는 차이가 있는 것으로 계산되었습니다."
    else:
        reason = "같은 넓은 기술 분야에 속하더라도 핵심 문제, 구성요소, 차별점의 축이 달라 낮은 유사도로 계산되었습니다."

    common_points = []
    for item in patent_a.differentiators:
        if any(word in " ".join(patent_b.differentiators) for word in item.split()[:3]):
            common_points.append(item)
    if not common_points:
        common_points = ["두 특허 모두 기술적 문제 해결을 위한 구체적 구성 조합을 청구항에서 보호하려는 문헌입니다."]

    return PairwiseComparison(
        patent_a=patent_a.title,
        patent_b=patent_b.title,
        similarity_score=similarity_score,
        similarity_level=level,  # type: ignore[arg-type]
        score_reason=reason,
        common_points=common_points[:3],
        key_differences=[
            f"특허 A의 핵심: {patent_a.one_line_summary}",
            f"특허 B의 핵심: {patent_b.one_line_summary}",
        ],
        practical_interpretation=(
            "이 점수는 법적 유사성이나 침해 가능성을 직접 의미하지 않습니다. "
            "엔지니어링 관점에서 두 특허의 문제 설정과 핵심 구성 설명이 얼마나 가까운지를 나타냅니다."
        ),
        recommended_review_points=[
            "각 특허의 독립항 필수 구성요소가 실제로 겹치는지 확인",
            "같은 냉각 대상인지, 같은 냉각 매체/유로/제어 원리인지 확인",
            "우선일과 출원인 기준으로 선후관계 확인",
        ],
        confidence="medium",
        notes=["LLM 비교 설명이 아닌 로컬 기본 비교 설명입니다."],
    )


def build_pair_indices(count: int, max_pairs: int = 15) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(count):
        for j in range(i + 1, count):
            pairs.append((i, j))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs
