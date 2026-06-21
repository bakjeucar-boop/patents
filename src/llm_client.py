from __future__ import annotations

from pydantic import ValidationError

from src.analyzer import analyze_without_llm
from src.config import Settings
from src.models import BatchComparisonSummary, MultiPatentComparison, PairwiseComparison, PatentAnalysis, PatentDocument, PatentPosition
from src.prompts import (
    build_batch_summary_prompt,
    build_multi_patent_comparison_prompt,
    build_pairwise_comparison_prompt,
    build_single_patent_prompt,
    extract_json_object,
)
from src.similarity import fallback_pairwise_comparison


class LlmClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.gemma_api_key)

    def analyze_patent(self, document: PatentDocument, max_input_chars: int | None = None) -> PatentAnalysis:
        if not self.is_configured:
            return analyze_without_llm(document)

        try:
            if self.settings.gemma_api_provider.lower() == "google":
                return self._analyze_with_google_genai(document, max_input_chars=max_input_chars)
        except Exception as exc:
            fallback = analyze_without_llm(document)
            fallback.notes.append(f"LLM API 호출 실패로 기본 분석을 사용했습니다: {exc}")
            return fallback

        fallback = analyze_without_llm(document)
        fallback.notes.append(f"지원하지 않는 LLM provider입니다: {self.settings.gemma_api_provider}")
        return fallback

    def _analyze_with_google_genai(
        self,
        document: PatentDocument,
        max_input_chars: int | None = None,
    ) -> PatentAnalysis:
        import httpx
        from google import genai
        from google.genai import types

        prompt = build_single_patent_prompt(document, max_input_chars or self.settings.gemma_max_input_chars)
        httpx_client = httpx.Client(trust_env=False, timeout=120)
        client = genai.Client(
            api_key=self.settings.gemma_api_key,
            http_options=types.HttpOptions(httpx_client=httpx_client),
        )
        last_error: Exception | None = None
        for attempt in range(2):
            contents = prompt
            if attempt == 1:
                contents += "\nCRITICAL: Return strict valid JSON only. Do not use backslashes or trailing commas."
            response = client.models.generate_content(
                model=self.settings.gemma_model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=self.settings.gemma_temperature if attempt == 0 else 0.0,
                    response_mime_type="application/json",
                ),
            )
            try:
                data = extract_json_object(response.text or "")
                analysis = PatentAnalysis.model_validate(data)
                analysis.notes.append(
                    f"LLM 분석 사용: provider={self.settings.gemma_api_provider}, model={self.settings.gemma_model}"
                )
                if attempt == 1:
                    analysis.notes.append("초기 JSON 응답 오류 후 자동 재시도로 복구했습니다.")
                return analysis
            except (ValidationError, ValueError) as exc:
                last_error = exc
        raise ValueError(f"LLM JSON schema validation failed after retry: {last_error}")

    def compare_patents(
        self,
        patent_a: PatentAnalysis,
        patent_b: PatentAnalysis,
        similarity_score: float,
    ) -> PairwiseComparison:
        if not self.is_configured:
            return fallback_pairwise_comparison(patent_a, patent_b, similarity_score)

        try:
            if self.settings.gemma_api_provider.lower() == "google":
                return self._compare_with_google_genai(patent_a, patent_b, similarity_score)
        except Exception as exc:
            fallback = fallback_pairwise_comparison(patent_a, patent_b, similarity_score)
            fallback.notes.append(f"LLM 비교 호출 실패로 기본 비교를 사용했습니다: {exc}")
            return fallback

        return fallback_pairwise_comparison(patent_a, patent_b, similarity_score)

    def _compare_with_google_genai(
        self,
        patent_a: PatentAnalysis,
        patent_b: PatentAnalysis,
        similarity_score: float,
    ) -> PairwiseComparison:
        import httpx
        from google import genai
        from google.genai import types

        prompt = build_pairwise_comparison_prompt(patent_a, patent_b, similarity_score)
        httpx_client = httpx.Client(trust_env=False, timeout=120)
        client = genai.Client(
            api_key=self.settings.gemma_api_key,
            http_options=types.HttpOptions(httpx_client=httpx_client),
        )
        response = client.models.generate_content(
            model=self.settings.gemma_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.settings.gemma_temperature,
                response_mime_type="application/json",
            ),
        )
        data = extract_json_object(response.text or "")
        comparison = PairwiseComparison.model_validate(data)
        comparison.notes.append(
            f"LLM 비교 사용: provider={self.settings.gemma_api_provider}, model={self.settings.gemma_model}"
        )
        return comparison

    def summarize_batch(
        self,
        analyses: list[PatentAnalysis],
        matrix: list[list[float]],
    ) -> BatchComparisonSummary:
        if not self.is_configured:
            return self._fallback_batch_summary(analyses, matrix)
        try:
            if self.settings.gemma_api_provider.lower() == "google":
                return self._summarize_batch_with_google(analyses, matrix)
        except Exception:
            return self._fallback_batch_summary(analyses, matrix)
        return self._fallback_batch_summary(analyses, matrix)

    def _summarize_batch_with_google(
        self,
        analyses: list[PatentAnalysis],
        matrix: list[list[float]],
    ) -> BatchComparisonSummary:
        import httpx
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self.settings.gemma_api_key,
            http_options=types.HttpOptions(httpx_client=httpx.Client(trust_env=False, timeout=120)),
        )
        response = client.models.generate_content(
            model=self.settings.gemma_model,
            contents=build_batch_summary_prompt(analyses, matrix),
            config=types.GenerateContentConfig(
                temperature=self.settings.gemma_temperature,
                response_mime_type="application/json",
            ),
        )
        return BatchComparisonSummary.model_validate(extract_json_object(response.text or ""))

    @staticmethod
    def _fallback_batch_summary(
        analyses: list[PatentAnalysis],
        matrix: list[list[float]],
    ) -> BatchComparisonSummary:
        pairs = []
        for i in range(len(analyses)):
            for j in range(i + 1, len(analyses)):
                pairs.append((matrix[i][j], i, j))
        pairs.sort(reverse=True)
        notable = [f"P{i + 1}-P{j + 1}: 유사도 {score:.3f}" for score, i, j in pairs[:5]]
        return BatchComparisonSummary(
            overview=f"총 {len(analyses)}건의 특허를 요약, 해결 문제, 차별점 기준으로 비교했습니다.",
            major_groups=["상세 기술 그룹은 LLM 연결 또는 사용자 선택 비교에서 확인할 수 있습니다."],
            notable_relationships=notable,
            interpretation_notes=["유사도는 기술 탐색용 참고 지표이며 법적 유사성이나 침해 가능성을 의미하지 않습니다."],
            confidence="medium",
        )

    def compare_multiple(
        self,
        labels: list[str],
        analyses: list[PatentAnalysis],
    ) -> MultiPatentComparison:
        if not self.is_configured:
            return self._fallback_multi_comparison(labels, analyses)
        try:
            if self.settings.gemma_api_provider.lower() == "google":
                return self._compare_multiple_with_google(labels, analyses)
        except Exception:
            return self._fallback_multi_comparison(labels, analyses)
        return self._fallback_multi_comparison(labels, analyses)

    def _compare_multiple_with_google(
        self,
        labels: list[str],
        analyses: list[PatentAnalysis],
    ) -> MultiPatentComparison:
        import httpx
        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=self.settings.gemma_api_key,
            http_options=types.HttpOptions(httpx_client=httpx.Client(trust_env=False, timeout=120)),
        )
        response = client.models.generate_content(
            model=self.settings.gemma_model,
            contents=build_multi_patent_comparison_prompt(labels, analyses),
            config=types.GenerateContentConfig(
                temperature=self.settings.gemma_temperature,
                response_mime_type="application/json",
            ),
        )
        return MultiPatentComparison.model_validate(extract_json_object(response.text or ""))

    @staticmethod
    def _fallback_multi_comparison(
        labels: list[str],
        analyses: list[PatentAnalysis],
    ) -> MultiPatentComparison:
        positions = [
            PatentPosition(
                label=label,
                technology_focus=analysis.one_line_summary,
                distinctive_points=analysis.differentiators[:3],
            )
            for label, analysis in zip(labels, analyses)
        ]
        return MultiPatentComparison(
            overview=f"선택한 {len(analyses)}건 특허의 핵심 기술과 차별점을 함께 비교했습니다.",
            common_points=["선택 특허의 공통점은 상세 LLM 비교 결과에서 확인할 수 있습니다."],
            patent_positions=positions,
            key_relationships=["각 특허의 해결 문제와 독립항 구성요소를 기준으로 추가 검토가 필요합니다."],
            design_around_insights=["공통 필수 구성요소와 각 특허 고유 구성요소를 구분해 대체 가능성을 검토합니다."],
            confidence="medium",
        )
