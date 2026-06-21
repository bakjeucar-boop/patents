from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st

from src.config import OUTPUT_DIR, ROOT_DIR, UPLOAD_DIR, ensure_directories, load_settings
from src.document_loader import (
    extract_patent_number,
    extract_pdf_text,
    fetch_url_text,
    google_patents_url,
    parse_multiline_inputs,
    save_uploaded_file,
)
from src.llm_client import LlmClient
from src.models import AnalysisMode, InputKind, PatentDocument, PatentInput
from src.patent_parser import structure_document
from src.report import build_docx_report, build_markdown_report, save_analysis
from src.similarity import build_similarity_matrix, similarity_level_ko


st.set_page_config(page_title="특허 검토 Agent", layout="wide")
ensure_directories()


def collect_inputs() -> list[PatentInput]:
    inputs: list[PatentInput] = []
    raw_entries = st.text_area(
        "🔎 특허번호 또는 URL 입력",
        placeholder="US1234567B2\nhttps://patents.google.com/patent/JP7727724B2/\nCN109876543A",
        height=140,
    )
    inputs.extend(parse_multiline_inputs(raw_entries))

    uploaded_files = st.file_uploader("📄 특허 PDF 업로드", type=["pdf"], accept_multiple_files=True)
    for file in uploaded_files:
        saved_path = save_uploaded_file(UPLOAD_DIR, file.name, file.getvalue())
        inputs.append(PatentInput(kind=InputKind.pdf, value=str(saved_path), label=file.name))
    return inputs


def depth_to_max_input_chars(depth: str, default_chars: int) -> int:
    if depth == "빠른 검토":
        return min(default_chars, 18000)
    if depth == "정밀 검토":
        return max(default_chars, 80000)
    return default_chars


def load_document(item: PatentInput) -> PatentDocument:
    if item.kind == InputKind.pdf:
        raw_text = extract_pdf_text(UPLOAD_DIR.parent / item.value if not item.value else item.value)
    elif item.kind == InputKind.url:
        raw_text = fetch_url_text(item.value)
    elif item.kind == InputKind.patent_number:
        raw_text = fetch_url_text(google_patents_url(item.value))
    else:
        raw_text = item.value
    document = structure_document(PatentDocument(source=item, raw_text=raw_text))
    document.publication_number = document.publication_number or extract_patent_number(item.value)
    return document


def compact_label(index: int) -> str:
    return f"P{index + 1}"


def title_without_patent_number(title: str, patent_number: str | None) -> str:
    cleaned = title.strip()
    if patent_number:
        cleaned = cleaned.replace(patent_number, "").strip(" -:|")
    return cleaned or title


def display_title(document: PatentDocument, analysis) -> str:
    return title_without_patent_number(analysis.title, document.publication_number)


def build_patent_display_rows(documents: list[PatentDocument], analyses) -> list[dict[str, str]]:
    rows = []
    for idx, (document, analysis) in enumerate(zip(documents, analyses)):
        rows.append(
            {
                "ID": compact_label(idx),
                "특허번호": document.publication_number or "-",
                "제목": title_without_patent_number(analysis.title, document.publication_number),
                "요약": analysis.one_line_summary,
            }
        )
    return rows


def looks_garbled(text: str) -> bool:
    if not text:
        return True
    sample = text[:5000]
    replacement_count = sample.count("\ufffd")
    readable_count = len(re.findall(r"[A-Za-z0-9가-힣ぁ-んァ-ン一-龥。、，．・ー\s]", sample))
    ratio = readable_count / max(len(sample), 1)
    return replacement_count > 10 or ratio < 0.45


def relabel_comparison_text(text: str, label_a: str, label_b: str) -> str:
    replacements = {
        "특허 A": label_a,
        "특허A": label_a,
        "Patent A": label_a,
        "patent A": label_a,
        "A의": f"{label_a}의",
        "특허 B": label_b,
        "특허B": label_b,
        "Patent B": label_b,
        "patent B": label_b,
        "B의": f"{label_b}의",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_single_result(document: PatentDocument, analysis) -> None:
    st.subheader(f"📌 {display_title(document, analysis)}")
    if document.title and document.title.strip() != analysis.title.strip():
        st.caption(f"원제: {document.title}")
    api_failures = [note for note in analysis.notes if "LLM API 호출 실패" in note]
    if api_failures:
        st.warning("AI 분석 응답 형식에 문제가 있어 기본 분석 결과를 표시했습니다. 다시 분석하면 정상 처리될 수 있습니다.")
    st.info(analysis.one_line_summary)
    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### 🧭 핵심 요약")
        st.write(analysis.simple_explanation)
        st.markdown("#### 💡 왜 특허가 되었는가")
        st.write(analysis.why_patentable)
    with cols[1]:
        st.markdown("#### 🎯 해결하려는 문제")
        st.write(analysis.problem_to_solve)
        st.markdown("#### 🔬 기술 차별점")
        for item in analysis.differentiators:
            st.write(f"- {item}")

    st.markdown("#### 🧩 청구항 구성요소")
    if analysis.key_claim_elements:
        st.dataframe(
            pd.DataFrame([element.model_dump() for element in analysis.key_claim_elements]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("청구항을 자동 추출하지 못했습니다. PDF 텍스트 품질 또는 문서 구조를 확인해 주세요.")

    st.markdown("#### 🛠️ 회피설계 검토 포인트")
    if analysis.design_around_points:
        for item in analysis.design_around_points:
            st.write(f"- {item}")
    else:
        st.write("- 별도 검토 포인트가 추출되지 않았습니다.")

    with st.expander("🧾 추출 원문 미리보기"):
        st.caption(f"총 추출 글자 수: {len(document.raw_text):,}자")
        if document.source.kind == InputKind.pdf and looks_garbled(document.raw_text):
            st.warning(
                "PDF 내부 문자 매핑 문제로 원문이 깨져 보일 수 있습니다. 특히 일본어 PDF에서 자주 발생합니다. "
                "가능하면 Google Patents URL을 입력하거나, OCR 기반 추출 기능을 추가하는 것이 좋습니다."
            )
        if document.abstract:
            st.text("... (원문 앞부분 생략) ...")
            st.text(document.abstract)
            st.text("... (원문 뒷부분 생략) ...")
        elif document.description:
            st.info("Abstract가 없어 Description 시작부분을 표시합니다.")
            st.text("... (Description 이전 내용 생략) ...")
            st.text(document.description[:2000])
            st.text("... (Description 이후 내용 생략) ...")
        else:
            st.warning("Abstract와 Description을 찾지 못해 추출 원문 앞부분을 대신 표시합니다.")
            st.text("... (이전 내용 생략) ...")
            st.text(document.raw_text[:2000])
            st.text("... (이후 내용 생략) ...")


def render_selected_multi_comparison(documents, analyses, llm_client: LlmClient) -> None:
    st.markdown("#### 🔍 선택 특허 종합 비교")
    options = list(range(len(analyses)))
    default_selection = options[: min(2, len(options))]
    selected = st.multiselect(
        "비교할 특허 선택",
        options,
        default=default_selection,
        max_selections=5,
        format_func=lambda idx: f"{compact_label(idx)} | {display_title(documents[idx], analyses[idx])}",
        key="multi_compare_selection",
    )

    if len(selected) < 2:
        st.info("비교할 특허를 2건 이상 선택해 주세요.")
        return

    key = "-".join(str(idx) for idx in selected)
    if st.button("선택 특허 종합 비교", key="run_multi_comparison"):
        labels = [compact_label(idx) for idx in selected]
        selected_analyses = [analyses[idx] for idx in selected]
        comparison = llm_client.compare_multiple(labels, selected_analyses)
        st.session_state.setdefault("multi_comparisons", {})[key] = comparison

    comparison = st.session_state.get("multi_comparisons", {}).get(key)
    if not comparison:
        return

    st.markdown("**🧭 종합 해석**")
    st.write(comparison.overview)
    st.markdown("**🔗 공통점**")
    for item in comparison.common_points:
        st.write(f"- {item}")
    st.markdown("**🧬 특허별 기술 포지션**")
    for position in comparison.patent_positions:
        with st.expander(f"{position.label} | {position.technology_focus}"):
            for item in position.distinctive_points:
                st.write(f"- {item}")
    st.markdown("**🔄 핵심 관계와 차이**")
    for item in comparison.key_relationships:
        st.write(f"- {item}")
    st.markdown("**🛠️ 회피설계 관점**")
    for item in comparison.design_around_insights:
        st.write(f"- {item}")


def render_ai_status(llm_client: LlmClient) -> None:
    st.sidebar.divider()
    if llm_client.is_configured:
        st.sidebar.success("🟢 AI API 연결 성공")
    else:
        st.sidebar.error("🔴 AI API 연결 실패")
        with st.sidebar.expander("API 키 설정 방법"):
            st.write("실제 LLM 분석을 사용하려면 프로젝트 루트에 `.env` 파일을 만듭니다.")
            st.code(
                f"{ROOT_DIR / '.env'}\n\n"
                "GEMMA_API_PROVIDER=google\n"
                "GEMMA_API_KEY=your_api_key_here\n"
                "GEMMA_MODEL=models/gemma-4-31b-it\n"
                "GEMMA_TEMPERATURE=0.2\n"
                "GEMMA_MAX_INPUT_CHARS=45000",
                language="text",
            )


def run_analysis(inputs: list[PatentInput], mode: AnalysisMode, max_input_chars: int, llm_client: LlmClient) -> None:
    documents: list[PatentDocument] = []
    analyses = []
    progress = st.progress(0, text="분석 준비 중")

    for idx, item in enumerate(inputs, start=1):
        progress.progress((idx - 1) / len(inputs), text=f"{item.label} 문서 수집 및 분석 중")
        try:
            document = load_document(item)
            analysis = llm_client.analyze_patent(document, max_input_chars=max_input_chars)
            save_analysis(OUTPUT_DIR, document, analysis)
            documents.append(document)
            analyses.append(analysis)
        except Exception as exc:
            st.error(f"{item.label} 처리 실패: {exc}")

    progress.progress(1.0, text="분석 완료")
    st.session_state["last_documents"] = documents
    st.session_state["last_analyses"] = analyses
    st.session_state["last_mode"] = mode
    st.session_state["report_timestamp"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state["multi_comparisons"] = {}
    if len(analyses) >= 2:
        matrix = build_similarity_matrix(analyses)
        st.session_state["batch_matrix"] = matrix
        st.session_state["batch_summary"] = llm_client.summarize_batch(analyses, matrix)
    else:
        st.session_state.pop("batch_matrix", None)
        st.session_state.pop("batch_summary", None)


def render_results(documents: list[PatentDocument], analyses, mode: AnalysisMode, analysis_goals, llm_client: LlmClient) -> None:
    if not analyses:
        return

    tabs = st.tabs(["📝 요약", "🔬 특허 분석", "📊 복수 비교", "📤 내보내기"])
    with tabs[0]:
        st.dataframe(
            pd.DataFrame(build_patent_display_rows(documents, analyses)),
            use_container_width=True,
            hide_index=True,
        )

    with tabs[1]:
        for document, analysis in zip(documents, analyses):
            with st.expander(display_title(document, analysis)):
                render_single_result(document, analysis)

    with tabs[2]:
        if len(analyses) < 2:
            st.info("복수 특허를 입력하면 유사도 비교 기능이 활성화됩니다.")
        elif not analysis_goals["유사도 비교"]:
            st.info("사이드바에서 '유사도 비교'를 선택하면 복수 비교 결과가 표시됩니다.")
        else:
            matrix = st.session_state.get("batch_matrix") or build_similarity_matrix(analyses)
            st.markdown("#### 🏷️ 특허 ID")
            st.dataframe(
                pd.DataFrame(build_patent_display_rows(documents, analyses)),
                use_container_width=True,
                hide_index=True,
            )

            labels = [compact_label(idx) for idx in range(len(analyses))]
            level_matrix = [
                ["-" if i == j else similarity_level_ko(float(matrix[i][j])) for j in range(len(analyses))]
                for i in range(len(analyses))
            ]
            st.markdown("#### 📐 유사도 단계")
            st.dataframe(pd.DataFrame(level_matrix, index=labels, columns=labels), use_container_width=True)
            with st.expander("숫자 점수 보기"):
                st.dataframe(pd.DataFrame(matrix, index=labels, columns=labels), use_container_width=True)

            summary = st.session_state.get("batch_summary")
            if summary:
                st.markdown("#### 🧭 전체 비교 요약")
                st.write(summary.overview)
                if summary.major_groups:
                    st.markdown("**주요 기술 그룹**")
                    for item in summary.major_groups:
                        st.write(f"- {item}")
                if summary.notable_relationships:
                    st.markdown("**주목할 관계**")
                    for item in summary.notable_relationships:
                        st.write(f"- {item}")

            render_selected_multi_comparison(documents, analyses, llm_client)

    with tabs[3]:
        st.markdown("#### 📤 분석 결과 내보내기")
        markdown_report = build_markdown_report(documents, analyses)
        docx_report = build_docx_report(documents, analyses)
        timestamp = st.session_state.get("report_timestamp") or datetime.now().strftime("%Y%m%d_%H%M%S")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "워드파일 내보내기",
                data=docx_report,
                file_name=f"patent_review_report_{timestamp}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="download_docx",
            )
        with col2:
            st.download_button(
                "MD 파일 내보내기",
                data=markdown_report.encode("utf-8"),
                file_name=f"patent_review_report_{timestamp}.md",
                mime="text/markdown",
                key="download_md",
            )
        st.caption("파일은 웹브라우저의 기본 다운로드 폴더에 저장됩니다.")


def main() -> None:
    st.title("📑 특허 검토 Agent")
    st.markdown(
        "특허번호, Google Patents URL 또는 PDF를 입력하면 핵심 기술과 청구항, 기술 차별점을 한국어로 분석합니다.  \n"
        "여러 특허를 한 번에 분류하고 유사도와 기술 포지션을 비교하며, 선택한 2~5건은 AI로 종합 비교할 수 있습니다.  \n"
        "분석 결과는 Word 및 Markdown 보고서로 내보낼 수 있습니다."
    )

    settings = load_settings()
    llm_client = LlmClient(settings)

    st.sidebar.header("⚙️ 분석 옵션")
    analysis_goals = {
        "특허 요약": st.sidebar.checkbox("특허 요약", value=True),
        "청구항 분석": st.sidebar.checkbox("청구항 분석", value=True),
        "차별점": st.sidebar.checkbox("차별점", value=True),
        "유사도 비교": st.sidebar.checkbox("유사도 비교", value=True),
    }
    depth = st.sidebar.radio("분석 깊이", ["빠른 검토", "표준 검토", "정밀 검토"], index=1)
    render_ai_status(llm_client)

    inputs = collect_inputs()
    mode = AnalysisMode.batch if len(inputs) >= 2 else AnalysisMode.single

    if inputs:
        st.write(f"📥 입력 {len(inputs)}건 감지: **{'복수 특허 분석' if mode == AnalysisMode.batch else '단일 특허 분석'}**")
        st.dataframe(
            pd.DataFrame([{"종류": item.kind.value, "입력": item.label} for item in inputs]),
            use_container_width=True,
            hide_index=True,
        )

    if st.button("🚀 분석 시작", type="primary"):
        if not inputs:
            st.warning("분석할 PDF, 특허번호 또는 URL을 먼저 입력해 주세요.")
            st.stop()
        max_input_chars = depth_to_max_input_chars(depth, settings.gemma_max_input_chars)
        run_analysis(inputs, mode, max_input_chars, llm_client)

    documents = st.session_state.get("last_documents", [])
    analyses = st.session_state.get("last_analyses", [])
    last_mode = st.session_state.get("last_mode", mode)
    if documents and analyses:
        render_results(documents, analyses, last_mode, analysis_goals, llm_client)


if __name__ == "__main__":
    main()
