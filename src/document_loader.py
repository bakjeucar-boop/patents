from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

from src.models import InputKind, PatentInput


PATENT_NUMBER_RE = re.compile(r"\b[A-Z]{0,2}\d{4,}[A-Z]?\d*(?:[A-Z]\d?)?\b", re.IGNORECASE)


def _http_get(url: str, timeout: int = 30) -> requests.Response:
    session = requests.Session()
    session.trust_env = False
    response = session.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 PatentTechnologyReviewAgent/0.1"},
    )
    response.raise_for_status()
    return response


def save_uploaded_file(upload_dir: Path, filename: str, content: bytes) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("_") or "uploaded.pdf"
    path = upload_dir / safe_name
    path.write_bytes(content)
    return path


def extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def extract_pdf_text_from_bytes(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages).strip()


def is_pdf_url(url: str, content_type: str | None = None) -> bool:
    path = urlparse(url).path.lower()
    if path.endswith(".pdf"):
        return True
    return bool(content_type and "pdf" in content_type.lower())


def decode_html_response(response: requests.Response) -> str:
    try:
        utf8_text = response.content.decode("utf-8")
        if "\ufffd" not in utf8_text:
            return utf8_text
    except UnicodeDecodeError:
        pass

    encoding = response.encoding or response.apparent_encoding or "utf-8"
    if encoding.lower() in {"iso-8859-1", "latin-1"}:
        encoding = response.apparent_encoding or "utf-8"
    try:
        return response.content.decode(encoding, errors="replace")
    except LookupError:
        return response.content.decode("utf-8", errors="replace")


def fetch_url_text(url: str, timeout: int = 20) -> str:
    response = _http_get(url, timeout=timeout)
    content_type = response.headers.get("Content-Type", "")
    if is_pdf_url(url, content_type):
        return extract_pdf_text_from_bytes(response.content)

    soup = BeautifulSoup(decode_html_response(response), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)


def extract_patent_number(value: str) -> str | None:
    parsed = urlparse(value)
    path_parts = [part for part in parsed.path.split("/") if part]
    if "patent" in path_parts:
        idx = path_parts.index("patent")
        if idx + 1 < len(path_parts):
            match = PATENT_NUMBER_RE.search(path_parts[idx + 1])
            if match:
                return match.group(0).upper()
    match = PATENT_NUMBER_RE.search(value)
    return match.group(0).upper() if match else None


def parse_multiline_inputs(raw_text: str) -> list[PatentInput]:
    inputs: list[PatentInput] = []
    for line in raw_text.splitlines():
        value = line.strip()
        if not value:
            continue
        if value.lower().startswith(("http://", "https://")):
            kind = InputKind.url
        elif PATENT_NUMBER_RE.search(value):
            kind = InputKind.patent_number
        else:
            kind = InputKind.text
        inputs.append(PatentInput(kind=kind, value=value, label=value))
    return inputs


def google_patents_url(patent_number: str) -> str:
    normalized = patent_number.strip().replace(" ", "")
    return f"https://patents.google.com/patent/{normalized}/"
