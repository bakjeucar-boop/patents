from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT_DIR / "uploads"
OUTPUT_DIR = ROOT_DIR / "outputs"
DATA_DIR = ROOT_DIR / "data"


@dataclass(frozen=True)
class Settings:
    gemma_api_key: str | None
    gemma_api_provider: str
    gemma_api_base_url: str | None
    gemma_model: str
    gemma_temperature: float
    gemma_max_input_chars: int


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)


def load_settings() -> Settings:
    load_dotenv(ROOT_DIR / ".env")
    return Settings(
        gemma_api_key=os.getenv("GEMMA_API_KEY"),
        gemma_api_provider=os.getenv("GEMMA_API_PROVIDER", "google"),
        gemma_api_base_url=os.getenv("GEMMA_API_BASE_URL"),
        gemma_model=os.getenv("GEMMA_MODEL", "gemma-4-31b"),
        gemma_temperature=float(os.getenv("GEMMA_TEMPERATURE", "0.2")),
        gemma_max_input_chars=int(os.getenv("GEMMA_MAX_INPUT_CHARS", "45000")),
    )
