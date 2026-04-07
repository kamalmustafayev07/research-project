"""General helper functions for logging, serialization, and text normalization."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any


def setup_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create and configure a logger with optional file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def save_json(path: Path, payload: Any) -> None:
    """Save JSON payload to disk with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=True, indent=2)


def load_json(path: Path) -> Any:
    """Load a JSON payload from disk."""
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def normalize_text(text: str) -> str:
    """Normalize text for robust EM/F1 evaluation."""
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_float(value: str | float | int, default: float = 0.0) -> float:
    """Safely convert an arbitrary value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            normalized = value.strip()
            if not normalized:
                return default
            normalized = normalized.replace("%", "").replace(",", ".")
            match = re.search(r"[-+]?\d*\.?\d+", normalized)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return default
        return default
