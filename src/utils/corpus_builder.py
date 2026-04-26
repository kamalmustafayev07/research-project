"""Utilities for building and caching a large unified passage corpus.

Pools supporting passages from all three benchmark datasets
(HotpotQA, MuSiQue, 2WikiMultiHopQA) and stores the result on disk so
subsequent Streamlit restarts are fast.

With a GPU the FAISS index can be built from hundreds of thousands of
passages in a few seconds; the index is also persisted to disk so it only
needs to be built once per corpus version.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.config import SETTINGS
from src.utils.data_loader import get_dataset_loader

logger = logging.getLogger(__name__)

_DATASETS = ["hotpotqa", "musique", "2wikimultihopqa"]

# Use a very large cap so the entire validation split of every dataset is
# loaded (HotpotQA ~7 405, MuSiQue ~2 417, 2Wiki ~12 576 examples).
# Set to None for truly unlimited (loads the entire split with no sampling).
DEFAULT_MAX_PER_SPLIT = 100_000

# Splits to load.  Adding "train" gives much more passage coverage at the
# cost of a longer first-run download.
DEFAULT_SPLITS = ["train", "validation"]


def _corpus_cache_key(
    max_per_split: int | None,
    splits: list[str],
) -> str:
    """Deterministic short key used in cache filenames."""
    raw = f"splits={'_'.join(sorted(splits))}_max={max_per_split or 'all'}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def corpus_paths(
    max_per_split: int | None = DEFAULT_MAX_PER_SPLIT,
    splits: list[str] | None = None,
) -> tuple[Path, Path]:
    """Return (passages_json_path, faiss_index_path) for the given settings."""
    key = _corpus_cache_key(max_per_split, splits or DEFAULT_SPLITS)
    base = SETTINGS.paths.data_processed / f"unified_corpus_{key}"
    return base.with_suffix(".json"), base.with_suffix(".faiss")


def build_unified_corpus(
    max_per_split: int | None = DEFAULT_MAX_PER_SPLIT,
    splits: list[str] | None = None,
    seed: int = 42,
    force_rebuild: bool = False,
) -> list[dict[str, Any]]:
    """Download all three datasets and return a deduplicated passage corpus.

    All unique context passages found across the requested splits are pooled
    and deduplicated by ``(title, text[:80])``.  Results are cached to
    ``data/processed/unified_corpus_<hash>.json`` so subsequent calls skip
    the download step entirely.

    Parameters
    ----------
    max_per_split:
        Maximum QA examples to load per dataset **per split**.  ``None`` loads
        the entire split (no sampling).
    splits:
        HuggingFace dataset splits to sample from, e.g. ``["train",
        "validation"]``.  Defaults to both train and validation.
    seed:
        Shuffle seed for reproducible sampling (only applies when
        ``max_per_split`` is set).
    force_rebuild:
        If ``True`` re-download and overwrite the cached file.

    Returns
    -------
    list[dict]
        Each item is a passage dict with keys ``title``, ``text``,
        ``passage_id``, ``source_type``, ``dataset``.
    """
    _splits = splits or DEFAULT_SPLITS
    passages_path, _ = corpus_paths(max_per_split, _splits)

    if not force_rebuild and passages_path.exists():
        logger.info("Loading cached unified corpus from %s", passages_path)
        with passages_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    cap_str = str(max_per_split) if max_per_split else "all"
    logger.info(
        "Building unified corpus: splits=%s, max=%s examples per dataset per split …",
        _splits,
        cap_str,
    )
    passages: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for dataset_name in _DATASETS:
        for split in _splits:
            try:
                loader = get_dataset_loader(dataset_name)
                if max_per_split is None:
                    # Load the entire split
                    raw = loader.load_raw_split(split=split)
                    examples = loader._to_examples(raw, split=split)
                else:
                    examples = loader.load_subset(
                        split=split,
                        subset_size=max_per_split,
                        seed=seed,
                    )
                n_before = len(passages)
                for ex in examples:
                    for ctx in ex.contexts:
                        title = str(ctx.get("title", ""))
                        text = str(ctx.get("text", ""))
                        key = (title, text[:80])
                        if key in seen:
                            continue
                        seen.add(key)
                        item = dict(ctx)
                        item["dataset"] = dataset_name
                        passages.append(item)
                added = len(passages) - n_before
                logger.info(
                    "  %s/%s: %d examples → %d new passages (total %d)",
                    dataset_name,
                    split,
                    len(examples),
                    added,
                    len(passages),
                )
            except Exception:
                logger.exception(
                    "  Failed to load dataset '%s' split '%s'", dataset_name, split
                )

    logger.info("Unified corpus: %d unique passages total", len(passages))

    passages_path.parent.mkdir(parents=True, exist_ok=True)
    with passages_path.open("w", encoding="utf-8") as fh:
        json.dump(passages, fh, ensure_ascii=False)
    logger.info("Corpus saved to %s", passages_path)

    return passages
