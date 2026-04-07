"""Data loading and preprocessing utilities for HotpotQA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

from src.config import SETTINGS
from src.utils.helpers import save_json


@dataclass(slots=True)
class HotpotExample:
    """Single processed HotpotQA sample."""

    qid: str
    question: str
    answer: str
    context: list[dict[str, Any]]
    supporting_facts: list[dict[str, Any]]


def _context_to_passages(context_item: dict[str, Any]) -> list[dict[str, Any]]:
    titles = context_item.get("title", [])
    sentences = context_item.get("sentences", [])
    passages: list[dict[str, Any]] = []
    for title, sent_list in zip(titles, sentences):
        text = " ".join(sent_list)
        passages.append({"title": title, "text": text})
    return passages


def prepare_hotpotqa_subset(subset_size: int | None = None, save: bool = True) -> list[HotpotExample]:
    """Download and prepare a HotpotQA subset for experiments."""
    subset_size = subset_size or SETTINGS.data.hotpot_subset_size

    dataset: Dataset = load_dataset(
        SETTINGS.data.hotpot_dataset_name,
        SETTINGS.data.hotpot_config_name,
        split=SETTINGS.data.hotpot_split,
    )
    dataset = dataset.shuffle(seed=SETTINGS.data.random_seed).select(range(subset_size))

    processed: list[HotpotExample] = []
    for item in dataset:
        context_passages = _context_to_passages(item["context"])
        supporting = [
            {"title": title, "sentence_id": int(sent_id)}
            for title, sent_id in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])
        ]
        processed.append(
            HotpotExample(
                qid=item["id"],
                question=item["question"],
                answer=item["answer"],
                context=context_passages,
                supporting_facts=supporting,
            )
        )

    if save:
        output_json = SETTINGS.paths.data_processed / f"hotpot_{SETTINGS.data.hotpot_split}_{subset_size}.json"
        save_json(output_json, [asdict(example) for example in processed])

        output_csv = SETTINGS.paths.data_processed / f"hotpot_{SETTINGS.data.hotpot_split}_{subset_size}.csv"
        pd.DataFrame([asdict(example) for example in processed]).to_csv(output_csv, index=False)

    return processed


def load_prepared_subset(subset_size: int | None = None) -> list[HotpotExample]:
    """Load prepared subset from disk or build it if missing."""
    subset_size = subset_size or SETTINGS.data.hotpot_subset_size
    cache_path = SETTINGS.paths.data_processed / f"hotpot_{SETTINGS.data.hotpot_split}_{subset_size}.json"
    if not cache_path.exists():
        return prepare_hotpotqa_subset(subset_size=subset_size, save=True)

    import json

    with cache_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return [HotpotExample(**item) for item in data]
