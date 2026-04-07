"""Data loading and preprocessing utilities for HotpotQA."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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


def _to_examples(dataset: Dataset) -> list[HotpotExample]:
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
    return processed


def _split_prefix(source_split: str, test_size: int, val_size: int, seed: int) -> str:
    return f"hotpot_{source_split}_disjoint_t{test_size}_v{val_size}_s{seed}"


def prepare_hotpotqa_subset(
    subset_size: int | None = None,
    save: bool = True,
    dataset_split: str | None = None,
) -> list[HotpotExample]:
    """Download and prepare a HotpotQA subset for experiments."""
    subset_size = subset_size or SETTINGS.data.hotpot_subset_size
    dataset_split = dataset_split or SETTINGS.data.hotpot_split

    dataset: Dataset = load_dataset(
        SETTINGS.data.hotpot_dataset_name,
        SETTINGS.data.hotpot_config_name,
        split=dataset_split,
    )
    dataset = dataset.shuffle(seed=SETTINGS.data.random_seed).select(range(subset_size))
    processed = _to_examples(dataset)

    if save:
        output_json = SETTINGS.paths.data_processed / f"hotpot_{dataset_split}_{subset_size}.json"
        save_json(output_json, [asdict(example) for example in processed])

        output_csv = SETTINGS.paths.data_processed / f"hotpot_{dataset_split}_{subset_size}.csv"
        pd.DataFrame([asdict(example) for example in processed]).to_csv(output_csv, index=False)

    return processed


def prepare_hotpotqa_disjoint_splits(
    test_size: int = 200,
    val_size: int = 1000,
    train_size: int | None = None,
    source_split: str | None = None,
    seed: int | None = None,
    save: bool = True,
) -> dict[str, list[HotpotExample]]:
    """Prepare deterministic train/validation/test splits with zero ID overlap."""
    source_split = source_split or SETTINGS.data.hotpot_split
    seed = SETTINGS.data.random_seed if seed is None else seed

    dataset: Dataset = load_dataset(
        SETTINGS.data.hotpot_dataset_name,
        SETTINGS.data.hotpot_config_name,
        split=source_split,
    )
    dataset = dataset.shuffle(seed=seed)

    total = len(dataset)
    if test_size <= 0 or val_size <= 0:
        raise ValueError("test_size and val_size must be positive.")
    if test_size + val_size >= total:
        raise ValueError("test_size + val_size must be smaller than dataset size.")

    remaining = total - test_size - val_size
    if train_size is None or train_size <= 0:
        train_size = remaining
    train_size = min(train_size, remaining)

    test_ds = dataset.select(range(0, test_size))
    val_ds = dataset.select(range(test_size, test_size + val_size))
    train_ds = dataset.select(range(test_size + val_size, test_size + val_size + train_size))

    splits = {
        "train": _to_examples(train_ds),
        "validation": _to_examples(val_ds),
        "test": _to_examples(test_ds),
    }

    if save:
        prefix = _split_prefix(source_split=source_split, test_size=test_size, val_size=val_size, seed=seed)
        for split_name, examples in splits.items():
            output_json = SETTINGS.paths.data_processed / f"{prefix}_{split_name}.json"
            save_json(output_json, [asdict(example) for example in examples])

            output_csv = SETTINGS.paths.data_processed / f"{prefix}_{split_name}.csv"
            pd.DataFrame([asdict(example) for example in examples]).to_csv(output_csv, index=False)

        split_ids = {name: {example.qid for example in examples} for name, examples in splits.items()}
        metadata = {
            "source_split": source_split,
            "seed": seed,
            "sizes": {name: len(examples) for name, examples in splits.items()},
            "overlap": {
                "train_val": len(split_ids["train"].intersection(split_ids["validation"])),
                "train_test": len(split_ids["train"].intersection(split_ids["test"])),
                "val_test": len(split_ids["validation"].intersection(split_ids["test"])),
            },
        }
        save_json(SETTINGS.paths.data_processed / f"{prefix}_metadata.json", metadata)

    return splits


def load_prepared_subset(subset_size: int | None = None, dataset_split: str | None = None) -> list[HotpotExample]:
    """Load prepared subset from disk or build it if missing."""
    subset_size = subset_size or SETTINGS.data.hotpot_subset_size
    dataset_split = dataset_split or SETTINGS.data.hotpot_split
    cache_path = SETTINGS.paths.data_processed / f"hotpot_{dataset_split}_{subset_size}.json"
    if not cache_path.exists():
        return prepare_hotpotqa_subset(subset_size=subset_size, save=True, dataset_split=dataset_split)

    import json

    with cache_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return [HotpotExample(**item) for item in data]


def load_prepared_disjoint_split(
    split_name: str,
    test_size: int = 200,
    val_size: int = 1000,
    train_size: int | None = None,
    source_split: str | None = None,
    seed: int | None = None,
) -> list[HotpotExample]:
    """Load a cached disjoint split, creating it if needed."""
    source_split = source_split or SETTINGS.data.hotpot_split
    seed = SETTINGS.data.random_seed if seed is None else seed
    prefix = _split_prefix(source_split=source_split, test_size=test_size, val_size=val_size, seed=seed)

    cache_path = SETTINGS.paths.data_processed / f"{prefix}_{split_name}.json"
    if not cache_path.exists():
        prepare_hotpotqa_disjoint_splits(
            test_size=test_size,
            val_size=val_size,
            train_size=train_size,
            source_split=source_split,
            seed=seed,
            save=True,
        )

    import json

    with cache_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return [HotpotExample(**item) for item in data]
