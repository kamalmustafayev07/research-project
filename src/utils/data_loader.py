"""Unified dataset loading and normalization utilities for multi-hop QA benchmarks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd
from datasets import Dataset, load_dataset

from src.config import SETTINGS
from src.utils.helpers import save_json


@dataclass(slots=True)
class MultiHopQAExample:
    """Single normalized multi-hop QA sample."""

    qid: str
    dataset: str
    split: str
    question: str
    answer: str
    contexts: list[dict[str, Any]]
    supporting_facts: list[dict[str, Any]]

    @property
    def context(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for existing pipeline code."""
        return self.contexts


# Backward-compatible export used by existing imports.
HotpotExample = MultiHopQAExample


class BaseDatasetLoader(ABC):
    """Abstract base interface for dataset-specific loaders."""

    dataset_key: str
    hf_dataset_name: str
    hf_config_name: str | None = None

    def __init__(self, dataset_key: str, hf_dataset_name: str, hf_config_name: str | None = None) -> None:
        self.dataset_key = dataset_key
        self.hf_dataset_name = hf_dataset_name
        self.hf_config_name = hf_config_name

    def load_raw_split(self, split: str) -> Dataset:
        """Load a split from HuggingFace datasets."""
        if self.hf_config_name:
            return load_dataset(self.hf_dataset_name, self.hf_config_name, split=split)
        return load_dataset(self.hf_dataset_name, split=split)

    def load_subset(self, split: str, subset_size: int, seed: int) -> list[MultiHopQAExample]:
        """Load and normalize a deterministic subset of records."""
        dataset = self.load_raw_split(split=split)
        sample_size = min(max(subset_size, 1), len(dataset))
        shuffled = dataset.shuffle(seed=seed).select(range(sample_size))
        return self._to_examples(shuffled, split=split)

    def load_disjoint_splits(
        self,
        source_split: str,
        test_size: int,
        val_size: int,
        train_size: int | None,
        seed: int,
    ) -> dict[str, list[MultiHopQAExample]]:
        """Build deterministic train/validation/test splits with no ID overlap."""
        dataset = self.load_raw_split(split=source_split)
        shuffled = dataset.shuffle(seed=seed)

        total = len(shuffled)
        if test_size <= 0 or val_size <= 0:
            raise ValueError("test_size and val_size must be positive.")
        if test_size + val_size >= total:
            raise ValueError("test_size + val_size must be smaller than dataset size.")

        remaining = total - test_size - val_size
        if train_size is None or train_size <= 0:
            train_size = remaining
        train_size = min(train_size, remaining)

        test_ds = shuffled.select(range(0, test_size))
        val_ds = shuffled.select(range(test_size, test_size + val_size))
        train_ds = shuffled.select(range(test_size + val_size, test_size + val_size + train_size))

        return {
            "train": self._to_examples(train_ds, split=f"{source_split}:train"),
            "validation": self._to_examples(val_ds, split=f"{source_split}:validation"),
            "test": self._to_examples(test_ds, split=f"{source_split}:test"),
        }

    def _to_examples(self, dataset: Dataset, split: str) -> list[MultiHopQAExample]:
        normalized: list[MultiHopQAExample] = []
        for idx, item in enumerate(dataset):
            normalized.append(self.normalize_item(item=item, index=idx, split=split))
        return normalized

    @abstractmethod
    def normalize_item(self, item: dict[str, Any], index: int, split: str) -> MultiHopQAExample:
        """Normalize one raw sample into the common schema."""


def _extract_qid(item: dict[str, Any], dataset_key: str, index: int) -> str:
    for key in ["id", "_id", "qid", "question_id", "uid"]:
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"{dataset_key}_{index}"


def _extract_answer(item: dict[str, Any]) -> str:
    for key in ["answer", "answers", "gold_answer", "target"]:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, str) and first.strip():
                return first.strip()
    return ""


def _normalize_contexts(contexts: Any, source_type: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []

    if isinstance(contexts, dict):
        titles = contexts.get("title", [])
        sentences = contexts.get("sentences", [])
        for idx, (title, sent_list) in enumerate(zip(titles, sentences)):
            if isinstance(sent_list, list):
                text = " ".join(str(s) for s in sent_list)
            else:
                text = str(sent_list)
            normalized.append(
                {
                    "title": str(title),
                    "text": text,
                    "passage_id": f"{source_type}_{idx}",
                    "source_type": source_type,
                }
            )
        return normalized

    if not isinstance(contexts, list):
        return normalized

    for idx, entry in enumerate(contexts):
        title = f"passage_{idx}"
        text = ""

        if isinstance(entry, dict):
            title = str(entry.get("title") or entry.get("paragraph_title") or entry.get("source") or title)
            text = str(
                entry.get("text")
                or entry.get("paragraph_text")
                or entry.get("paragraph")
                or entry.get("context")
                or entry.get("sentences")
                or ""
            )
            if isinstance(entry.get("sentences"), list):
                text = " ".join(str(s) for s in entry["sentences"])
        elif isinstance(entry, list) and len(entry) >= 2:
            title = str(entry[0])
            body = entry[1]
            if isinstance(body, list):
                text = " ".join(str(s) for s in body)
            else:
                text = str(body)
        elif isinstance(entry, str):
            text = entry

        if text.strip():
            normalized.append(
                {
                    "title": title,
                    "text": text,
                    "passage_id": f"{source_type}_{idx}",
                    "source_type": source_type,
                }
            )

    return normalized


def _normalize_supporting_facts(value: Any) -> list[dict[str, Any]]:
    supporting: list[dict[str, Any]] = []

    if isinstance(value, dict):
        titles = value.get("title", [])
        sent_ids = value.get("sent_id", [])
        for title, sent_id in zip(titles, sent_ids):
            supporting.append(
                {
                    "title": str(title),
                    "sentence_id": int(sent_id) if str(sent_id).isdigit() else sent_id,
                }
            )
        return supporting

    if isinstance(value, list):
        for idx, fact in enumerate(value):
            if isinstance(fact, dict):
                title = str(fact.get("title") or fact.get("paragraph_title") or fact.get("source") or f"support_{idx}")
                sentence_id = fact.get("sentence_id")
                if sentence_id is None:
                    sentence_id = fact.get("sent_id")
                supporting.append({"title": title, "sentence_id": sentence_id})
            elif isinstance(fact, list) and len(fact) >= 2:
                supporting.append({"title": str(fact[0]), "sentence_id": fact[1]})
            elif isinstance(fact, str):
                supporting.append({"title": fact, "sentence_id": None})
    return supporting


class HotpotQALoader(BaseDatasetLoader):
    """Loader for HotpotQA."""

    def __init__(self) -> None:
        super().__init__(
            dataset_key="hotpotqa",
            hf_dataset_name=SETTINGS.data.hotpot_dataset_name,
            hf_config_name=SETTINGS.data.hotpot_config_name,
        )

    def normalize_item(self, item: dict[str, Any], index: int, split: str) -> MultiHopQAExample:
        contexts = _normalize_contexts(item.get("context", {}), source_type="hotpot_passage")
        supporting = _normalize_supporting_facts(item.get("supporting_facts", []))

        return MultiHopQAExample(
            qid=_extract_qid(item, self.dataset_key, index),
            dataset=self.dataset_key,
            split=split,
            question=str(item.get("question", "")).strip(),
            answer=_extract_answer(item),
            contexts=contexts,
            supporting_facts=supporting,
        )


class MuSiQueLoader(BaseDatasetLoader):
    """Loader for MuSiQue."""

    def __init__(self) -> None:
        super().__init__(
            dataset_key="musique",
            hf_dataset_name=SETTINGS.data.musique_dataset_name,
            hf_config_name=SETTINGS.data.musique_config_name,
        )

    def normalize_item(self, item: dict[str, Any], index: int, split: str) -> MultiHopQAExample:
        raw_contexts = (
            item.get("paragraphs")
            or item.get("context")
            or item.get("contexts")
            or item.get("passages")
            or []
        )
        contexts = _normalize_contexts(raw_contexts, source_type="musique_paragraph")

        supporting = _normalize_supporting_facts(
            item.get("supporting_facts")
            or item.get("supporting_paragraphs")
            or item.get("evidence")
            or []
        )
        if not supporting and isinstance(raw_contexts, list):
            for idx_ctx, paragraph in enumerate(raw_contexts):
                if isinstance(paragraph, dict) and bool(
                    paragraph.get("is_supporting")
                    or paragraph.get("is_support")
                    or paragraph.get("supporting")
                ):
                    supporting.append(
                        {
                            "title": str(
                                paragraph.get("title")
                                or paragraph.get("paragraph_title")
                                or paragraph.get("source")
                                or f"paragraph_{idx_ctx}"
                            ),
                            "sentence_id": paragraph.get("sentence_id"),
                        }
                    )

        return MultiHopQAExample(
            qid=_extract_qid(item, self.dataset_key, index),
            dataset=self.dataset_key,
            split=split,
            question=str(item.get("question") or item.get("query") or "").strip(),
            answer=_extract_answer(item),
            contexts=contexts,
            supporting_facts=supporting,
        )


class TwoWikiMultiHopQALoader(BaseDatasetLoader):
    """Loader for 2WikiMultiHopQA."""

    def __init__(self) -> None:
        super().__init__(
            dataset_key="2wikimultihopqa",
            hf_dataset_name=SETTINGS.data.twowiki_dataset_name,
            hf_config_name=SETTINGS.data.twowiki_config_name,
        )

    def normalize_item(self, item: dict[str, Any], index: int, split: str) -> MultiHopQAExample:
        raw_contexts = item.get("context") or item.get("contexts") or item.get("passages") or []
        contexts = _normalize_contexts(raw_contexts, source_type="wiki_chain")
        supporting = _normalize_supporting_facts(
            item.get("supporting_facts")
            or item.get("evidences")
            or item.get("evidence")
            or item.get("supporting_sentences")
            or []
        )

        return MultiHopQAExample(
            qid=_extract_qid(item, self.dataset_key, index),
            dataset=self.dataset_key,
            split=split,
            question=str(item.get("question", "")).strip(),
            answer=_extract_answer(item),
            contexts=contexts,
            supporting_facts=supporting,
        )


def get_dataset_loader(dataset: str | None = None) -> BaseDatasetLoader:
    """Return a dataset loader by key."""
    selected = (dataset or SETTINGS.data.dataset).strip().lower()
    aliases = {
        "hotpotqa": "hotpotqa",
        "hotpot": "hotpotqa",
        "musique": "musique",
        "2wiki": "2wikimultihopqa",
        "2wikimultihop": "2wikimultihopqa",
        "2wikimultihopqa": "2wikimultihopqa",
    }
    resolved = aliases.get(selected)
    if resolved is None:
        raise ValueError(f"Unsupported dataset '{dataset}'.")

    if resolved == "hotpotqa":
        return HotpotQALoader()
    if resolved == "musique":
        return MuSiQueLoader()
    return TwoWikiMultiHopQALoader()


def _split_prefix(dataset: str, source_split: str, test_size: int, val_size: int, seed: int) -> str:
    return f"{dataset}_{source_split}_disjoint_t{test_size}_v{val_size}_s{seed}"


def _to_payload(example: MultiHopQAExample) -> dict[str, Any]:
    payload = asdict(example)
    payload["context"] = payload["contexts"]
    return payload


def _from_payload(item: dict[str, Any], default_dataset: str, default_split: str) -> MultiHopQAExample:
    return MultiHopQAExample(
        qid=str(item.get("qid") or item.get("id") or ""),
        dataset=str(item.get("dataset") or default_dataset),
        split=str(item.get("split") or default_split),
        question=str(item.get("question") or ""),
        answer=str(item.get("answer") or ""),
        contexts=list(item.get("contexts") or item.get("context") or []),
        supporting_facts=list(item.get("supporting_facts") or []),
    )


def prepare_dataset_subset(
    dataset: str | None = None,
    subset_size: int | None = None,
    save: bool = True,
    dataset_split: str | None = None,
) -> list[MultiHopQAExample]:
    """Download and normalize a subset for the selected dataset."""
    selected_dataset = dataset or SETTINGS.data.dataset
    subset = subset_size or SETTINGS.data.dataset_subset_size
    split = dataset_split or SETTINGS.data.dataset_split

    loader = get_dataset_loader(selected_dataset)
    processed = loader.load_subset(split=split, subset_size=subset, seed=SETTINGS.data.random_seed)

    if save:
        output_json = SETTINGS.paths.data_processed / f"{loader.dataset_key}_{split}_{subset}.json"
        save_json(output_json, [_to_payload(example) for example in processed])

        output_csv = SETTINGS.paths.data_processed / f"{loader.dataset_key}_{split}_{subset}.csv"
        pd.DataFrame([_to_payload(example) for example in processed]).to_csv(output_csv, index=False)

    return processed


def prepare_dataset_disjoint_splits(
    dataset: str | None = None,
    test_size: int = 200,
    val_size: int = 1000,
    train_size: int | None = None,
    source_split: str | None = None,
    seed: int | None = None,
    save: bool = True,
) -> dict[str, list[MultiHopQAExample]]:
    """Prepare deterministic disjoint train/validation/test splits for any supported dataset."""
    selected_dataset = dataset or SETTINGS.data.dataset
    split = source_split or SETTINGS.data.dataset_split
    random_seed = SETTINGS.data.random_seed if seed is None else seed

    loader = get_dataset_loader(selected_dataset)
    splits = loader.load_disjoint_splits(
        source_split=split,
        test_size=test_size,
        val_size=val_size,
        train_size=train_size,
        seed=random_seed,
    )

    if save:
        prefix = _split_prefix(
            dataset=loader.dataset_key,
            source_split=split,
            test_size=test_size,
            val_size=val_size,
            seed=random_seed,
        )
        for split_name, examples in splits.items():
            output_json = SETTINGS.paths.data_processed / f"{prefix}_{split_name}.json"
            save_json(output_json, [_to_payload(example) for example in examples])

            output_csv = SETTINGS.paths.data_processed / f"{prefix}_{split_name}.csv"
            pd.DataFrame([_to_payload(example) for example in examples]).to_csv(output_csv, index=False)

        split_ids = {name: {example.qid for example in examples} for name, examples in splits.items()}
        metadata = {
            "dataset": loader.dataset_key,
            "source_split": split,
            "seed": random_seed,
            "sizes": {name: len(examples) for name, examples in splits.items()},
            "overlap": {
                "train_val": len(split_ids["train"].intersection(split_ids["validation"])),
                "train_test": len(split_ids["train"].intersection(split_ids["test"])),
                "val_test": len(split_ids["validation"].intersection(split_ids["test"])),
            },
        }
        save_json(SETTINGS.paths.data_processed / f"{prefix}_metadata.json", metadata)

    return splits


def load_prepared_subset(
    subset_size: int | None = None,
    dataset_split: str | None = None,
    dataset: str | None = None,
) -> list[MultiHopQAExample]:
    """Load prepared subset from disk or build it if missing."""
    selected_dataset = dataset or SETTINGS.data.dataset
    subset = subset_size or SETTINGS.data.dataset_subset_size
    split = dataset_split or SETTINGS.data.dataset_split

    loader = get_dataset_loader(selected_dataset)
    cache_path = SETTINGS.paths.data_processed / f"{loader.dataset_key}_{split}_{subset}.json"
    if not cache_path.exists():
        return prepare_dataset_subset(
            dataset=loader.dataset_key,
            subset_size=subset,
            save=True,
            dataset_split=split,
        )

    import json

    with cache_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return [_from_payload(item, default_dataset=loader.dataset_key, default_split=split) for item in data]


def load_prepared_disjoint_split(
    split_name: str,
    test_size: int = 200,
    val_size: int = 1000,
    train_size: int | None = None,
    source_split: str | None = None,
    seed: int | None = None,
    dataset: str | None = None,
) -> list[MultiHopQAExample]:
    """Load a cached disjoint split, creating it if needed."""
    selected_dataset = dataset or SETTINGS.data.dataset
    split = source_split or SETTINGS.data.dataset_split
    random_seed = SETTINGS.data.random_seed if seed is None else seed

    loader = get_dataset_loader(selected_dataset)
    prefix = _split_prefix(
        dataset=loader.dataset_key,
        source_split=split,
        test_size=test_size,
        val_size=val_size,
        seed=random_seed,
    )

    cache_path = SETTINGS.paths.data_processed / f"{prefix}_{split_name}.json"
    if not cache_path.exists():
        prepare_dataset_disjoint_splits(
            dataset=loader.dataset_key,
            test_size=test_size,
            val_size=val_size,
            train_size=train_size,
            source_split=split,
            seed=random_seed,
            save=True,
        )

    import json

    with cache_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return [_from_payload(item, default_dataset=loader.dataset_key, default_split=f"{split}:{split_name}") for item in data]


# Backward-compatible wrappers for existing Hotpot-oriented commands.
def prepare_hotpotqa_subset(
    subset_size: int | None = None,
    save: bool = True,
    dataset_split: str | None = None,
) -> list[MultiHopQAExample]:
    return prepare_dataset_subset(
        dataset="hotpotqa",
        subset_size=(subset_size or SETTINGS.data.hotpot_subset_size),
        save=save,
        dataset_split=(dataset_split or SETTINGS.data.hotpot_split),
    )


def prepare_hotpotqa_disjoint_splits(
    test_size: int = 200,
    val_size: int = 1000,
    train_size: int | None = None,
    source_split: str | None = None,
    seed: int | None = None,
    save: bool = True,
) -> dict[str, list[MultiHopQAExample]]:
    return prepare_dataset_disjoint_splits(
        dataset="hotpotqa",
        test_size=test_size,
        val_size=val_size,
        train_size=train_size,
        source_split=(source_split or SETTINGS.data.hotpot_split),
        seed=(SETTINGS.data.random_seed if seed is None else seed),
        save=save,
    )
