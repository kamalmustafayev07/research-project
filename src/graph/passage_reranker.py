"""Lightweight trainable passage reranker for retrieval refinement."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from src.config import SETTINGS
from src.utils.embeddings import EmbeddingEncoder


@dataclass(slots=True)
class RerankerTrainMetrics:
    """Training and validation metrics for reranker quality."""

    train_pairs: int
    positive_rate: float
    validation_accuracy: float
    trained: bool
    training_history: list[dict[str, float | int]]


class PassageReranker:
    """Train and apply a shallow reranker using lexical and semantic features."""

    def __init__(self, encoder: EmbeddingEncoder, model_path: Path | None = None) -> None:
        self.encoder = encoder
        self.model_path = model_path or SETTINGS.paths.reranker_model
        self.model = self._new_model(max_iter=400)
        self.is_trained = False
        self._load_if_available()

    @staticmethod
    def _new_model(max_iter: int) -> LogisticRegression:
        return LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            random_state=SETTINGS.data.random_seed,
            warm_start=True,
        )

    @staticmethod
    def _iteration_schedule(max_iter: int) -> list[int]:
        # Fixed checkpoints keep history lightweight while still showing learning dynamics.
        checkpoints = [20, 40, 80, 120, 160, 220, 300, max_iter]
        schedule = sorted({step for step in checkpoints if 0 < step <= max_iter})
        return schedule or [max_iter]

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _pair_features(self, query: str, passages: list[dict[str, Any]]) -> np.ndarray:
        if not passages:
            return np.zeros((0, 5), dtype=np.float32)

        passage_texts = [f"{p.get('title', '')}: {p.get('text', '')}" for p in passages]
        vectors = self.encoder.encode([query] + passage_texts, normalize=True)
        query_vec = vectors[0]
        passage_vecs = vectors[1:]

        query_tokens = self._tokenize(query)
        features: list[list[float]] = []

        for idx, passage in enumerate(passages):
            p_text = str(passage.get("text", ""))
            p_title = str(passage.get("title", ""))
            p_tokens = self._tokenize(f"{p_title} {p_text}")
            overlap = len(query_tokens & p_tokens) / max(1, len(query_tokens))
            title_overlap = len(query_tokens & self._tokenize(p_title)) / max(1, len(query_tokens))
            length_signal = min(1.0, len(p_text.split()) / 160.0)
            semantic = float(np.dot(query_vec, passage_vecs[idx]))
            dense_score = float(passage.get("score", semantic))
            dense_norm = max(0.0, min(1.0, (dense_score + 1.0) / 2.0))
            features.append([semantic, dense_norm, overlap, title_overlap, length_signal])

        return np.asarray(features, dtype=np.float32)

    def _sample_training_pairs(
        self,
        examples: list[dict[str, Any]],
        max_examples: int,
        negatives_per_positive: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not examples:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        rng = random.Random(SETTINGS.data.random_seed)
        sampled = examples
        if len(sampled) > max_examples:
            sampled = rng.sample(sampled, max_examples)

        all_features: list[np.ndarray] = []
        labels: list[int] = []

        for example in sampled:
            passages = list(example.get("context", []))
            if not passages:
                continue

            supporting_titles = {
                str(item.get("title", ""))
                for item in example.get("supporting_facts", [])
                if isinstance(item, dict)
            }
            if not supporting_titles:
                continue

            positive_indices = [idx for idx, p in enumerate(passages) if str(p.get("title", "")) in supporting_titles]
            if not positive_indices:
                continue

            negative_indices = [idx for idx in range(len(passages)) if idx not in positive_indices]
            max_negatives = max(1, len(positive_indices) * negatives_per_positive)
            if len(negative_indices) > max_negatives:
                negative_indices = rng.sample(negative_indices, max_negatives)

            selected_indices = positive_indices + negative_indices
            selected_passages = [dict(passages[idx]) for idx in selected_indices]
            pair_features = self._pair_features(str(example.get("question", "")), selected_passages)
            all_features.append(pair_features)

            positive_set = set(positive_indices)
            labels.extend([1 if idx in positive_set else 0 for idx in selected_indices])

        if not all_features:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        matrix = np.vstack(all_features)
        y = np.asarray(labels, dtype=np.int64)
        return matrix, y

    def fit(
        self,
        train_examples: list[dict[str, Any]],
        validation_examples: list[dict[str, Any]] | None = None,
        max_examples: int | None = None,
        negatives_per_positive: int | None = None,
    ) -> RerankerTrainMetrics:
        max_examples = max_examples or SETTINGS.retrieval.reranker_max_train_examples
        negatives_per_positive = negatives_per_positive or SETTINGS.retrieval.reranker_negatives_per_positive

        x_train, y_train = self._sample_training_pairs(
            examples=train_examples,
            max_examples=max_examples,
            negatives_per_positive=negatives_per_positive,
        )

        x_val = np.zeros((0, 5), dtype=np.float32)
        y_val = np.zeros((0,), dtype=np.int64)
        has_val = False
        if validation_examples:
            x_val, y_val = self._sample_training_pairs(
                examples=validation_examples,
                max_examples=max(200, max_examples // 2),
                negatives_per_positive=negatives_per_positive,
            )
            has_val = x_val.shape[0] > 0 and len(set(y_val.tolist())) >= 2

        if x_train.shape[0] == 0 or len(set(y_train.tolist())) < 2:
            self.is_trained = False
            return RerankerTrainMetrics(
                train_pairs=int(x_train.shape[0]),
                positive_rate=0.0,
                validation_accuracy=0.0,
                trained=False,
                training_history=[],
            )

        training_history: list[dict[str, float | int]] = []
        schedule = self._iteration_schedule(max_iter=400)
        self.model = self._new_model(max_iter=schedule[-1])

        for iter_cap in schedule:
            self.model.max_iter = iter_cap
            self.model.fit(x_train, y_train)

            train_proba = self.model.predict_proba(x_train)[:, 1]
            train_loss = float(log_loss(y_train, train_proba, labels=[0, 1]))

            val_accuracy = 0.0
            if has_val:
                val_preds = self.model.predict(x_val)
                val_accuracy = float((val_preds == y_val).mean())

            training_history.append(
                {
                    "iteration": int(iter_cap),
                    "train_log_loss": train_loss,
                    "validation_accuracy": val_accuracy,
                }
            )

        self.is_trained = True
        self._save()

        val_accuracy = training_history[-1]["validation_accuracy"] if training_history else 0.0

        return RerankerTrainMetrics(
            train_pairs=int(x_train.shape[0]),
            positive_rate=float(y_train.mean()) if len(y_train) else 0.0,
            validation_accuracy=float(val_accuracy),
            trained=True,
            training_history=training_history,
        )

    def rerank(self, query: str, passages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not passages or not self.is_trained:
            return passages

        features = self._pair_features(query, passages)
        if features.shape[0] == 0:
            return passages

        probs = self.model.predict_proba(features)[:, 1]
        weight = SETTINGS.retrieval.reranker_weight

        reranked: list[dict[str, Any]] = []
        for idx, passage in enumerate(passages):
            item = dict(passage)
            dense_score = float(item.get("score", 0.0))
            dense_norm = max(0.0, min(1.0, (dense_score + 1.0) / 2.0))
            rerank_score = float(probs[idx])
            combined = (1.0 - weight) * dense_norm + weight * rerank_score
            item["dense_score"] = dense_score
            item["rerank_score"] = rerank_score
            item["score"] = combined
            reranked.append(item)

        reranked.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        return reranked

    def _save(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, self.model_path)

    def _load_if_available(self) -> None:
        if not self.model_path.exists():
            return
        try:
            loaded = joblib.load(self.model_path)
            self.model = loaded
            self.is_trained = True
        except Exception:
            self.is_trained = False
