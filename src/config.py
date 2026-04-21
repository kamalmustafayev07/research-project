"""Global configuration and runtime settings for Agent-Enhanced GraphRAG."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import torch
from dotenv import load_dotenv

# Load .env values once before reading environment variables into settings.
load_dotenv()


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


def detect_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass(slots=True)
class ProjectPaths:
    """Filesystem paths used by the project."""

    root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])

    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def output_results(self) -> Path:
        return self.root / "outputs" / "results"

    @property
    def output_experiments(self) -> Path:
        return self.root / "outputs" / "experiments"

    @property
    def output_evidence(self) -> Path:
        return self.root / "outputs" / "evidence_chains"

    @property
    def output_logs(self) -> Path:
        return self.root / "outputs" / "logs"

    @property
    def local_logs(self) -> Path:
        return self.root / "logs"

    @property
    def output_models(self) -> Path:
        return self.root / "outputs" / "models"

    @property
    def reranker_model(self) -> Path:
        return self.output_models / "passage_reranker.joblib"


@dataclass(slots=True)
class ModelConfig:
    """Model and inference backend configuration."""

    llm_backend: Literal["transformers", "ollama", "mock"] = "transformers"
    hf_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    use_4bit: bool = True
    temperature: float = 0.1
    max_new_tokens: int = 384


@dataclass(slots=True)
class RetrievalConfig:
    """Parameters for graph retrieval and traversal."""

    top_k_passages: int = 8
    top_k_edges: int = 10
    max_hops: int = 3
    max_nodes: int = 400
    max_retrieval_loops: int = 2
    use_reranker: bool = True
    reranker_weight: float = 0.4
    reranker_negatives_per_positive: int = 3
    reranker_max_train_examples: int = 1200
    reranker_epochs: int = 12


@dataclass(slots=True)
class DataConfig:
    """Dataset selection and sampling settings."""

    dataset: Literal["hotpotqa", "musique", "2wikimultihopqa"] = "hotpotqa"
    dataset_split: str = "validation"
    dataset_subset_size: int = 200
    hotpot_dataset_name: str = "hotpot_qa"
    hotpot_config_name: str = "fullwiki"
    musique_dataset_name: str = "dgslibisey/MuSiQue"
    musique_config_name: str | None = None
    twowiki_dataset_name: str = "voidful/2WikiMultihopQA"
    twowiki_config_name: str | None = None
    hotpot_split: str = "validation"
    hotpot_subset_size: int = 200
    random_seed: int = 42


@dataclass(slots=True)
class Settings:
    """Top-level settings object."""

    paths: ProjectPaths = field(default_factory=ProjectPaths)
    model: ModelConfig = field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    device: str = field(default_factory=detect_device)

    @property
    def torch_dtype(self) -> torch.dtype:
        if self.device == "cuda":
            return torch.float16
        return torch.float32

    @classmethod
    def from_env(cls) -> "Settings":
        """Construct settings from environment variables."""
        dataset_raw = os.getenv("DATASET", "hotpotqa").strip().lower()
        dataset_aliases = {
            "hotpot": "hotpotqa",
            "hotpot_qa": "hotpotqa",
            "musique": "musique",
            "2wiki": "2wikimultihopqa",
            "2wikimultihop": "2wikimultihopqa",
            "2wikimultihopqa": "2wikimultihopqa",
        }
        dataset = dataset_aliases.get(dataset_raw, "hotpotqa")

        model = ModelConfig(
            llm_backend=os.getenv("LLM_BACKEND", "transformers"),
            hf_model_name=os.getenv("HF_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
            ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            use_4bit=_to_bool(os.getenv("USE_4BIT"), True),
            temperature=float(os.getenv("TEMPERATURE", "0.1")),
            max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "384")),
        )
        retrieval = RetrievalConfig(
            max_retrieval_loops=int(os.getenv("MAX_RETRIEVAL_LOOPS", "2")),
            use_reranker=_to_bool(os.getenv("USE_RERANKER"), True),
            reranker_weight=float(os.getenv("RERANKER_WEIGHT", "0.4")),
            reranker_negatives_per_positive=int(os.getenv("RERANKER_NEGATIVES_PER_POSITIVE", "3")),
            reranker_max_train_examples=int(os.getenv("RERANKER_MAX_TRAIN_EXAMPLES", "1200")),
            reranker_epochs=int(os.getenv("RERANKER_EPOCHS", "12")),
        )
        data = DataConfig(
            dataset=dataset,
            dataset_subset_size=int(os.getenv("DATASET_SUBSET_SIZE", os.getenv("HOTPOT_SUBSET_SIZE", "200"))),
            dataset_split=os.getenv("DATASET_SPLIT", os.getenv("HOTPOT_SPLIT", "validation")),
            hotpot_dataset_name=os.getenv("HOTPOT_DATASET_NAME", "hotpot_qa"),
            hotpot_config_name=os.getenv("HOTPOT_CONFIG_NAME", "fullwiki"),
            musique_dataset_name=os.getenv("MUSIQUE_DATASET_NAME", "dgslibisey/MuSiQue"),
            musique_config_name=os.getenv("MUSIQUE_CONFIG_NAME") or None,
            twowiki_dataset_name=os.getenv("TWOWIKI_DATASET_NAME", "voidful/2WikiMultihopQA"),
            twowiki_config_name=os.getenv("TWOWIKI_CONFIG_NAME") or None,
            hotpot_subset_size=int(os.getenv("HOTPOT_SUBSET_SIZE", "200")),
            hotpot_split=os.getenv("HOTPOT_SPLIT", "validation"),
        )
        settings = cls(model=model, retrieval=retrieval, data=data)

        for path in [
            settings.paths.data_raw,
            settings.paths.data_processed,
            settings.paths.output_results,
            settings.paths.output_experiments,
            settings.paths.output_evidence,
            settings.paths.output_logs,
            settings.paths.output_models,
            settings.paths.local_logs,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        return settings


SETTINGS = Settings.from_env()
