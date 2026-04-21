"""Utilities for run-scoped experiment artifact directories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import re


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text or "run"


def _build_base_run_id(run_name: str | None = None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if run_name:
        return f"{stamp}_{_slugify(run_name)}"
    return f"{stamp}_run"


def _ensure_unique_run_id(runs_root: Path, base_run_id: str) -> str:
    candidate = base_run_id
    suffix = 1
    while (runs_root / candidate).exists():
        suffix += 1
        candidate = f"{base_run_id}_v{suffix:02d}"
    return candidate


@dataclass(slots=True)
class DatasetRunPaths:
    """Directory layout for one dataset in one run."""

    dataset: str
    run_id: str
    root: Path

    @property
    def metadata_dir(self) -> Path:
        return self.root / "metadata"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def predictions_dir(self) -> Path:
        return self.root / "predictions"

    @property
    def latency_dir(self) -> Path:
        return self.root / "latency"

    @property
    def evidence_dir(self) -> Path:
        return self.root / "evidence"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"

    @property
    def stages_dir(self) -> Path:
        return self.root / "stages"

    def stage_dir(self, stage_name: str) -> Path:
        return self.stages_dir / _slugify(stage_name)

    def ensure(self) -> None:
        for path in [
            self.root,
            self.metadata_dir,
            self.results_dir,
            self.predictions_dir,
            self.latency_dir,
            self.evidence_dir,
            self.plots_dir,
            self.logs_dir,
            self.stages_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ExperimentPaths:
    """Run-level and dataset-level experiment directories."""

    base_dir: Path
    runs_root: Path
    datasets_root: Path
    run_id: str
    created_at_utc: str
    run_root: Path
    dataset_paths: dict[str, DatasetRunPaths]

    @property
    def metadata_dir(self) -> Path:
        return self.run_root / "metadata"

    @property
    def logs_dir(self) -> Path:
        return self.run_root / "logs"

    @property
    def plots_dir(self) -> Path:
        return self.run_root / "plots"

    @property
    def summaries_dir(self) -> Path:
        return self.run_root / "summaries"

    @property
    def run_manifest_path(self) -> Path:
        return self.metadata_dir / "run_manifest.json"

    def for_dataset(self, dataset: str) -> DatasetRunPaths:
        if dataset not in self.dataset_paths:
            raise KeyError(f"Dataset '{dataset}' is not part of this run.")
        return self.dataset_paths[dataset]

    def ensure(self) -> None:
        for path in [
            self.base_dir,
            self.runs_root,
            self.datasets_root,
            self.run_root,
            self.metadata_dir,
            self.logs_dir,
            self.plots_dir,
            self.summaries_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
        for ds_paths in self.dataset_paths.values():
            ds_paths.ensure()


def create_experiment_paths(base_output_dir: Path, datasets: list[str], run_name: str | None = None) -> ExperimentPaths:
    """Create run-scoped directories for all datasets and return structured paths."""
    base_dir = Path(base_output_dir)
    runs_root = base_dir / "runs"
    datasets_root = base_dir / "datasets"

    runs_root.mkdir(parents=True, exist_ok=True)
    datasets_root.mkdir(parents=True, exist_ok=True)

    base_run_id = _build_base_run_id(run_name=run_name)
    run_id = _ensure_unique_run_id(runs_root, base_run_id)

    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_root = runs_root / run_id

    dataset_paths: dict[str, DatasetRunPaths] = {}
    seen: set[str] = set()
    for dataset in datasets:
        if dataset in seen:
            continue
        seen.add(dataset)
        ds_root = datasets_root / dataset / run_id
        dataset_paths[dataset] = DatasetRunPaths(dataset=dataset, run_id=run_id, root=ds_root)

    paths = ExperimentPaths(
        base_dir=base_dir,
        runs_root=runs_root,
        datasets_root=datasets_root,
        run_id=run_id,
        created_at_utc=created_at_utc,
        run_root=run_root,
        dataset_paths=dataset_paths,
    )
    paths.ensure()
    return paths
