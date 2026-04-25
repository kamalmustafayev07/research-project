# Agent-Enhanced GraphRAG Experiment Guide (3-Run Plan)

## 1) Project Goal and Research Framing

Research question:
To what extent does a multi-agent system integrated with GraphRAG improve accuracy, explainability, and automation in multi-hop QA compared with standard RAG?

What the pipeline actually evaluates in each run:
- B1: Standard Dense RAG
- B2: Basic GraphRAG
- OURS: Agent-Enhanced GraphRAG (decomposer, retriever, reasoner, critic)

Each run command in this guide evaluates all three methods above, on all supported datasets.

Supported datasets:
- hotpotqa
- musique
- 2wikimultihopqa

## 2) Configuration Files

Runtime/config files that control behavior:
- .env (primary runtime switches: model backend, model name, retrieval loops, reranker settings, dataset defaults)
- src/config.py (reads .env and defines defaults)
- run_pipeline.py (top-level benchmark runner used in this guide)
- environment.yml (optional conda environment specification; not required for venv flow)

YAML/JSON notes:
- environment.yml is optional if you use venv.
- JSON files under data/processed are prepared dataset caches/splits, not manual config files.
- JSON files under outputs/experiments are generated artifacts and metadata for reproducibility.

## 3) Full Environment Setup (Windows PowerShell)

Run from project root:

    cd C:\Users\Kamal Mustafayev\Desktop\research\research-project

Create and activate virtual environment:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Upgrade pip and install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

Install spaCy model required by knowledge graph entity extraction:

    python -m spacy download en_core_web_sm

Ensure Ollama models are available for the two model runs:

    ollama pull qwen2.5:7b
    ollama pull llama3.2:3b

## 4) Required .env Settings

Use .env in project root. Set these base values before runs:

    LLM_BACKEND=ollama
    EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
    USE_4BIT=true
    TEMPERATURE=0.1
    MAX_NEW_TOKENS=384
    MAX_RETRIEVAL_LOOPS=2
    DATASET_SUBSET_SIZE=200
    DATASET_SPLIT=validation
    TWOWIKI_DATASET_NAME=voidful/2WikiMultihopQA

For reranker-enabled behavior (recommended default):

    USE_RERANKER=true
    RERANKER_WEIGHT=0.4
    RERANKER_NEGATIVES_PER_POSITIVE=3
    RERANKER_MAX_TRAIN_EXAMPLES=1200
    RERANKER_EPOCHS=12

Important: do not need to edit src/config.py directly. It already reads these values from .env.

## 5) Three Experiment Run Commands

These are the only three experiment executions. Each uses all datasets.

### Run 1 (Qwen main run)
Why: strongest-quality model run for primary paper numbers across all datasets.

    $env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="qwen2.5:7b"; $env:USE_RERANKER="true"; $env:MAX_RETRIEVAL_LOOPS="2"; python run_pipeline.py --datasets all --prepare-data --subset-size 200 --split validation --run-name run1_qwen_all

### Run 2 (LLaMA model comparison)
Why: model sensitivity check under same pipeline/settings for fair comparison.

    $env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="llama3.2:3b"; $env:USE_RERANKER="true"; $env:MAX_RETRIEVAL_LOOPS="2"; python run_pipeline.py --datasets all --subset-size 200 --split validation --run-name run2_llama_all

### Run 3 (Methodological variation: disjoint split + reranker training)
Why: demonstrates retrieval-learning contribution with strict train/validation/test separation, and generates high-impact training/confusion visuals.

    $env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="qwen2.5:7b"; $env:USE_RERANKER="true"; $env:RERANKER_MAX_TRAIN_EXAMPLES="1200"; $env:RERANKER_EPOCHS="12"; python run_pipeline.py --datasets all --prepare-data --use-disjoint-splits --source-split validation --test-size 200 --val-size 1000 --train-size 1200 --train-reranker --run-name run3_qwen_disjoint_reranker

## 6) Why These 3 Runs Are the Most Impactful

- Run 1 isolates your best practical model choice (Qwen) with full benchmark coverage and delivers primary headline results.
- Run 2 changes only the LLM backbone (LLaMA) while keeping pipeline structure comparable, giving a clean model-level comparison.
- Run 3 tests a meaningful methodological variation (disjoint training and explicit reranker training), directly tied to retrieval quality and explainability claims.

Together, these three runs show:
- Contribution robustness across model backbones
- Added value of retrieval learning and stricter evaluation protocol
- Full dataset generalization (not single-dataset overfitting)

## 7) Archive Results Safely (No Overwrite)

Create archive root once:

    New-Item -ItemType Directory -Path outputs/archive -Force | Out-Null

After Run 1:

    $runId=(Get-ChildItem outputs/experiments/runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name; $dest="outputs/archive/run_qwen"; New-Item -ItemType Directory -Path $dest -Force | Out-Null; New-Item -ItemType Directory -Path "$dest/runs" -Force | Out-Null; Copy-Item "outputs/experiments/runs/$runId" "$dest/runs/$runId" -Recurse -Force; foreach($ds in "hotpotqa","musique","2wikimultihopqa"){ New-Item -ItemType Directory -Path "$dest/datasets/$ds" -Force | Out-Null; Copy-Item "outputs/experiments/datasets/$ds/$runId" "$dest/datasets/$ds/$runId" -Recurse -Force }

After Run 2:

    $runId=(Get-ChildItem outputs/experiments/runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name; $dest="outputs/archive/run_llama"; New-Item -ItemType Directory -Path $dest -Force | Out-Null; New-Item -ItemType Directory -Path "$dest/runs" -Force | Out-Null; Copy-Item "outputs/experiments/runs/$runId" "$dest/runs/$runId" -Recurse -Force; foreach($ds in "hotpotqa","musique","2wikimultihopqa"){ New-Item -ItemType Directory -Path "$dest/datasets/$ds" -Force | Out-Null; Copy-Item "outputs/experiments/datasets/$ds/$runId" "$dest/datasets/$ds/$runId" -Recurse -Force }

After Run 3:

    $runId=(Get-ChildItem outputs/experiments/runs -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name; $dest="outputs/archive/run_qwen_disjoint_reranker"; New-Item -ItemType Directory -Path $dest -Force | Out-Null; New-Item -ItemType Directory -Path "$dest/runs" -Force | Out-Null; Copy-Item "outputs/experiments/runs/$runId" "$dest/runs/$runId" -Recurse -Force; foreach($ds in "hotpotqa","musique","2wikimultihopqa"){ New-Item -ItemType Directory -Path "$dest/datasets/$ds" -Force | Out-Null; Copy-Item "outputs/experiments/datasets/$ds/$runId" "$dest/datasets/$ds/$runId" -Recurse -Force }

## 8) Where Outputs Are Stored and What They Mean

Per run root:
- outputs/experiments/runs/<run_id>/metadata/run_manifest.json
  - complete run configuration snapshot (args, settings, dataset layout)
- outputs/experiments/runs/<run_id>/logs/pipeline.log
  - pipeline execution log
- outputs/experiments/runs/<run_id>/summaries/benchmark_summary_all_datasets.json
  - main cross-dataset metrics summary
- outputs/experiments/runs/<run_id>/summaries/benchmark_summary_all_datasets.csv
  - table-friendly summary (dataset, method, EM, F1, explainability, latency)
- outputs/experiments/runs/<run_id>/summaries/latency_comparison_all_datasets.json
  - latency means and p95 by dataset/method

Per dataset, per run:
- outputs/experiments/datasets/<dataset>/<run_id>/metadata/dataset_selection.json
  - dataset and split used, limits/sizes, split mode
- outputs/experiments/datasets/<dataset>/<run_id>/metadata/dataset_run_artifacts.json
  - index of generated files for that dataset
- outputs/experiments/datasets/<dataset>/<run_id>/results/benchmark_summary.json
  - B1/B2/OURS metrics for that dataset
- outputs/experiments/datasets/<dataset>/<run_id>/results/benchmark_summary.csv
  - same metrics in CSV form
- outputs/experiments/datasets/<dataset>/<run_id>/results/evidence_chain_analysis.json
  - evidence-chain diagnostics (empty rate, avg hops, malformed rate, relation frequency)
- outputs/experiments/datasets/<dataset>/<run_id>/predictions/*.json and *.csv
  - per-question predictions for B1/B2/OURS
- outputs/experiments/datasets/<dataset>/<run_id>/latency/*.json and *.csv
  - per-question latency totals and per-agent latency breakdown
- outputs/experiments/datasets/<dataset>/<run_id>/evidence/benchmark_ours_evidence_chains.json
  - detailed evidence chains for OURS
- outputs/experiments/datasets/<dataset>/<run_id>/plots/*.png
  - benchmark metrics bars, latency bars, confidence confusion matrices

Run-3-only reranker artifacts (because run3 trains reranker):
- outputs/experiments/datasets/<dataset>/<run_id>/results/reranker_metrics.json
- outputs/experiments/datasets/<dataset>/<run_id>/plots/reranker/reranker_training_history.png
- outputs/experiments/datasets/<dataset>/<run_id>/plots/reranker/reranker_test_confusion_matrix.png

Reusable trained reranker model path:
- outputs/models/passage_reranker.joblib

## 9) What to Include in the Research Paper

Primary metrics to report (for each dataset and method):
- Exact Match (exact_match)
- F1 (f1)
- Explainability (explainability)
- Latency mean (latency_mean)
- Latency p95 (latency_p95)

Required comparison table (recommended structure):
- Rows: dataset x run x method (or dataset x method with separate run columns)
- Columns:
  - Run ID
  - Dataset
  - Method (B1/B2/OURS)
  - EM
  - F1
  - Explainability
  - Latency Mean
  - Latency P95

High-value quantitative highlights:
- Ours vs B1 and Ours vs B2 margins per dataset (EM/F1 deltas)
- Run1 vs Run2 change (model backbone effect)
- Run1 vs Run3 change (methodological variation effect)
- Any trade-off between accuracy gain and latency increase

Most important insights to discuss:
- Whether gains are consistent across all three datasets
- Whether improved retrieval quality (run3) improves explainability and/or EM/F1
- Whether one model is uniformly better, or dataset-dependent
- Whether additional complexity is justified by the performance lift

## 10) What to Include in the Presentation

Use only high-impact visuals directly from generated artifacts.

1. Training curves (run3 reranker)
- File: reranker_training_history.png
- Show train vs validation loss and accuracy by epoch
- Interpretation: stable convergence, overfitting signs, best epoch region

2. Validation/test performance bars
- File: benchmark_metrics_<dataset>.png
- Show B1 vs B2 vs OURS for each dataset
- Interpretation: where multi-agent GraphRAG gives largest gains

3. Confusion matrices
- Reranker confusion (run3): reranker_test_confusion_matrix.png
- Confidence confusion (all runs): confidence_confusion_<dataset>_<method>.png
- Interpretation: calibration quality and error types (false confident vs low-confidence correct)

4. Latency plots
- File: benchmark_latency_<dataset>.png
- Interpretation: operational cost of quality improvements (mean and p95)

Presentation conclusions to emphasize:
- Which run is best overall and why
- Whether quality gains hold across all datasets
- Whether reranker training/disjoint protocol strengthens evidence quality and confidence behavior
- Practical recommendation for deployment setting (quality-first vs speed-first)

## 11) Efficiency and Reproducibility Notes

- We are limited to exactly 3 total runs: this plan already uses that full budget optimally.
- Run 1 prepares cached subset data for all datasets.
- Run 2 reuses cached data for efficiency.
- Run 3 uses disjoint splits and trains reranker with controlled size for meaningful extra signal.
- Sampling is deterministic in code via random seed 42.
- run_name in each command prevents accidental naming confusion.
- Archive commands copy each completed run into a dedicated outputs/archive subfolder to avoid overwrite.

## 12) Quick Execution Checklist

1. Complete environment setup section once.
2. Set .env base values once.
3. Execute Run 1 command, then Run 1 archive command.
4. Execute Run 2 command, then Run 2 archive command.
5. Execute Run 3 command, then Run 3 archive command.
6. Build paper tables from benchmark_summary_all_datasets.csv in each archived run.
7. Build presentation from plot PNG files listed above.
