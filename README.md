# Agent-Enhanced GraphRAG

Code for multi-hop QA over text: LangGraph wires entity linking, query decomposition, dense + graph retrieval, a ReAct-style answerer, and a critic that can trigger more retrieval. Write-up and experiment tables: *Agent-Enhanced GraphRAG: Automating Multi-Hop Question Answering with Multi-Agent Systems and Graph Learning*.

---

## Overview

Given a question and a passage list (benchmark contexts or the unified corpus), the pipeline builds sub-questions, runs hybrid retrieval, constructs a small knowledge graph from hits, scores an evidence chain, asks the LLM for an answer and confidence, and runs a critic. Failed checks bump retrieval again until limits in `MAX_RETRIEVAL_LOOPS` / hop logic stop it.

Dense single-shot RAG is weak when evidence sits in different chunks and entity names must stay consistent across steps. This repo includes **B1** (dense retrieve + one generate call) and **B2** (graph retrieval without the full agent loop) beside **OURS** so everything shares the same embedder, passage pool, and LLM settings where it applies.

Datasets in active use: HotpotQA, MuSiQue, 2WikiMultihopQA. Reports track EM, F1, a simple explainability flag from evidence chains, and latency. Disjoint splits + reranker training live in `run_pipeline.py` behind flags.

---

## Research goal

| Topic | Code |
|-------|------|
| Multi-agent loop | `AgentEnhancedGraphRAG` in `src/pipeline.py` - LangGraph `StateGraph`: linker, decomposer, retriever, reasoner, critic, hop advance / same-hop retry. |
| Graph + dense retrieval | `src/graph/knowledge_graph.py`, `src/graph/retriever.py` - NetworkX graph from text, FAISS over sentence embeddings, BFS from seed entities. |
| Multi-hop plan | `QueryDecomposerAgent`, `hop_safety.py` - sub-questions, relation tags, `HopMemory`. |
| Ranking / rerank | `entity_scoring.py`, `passage_reranker.py`, `retrieval_query_builder.py`; critic text can alter the next retrieval query. |
| Entry points | `main.py` (Typer), `run_pipeline.py` (batch), `app.py` (Streamlit). |

Flow: link → decompose → retrieve → answer → critique; retries until caps in config / graph routing.

---

## Architecture

**Graph.** `src/pipeline.py` defines `GraphRAGState` and compiles the LangGraph.

**Stages.**

- `entity_linker.py` - mentions, canonical strings, referential spans; shared by decomposer and retriever.
- `query_decomposer.py` - LLM JSON plan: sub-questions, relations, per-hop entities; strings in `prompts.py`.
- `graph_retriever.py` - calls `HybridGraphRetriever.retrieve`; exposes `fit_reranker`.
- `react_reasoner.py` - few-step ReAct over the evidence blob; `evidence_validation.py`, `confidence_calibration.py`.
- `critic.py` - hard checks then LLM pass/fail; feeds retries.

**Retrieval.** `HybridGraphRetriever`: `EmbeddingEncoder`, FAISS `IndexFlatIP`, merged query embeddings, `DynamicKnowledgeGraph`, hybrid passage scores, `PassageReranker` when `USE_RERANKER` is on. `preload_corpus()` loads or builds a saved index; `app.py` uses it for the full merged corpus.

**LLM.** `LLMClient` in `src/utils/llm.py`: Hugging Face, Ollama, Azure-compatible HTTP, `mock`. `src/config.py` + `.env`.

**Eval.** `hotpotqa_eval.py` (EM, F1), `utils/evaluation.py` (`run_method_on_dataset`, checkpoints, CSV). `plotting.py` for figures.

**Data.** `data_loader.py` pulls HF examples, subsets, disjoint splits. `corpus_builder.py` merges passages and names the FAISS file under `data/processed/`.

---

## Repository structure

```text
.
├── app.py                  # Streamlit UI; unified corpus + FAISS preload
├── main.py                 # Typer: prepare-data, query, evaluate, smoke-test
├── run_pipeline.py         # Multi-dataset benchmark; B1/B2/OURS; manifests; checkpoints
├── compare_experiments.py  # Read benchmark_summary JSON/CSV from disk
├── setup.py
├── requirements.txt
├── environment.yml         # Conda + PyTorch pins
├── .env                    # Local; read by config (not in git)
├── data/
│   ├── raw/
│   │   └── huggingface_datasets/   # HF cache
│   └── processed/                  # Subsets, disjoint splits, unified_corpus_*.json, *.faiss
├── logs/                   # ProjectPaths.local_logs
├── outputs/
│   ├── archive/            # Frozen copies of past `experiments/` trees
│   ├── experiments/
│   │   ├── runs/           # Per run: metadata/, logs/, summaries/
│   │   └── datasets/       # Per dataset: metadata/, results/, predictions/, latency/, evidence/, plots/
│   ├── models/             # passage_reranker.joblib
│   ├── results/            # ProjectPaths.output_results
│   ├── evidence_chains/    # ProjectPaths.output_evidence
│   └── logs/               # ProjectPaths.output_logs
├── scripts/
└── src/
    ├── config.py           # Paths, model, retrieval, data; SETTINGS.from_env()
    ├── pipeline.py         # AgentEnhancedGraphRAG; B1/B2; reranker train hook
    ├── agents/
    │   ├── entity_linker.py          # Entities and anchors for the question
    │   ├── query_decomposer.py       # LLM plan: sub_questions, relations, entities
    │   ├── graph_retriever.py        # Wraps HybridGraphRetriever
    │   ├── react_reasoner.py         # ReAct over evidence; confidence
    │   ├── critic.py                 # Checks + LLM; retry signal
    │   ├── prompts.py                # Prompt strings
    │   ├── hop_safety.py             # HopMemory, checks, `<answer of hop k>` handling
    │   ├── hop_execution_context.py  # Hop index / retry fields into retriever+reasoner
    │   ├── relation_scorer.py       # Relation hints for retrieval
    │   ├── evidence_validation.py    # Text checks vs evidence
    │   ├── confidence_calibration.py # Adjust reasoner confidence
    │   └── reranker.py               # Re-exports from graph scoring / reranker
    ├── graph/
    │   ├── retriever.py              # HybridGraphRetriever
    │   ├── knowledge_graph.py        # Build graph; spaCy; BFS chains
    │   ├── entity_scoring.py         # Passage and edge scores
    │   ├── passage_reranker.py       # sklearn classifier
    │   └── retrieval_query_builder.py # Query strings and placeholder fill
    ├── evaluation/
    │   └── hotpotqa_eval.py          # EM, F1, aggregates
    └── utils/
        ├── llm.py
        ├── embeddings.py
        ├── data_loader.py
        ├── corpus_builder.py
        ├── evaluation.py
        ├── experiment_paths.py
        ├── plotting.py
        └── helpers.py
```

`run_pipeline.py` and `main.py evaluate` write through `experiment_paths.py` under `outputs/experiments/`. `compare_experiments.py` reads that tree or `outputs/archive/`. `app.py` calls `build_unified_corpus` and `preload_corpus` so the UI searches the merged index in `data/processed/`.

Deleting `data/raw/huggingface_datasets/` forces HF to download again.

Each folder under `outputs/experiments/runs/` is one benchmark run (`run_manifest.json`, `pipeline.log`, `benchmark_summary_all_datasets.*`). Under `datasets/<name>/`, folder names match that run id; inside are the usual `results/`, `predictions/`, etc. After `--train-reranker`, expect `reranker_metrics.json` and `plots/reranker/`.

`archive/` holds copies of the same layout when you snapshot a run before reusing the same run name.

`pipeline.py` owns the graph; `agents/*` call the LLM and mutate state; `graph/*` does vectors, FAISS, and NetworkX. `utils/evaluation.py` only sees dicts from `invoke()` / baselines, not agent imports.

---

## Key components

| Component | Role | Called from |
|-----------|------|-------------|
| `AgentEnhancedGraphRAG` | Compiled graph; `invoke()`; B1/B2; train reranker | `main.py`, `run_pipeline.py`, `app.py` |
| `EntityLinker` | Question entities / anchors | Linker node |
| `QueryDecomposerAgent` | Sub-question plan | Decomposer node |
| `GraphRetrieverAgent` / `HybridGraphRetriever` | Retrieval + graph | Retriever node |
| `ReActReasonerAgent` | Answer from evidence | Reasoner node |
| `CriticAgent` | Pass/fail + critique | Critic node |
| `LLMClient` | Generate / chat backends | Agents |
| `EmbeddingEncoder` | Encode text | Retriever, reranker |
| `DynamicKnowledgeGraph` | Graph build + walk | `HybridGraphRetriever` |

---

## Data flow

1. Input: question + `list` of passages (`title`, `text`, ids if the loader provides them) from a dataset row or corpus slice.
2. Linker fills entity fields on state.
3. Decomposer returns sub-questions and relations; `HopMemory` starts; `retrieval_query_builder` may add query variants.
4. Retriever: embed, FAISS pool, score passages, build graph, walk, return evidence chain + passage list; reranker may mix in.
5. Reasoner: short ReAct; may return early on failed checks.
6. Critic: rules + LLM; graph routes to retry hop or next hop or stop.
7. Output: answer, confidence, evidence chain, latency dict, loop counts - CSV/JSON in `outputs/` or UI.

---

## Installation

PowerShell, repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
```

`.env`:

```env
LLM_BACKEND=ollama
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
USE_4BIT=true
TEMPERATURE=0.1
MAX_NEW_TOKENS=384
MAX_RETRIEVAL_LOOPS=2
DATASET_SUBSET_SIZE=200
DATASET_SPLIT=validation
TWOWIKI_DATASET_NAME=voidful/2WikiMultihopQA
USE_RERANKER=true
RERANKER_WEIGHT=0.4
RERANKER_NEGATIVES_PER_POSITIVE=3
RERANKER_MAX_TRAIN_EXAMPLES=1200
RERANKER_EPOCHS=12
```

`LLM_BACKEND=azure`:

```env
LLM_BACKEND=azure
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_DEPLOYMENT=<deployment>
AZURE_OPENAI_API_VERSION=2024-12-01
```

`src/config.py` imports `dotenv` and reads `.env` at startup.

---

## Usage / running

`--datasets all` → `hotpotqa`, `musique`, `2wikimultihopqa`.

**Ollama - full benchmark, Qwen, refresh cached subsets.** Runs B1/B2/OURS on all three datasets; `--prepare-data` rewrites prepared JSON under `data/processed/`.

```powershell
$env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="qwen2.5:7b"; $env:USE_RERANKER="true"; $env:MAX_RETRIEVAL_LOOPS="2"; python run_pipeline.py --datasets all --prepare-data --subset-size 200 --split validation --run-name run1_qwen_all
```

**Ollama - same benchmark, LLaMA.** Same flags as above but no `--prepare-data` so it reuses data from the last prepare if still on disk.

```powershell
$env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="llama3.2:3b"; $env:USE_RERANKER="true"; $env:MAX_RETRIEVAL_LOOPS="2"; python run_pipeline.py --datasets all --subset-size 200 --split validation --run-name run2_llama_all
```

**Ollama - disjoint train/val/test, train reranker, evaluate on held-out test.** Writes `reranker_metrics.json` and reranker plots under each dataset run folder when training finishes.

```powershell
$env:LLM_BACKEND="ollama"; $env:OLLAMA_MODEL="qwen2.5:7b"; $env:USE_RERANKER="true"; $env:RERANKER_MAX_TRAIN_EXAMPLES="1200"; $env:RERANKER_EPOCHS="12"; python run_pipeline.py --datasets all --prepare-data --use-disjoint-splits --source-split validation --test-size 200 --val-size 1000 --train-size 1200 --train-reranker --run-name run3_qwen_disjoint_reranker
```

Azure:

**Azure - full benchmark with prepare-data.** Same role as the first Ollama command; LLM traffic goes to the deployment in `AZURE_OPENAI_*`.

```powershell
$env:LLM_BACKEND="azure"; $env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/v1"; $env:AZURE_OPENAI_API_KEY="<key>"; $env:AZURE_OPENAI_DEPLOYMENT="<deployment>"; $env:AZURE_OPENAI_API_VERSION="2024-12-01"; $env:USE_RERANKER="true"; $env:MAX_RETRIEVAL_LOOPS="2"; python run_pipeline.py --datasets all --prepare-data --subset-size 200 --split validation --run-name run1_azure_all
```

**Azure - no `--prepare-data`.** Reuses prepared files under `data/processed/`; set `AZURE_OPENAI_DEPLOYMENT` (and related vars) to a different deployment if you want another model.

```powershell
$env:LLM_BACKEND="azure"; $env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/v1"; $env:AZURE_OPENAI_API_KEY="<key>"; $env:AZURE_OPENAI_DEPLOYMENT="<deployment>"; $env:AZURE_OPENAI_API_VERSION="2024-12-01"; $env:USE_RERANKER="true"; $env:MAX_RETRIEVAL_LOOPS="2"; python run_pipeline.py --datasets all --subset-size 200 --split validation --run-name run2_azure_all
```

**Azure - disjoint splits + reranker training.** Same layout as the third Ollama command.

```powershell
$env:LLM_BACKEND="azure"; $env:AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/openai/v1"; $env:AZURE_OPENAI_API_KEY="<key>"; $env:AZURE_OPENAI_DEPLOYMENT="<deployment>"; $env:AZURE_OPENAI_API_VERSION="2024-12-01"; $env:USE_RERANKER="true"; $env:RERANKER_MAX_TRAIN_EXAMPLES="1200"; $env:RERANKER_EPOCHS="12"; python run_pipeline.py --datasets all --prepare-data --use-disjoint-splits --source-split validation --test-size 200 --val-size 1000 --train-size 1200 --train-reranker --run-name run3_azure_disjoint_reranker
```

**Typer evaluate - one or more datasets via flags.** Lighter than `run_pipeline.py`; still runs B1/B2/OURS but uses `main.py`’s output layout under `outputs/experiments/`.

```powershell
python main.py evaluate --datasets hotpotqa --subset-size 200 --split validation
```

**Compare saved summaries.** Reads `benchmark_summary_all_datasets.json` / per-dataset summaries under `outputs/experiments` (or a run id you pass).

```powershell
python compare_experiments.py --results-dir outputs/experiments
```

Summaries: `outputs/experiments/runs/*/summaries/`. Per-dataset metrics: `outputs/experiments/datasets/*/*/results/`. Reranker weights: `outputs/models/passage_reranker.joblib`. Copy a run tree into `outputs/archive/` before overwriting if you need to keep the old files.

---

## Streamlit app

`app.py` runs the same `AgentEnhancedGraphRAG` graph as batch code. Sidebar sets backend, model, temperature, and retrieval loop count. First load builds or loads the unified corpus and FAISS index under `data/processed/`. `app.py` has extra hop bookkeeping (LLM extract for early hops) so the UI stays consistent on multi-hop questions.

```powershell
python -m streamlit run app.py
```

---

## Experiments / evaluation

- **Methods:** `B1_Standard_Dense_RAG`, `B2_Basic_GraphRAG`, `Ours_Agent_Enhanced_GraphRAG` - `dense_rag_baseline` / `basic_graphrag_baseline` in `src/pipeline.py`.
- **Metrics:** Exact match, token F1, explainability helper from `heuristic_explainability_score`, mean and p95 latency; per-stage times in the latency breakdown.
- **Artifacts:** `benchmark_summary.*`, per-method prediction CSV/JSON, latency files, `evidence_chain_analysis.json`, PNG plots from `plotting.py`.
- **Disjoint splits:** `prepare_dataset_disjoint_splits` + `--train-reranker` in `run_pipeline.py` (train / val / test indices, reranker fit on train).
- **Checkpoints:** `run_method_on_dataset` resumes from `checkpoint_dir` (batch uses `results/checkpoints`).

---

## Tech stack

| Area | Stack |
|------|--------|
| Graph runner | LangGraph |
| LLM | `transformers`, `bitsandbytes`, `ollama`, Azure HTTP client, `mock` |
| Vectors | `sentence-transformers`, `faiss-cpu` |
| Graph | NetworkX |
| NER / KG | spaCy |
| Reranker | scikit-learn |
| Data | `datasets`, pandas |
| UI / CLI | Streamlit, Typer, Rich |
| Other | PyTorch, python-dotenv, tqdm, matplotlib, graphviz, rapidfuzz, `evaluate` |

`langchain` and `langchain-community` are in `requirements.txt` but unused in `src/` imports.

---

## Current status

Benchmark / research codebase: not a hosted API. Hop and retriever behavior changes with prompts and LLM backend. `run_pipeline.py` normalizes some fields when it reuses `app.run_pipeline` for OURS so eval keys stay consistent. Reranker + disjoint path is off unless you pass the flags.

---

## Notes

- **Device:** `config.detect_device()` picks CUDA, then MPS, else CPU.
- **Windows:** `pathlib` throughout; `faiss-cpu` wheel in requirements.
- **Cost:** Azure and large HF models cost money; `LLM_BACKEND=mock` and small `--limit` are cheap for debugging.
- **Authors:** Kamal Mustafayev, Fidan Maharlamova, Mahabbat Zakariyayev
