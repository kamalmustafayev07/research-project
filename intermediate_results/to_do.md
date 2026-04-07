# To-Do Runbook (Ordered)

This checklist reproduces everything already executed in this project, in order.

## 0) Open project root
- Ensure you are in the repository root:
  - `cd /home/ufaz/Desktop/research_project/research-project`

## 1) Create and activate virtual environment
- Create venv:
  - `python3 -m venv .venv`
- Activate it:
  - `source .venv/bin/activate`

## 2) Install dependencies
- Install all packages:
  - `.venv/bin/pip install -r requirements.txt`

## 3) Verify critical imports
- Run import smoke check:
  - `.venv/bin/python -c "import typer, dotenv, datasets, networkx, faiss, sentence_transformers, langgraph; print('deps_ok')"`
- Expected result:
  - `deps_ok`

## 4) Prepare HotpotQA subset (200 samples)
- Download + preprocess:
  - `LLM_BACKEND=mock .venv/bin/python main.py prepare-data --subset-size 200`
- Expected result:
  - Prepared 200 samples in `data/processed`

## 5) Run smoke test (quick end-to-end)
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py smoke-test`
- Expected artifact:
  - `outputs/results/smoke_test_results.json`

## 6) Run 20-sample benchmark (B1, B2, Ours)
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py evaluate --subset-size 200 --limit 20`
- Expected artifact:
  - `outputs/results/evaluation_results.json` (contains current run output)

## 7) Run full 200-sample benchmark (B1, B2, Ours)
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py evaluate --subset-size 200`
- Expected artifacts:
  - `outputs/results/evaluation_results.json`
  - `outputs/evidence_chains/ours_evidence_chains.json`

## 8) Run single-query demo
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py query "Which country is the city where Tesla was founded located in?" --sample-index 0`

## 9) Run individual agent checks
- Command (script already used during validation):
  - `PYTHONPATH=. LLM_BACKEND=mock .venv/bin/python /tmp/agent_tests.py`
- Purpose:
  - Verifies Query Decomposer, Graph Retriever, ReAct Reasoner, and Critic paths run successfully.

## 10) Optional: switch from mock to real model backend
- Ollama path (local):
  - Set env: `LLM_BACKEND=ollama`
  - Ensure model exists: `ollama pull llama3.2:3b`
- Transformers path:
  - Set env: `LLM_BACKEND=transformers`
  - Ensure access to model in config/environment.

## Notes
- The reported benchmark numbers in this validation were generated with `LLM_BACKEND=mock` for deterministic and fast reproducibility.
- For meaningful EM/F1 quality, rerun with a real backend (`ollama` or `transformers`).
