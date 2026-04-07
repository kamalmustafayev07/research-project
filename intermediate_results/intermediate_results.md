# Intermediate Results (Current Session)

This file records what was run and what was observed in order.

## 1) Dependency import check (before install)
- Command:
  - `.venv/bin/python -c "import typer, dotenv, datasets, networkx, faiss, sentence_transformers, langgraph; print('deps_ok')"`
- Result:
  - Failed with `ModuleNotFoundError: No module named 'datasets'`

## 2) Install requirements
- Command:
  - `.venv/bin/pip install -r requirements.txt`
- Result:
  - Completed successfully

## 3) Dependency import check (after install)
- Command:
  - `.venv/bin/python -c "import typer, dotenv, datasets, networkx, faiss, sentence_transformers, langgraph; print('deps_ok')"`
- Result:
  - `deps_ok`

## 4) Data preparation (HotpotQA subset = 200)
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py prepare-data --subset-size 200`
- Result:
  - Download/preprocessing finished
  - `Prepared 200 samples in data/processed.`

## 5) Smoke test (10 samples)
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py smoke-test`
- Result:
  - Completed successfully
  - Output saved to `outputs/results/smoke_test_results.json`
- Metrics in output file:
  - exact_match: `0.0`
  - f1: `0.0`
  - explainability: `0.66`

## 6) Evaluation on 20 samples
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py evaluate --subset-size 200 --limit 20`
- Console summary:
  - B1: EM `0.000`, F1 `0.000`, Explainability `0.600`
  - B2: EM `0.000`, F1 `0.000`, Explainability `0.540`
  - Ours: EM `0.000`, F1 `0.003`, Explainability `0.590`
- Result path:
  - `outputs/results/evaluation_results.json`

## 7) Full evaluation on 200 samples
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py evaluate --subset-size 200`
- Console summary:
  - B1: EM `0.000`, F1 `0.000`, Explainability `0.598`
  - B2: EM `0.000`, F1 `0.000`, Explainability `0.529`
  - Ours: EM `0.000`, F1 `0.000`, Explainability `0.573`
- Result paths:
  - `outputs/results/evaluation_results.json`
  - `outputs/evidence_chains/ours_evidence_chains.json`

## 8) Single query run
- Command:
  - `LLM_BACKEND=mock .venv/bin/python main.py query "Which country is the city where Tesla was founded located in?" --sample-index 0`
- Console output (key fields):
  - Answer: `Mock answer based on retrieved evidence.`
  - Confidence: `0.780`
  - Retrieval Loops: `2`
  - Evidence Hops: `0`

## 9) Individual agent functional check
- Command:
  - `PYTHONPATH=. LLM_BACKEND=mock .venv/bin/python /tmp/agent_tests.py`
- Output:
  - DEVICE: `cpu`
  - BACKEND: `mock`
  - QUERY_DECOMPOSER_SUBQS: `2`
  - GRAPH_RETRIEVER_HOPS: `10`
  - REASONER_CONF: `0.72`
  - CRITIC_APPROVED: `True`
  - CRITIC_CONF: `0.78`

## 10) Notebook lint/runtime fix applied
- File edited:
  - `colab_notebook.ipynb`
- Change:
  - Replaced `!pip install ...` with `%pip install ...` in install cell
- Validation:
  - No notebook errors reported after edit

## Important context for interpreting these numbers
- All reported benchmark values here were produced with `LLM_BACKEND=mock`.
- This validates pipeline orchestration/reproducibility, not final model quality.
- Re-run the same commands with `LLM_BACKEND=ollama` or `LLM_BACKEND=transformers` for real QA performance.
