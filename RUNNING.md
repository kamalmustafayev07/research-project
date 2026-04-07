# Running Agent-Enhanced GraphRAG

## 1) Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Optional CUDA stack:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 2) Configure environment

```bash
cp .env.example .env
```

Set one of these backends in `.env`:

- `LLM_BACKEND=transformers` for HuggingFace local inference
- `LLM_BACKEND=ollama` for local Ollama serving (`ollama pull llama3.2:3b`)
- `LLM_BACKEND=mock` for deterministic smoke tests

## 3) Prepare data

```bash
python main.py prepare-data --subset-size 200
```

## 4) Run single query

```bash
python main.py query "Which country is the city where Tesla was founded located in?"
```

## 5) Evaluate B1/B2/Ours

```bash
python main.py evaluate --subset-size 200
```

Quick test run:

```bash
LLM_BACKEND=mock python main.py evaluate --subset-size 20 --limit 10
```

## 6) End-to-end script

```bash
python run_pipeline.py --prepare-data --subset-size 200
```

Outputs are written to:

- `outputs/results/`
- `outputs/evidence_chains/`
- `outputs/logs/`

## 7) Google Colab

Use `colab_notebook.ipynb` directly in Colab. It installs dependencies, prepares data, and runs a full benchmark pipeline.
