# Running Agent-Enhanced GraphRAG

## 1) Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Windows PowerShell equivalent:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
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

Windows PowerShell equivalent:

```powershell
Copy-Item .env.example .env
```

Set one of these backends in `.env`:

- `LLM_BACKEND=transformers` for HuggingFace local inference
- `LLM_BACKEND=ollama` for local Ollama serving (`ollama pull llama3.2:3b`)
- `LLM_BACKEND=mock` for deterministic smoke tests

### Run with real model outputs (Ollama + NVIDIA GPU)

1. Install/update NVIDIA driver and verify GPU is visible:

```powershell
nvidia-smi
```

2. Pull a model in Ollama:

```powershell
ollama pull llama3.2:3b
```

3. Ensure `.env` uses Ollama (not mock):

```env
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3.2:3b
```

4. Start Ollama service/app, then verify model responds:

```powershell
ollama run llama3.2:3b "Say hello in one sentence"
```

5. Run project commands normally (the pipeline will call Ollama and produce real outputs).

If your RTX 5060 has enough VRAM and you want better quality, try:

```powershell
ollama pull qwen2.5:7b
```

then set `OLLAMA_MODEL=qwen2.5:7b` in `.env`.

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

Quick real-output test (PowerShell):

```powershell
$env:LLM_BACKEND="ollama"
python main.py evaluate --subset-size 20 --limit 10
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
