# Agent-Enhanced GraphRAG: Automating Multi-Hop Question Answering with Multi-Agent Systems & Graph Learning

> **Group 09** · Kamal Mustafayev · Fidan Maharlamova · Mahabbat Zakariyayev

---

## Overview

This project builds an **Agent-Enhanced GraphRAG** system that combines knowledge graph traversal with a multi-agent pipeline to answer multi-hop questions — questions that require chaining evidence across multiple documents. The system supports **HotpotQA, MuSiQue, and 2WikiMultiHopQA**, and benchmarks EM/F1 plus latency across baselines and the full agent workflow.

---

## Research Question

> *To what extent does a multi-agent system integrated with GraphRAG improve accuracy, explainability, and automation in answering multi-hop questions on knowledge-intensive tasks compared to standard RAG?*

---

## System Architecture

The system is built on the **LangGraph** framework and consists of four specialized agents operating in sequence, with a feedback loop from the Critic back to the Retriever.

```
Natural Language Query
        │
        ▼
┌─────────────────────┐
│  1. Query Decomposer │  → Structured sub-questions + relation sequence
└────────┬────────────┘     (Llama-3.2-3B, 1 LLM call)
         │
         ▼
┌─────────────────────┐
│  2. Graph Retriever  │  → Explicit evidence chain with node provenance
└────────┬────────────┘     (NetworkX BFS + FAISS + sentence-transformers)
         │
         ▼
┌─────────────────────┐
│  3. ReAct Reasoner  │  → Intermediate answer + confidence score
└────────┬────────────┘     (ReAct-style Reason + Act loop, Llama-3.2-3B)
         │
         ▼
┌─────────────────────┐
│  4. Critic /         │  → Final answer + evidence chain
│     Self-Refine      │     OR refinement signal back to Retriever
└─────────────────────┘
```

### Agent Descriptions

| # | Agent | Role | Output |
|---|-------|------|--------|
| 1 | **Query Decomposer** | Receives the natural language query. Produces an ordered list of sub-questions and a predicted relation sequence using a single Llama-3.2-3B call. | Structured decomposition plan |
| 2 | **Graph Retriever** | Constructs a local KG on-the-fly via BFS over a NetworkX graph + FAISS vector index. Executes embedding-guided edge scoring. | Evidence chain with node-level provenance |
| 3 | **ReAct Reasoner** | Applies ReAct-style (Reason + Act) inference loop to synthesize partial answers across hops. Grounded entirely in retrieved evidence. | Intermediate answer with confidence score |
| 4 | **Critic / Self-Refine** | Evaluates answer completeness. If confidence is below threshold, issues a retrieval-insufficiency signal back to the Graph Retriever. Otherwise approves the final answer. | Final answer + evidence chain OR refinement signal |

### Knowledge Graph Layer

The KG is **not pre-built** — it is constructed dynamically at query time:

- Retrieved passages are parsed into entity-relation triples using lightweight NER and relation extraction
- Entity nodes and relation edges are stored in a **NetworkX** in-memory graph
- Semantic similarity search is provided by a **FAISS** index backed by **sentence-transformers** embeddings
- The entire infrastructure runs within a free Google Colab session at zero cost

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Agent orchestration | LangGraph |
| Language model | Llama-3.2-3B (via Ollama or HuggingFace) |
| Graph storage | NetworkX (in-memory) |
| Vector search | FAISS |
| Embeddings | sentence-transformers |
| Runtime environment | Google Colab (free tier, ≤15 GB VRAM) |
| Datasets | HuggingFace Datasets |

**No paid APIs are required at any stage of the pipeline.**

---

## Datasets

| Dataset | Priority | Size | HuggingFace ID |
|---------|----------|------|----------------|
| **HotpotQA** | Supported | 113K pairs | `hotpot_qa` |
| **MuSiQue** | Supported | 25K pairs | `dgslibisey/MuSiQue` |
| **2WikiMultiHopQA** | Supported | 167K pairs | `voidful/2WikiMultihopQA` |

Evaluation can run on a subset per dataset (default `200`) for reproducible comparisons under limited compute.

---

## Evaluation

Three system configurations are benchmarked against the same test set:

- **B1 — Standard Dense RAG:** sentence-transformers + FAISS, no KG, no agents
- **B2 — Basic GraphRAG:** Edge et al. (2024) pipeline, graph structure but no agents
- **Ours — Agent-Enhanced GraphRAG:** full four-agent pipeline with BFS graph traversal

| Metric | Method | Target vs. Standard RAG |
|--------|--------|--------------------------|
| Exact Match (EM) | HotpotQA official scorer | ≥10% improvement |
| F1 Score | Token-level precision/recall | ≥10% improvement |
| Explainability | LLM-judge (GPT-4o-mini rubric) | ≥0.7 / 1.0 average score |
| Evidence Chain Quality | Manual spot-check (20 samples) | Traceable hops per answer |
| Latency | Wall-clock time per query + per-agent timings | <15s per query on Colab |

Per-query latency logs include total latency and the breakdown for:

- Query Decomposer
- Graph Retriever
- ReAct Reasoner
- Critic

---

## Project Timeline

| Weeks | Phase | Key Activities | Milestone |
|-------|-------|----------------|-----------|
| Wk 1–3 | **Literature Review** | Synthesize 12+ papers; annotated reference table; finalize TOP 3 papers; confirm datasets and feasibility | Milestone 2 — Full paper proposal; state of the art; methodology confirmed |
| Wk 4–8 | **Prototype Implementation** | LangGraph agents (Decomposer, Retriever, ReAct Reasoner, Critic); NetworkX + FAISS KG construction; Llama-3.2-3B via Ollama/HuggingFace on Colab | Milestone 3 — Working prototype; initial qualitative results on HotpotQA |
| Wk 9–10 | **Evaluation** | 200-sample HotpotQA evaluation; ablation study; comparison vs. baselines B1 and B2 | Milestone 4 — Quantitative evaluation; ablation results; comparison table |
| Wk 11–12 | **Analysis & Reporting** | Statistical significance tests; error analysis; final paper write-up; presentation preparation | Final Submission — Full paper; presentation; code repository & reproducibility package |

---

## Feasibility

- All datasets are freely available on HuggingFace
- Llama-3.2-3B runs on the Colab free tier (≤15 GB VRAM)
- NetworkX + FAISS = zero-cost graph infrastructure
- 200-sample eval subset fits in 1–2 hrs Colab runtime
- No paid APIs required at any stage

---

## Key References

1. Shrestha & Kim (2025). *Efficient Multi-Hop QA over KGs via LLM Planning and Embedding-Guided Search.* arXiv:2511.19648
2. Ni et al. (2025). *StepChain GraphRAG: Reasoning Over Knowledge Graphs for Multi-Hop QA.* arXiv:2510.02827
3. Song et al. (2026). *Rethinking Retrieval-Augmented Generation as a Cooperative Decision-Making Problem.* arXiv:2602.18734
4. Edge et al. (2024). *From Local to Global: A GraphRAG Approach to Query-Focused Summarization.* arXiv:2404.16130
5. Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.* arXiv:2210.03629
