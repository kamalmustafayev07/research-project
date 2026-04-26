# app.py
"""
Streamlit UI for Agent-Enhanced GraphRAG
Uses same pipeline logic as run_pipeline.py

Features:
- User asks short natural question
- Query goes through:
    1. Decomposer
    2. Retriever
    3. Passage Reranker (outputs/models/)
    4. Reasoner
    5. Critic
- Final answer shown
- Evidence chain visualization
- Retrieved passages
- Agent traces
- Performance metrics
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import SETTINGS
from src.agents.entity_linker import build_retrieval_entity_context
from src.graph.retrieval_query_builder import build_retrieval_query_pack, resolve_placeholders
from src.pipeline import AgentEnhancedGraphRAG
from src.utils.corpus_builder import build_unified_corpus, corpus_paths, DEFAULT_SPLITS
from src.evaluation.hotpotqa_eval import heuristic_explainability_score

# ==========================================================
# CONFIG
# ==========================================================

DEFAULT_MODEL = "qwen2.5:7b"

# Load the complete train + validation splits of all three datasets.
# None = no cap (download everything available).
# The passages JSON and FAISS index are cached to disk so this only
# happens once; subsequent restarts load from cache in seconds.
CORPUS_MAX_PER_SPLIT: int | None = None
CORPUS_SPLITS = DEFAULT_SPLITS  # ["train", "validation"]


# ==========================================================
# SETTINGS
# ==========================================================

def apply_settings(model_name: str, temperature: float, loops: int):
    root = Path(__file__).resolve().parent

    SETTINGS.paths.root = root
    SETTINGS.model.llm_backend = "ollama"
    SETTINGS.model.ollama_model = model_name
    SETTINGS.model.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    SETTINGS.model.temperature = temperature
    SETTINGS.model.max_new_tokens = 384

    SETTINGS.retrieval.max_retrieval_loops = loops
    SETTINGS.retrieval.use_reranker = True

    SETTINGS.paths.output_models.mkdir(parents=True, exist_ok=True)


# ==========================================================
# LOAD PIPELINE + CORPUS (cached together so the FAISS index
# is built once per session and reused across queries)
# ==========================================================

@st.cache_resource(show_spinner=False)
def get_pipeline_and_corpus(model_name: str, temperature: float, loops: int):
    """Initialise pipeline and pre-load the FAISS index over the full corpus.

    Both objects are cached by ``@st.cache_resource`` so re-runs of the
    Streamlit script reuse the same pipeline and index without re-encoding
    passages on every user interaction.

    On first run the three datasets are downloaded from HuggingFace and saved
    to disk; the FAISS index is built from the GPU-encoded embeddings and also
    persisted to disk.  Subsequent restarts skip both steps and are fast.
    """
    apply_settings(model_name, temperature, loops)
    pipeline = AgentEnhancedGraphRAG()

    # Build (or load from disk) the unified passage corpus.
    corpus = build_unified_corpus(
        max_per_split=CORPUS_MAX_PER_SPLIT,
        splits=CORPUS_SPLITS,
    )

    # Resolve the persistent FAISS index path that matches this corpus version.
    _, faiss_path = corpus_paths(CORPUS_MAX_PER_SPLIT, CORPUS_SPLITS)

    # Load or build the FAISS index; save it to disk for future restarts.
    pipeline.retriever.retriever.preload_corpus(corpus, index_path=faiss_path)

    return pipeline, corpus


# ==========================================================
# RUN WITH TRACES
# ==========================================================

def run_pipeline(
    pipeline: AgentEnhancedGraphRAG,
    question: str,
    contexts: list[dict[str, Any]],
):
    traces = []
    latency = {}

    total_start = perf_counter()

    # ------------------------------------------------------
    # 1. ENTITY LINKING
    # ------------------------------------------------------
    t_link = perf_counter()
    linked = pipeline.entity_linker.link(question)
    latency["entity_linker"] = perf_counter() - t_link
    traces.append({"agent": "entity_linker", "output": linked.as_dict()})

    # ------------------------------------------------------
    # 2. DECOMPOSER
    # ------------------------------------------------------
    t0 = perf_counter()
    decomposition = pipeline.decomposer.run(question, entity_linking=linked)
    latency["decomposer"] = perf_counter() - t0

    traces.append({
        "agent": "decomposer",
        "output": asdict(decomposition)
    })

    entity_context = build_retrieval_entity_context(
        linked, decomposition.sub_question_entities, decomposition.relation_sequence
    )

    # ------------------------------------------------------
    # 3. RETRIEVER + RERANKER
    # ------------------------------------------------------
    ref_anchors = linked.as_dict().get("referential_anchors", [])
    query = question
    if decomposition.sub_questions:
        # Resolve <answer of hop N> so the primary query string is grammatically
        # correct and semantically meaningful (e.g. "Which university did the
        # writer of The Hobbit attend?" instead of raw placeholder text).
        resolved_sqs = [
            resolve_placeholders(sq, ref_anchors)
            for sq in decomposition.sub_questions[:2]
        ]
        query += " ; " + " ; ".join(resolved_sqs)
    extra = []
    for row in decomposition.sub_question_entities[:3]:
        extra.extend(row.get("resolved_entities") or [])
    if extra:
        query += " ; " + " ; ".join(str(x) for x in extra[:4])

    t1 = perf_counter()
    ec: dict[str, Any] | None = None
    if entity_context.get("match_strings"):
        ec = dict(entity_context)
        variants, q_dbg = build_retrieval_query_pack(question, asdict(decomposition), ec)
        ec["retrieval_queries"] = variants
        ec["retrieval_query_debug"] = q_dbg
    retrieval = pipeline.retriever.run(query, contexts, entity_context=ec)
    latency["retriever+rereanker"] = perf_counter() - t1

    selected = retrieval.selected_passages
    chain = retrieval.evidence_chain

    traces.append({
        "agent": "retriever",
        "query": query,
        "retrieval_queries": (ec or {}).get("retrieval_queries") if ec else None,
        "retrieval_query_debug": (ec or {}).get("retrieval_query_debug") if ec else None,
        "selected_passages": selected[:5],
        "graph_stats": retrieval.graph_stats,
    })

    # ------------------------------------------------------
    # 3. REASONER
    # ------------------------------------------------------
    t2 = perf_counter()
    ec_for_agents = ec if ec is not None else entity_context
    reasoned = pipeline.reasoner.run(
        question,
        evidence_chain=chain,
        selected_passages=selected,
        entity_context=ec_for_agents,
        decomposition=asdict(decomposition),
    )
    latency["reasoner"] = perf_counter() - t2

    traces.append({
        "agent": "reasoner",
        "thoughts": reasoned.thoughts,
        "answer": reasoned.answer,
        "confidence": reasoned.confidence,
    })

    # ------------------------------------------------------
    # 4. CRITIC
    # ------------------------------------------------------
    t3 = perf_counter()
    judged = pipeline.critic.run(
        query=question,
        answer=reasoned.answer,
        confidence=reasoned.confidence,
        evidence_count=len(chain),
        evidence_chain=chain,
        selected_passages=selected,
        entity_context=ec_for_agents,
        decomposition=asdict(decomposition),
    )
    latency["critic"] = perf_counter() - t3

    traces.append({
        "agent": "critic",
        "approved": judged.approved,
        "critique": judged.critique,
        "confidence": judged.confidence,
    })

    # ------------------------------------------------------
    # FINALIZE
    # ------------------------------------------------------
    answer = pipeline._finalize_answer(
        question=question,
        raw_answer=reasoned.answer,
        evidence_chain=chain,
        selected_passages=selected,
    )

    total = perf_counter() - total_start

    explainability = heuristic_explainability_score(chain, answer)

    return {
        "answer": answer,
        "confidence": judged.confidence,
        "approved": judged.approved,
        "critique": judged.critique,
        "traces": traces,
        "chain": chain,
        "selected": selected,
        "latency": latency,
        "total": total,
        "explainability": explainability,
        "sub_questions": decomposition.sub_questions,
    }


# ==========================================================
# UI helpers
# ==========================================================

def _format_latency(seconds: float) -> str:
    """Human-readable latency: always show milliseconds; show seconds when >= 1 s."""
    ms = float(seconds) * 1000.0
    if seconds >= 1.0:
        return f"{ms:.1f} ms ({seconds:.3f} s)"
    return f"{ms:.1f} ms"


def main():
    st.set_page_config(
        page_title="Agent GraphRAG",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Agent-Enhanced GraphRAG")
    st.caption("4 Agents + Passage Reranker + Evidence Chain")

    # ------------------------------------------------------
    # SIDEBAR
    # ------------------------------------------------------
    st.sidebar.header("Settings")

    model_name = st.sidebar.selectbox(
        "Model",
        [DEFAULT_MODEL, "qwen2.5:14b", "llama3.2:3b"]
    )

    temperature = st.sidebar.slider(
        "Temperature",
        0.0, 1.0, 0.1, 0.05
    )

    loops = st.sidebar.slider(
        "Max Retrieval Loops",
        1, 4, 2
    )

    # ------------------------------------------------------
    # LOAD
    # ------------------------------------------------------
    with st.spinner(
        "Loading corpus + models … "
        "(first run: downloads all 3 datasets + builds FAISS index — takes a few minutes)"
    ):
        pipeline, contexts = get_pipeline_and_corpus(model_name, temperature, loops)

    st.success(f"Loaded {len(contexts):,} passages from 3 datasets (train + validation)")

    # ------------------------------------------------------
    # INPUT
    # ------------------------------------------------------
    question = st.text_input(
        "Ask short question",
        placeholder="director of dunkirk studied where?"
    )

    if st.button("Get Answer", use_container_width=True):

        if not question.strip():
            st.warning("Enter question.")
            st.stop()

        with st.spinner("Running 4-agent pipeline..."):
            result = run_pipeline(pipeline, question, contexts)

        # --------------------------------------------------
        # TABS
        # --------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "Answer",
            "Evidence Chain",
            "Retrieved Passages",
            "Agents / Metrics"
        ])

        # ==================================================
        # ANSWER
        # ==================================================
        with tab1:
            st.subheader("Final Answer")
            st.markdown(f"## {result['answer']}")

            c1, c2, c3 = st.columns(3)

            c1.metric("Confidence", f"{result['confidence']:.3f}")
            expl = float(result["explainability"])
            c2.metric(
                "Explainability (0–100)",
                f"{expl * 100:.1f}",
                help="Scaled from the 0–1 heuristic used in evaluation (×100).",
            )
            c3.metric("Approved", "YES" if result["approved"] else "NO")

            st.info(result["critique"])

            if result["sub_questions"]:
                st.markdown("### Multi-hop decomposition")
                for i, q in enumerate(result["sub_questions"], 1):
                    st.markdown(f"- Hop {i}: {q}")

        # ==================================================
        # EVIDENCE CHAIN
        # ==================================================
        with tab2:
            st.subheader("Evidence Chain")

            for i, hop in enumerate(result["chain"], 1):
                with st.expander(f"Hop {i}"):

                    st.write("From:", hop.get("node_from"))
                    st.write("To:", hop.get("node_to"))
                    st.write("Relation:", hop.get("relation"))
                    st.write("Score:", round(float(hop.get("score", 0)), 3))
                    st.write(hop.get("text", ""))

        # ==================================================
        # PASSAGES
        # ==================================================
        with tab3:
            st.subheader("Top Retrieved Passages")

            for i, p in enumerate(result["selected"][:10], 1):
                with st.expander(f"Passage {i}"):

                    st.write("Dataset:", p.get("dataset"))
                    st.write("Title:", p.get("title"))

                    st.write(
                        "Final Score:",
                        round(float(p.get("score", 0)), 3)
                    )

                    if "rerank_score" in p:
                        st.write(
                            "Reranker:",
                            round(float(p["rerank_score"]), 3)
                        )

                    st.write(p.get("text", ""))

        # ==================================================
        # TRACES / METRICS
        # ==================================================
        with tab4:
            st.subheader("Agent Traces")

            for tr in result["traces"]:
                with st.expander(tr["agent"].upper()):
                    st.json(tr)

            if result.get("debug_metrics"):
                st.subheader("Benchmark debug metrics")
                st.json(result["debug_metrics"])

            st.subheader("Latency")

            lat = result["latency"]
            order = [
                ("Entity linker", "entity_linker"),
                ("Decomposer", "decomposer"),
                ("Retriever + reranker", "retriever+rereanker"),
                ("Reasoner", "reasoner"),
                ("Critic", "critic"),
            ]
            labels = [lbl for lbl, key in order if key in lat]
            cols = st.columns(len(labels) or 1)
            col_idx = 0
            for label, key in order:
                if key not in lat:
                    continue
                cols[col_idx].metric(label, _format_latency(lat[key]))
                col_idx += 1

            st.metric("Total wall time", _format_latency(result["total"]))


if __name__ == "__main__":
    main()