"""
Definitive trace of the pipeline for the primary test question.
Prints exactly what each agent outputs so you can compare against the old broken Streamlit log.
"""
import json

PASSAGES = [
    {
        "title": "The Hobbit",
        "text": (
            "The Hobbit, or There and Back Again is a children's fantasy novel by "
            "J. R. R. Tolkien. It was published in 1937. Tolkien attended Exeter College, "
            "Oxford, where he studied English Language and Literature."
        ),
    },
    {
        "title": "J. R. R. Tolkien",
        "text": (
            "John Ronald Reuel Tolkien was an English author and philologist. "
            "He studied at King Edward's School, Birmingham, and later at Exeter College, "
            "Oxford. He is best known for The Hobbit and The Lord of the Rings."
        ),
    },
    {
        "title": "Exeter College, Oxford",
        "text": (
            "Exeter College is one of the constituent colleges of the University of Oxford. "
            "Notable alumni include J. R. R. Tolkien, who read English at Exeter."
        ),
    },
]

Q = "Which university did the writer of The Hobbit attend?"

SEP = "=" * 64

# ------------------------------------------------------------------ #
# ENTITY LINKER
# ------------------------------------------------------------------ #
from src.agents.entity_linker import (
    EntityLinker,
    build_retrieval_entity_context,
)

linker = EntityLinker()
link = linker.link(Q)
link_dict = link.as_dict()

print(SEP)
print("ENTITY_LINKER")
print(SEP)
print(json.dumps({"agent": "entity_linker", "output": link_dict}, indent=2))

# ------------------------------------------------------------------ #
# DECOMPOSER
# ------------------------------------------------------------------ #
from src.agents.query_decomposer import QueryDecomposerAgent
from src.utils.llm import LLMClient

llm = LLMClient()
decomp_agent = QueryDecomposerAgent(llm)
decomp = decomp_agent.run(Q, entity_linking=link)

ec = build_retrieval_entity_context(
    link, decomp.sub_question_entities, decomp.relation_sequence
)

decomp_dict = {
    "sub_questions": decomp.sub_questions,
    "relation_sequence": decomp.relation_sequence,
    "sub_question_entities": decomp.sub_question_entities,
}

print()
print(SEP)
print("DECOMPOSER")
print(SEP)
print(json.dumps({"agent": "decomposer", "output": decomp_dict}, indent=2))

# Key assertions
print()
print("--- ASSERTION CHECKS ---")
sqs = decomp.sub_questions
assert len(sqs) >= 2, f"Expected >=2 sub_questions, got {len(sqs)}"
assert "<answer of hop 1>" in sqs[1] or "the writer of The Hobbit" in sqs[1].lower(), (
    f"Sub-question 2 must contain placeholder or resolved text, got: {sqs[1]!r}"
)
print(f"  sub_question[1] = {sqs[1]!r}  -- OK (contains placeholder or resolved text)")

for sq_ent in decomp.sub_question_entities:
    for k in (sq_ent.get("entity_map") or {}):
        assert "<answer" not in k.lower(), f"Placeholder leaked into entity_map key: {k!r}"
print("  No <answer of hop N> keys in entity_map  -- OK")

# ------------------------------------------------------------------ #
# RETRIEVER query pack
# ------------------------------------------------------------------ #
from src.graph.retrieval_query_builder import build_retrieval_query_pack

variants, debug = build_retrieval_query_pack(Q, decomp_dict, ec)

print()
print(SEP)
print("RETRIEVER — query pack")
print(SEP)
print(json.dumps({"agent": "retriever", "debug": debug, "retrieval_queries": variants}, indent=2))

# Key assertions
print()
print("--- ASSERTION CHECKS ---")
joined = " | ".join(variants).lower()
assert "answer of hop" not in joined, "Literal <answer of hop N> leaked into queries!"
print("  No literal placeholder in query variants  -- OK")
assert any("the writer of the hobbit" in v.lower() or "who wrote the hobbit" in v.lower() for v in variants), (
    "Expected a resolved sub-question query like 'Which university did the writer of The Hobbit attend?'"
)
print("  Resolved sub-question present in query variants  -- OK")

# ------------------------------------------------------------------ #
# FULL PIPELINE
# ------------------------------------------------------------------ #
from src.pipeline import AgentEnhancedGraphRAG

print()
print(SEP)
print("FULL PIPELINE")
print(SEP)

pipeline = AgentEnhancedGraphRAG()
result = pipeline.invoke(Q, PASSAGES)

print(json.dumps(
    {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "retrieval_loops": result["retrieval_loops"],
        "critique": result["critique"],
        "evidence_count": len(result["evidence_chain"]),
        "top_evidence": [
            {"source": h.get("source", "?"), "text_snippet": str(h.get("text", ""))[:100]}
            for h in result["evidence_chain"][:4]
        ],
    },
    indent=2,
))

print()
ans = result["answer"].lower()
assert "oxford" in ans or "exeter" in ans, f"Expected Oxford/Exeter, got: {result['answer']!r}"
print(f"[FINAL PASS]  answer={result['answer']!r}  confidence={result['confidence']}")
