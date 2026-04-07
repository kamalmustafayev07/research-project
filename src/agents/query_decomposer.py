"""Query decomposition agent for multi-hop planning."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.llm import LLMClient


@dataclass(slots=True)
class DecompositionOutput:
    """Decomposition result schema."""

    sub_questions: list[str]
    relation_sequence: list[str]


class QueryDecomposerAgent:
    """Create a structured decomposition from a natural-language question."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    def run(self, question: str) -> DecompositionOutput:
        """Run decomposition with robust fallback parsing."""
        prompt = (
            "You are a QA planning agent. Given a multi-hop question, return strict JSON with keys "
            "sub_questions (list[str]) and relation_sequence (list[str]).\n"
            f"Question: {question}"
        )
        response = self.llm.generate(prompt)
        parsed = self.llm.extract_json(response.text)

        sub_questions = parsed.get("sub_questions") if isinstance(parsed.get("sub_questions"), list) else []
        relation_sequence = parsed.get("relation_sequence") if isinstance(parsed.get("relation_sequence"), list) else []

        if not sub_questions:
            # Fallback decomposition heuristic.
            chunks = [piece.strip() for piece in question.replace("?", "").split(" and ") if piece.strip()]
            if len(chunks) <= 1:
                chunks = [question.strip()]
            sub_questions = chunks

        if not relation_sequence:
            relation_sequence = ["related_to" for _ in sub_questions]

        return DecompositionOutput(sub_questions=sub_questions, relation_sequence=relation_sequence)
