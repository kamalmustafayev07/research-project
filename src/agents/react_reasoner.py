"""ReAct-style reasoner agent grounded in retrieved evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.helpers import safe_float
from src.utils.llm import LLMClient


@dataclass(slots=True)
class ReasonerOutput:
    """Reasoner result schema."""

    thoughts: list[str]
    answer: str
    confidence: float


class ReActReasonerAgent:
    """Run a compact ReAct loop to synthesize an answer from evidence."""

    def __init__(self, llm: LLMClient | None = None, max_steps: int = 3) -> None:
        self.llm = llm or LLMClient()
        self.max_steps = max_steps

    def run(
        self,
        query: str,
        evidence_chain: list[dict[str, Any]],
        selected_passages: list[dict[str, Any]] | None = None,
    ) -> ReasonerOutput:
        """Generate iterative reasoning traces and final answer."""
        thoughts: list[str] = []
        answer = ""
        confidence = 0.0
        selected_passages = selected_passages or []

        evidence_text = "\n".join(
            [
                f"Hop {idx + 1}: {hop.get('node_from', '')} --{hop.get('relation', '')}--> {hop.get('node_to', '')} | {hop.get('source', '')}"
                for idx, hop in enumerate(evidence_chain[:10])
            ]
        )
        passage_text = "\n".join(
            [
                f"Passage {idx + 1} ({p.get('title', 'unknown')}): {p.get('text', '')[:320]}"
                for idx, p in enumerate(selected_passages[:4])
            ]
        )

        for step in range(1, self.max_steps + 1):
            prompt = (
                "You are a ReAct reasoner. Think step-by-step over the evidence and then output strict JSON with "
                "keys: thought (str), answer (str), confidence (float from 0 to 1). "
                "The answer must be a short span (1-8 words), not an explanation.\n"
                f"Question: {query}\n"
                f"Evidence:\n{evidence_text}\n"
                f"Passages:\n{passage_text}\n"
                f"Current step: {step}/{self.max_steps}"
            )
            response = self.llm.generate(prompt)
            parsed = self.llm.extract_json(response.text)

            thought = str(parsed.get("thought", "")).strip()
            if thought:
                thoughts.append(thought)

            candidate_answer = str(parsed.get("answer", "")).strip()
            candidate_conf = safe_float(parsed.get("confidence", 0.0), default=0.0)

            if candidate_answer:
                answer = candidate_answer
            confidence = max(confidence, candidate_conf)

            if confidence >= 0.75 and answer:
                break

        if not answer:
            answer = "I could not derive a reliable answer from the available evidence."

        return ReasonerOutput(thoughts=thoughts, answer=answer, confidence=min(confidence, 1.0))
