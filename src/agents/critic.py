"""Critic/self-refinement agent with retrieval feedback decisions."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.helpers import safe_float
from src.utils.llm import LLMClient


@dataclass(slots=True)
class CriticOutput:
    """Critic decision for graph loop control."""

    approved: bool
    critique: str
    confidence: float


class CriticAgent:
    """Evaluate reasoning outputs and decide whether retrieval refinement is needed."""

    def __init__(self, llm: LLMClient | None = None, threshold: float = 0.65) -> None:
        self.llm = llm or LLMClient()
        self.threshold = threshold

    def run(self, query: str, answer: str, confidence: float, evidence_count: int) -> CriticOutput:
        """Return approval signal or feedback for additional retrieval."""
        prompt = (
            "You are a critic agent. Assess if the answer is complete and grounded. Output strict JSON with keys "
            "approved (bool), critique (str), confidence (float).\n"
            f"Question: {query}\nAnswer: {answer}\nReasoner confidence: {confidence}\nEvidence count: {evidence_count}"
        )
        response = self.llm.generate(prompt)
        parsed = self.llm.extract_json(response.text)

        approved = bool(parsed.get("approved", False))
        critique = str(parsed.get("critique", "Need stronger supporting evidence.")).strip()
        critic_conf = safe_float(parsed.get("confidence", confidence), default=confidence)

        if confidence < self.threshold or evidence_count < 2:
            approved = False
        if approved and critic_conf < self.threshold:
            approved = False

        return CriticOutput(approved=approved, critique=critique, confidence=min(critic_conf, 1.0))
