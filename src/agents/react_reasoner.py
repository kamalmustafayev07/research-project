"""ReAct-style reasoner agent grounded in retrieved evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.evidence_validation import (
    answer_strings_supported_by_evidence,
    chain_covers_resolved_entities,
    evidence_blob,
    resolved_entities_from_context,
)
from src.agents.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    REASONER_PROMPT_TEMPLATE,
)
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
        entity_context: dict[str, Any] | None = None,
        decomposition: dict[str, Any] | None = None,  # reserved for hop-consistent prompts
    ) -> ReasonerOutput:
        """Generate iterative reasoning traces and final answer."""
        thoughts: list[str] = []
        answer = ""
        confidence = 0.0
        selected_passages = selected_passages or []

        resolved = resolved_entities_from_context(entity_context)
        full_blob = evidence_blob(evidence_chain, selected_passages)
        if resolved and not chain_covers_resolved_entities(evidence_chain, selected_passages, resolved):
            return ReasonerOutput(
                thoughts=[INSUFFICIENT_EVIDENCE_ANSWER],
                answer=INSUFFICIENT_EVIDENCE_ANSWER,
                confidence=0.15,
            )

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

        resolved_block = ""
        if entity_context:
            names = resolved_entities_from_context(entity_context)[:16]
            if names:
                resolved_block = (
                    "Resolved entity chain (do NOT introduce different people, works, or places): "
                    + ", ".join(names)
                    + "\n"
                )

        for step in range(1, self.max_steps + 1):
            prompt = REASONER_PROMPT_TEMPLATE.format(
                question=query,
                resolved_entity_block=resolved_block,
                evidence_text=evidence_text,
                passage_text=passage_text,
                step=step,
                max_steps=self.max_steps,
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
            answer = INSUFFICIENT_EVIDENCE_ANSWER

        if answer != INSUFFICIENT_EVIDENCE_ANSWER:
            if not answer_strings_supported_by_evidence(answer, full_blob.lower()):
                answer = INSUFFICIENT_EVIDENCE_ANSWER
                confidence = min(confidence, 0.25)
                thoughts.append("Unsupported answer span relative to resolved-entity evidence; abstaining.")

        return ReasonerOutput(thoughts=thoughts, answer=answer, confidence=min(confidence, 1.0))
