"""Critic/self-refinement agent with retrieval feedback decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.evidence_validation import (
    answer_strings_supported_by_evidence,
    chain_covers_resolved_entities,
    evidence_blob,
    resolved_entities_from_context,
)
from src.agents.prompts import CRITIC_PROMPT_TEMPLATE, INSUFFICIENT_EVIDENCE_ANSWER
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

    def run(
        self,
        query: str,
        answer: str,
        confidence: float,
        evidence_count: int,
        evidence_chain: list[dict[str, Any]] | None = None,
        selected_passages: list[dict[str, Any]] | None = None,
        entity_context: dict[str, Any] | None = None,
        decomposition: dict[str, Any] | None = None,
    ) -> CriticOutput:
        """Return approval signal or feedback for additional retrieval."""
        evidence_chain = evidence_chain or []
        selected_passages = selected_passages or []
        blob = evidence_blob(evidence_chain, selected_passages)
        resolved = resolved_entities_from_context(entity_context)

        prog_failures: list[str] = []
        if resolved and not chain_covers_resolved_entities(evidence_chain, selected_passages, resolved):
            prog_failures.append("Retrieved evidence does not cover all resolved entities in the chain.")
        canon_only = list((entity_context or {}).get("canonical_entities") or [])
        if canon_only and not chain_covers_resolved_entities(evidence_chain, selected_passages, canon_only):
            prog_failures.append("Canonical subjects from entity linking are missing in retrieved evidence.")
        amain = (answer or "").strip()
        insuff = amain == INSUFFICIENT_EVIDENCE_ANSWER or "not enough information" in amain.lower()
        if (answer or "").strip() == INSUFFICIENT_EVIDENCE_ANSWER or insuff:
            prog_failures.append("Abstaining answer; retrieval or grounding must improve.")
        elif not answer_strings_supported_by_evidence(str(answer or ""), blob.lower()):
            prog_failures.append("Final answer is not a direct substring of retrieved evidence.")
        if decomposition and len(decomposition.get("sub_questions") or []) > 1 and evidence_count < 2:
            prog_failures.append("Multi-hop plan requires at least two evidence items.")
        n_sub = len((decomposition or {}).get("sub_questions") or [])
        if n_sub > 1 and len(selected_passages) < 2:
            prog_failures.append("Hop mismatch: need at least two entity-grounded passages for the planned hops.")

        ev_summary = "\n".join(
            [
                f"- {h.get('node_from', '')} -{h.get('relation', '')}-> {h.get('node_to', '')}"
                for h in evidence_chain[:6]
            ]
        )
        if not ev_summary.strip():
            ev_summary = "(no graph chain; using passages only)"

        ent_val = "resolved entities: " + (", ".join(resolved) if resolved else "none")
        if prog_failures:
            ent_val += " | issues: " + " ; ".join(prog_failures)

        prompt = CRITIC_PROMPT_TEMPLATE.format(
            query=query,
            answer=answer,
            reasoner_confidence=confidence,
            evidence_count=evidence_count,
            evidence_summary=ev_summary,
            entity_validation=ent_val,
        )
        response = self.llm.generate(prompt)
        parsed = self.llm.extract_json(response.text)

        approved = bool(parsed.get("approved", False))
        critique = str(parsed.get("critique", "Need stronger supporting evidence.")).strip()
        critic_conf = safe_float(parsed.get("confidence", confidence), default=confidence)

        if prog_failures:
            approved = False
            critique = prog_failures[0] if critique == "Need stronger supporting evidence." or not critique else critique
            critic_conf = min(critic_conf, 0.35)

        subqs = len((decomposition or {}).get("sub_questions") or [])
        need_evidence = 2 if subqs > 1 else 1
        if evidence_count < need_evidence or not selected_passages:
            approved = False
        if confidence < self.threshold:
            approved = False
        if approved and critic_conf < self.threshold:
            approved = False

        return CriticOutput(approved=approved, critique=critique, confidence=min(critic_conf, 1.0))
