"""Critic/self-refinement agent with retrieval feedback decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.agents.evidence_validation import (
    answer_grounded_in_evidence,
    chain_covers_resolved_entities,
    evidence_blob,
    resolved_entities_from_context,
)
from src.agents.hop_safety import (
    expected_answer_granularity as _hop_expected_granularity,
    expected_answer_type as _hop_expected_type,
    infer_entity_type as _hop_infer_type,
    looks_like_country as _hop_looks_like_country,
    normalize_entity as _hop_normalize,
)
from src.agents.prompts import CRITIC_PROMPT_TEMPLATE, INSUFFICIENT_EVIDENCE_ANSWER
from src.utils.helpers import safe_float
from src.utils.llm import LLMClient


@dataclass(slots=True)
class CriticOutput:
    approved: bool
    critique: str
    confidence: float


class CriticAgent:
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
        active_hop: dict[str, Any] | None = None,
    ) -> CriticOutput:
        """Hard fails (abstain, no passages, active-hop mismatch) reject; soft hints only go into critic prompt."""
        evidence_chain = evidence_chain or []
        selected_passages = selected_passages or []
        blob = evidence_blob(evidence_chain, selected_passages)
        resolved = resolved_entities_from_context(entity_context)

        hard_failures: list[str] = []
        amain = (answer or "").strip()
        insuff = amain == INSUFFICIENT_EVIDENCE_ANSWER or "not enough information" in amain.lower()
        if insuff:
            hard_failures.append("Abstaining answer; retrieval or grounding must improve.")
        if not selected_passages:
            hard_failures.append("No passages retrieved; cannot evaluate answer quality.")

        if active_hop and amain and not insuff:
            hop_failures = self._active_hop_hard_checks(amain, confidence, active_hop)
            hard_failures.extend(hop_failures)

        soft_hints: list[str] = []
        if resolved and not chain_covers_resolved_entities(evidence_chain, selected_passages, resolved):
            soft_hints.append(
                "Current-loop evidence may not cover all expected pivot entities "
                "(normal in multi-hop reasoning where prior hops are not re-retrieved)."
            )
        canon_only = list((entity_context or {}).get("canonical_entities") or [])
        if canon_only and not chain_covers_resolved_entities(evidence_chain, selected_passages, canon_only):
            soft_hints.append(
                "Canonical pivot entity not found in retrieved evidence; "
                "retrieval may need refinement."
            )
        if not insuff and not answer_grounded_in_evidence(str(answer or ""), blob.lower()):
            soft_hints.append(
                "Answer has weak lexical overlap with retrieved evidence; "
                "verify grounding and paraphrase risk."
            )
        n_sub = len((decomposition or {}).get("sub_questions") or [])
        if n_sub > 1 and evidence_count < 2:
            soft_hints.append(
                "Multi-hop plan used fewer than two evidence items; "
                "consider whether more evidence would strengthen the answer."
            )

        # Build entity validation string (context for the LLM prompt).
        ent_val = "resolved entities: " + (", ".join(resolved) if resolved else "none")
        if active_hop:
            ent_val += (
                " | active_hop=" + str(active_hop.get("hop_index", "?"))
                + " subject=" + repr(str(active_hop.get("subject_text") or ""))
                + " relation=" + repr(str(active_hop.get("relation") or ""))
                + " expected_answer_type="
                + repr(str(active_hop.get("expected_answer_type") or ""))
            )
            prior_ans = [a for a in (active_hop.get("prior_answers") or []) if a]
            if prior_ans:
                ent_val += " | prior_hop_answers=" + ",".join(repr(a) for a in prior_ans)
        if hard_failures:
            ent_val += " | HARD: " + " ; ".join(hard_failures)
        if soft_hints:
            ent_val += " | hints: " + " ; ".join(soft_hints)

        ev_summary = "\n".join(
            [
                f"- {h.get('node_from', '')} -{h.get('relation', '')}-> {h.get('node_to', '')}"
                for h in evidence_chain[:6]
            ]
        )
        if not ev_summary.strip():
            ev_summary = "(no graph chain; using passages only)"

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

        if hard_failures:
            approved = False
            critique = hard_failures[0] if not critique or critique == "Need stronger supporting evidence." else critique
            critic_conf = min(critic_conf, 0.35)

        if evidence_count < 1 or not selected_passages:
            approved = False

        if confidence < self.threshold:
            approved = False
        if approved and critic_conf < self.threshold:
            approved = False

        return CriticOutput(approved=approved, critique=critique, confidence=min(critic_conf, 1.0))

    @staticmethod
    def _active_hop_hard_checks(
        candidate_answer: str,
        reasoner_confidence: float,
        active_hop: dict[str, Any],
    ) -> list[str]:
        """Reject if answer == subject, regresses to prior hop, or type/granularity mismatches hop."""
        failures: list[str] = []
        candidate_norm = _hop_normalize(candidate_answer)
        if not candidate_norm:
            return failures

        subject_text = str(active_hop.get("subject_text") or "")
        if subject_text and _hop_normalize(subject_text) == candidate_norm:
            failures.append(
                f"Hop {active_hop.get('hop_index', '?')} answer equals the hop subject "
                "({subject!r}); the relation was not resolved.".format(
                    subject=subject_text
                )
            )
            return failures

        prior_subjects = [str(s) for s in (active_hop.get("prior_subjects") or [])]
        prior_answers = [str(a) for a in (active_hop.get("prior_answers") or [])]
        for prior in (*prior_subjects, *prior_answers):
            if prior and _hop_normalize(prior) == candidate_norm:
                failures.append(
                    f"Hop {active_hop.get('hop_index', '?')} answer regresses to a "
                    f"prior hop entity ({prior!r}); reject and retry the active hop."
                )
                return failures

        relation = str(active_hop.get("relation") or "")
        subq = str(
            active_hop.get("resolved_subquestion")
            or active_hop.get("subquestion_text")
            or ""
        )
        expected_type = str(
            active_hop.get("expected_answer_type")
            or _hop_expected_type(relation, subq)
        )
        inferred_type = _hop_infer_type(candidate_answer)
        if (
            expected_type
            and inferred_type
            and expected_type != "Unknown"
            and inferred_type != "Unknown"
            and expected_type != inferred_type
        ):
            failures.append(
                f"Hop {active_hop.get('hop_index', '?')} answer type mismatch: "
                f"expected {expected_type}, inferred {inferred_type}."
            )
            return failures

        granularity = str(
            active_hop.get("expected_answer_granularity")
            or _hop_expected_granularity(relation, subq)
        )
        if granularity in ("City", "State", "Region") and _hop_looks_like_country(
            candidate_answer
        ):
            failures.append(
                f"Hop {active_hop.get('hop_index', '?')} granularity mismatch: "
                f"question expects {granularity}, but the answer looks like a country-level surface form."
            )

        return failures
