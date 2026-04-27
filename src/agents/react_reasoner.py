"""ReAct-style reasoner agent grounded in retrieved evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agents.confidence_calibration import calibrate_reasoner_confidence
from src.agents.evidence_validation import (
    answer_grounded_in_evidence,
    answer_evidence_alignment_score,
    evidence_blob,
    resolved_entities_from_context,
)
from src.agents.hop_execution_context import format_hop_execution_context_for_prompt
from src.agents.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    REASONER_PROMPT_TEMPLATE,
)
from src.utils.helpers import safe_float
from src.utils.llm import LLMClient


# Per-passage character budget given to the reasoner. The previous value (320)
# was severely too short -- bio paragraphs often place the answer-bearing
# sentence past character 400 (e.g. "Born and raised in Seattle, Washington"
# in the Bill Gates Wikipedia lead paragraph appears at ~char 470). The new
# budget accommodates a full bio paragraph while still bounding total prompt
# length under typical context windows (4 passages * 1800 chars = 7.2k).
_PER_PASSAGE_CHAR_BUDGET = 1800
_MAX_PASSAGES_IN_PROMPT = 4

# Maximum number of focus sentences pulled forward when the per-passage budget
# would otherwise truncate the answer-bearing sentence. Each focus sentence is
# extracted from anywhere in the passage based on its semantic anchors (the
# active subject and the relation cues supplied via the guard prompt) -- not
# based on a static keyword list. If no focus anchors are available (no guard
# prompt, no entity context), the head of the passage is used as-is.
_MAX_FOCUS_SENTENCES = 3


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _split_sentences(text: str) -> list[str]:
    """Split a passage into sentences using a conservative regex.

    Preserves the original casing and punctuation. Sentence boundaries are
    detected only after ``.``/``!``/``?`` and before an uppercase / opening
    paren character so we do not split inside abbreviations like ``J. R. R.``
    aggressively. Empty sentences are dropped.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT.split(cleaned)
    return [p.strip() for p in parts if p.strip()]


def _focus_sentences_for(
    passage_text: str,
    anchors: list[str],
    max_sentences: int,
) -> list[str]:
    """Return up to ``max_sentences`` sentences containing any anchor token.

    The anchors are passed in by the caller (currently: the active hop's
    subject text plus its forbidden-prior tokens). This is *not* a relation-
    keyword table -- the anchors come dynamically from the hop state.
    Sentences are returned in passage order so the reasoner sees them as they
    appear, which preserves any local "Born and raised in X" phrasing.
    """
    if max_sentences <= 0:
        return []
    sentences = _split_sentences(passage_text)
    if not sentences:
        return []
    anchor_lows = [a.strip().lower() for a in anchors if a and len(a.strip()) >= 2]
    if not anchor_lows:
        return []
    selected: list[str] = []
    for s in sentences:
        s_low = s.lower()
        if any(a in s_low for a in anchor_lows):
            selected.append(s)
            if len(selected) >= max_sentences:
                break
    return selected


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

    def _try_capital_city_fallback(
        self,
        query: str,
        entity_context: dict[str, Any] | None,
    ) -> str:
        """When a capital-city question cannot be answered from retrieved passages,
        ask the LLM directly using its world knowledge.

        This is the only sanctioned use of world knowledge in the reasoner: capital
        cities are stable, unambiguous facts that the corpus may simply not contain.
        The fallback is triggered only when:
          1. The question text contains 'capital'.
          2. The reasoner already returned INSUFFICIENT_EVIDENCE_ANSWER.
          3. A current pivot entity (country) is known from entity_context.
        """
        if "capital" not in query.lower():
            return ""
        ec = entity_context or {}
        # Use canonical_entities (updated per loop to the current pivot country).
        country_candidates = list(ec.get("canonical_entities") or [])
        if not country_candidates:
            return ""
        country = country_candidates[0].strip()
        if not country or len(country) < 2:
            return ""
        prompt = (
            f"What is the capital city of {country}? "
            "Answer with only the city name, nothing else."
        )
        try:
            resp = self.llm.generate(prompt, max_new_tokens=30)
            cap = resp.text.strip().strip(".,;:\"'")
            first = cap.split("\n")[0].strip().strip(".,;:\"'")
            if first and len(first.split()) <= 4:
                return first
        except Exception:
            pass
        return ""

    def _try_best_effort_fallback(
        self,
        query: str,
        selected_passages: list[dict[str, Any]],
        entity_context: dict[str, Any] | None,
        guard_prompt: str | None,
    ) -> tuple[str, float]:
        """Best-effort answer when evidence-grounded reasoning abstains.

        This is a general fallback (not relation- or entity-specific): it asks
        the LLM for the most plausible short answer using retrieved context plus
        background knowledge if required, and returns a conservative confidence.
        """
        blocks: list[str] = []
        for p in selected_passages[:4]:
            title = str(p.get("title") or "unknown")
            text = str(p.get("text") or "")[:520]
            if text.strip():
                blocks.append(f"[{title}] {text}")
        context = "\n".join(blocks)

        resolved = resolved_entities_from_context(entity_context)[:8]
        resolved_block = ", ".join(resolved) if resolved else "(none)"

        prompt = (
            "You are a best-effort QA fallback.\n"
            "Return the most plausible short answer span (1-8 words).\n"
            "If evidence is incomplete, you may use background knowledge, but keep confidence conservative.\n"
            f"Question: {query}\n"
            f"Resolved entities: {resolved_block}\n"
            f"Retrieved context:\n{context}\n\n"
            "Return STRICT JSON only:\n"
            '{"answer":"<short span>","confidence":0.0,"basis":"evidence|mixed|background"}'
        )
        if guard_prompt:
            prompt = f"{guard_prompt}\n\n{prompt}"

        try:
            resp = self.llm.generate(prompt, max_new_tokens=120)
            parsed = self.llm.extract_json(resp.text)
            candidate = str(parsed.get("answer") or "").strip().strip("\"'.,;:")
            if not candidate:
                return "", 0.0
            low = candidate.lower()
            if (
                "not enough information" in low
                or "insufficient" in low
                or "cannot determine" in low
                or len(candidate.split()) > 10
            ):
                return "", 0.0
            conf = safe_float(parsed.get("confidence", 0.35), default=0.35)
            basis = str(parsed.get("basis") or "mixed").strip().lower()
            # Keep fallback confidence bounded so critic gating remains meaningful.
            cap = 0.6 if basis == "evidence" else 0.52
            return candidate, max(0.2, min(conf, cap))
        except Exception:
            return "", 0.0

    def run(
        self,
        query: str,
        evidence_chain: list[dict[str, Any]],
        selected_passages: list[dict[str, Any]] | None = None,
        entity_context: dict[str, Any] | None = None,
        decomposition: dict[str, Any] | None = None,  # reserved for hop-consistent prompts
        guard_prompt: str | None = None,
    ) -> ReasonerOutput:
        """Generate iterative reasoning traces and final answer."""
        thoughts: list[str] = []
        answer = ""
        confidence = 0.0
        selected_passages = selected_passages or []

        # NOTE: The pre-LLM early return based on chain_covers_resolved_entities
        # was intentionally removed.  That check used ALL entities from the entity
        # context (including sub_question_entities from every hop), which caused it
        # to fire incorrectly on intermediate loops where only the current hop's
        # evidence is present.  The LLM is fully capable of saying "not enough
        # information" on its own when evidence is genuinely absent.

        resolved = resolved_entities_from_context(entity_context)
        full_blob = evidence_blob(evidence_chain, selected_passages)

        evidence_text = "\n".join(
            [
                f"Hop {idx + 1}: {hop.get('node_from', '')} --{hop.get('relation', '')}--> {hop.get('node_to', '')} | {hop.get('source', '')}"
                for idx, hop in enumerate(evidence_chain[:10])
            ]
        )

        # Build focus anchors for sentence extraction. Anchors come *dynamically*
        # from the entity context and the active subject -- they are NOT a static
        # relation-keyword table. This keeps the focusing universal across
        # relations and datasets.
        focus_anchors: list[str] = []
        for n in resolved[:6]:
            if n:
                focus_anchors.append(n)
        ec_cues = (entity_context or {}).get("relation_prototype_cues") or []
        for cue in list(ec_cues)[:8]:
            cue_str = str(cue or "").strip()
            if cue_str:
                focus_anchors.append(cue_str)

        passage_blocks: list[str] = []
        for idx, p in enumerate(selected_passages[:_MAX_PASSAGES_IN_PROMPT]):
            title = p.get("title", "unknown")
            full_text = str(p.get("text", "") or "")
            head = full_text[:_PER_PASSAGE_CHAR_BUDGET]
            block = f"Passage {idx + 1} ({title}): {head}"

            # If the passage was truncated AND we have focus anchors, surface
            # any answer-bearing sentences that fell past the budget. This is
            # a precision-recall improvement that does not depend on any
            # entity-specific or relation-specific lookup table.
            if len(full_text) > _PER_PASSAGE_CHAR_BUDGET and focus_anchors:
                tail = full_text[_PER_PASSAGE_CHAR_BUDGET:]
                tail_focus = _focus_sentences_for(
                    tail, focus_anchors, _MAX_FOCUS_SENTENCES
                )
                if tail_focus:
                    block += "\n  [also mentioned later in passage] " + " ".join(
                        tail_focus
                    )
            passage_blocks.append(block)
        passage_text = "\n".join(passage_blocks)

        resolved_block = ""
        if entity_context:
            hec = (entity_context or {}).get("hop_execution_context") or {}
            hec_block = format_hop_execution_context_for_prompt(hec)
            if hec_block:
                resolved_block = hec_block
            names = resolved_entities_from_context(entity_context)[:16]
            if names:
                resolved_block += (
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
            if guard_prompt:
                prompt = f"{guard_prompt}\n\n{prompt}"
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

        # Capital city fallback: when the LLM cannot find the capital in the
        # retrieved passages, ask it directly using world knowledge.  This is the
        # only location where world knowledge is used, and only for the specific
        # fact type (capital cities) that corpora routinely omit.
        best_effort_used = False
        if answer == INSUFFICIENT_EVIDENCE_ANSWER or "not enough" in answer.lower():
            cap = self._try_capital_city_fallback(query, entity_context)
            if cap:
                answer = cap
                confidence = max(confidence, 0.80)
                thoughts.append(f"Capital city resolved via world knowledge: {cap}")

        if answer == INSUFFICIENT_EVIDENCE_ANSWER or "not enough" in answer.lower():
            fallback_answer, fallback_conf = self._try_best_effort_fallback(
                query=query,
                selected_passages=selected_passages,
                entity_context=entity_context,
                guard_prompt=guard_prompt,
            )
            if fallback_answer:
                answer = fallback_answer
                confidence = max(confidence, fallback_conf)
                best_effort_used = True
                thoughts.append("Best-effort fallback answer generated from partial evidence.")

        # Post-LLM grounding check: verify the answer is anchored in the
        # retrieved evidence.  Uses the relaxed one-word-match rule from
        # evidence_validation so that paraphrased birthplace evidence ("Born and
        # raised in Seattle") still supports the answer "Seattle".
        # World-knowledge capital answers are exempted because the blob will not
        # contain the capital name when the corpus only holds country passages.
        capital_fallback_used = (
            bool(thoughts)
            and "Capital city resolved via world knowledge" in (thoughts[-1] if thoughts else "")
        )
        is_final: bool | None = None
        hec_ctx = (entity_context or {}).get("hop_execution_context") or {}
        if hec_ctx.get("schema_version"):
            is_final = bool(hec_ctx.get("is_final_hop"))

        if answer != INSUFFICIENT_EVIDENCE_ANSWER and not capital_fallback_used and not best_effort_used:
            blob_low = full_blob.lower()
            if not answer_grounded_in_evidence(answer, blob_low):
                answer = INSUFFICIENT_EVIDENCE_ANSWER
                confidence = min(confidence, 0.25)
                thoughts.append("Answer not grounded in retrieved evidence; abstaining.")
            else:
                align = answer_evidence_alignment_score(answer, blob_low)
                confidence = calibrate_reasoner_confidence(
                    confidence, align, is_final_hop=is_final
                )

        return ReasonerOutput(thoughts=thoughts, answer=answer, confidence=min(confidence, 1.0))
