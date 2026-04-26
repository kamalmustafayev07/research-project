"""Query decomposition agent for multi-hop planning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agents.entity_linker import EntityLinkingResult
from src.agents.prompts import DECOMPOSER_PROMPT_TEMPLATE
from src.utils.llm import LLMClient


@dataclass(slots=True)
class DecompositionOutput:
    """Decomposition result schema."""

    sub_questions: list[str]
    relation_sequence: list[str]
    sub_question_entities: list[dict[str, Any]]


class QueryDecomposerAgent:
    """Create a structured decomposition from a natural-language question."""

    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm or LLMClient()

    @staticmethod
    def _merge_entity_plans(
        sub_questions: list[str],
        relation_sequence: list[str],
        parsed_plans: list[dict[str, Any]] | None,
        entity_linking: EntityLinkingResult | None,
    ) -> list[dict[str, Any]]:
        plans: list[dict[str, Any]] = []
        link_map = dict(entity_linking.mention_to_canonical) if entity_linking else {}
        match_strings = list(entity_linking.match_strings) if entity_linking else []

        for idx, sq in enumerate(sub_questions):
            rel = relation_sequence[idx] if idx < len(relation_sequence) else "related_to"
            row: dict[str, Any] = {
                "sub_question": sq,
                "relation": rel,
                "resolved_entities": [],
                "entity_map": {},
            }
            if parsed_plans and idx < len(parsed_plans) and isinstance(parsed_plans[idx], dict):
                cand = parsed_plans[idx]
                if isinstance(cand.get("resolved_entities"), list):
                    row["resolved_entities"] = [str(x) for x in cand["resolved_entities"]]
                if isinstance(cand.get("entity_map"), dict):
                    row["entity_map"] = {str(k): str(v) for k, v in cand["entity_map"].items()}
            if entity_linking:
                em = dict(link_map)
                for k, v in em.items():
                    if k.lower() in sq.lower() or (v and v.lower() in sq.lower()):
                        row["entity_map"].setdefault(k, v)
                for span in match_strings:
                    if span.lower() in sq.lower():
                        if span not in row["resolved_entities"]:
                            row["resolved_entities"].append(span)
                if idx == 0 and link_map:
                    for mention, canon in link_map.items():
                        row["entity_map"].setdefault(mention, canon)
                        if canon not in row["resolved_entities"]:
                            row["resolved_entities"].append(canon)
            plans.append(row)

        return plans

    @staticmethod
    def _work_in_writer_of_object_phrase(text: str, work_surface: str) -> bool:
        """True if work_surface appears as the object of ... writer/author ... of ..."""
        if not work_surface or work_surface.lower() not in text.lower():
            return False
        for m in re.finditer(
            r"\b(writer|author|authors|wrote|penned)\s+of\s+",
            text,
            re.IGNORECASE,
        ):
            tail = text[m.end() :]
            if work_surface.lower() in tail.lower()[: 120]:
                return True
        return False

    def _rewrite_sub_questions_with_canonicals(
        self,
        sub_questions: list[str],
        entity_linking: EntityLinkingResult | None,
    ) -> list[str]:
        """Rewrite hops using canonicals without turning 'writer of The Hobbit' into 'writer of <person>'."""
        if not entity_linking or not sub_questions:
            return sub_questions
        out: list[str] = []
        pairs = sorted(entity_linking.mention_to_canonical.items(), key=lambda kv: len(kv[0]), reverse=True)
        for sq in sub_questions:
            s = sq
            for surf, canon in pairs:
                if len(surf) < 3 or not canon or surf.lower() == canon.lower():
                    continue
                if surf.lower() not in s.lower():
                    continue
                if self._work_in_writer_of_object_phrase(s, surf):
                    continue
                s = re.sub(re.escape(surf), canon, s, count=0, flags=re.IGNORECASE)
            for w, auth in entity_linking.work_to_author_edges:
                if len(w) < 3 or not auth:
                    continue
                if w.lower() not in s.lower():
                    continue
                if self._work_in_writer_of_object_phrase(s, w):
                    continue
                s = re.sub(re.escape(w), auth, s, count=0, flags=re.IGNORECASE)
            out.append(s)
        return out

    def run(self, question: str, entity_linking: EntityLinkingResult | None = None) -> DecompositionOutput:
        """Run decomposition with robust fallback parsing."""
        hints = ""
        if entity_linking:
            ref_anchors = getattr(entity_linking, "referential_anchors", [])
            hints = (
                f"anchors={entity_linking.anchor_mentions}; "
                f"canonical_map={entity_linking.mention_to_canonical}; "
                f"required_spans={entity_linking.required_surface_forms}; "
                f"referential_anchors={ref_anchors}"
            )
        prompt = DECOMPOSER_PROMPT_TEMPLATE.format(question=question, entity_hints=hints or "(none)")
        # Decomposer needs more tokens: 2-3 hops × ~80 tokens each + JSON framing.
        response = self.llm.generate(prompt, max_new_tokens=1024)
        parsed = self.llm.extract_json(response.text)

        sub_questions = parsed.get("sub_questions") if isinstance(parsed.get("sub_questions"), list) else []
        relation_sequence = parsed.get("relation_sequence") if isinstance(parsed.get("relation_sequence"), list) else []
        parsed_plans = parsed.get("sub_question_entities") if isinstance(parsed.get("sub_question_entities"), list) else []

        if not sub_questions:
            chunks = [piece.strip() for piece in question.replace("?", "").split(" and ") if piece.strip()]
            if len(chunks) <= 1:
                chunks = [question.strip()]
            sub_questions = chunks

        # Do NOT strip hop placeholders here — <answer of hop N> is structurally
        # important for the reasoner and must be preserved in the state.
        # The query builder resolves placeholders to descriptive text before embedding.

        if not relation_sequence:
            relation_sequence = ["related_to" for _ in sub_questions]

        sub_question_entities = self._merge_entity_plans(sub_questions, relation_sequence, parsed_plans, entity_linking)
        sub_questions = self._rewrite_sub_questions_with_canonicals(sub_questions, entity_linking)
        for idx, row in enumerate(sub_question_entities):
            if idx < len(sub_questions):
                row["sub_question"] = sub_questions[idx]

        return DecompositionOutput(
            sub_questions=sub_questions,
            relation_sequence=relation_sequence,
            sub_question_entities=sub_question_entities,
        )
