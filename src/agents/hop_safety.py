"""Hop-level safety utilities for multi-hop question answering.

This module provides the core state machine and verification primitives the
pipeline uses to keep multi-hop reasoning consistent across stages:

1. ``HopMemory`` / ``HopRecord`` — persistent, immutable-subject state for each
   hop. ``main_entity`` is set when the hop is created and never rewritten by
   later answer attempts; this is what stops the reasoner from regressing from
   hop ``k`` back to hop ``k-1``.
2. ``build_retrieval_query`` — chain-anchored retrieval query construction.
3. ``get_reasoner_prompt`` — structured prompt that names the active sub-
   question, subject and relation and enumerates the prior-hop answers the
   reasoner is forbidden to repeat.
4. ``verify_hop_answer`` — relation-conditioned, type-checked verifier with
   explicit regression detection against prior-hop subjects and answers.

Plus serialisation helpers (``hop_memory_to_dict`` / ``hop_memory_from_dict``)
so the LangGraph state can carry the structure across nodes.

The implementation is intentionally general and contains no domain-specific
entities, benchmark patterns, or hard-coded examples. Type inference and
relation classification both fall back to ``Unknown`` rather than guessing,
so the verifier accepts valid short answers (``"Oxford"``, ``"France"``)
instead of rejecting them as ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Literal, Sequence


EntityType = Literal[
    "Person",
    "Location",
    "Organization",
    "Work",
    "Event",
    "Date",
    "Number",
    "Other",
    "Unknown",
]


@dataclass(slots=True, frozen=True)
class EntityState:
    """Normalized entity information used as a hop subject anchor."""

    text: str
    entity_type: EntityType
    role: str = "subject"


@dataclass(slots=True)
class HopRecord:
    """Single hop state tracked by ``HopMemory``.

    The key invariant is that ``main_entity`` is immutable for this hop after
    creation. This prevents accidental subject replacement by noisy answer spans.
    """

    hop_index: int
    subquestion_text: str
    main_entity: EntityState
    relation: str
    answer: str | None = None
    confidence: float | None = None


@dataclass(slots=True)
class HopMemory:
    """Persistent, explicit memory for a full multi-hop reasoning chain.

    Required fields retained:
    - Original question
    - Ordered list of hop records (sub-question + main entity/type + relation + answer + confidence)

    Notes on drift prevention:
    - ``record_answer`` updates only answer/confidence; it never rewrites
      ``main_entity`` for the hop.
    - The next hop's main subject must be explicitly provided by the caller (e.g.
      from decomposition output), rather than inferred implicitly from text.
    """

    original_question: str
    hops: list[HopRecord] = field(default_factory=list)

    def add_hop(
        self,
        subquestion_text: str,
        main_entity_text: str,
        main_entity_type: EntityType,
        relation: str,
        role: str = "subject",
    ) -> HopRecord:
        """Append a new hop with an explicit subject anchor and relation."""
        hop = HopRecord(
            hop_index=len(self.hops) + 1,
            subquestion_text=subquestion_text.strip(),
            main_entity=EntityState(
                text=main_entity_text.strip(),
                entity_type=main_entity_type,
                role=role.strip() or "subject",
            ),
            relation=relation.strip(),
        )
        self.hops.append(hop)
        return hop

    def record_answer(self, hop_index: int, answer: str, confidence: float) -> None:
        """Store answer metadata for a hop without mutating the hop subject anchor."""
        hop = self.get_hop(hop_index)
        hop.answer = answer.strip()
        hop.confidence = _clamp_confidence(confidence)

    def get_hop(self, hop_index: int) -> HopRecord:
        """Return the hop by 1-based index."""
        if hop_index < 1 or hop_index > len(self.hops):
            raise IndexError(f"hop_index {hop_index} is out of range")
        return self.hops[hop_index - 1]

    def find_hop_by_subquestion(self, subquestion_text: str) -> HopRecord:
        """Locate a hop by normalized sub-question text.

        If no exact match exists, the latest unanswered hop is returned as a safe
        fallback for sequential pipelines.
        """
        target = _normalize_ws(subquestion_text)
        for hop in self.hops:
            if _normalize_ws(hop.subquestion_text) == target:
                return hop

        for hop in self.hops:
            if hop.answer is None:
                return hop

        if not self.hops:
            raise ValueError("hop memory is empty")
        return self.hops[-1]

    def previous_hops(self, hop_index: int) -> list[HopRecord]:
        """Return hops strictly before the given hop index."""
        if hop_index <= 1:
            return []
        return self.hops[: hop_index - 1]

    def chain_lines(self, upto_hop_index: int | None = None) -> list[str]:
        """Return human-readable chain lines with types and roles.

        ``upto_hop_index`` is inclusive and 1-based. If omitted, includes all hops.
        """
        if upto_hop_index is None:
            records = self.hops
        else:
            records = self.hops[: max(0, upto_hop_index)]

        lines: list[str] = []
        for hop in records:
            line = (
                f"Hop {hop.hop_index}: subject={hop.main_entity.text}"
                f" ({hop.main_entity.entity_type}, role={hop.main_entity.role}); "
                f"relation={hop.relation}; answer={hop.answer or '[pending]'}; "
                f"confidence={hop.confidence if hop.confidence is not None else '[pending]'}"
            )
            lines.append(line)
        return lines


@dataclass(slots=True, frozen=True)
class VerificationResult:
    """Validation result for a proposed hop answer."""

    approved: bool
    explanation: str
    recovery_action: str | None = None


def build_retrieval_query(current_hop: HopRecord, hop_memory: HopMemory) -> str:
    """Build a retrieval query anchored to the original question and hop chain.

    The query always includes:
    - original question anchor
    - all previous hop subjects with entity type + role
    - current hop subject and target relation

    This context makes entity-role transitions explicit and prevents accidental
    pivoting to unrelated co-occurring entities.
    """
    previous_context = hop_memory.previous_hops(current_hop.hop_index)
    chain_bits: list[str] = []
    for hop in previous_context:
        chain_bits.append(
            f"hop{hop.hop_index}: {hop.main_entity.text} ({hop.main_entity.entity_type}, {hop.main_entity.role})"
            f" --{hop.relation}--> {hop.answer or '[pending]'}"
        )

    chain_text = " | ".join(chain_bits) if chain_bits else "[no previous hops]"
    current_target = (
        f"Find {current_hop.relation} for {current_hop.main_entity.text} "
        f"({current_hop.main_entity.entity_type}) who is {current_hop.main_entity.role}."
    )
    return "\n".join(
        [
            f"Original question: {hop_memory.original_question}",
            f"Reasoning chain: {chain_text}",
            f"Current sub-question: {current_hop.subquestion_text}",
            f"Current relation: {current_hop.relation}",
            current_target,
        ]
    )


def get_reasoner_prompt(current_subquestion: str, hop_memory: HopMemory) -> str:
    """Create a strict reasoner prompt that enforces subject/relation discipline.

    The prompt requires the model to explicitly state:
    - main subject entity and entity type
    - target relation
    - why evidence supports or does not support the candidate answer

    It also includes an explicit relation-disambiguation requirement between
    education-style and employment/association-style relations.
    """
    hop = hop_memory.find_hop_by_subquestion(current_subquestion)
    chain_text = "\n".join(hop_memory.chain_lines(upto_hop_index=hop.hop_index))

    return (
        "You are a hop-aware multi-hop QA reasoner.\n"
        "Follow the chain memory exactly and do not replace the hop subject with a co-occurring entity.\n\n"
        "Reasoning context:\n"
        f"Original question: {hop_memory.original_question}\n"
        f"Current sub-question: {hop.subquestion_text}\n"
        f"Current subject entity: {hop.main_entity.text}\n"
        f"Current subject type: {hop.main_entity.entity_type}\n"
        f"Current relation: {hop.relation}\n"
        "Hop chain:\n"
        f"{chain_text}\n\n"
        "Critical relation disambiguation:\n"
        "- Treat studied_at / attended / educated_at as education relations.\n"
        "- Treat worked_at / taught_at / associated_with as employment-or-association relations.\n"
        "- Do not accept employment evidence as proof of education relations, and do not accept "
        "education evidence as proof of employment-or-association relations.\n\n"
        "Output STRICT JSON only with this schema:\n"
        "{\n"
        "  \"main_subject_entity\": str,\n"
        "  \"main_subject_type\": str,\n"
        "  \"target_relation\": str,\n"
        "  \"candidate_answer\": str,\n"
        "  \"candidate_answer_type\": str,\n"
        "  \"why_answer_fits\": str,\n"
        "  \"why_other_candidates_do_not_fit\": str,\n"
        "  \"is_answer_supported\": bool,\n"
        "  \"confidence\": float\n"
        "}\n"
    )


def verify_hop_answer(
    proposed_answer: str,
    current_subquestion: str,
    hop_memory: HopMemory,
) -> VerificationResult:
    """Validate whether a proposed answer fits the current hop.

    Checks performed (all generic — no entity- or example-specific logic):

    1. Non-empty, non-abstain marker.
    2. Not a regression to the current hop's locked subject anchor.
    3. Not a regression to any *prior* hop's resolved answer or subject
       (loop-regression prevention — the central fix for hop drift).
    4. Type compatibility between the inferred answer type and the expected
       answer type derived from the relation and sub-question wording.
       The check is conservative: when either side is ``Unknown`` it is
       skipped rather than rejected, so valid short answers ("Oxford",
       "France") are not rejected just because the lightweight typing
       heuristic could not classify them.

    The previous "drift-risk" heuristic (rejecting single-token Unknown-typed
    answers on education relations) was intentionally removed: it produced
    false positives for valid short organisation answers. Regression now goes
    through (3) — explicit comparison with prior-hop state — which is exact,
    not heuristic.
    """
    hop = hop_memory.find_hop_by_subquestion(current_subquestion)
    candidate = (proposed_answer or "").strip()

    if not candidate:
        return VerificationResult(
            approved=False,
            explanation="Rejected: empty answer for current hop.",
            recovery_action=(
                "Re-run retrieval for the same subject and relation with stronger relation phrasing."
            ),
        )

    lowered = candidate.lower()
    abstain_markers = ("not enough", "insufficient", "unknown", "cannot determine")
    if any(marker in lowered for marker in abstain_markers):
        return VerificationResult(
            approved=False,
            explanation="Rejected: abstention marker detected; no concrete hop answer available.",
            recovery_action=(
                "Retrieve additional passages anchored to the same subject entity and relation."
            ),
        )

    if _normalize_entity(candidate) == _normalize_entity(hop.main_entity.text):
        return VerificationResult(
            approved=False,
            explanation=(
                "Rejected: proposed answer repeats the hop subject instead of resolving the relation target."
            ),
            recovery_action=(
                "Require evidence sentences that explicitly connect the subject to a distinct relation object."
            ),
        )

    # Loop-regression prevention: an answer that matches *any* previously
    # approved hop's answer (or subject) is by construction a regression to a
    # prior hop, not an answer to the current one.
    candidate_norm = _normalize_entity(candidate)
    for prior in hop_memory.previous_hops(hop.hop_index):
        prior_subj = _normalize_entity(prior.main_entity.text)
        prior_ans = _normalize_entity(prior.answer or "")
        if candidate_norm and (candidate_norm == prior_subj or candidate_norm == prior_ans):
            return VerificationResult(
                approved=False,
                explanation=(
                    f"Rejected: answer regresses to hop {prior.hop_index} "
                    "(matches a prior-hop subject or answer)."
                ),
                recovery_action=(
                    "Re-run reasoning with the active sub-question locked and explicitly "
                    "forbid prior-hop subjects/answers as candidate outputs."
                ),
            )

    expected_type = _expected_answer_type(hop.relation, hop.subquestion_text)
    inferred_type = _infer_entity_type(candidate)
    if expected_type != "Unknown" and inferred_type != "Unknown" and expected_type != inferred_type:
        return VerificationResult(
            approved=False,
            explanation=(
                f"Rejected: type mismatch for this hop. Expected {expected_type}, got {inferred_type}."
            ),
            recovery_action=(
                "Re-run retrieval using the same subject anchor and relation, and rank candidates by expected entity type."
            ),
        )

    # 5. Granularity check (fine-grained subtype within ``Location``/``Date``/...).
    #    When the sub-question explicitly asks for a *specific* granularity
    #    (e.g. "which city") and the candidate looks like a coarser unit
    #    (e.g. a country surface form), reject -- this is the failure mode
    #    where the reasoner returns a country for a city question. The check
    #    is conservative: it only fires for proven mismatches based on
    #    language markers in the surface form, never on entity lookup.
    granularity = _expected_answer_granularity(hop.relation, hop.subquestion_text)
    if granularity in ("City", "State", "Region") and _looks_like_country(candidate):
        return VerificationResult(
            approved=False,
            explanation=(
                f"Rejected: question expects a {granularity}, but answer "
                f"contains a country-level marker."
            ),
            recovery_action=(
                "Re-read the evidence and return the most specific named place "
                f"(a {granularity}) that is explicitly written in the passage; "
                "do not generalise nationalities or countries upward."
            ),
        )

    return VerificationResult(
        approved=True,
        explanation="Approved: answer is aligned with the hop subject, relation, and type constraints.",
        recovery_action=None,
    )


@dataclass(slots=True, frozen=True)
class HopPlanItem:
    """Minimal plan unit for pipeline integration examples."""

    subquestion_text: str
    main_entity_text: str
    main_entity_type: EntityType
    relation: str
    role: str = "subject"


def run_multihop_with_guards(
    question: str,
    hop_plan: Sequence[HopPlanItem],
    retrieve_fn: Callable[[str], str],
    reason_fn: Callable[[str, str], tuple[str, float]],
) -> HopMemory:
    """Short integration example for a multi-hop loop.

    Contract:
    - ``retrieve_fn`` receives a retrieval query and returns evidence text.
    - ``reason_fn`` receives (prompt, evidence_text) and returns (answer, confidence).

    The loop demonstrates the safe order of operations:
    1. Add hop with explicit subject anchor.
    2. Build retrieval query from full chain memory.
    3. Generate reasoner prompt from current hop memory.
    4. Verify answer before committing it to memory.
    """
    memory = HopMemory(original_question=question)

    for plan_item in hop_plan:
        hop = memory.add_hop(
            subquestion_text=plan_item.subquestion_text,
            main_entity_text=plan_item.main_entity_text,
            main_entity_type=plan_item.main_entity_type,
            relation=plan_item.relation,
            role=plan_item.role,
        )

        retrieval_query = build_retrieval_query(hop, memory)
        evidence_text = retrieve_fn(retrieval_query)

        prompt = get_reasoner_prompt(hop.subquestion_text, memory)
        candidate_answer, candidate_conf = reason_fn(prompt, evidence_text)

        verification = verify_hop_answer(candidate_answer, hop.subquestion_text, memory)
        if not verification.approved:
            raise RuntimeError(
                "Hop answer rejected. "
                f"hop={hop.hop_index}; reason={verification.explanation}; "
                f"recovery={verification.recovery_action}"
            )

        memory.record_answer(hop.hop_index, candidate_answer, candidate_conf)

    return memory


def _normalize_ws(text: str) -> str:
    return " ".join((text or "").strip().split())


def _normalize_entity(text: str) -> str:
    norm = _normalize_ws(text).lower()
    norm = re.sub(r"[^a-z0-9\s]", "", norm)
    return " ".join(norm.split())


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _classify_relation_family(relation: str) -> str:
    """Map arbitrary relation labels to a broad semantic family."""
    rel = (relation or "").strip().lower().replace("-", "_")

    education_keywords = {"study", "studied", "attend", "attended", "education", "educated", "alma"}
    employment_keywords = {"work", "worked", "employment", "employed", "teach", "taught", "associate", "affiliation"}
    location_keywords = {"born", "birth", "located", "location", "capital", "city", "country", "region"}

    if any(token in rel for token in education_keywords):
        return "education"
    if any(token in rel for token in employment_keywords):
        return "employment_or_association"
    if any(token in rel for token in location_keywords):
        return "location"
    return "generic"


def _expected_answer_type(relation: str, subquestion_text: str) -> EntityType:
    """Infer expected answer type from relation and question wording.

    This function uses broad lexical cues so it remains robust to arbitrary relation
    label conventions in different pipelines.
    """
    relation_family = _classify_relation_family(relation)
    q = (subquestion_text or "").lower()

    if relation_family in {"education", "employment_or_association"}:
        return "Organization"
    if relation_family == "location":
        return "Location"
    if "who" in q:
        return "Person"
    if "where" in q or "which city" in q or "which country" in q:
        return "Location"
    if "which university" in q or "which company" in q or "which organization" in q:
        return "Organization"
    return "Unknown"


# Linguistic cues for fine-grained answer granularity. The lists are intentionally
# small, language-only patterns -- they describe how a question phrases a finer
# subtype, not which entities qualify. No entity database, no benchmark patterns.
_GRANULARITY_QUESTION_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    # (granularity_label, sub-question phrases that imply this granularity)
    ("City",       ("which city", "what city", "in which city", "in what city",
                    "name of the city", "city was", "city is", "city in which")),
    ("Country",    ("which country", "what country", "in which country",
                    "in what country", "country was", "country is",
                    "what nation", "which nation")),
    ("State",      ("which state", "what state", "which province",
                    "what province", "in which state", "in what state")),
    ("Region",     ("which region", "what region", "which area", "what area")),
    ("Year",       ("what year", "which year", "in what year", "in which year",
                    "year did", "year was", "year of", "when did", "when was")),
    ("Month",      ("which month", "what month", "in what month",
                    "in which month")),
    ("Date",       ("on what date", "on which date", "what date")),
    ("University", ("which university", "what university", "which college",
                    "what college")),
    ("Company",    ("which company", "what company", "which corporation",
                    "what corporation", "which firm", "what firm")),
    ("Organization", ("which organization", "what organization",
                      "which organisation", "what organisation",
                      "which institution", "what institution")),
    ("Book",       ("which book", "what book", "which novel", "what novel")),
    ("Film",       ("which film", "what film", "which movie", "what movie")),
)


def _expected_answer_granularity(relation: str, subquestion_text: str) -> str:
    """Detect fine-grained answer granularity from sub-question wording.

    Returns labels like ``"City"``, ``"Country"``, ``"Year"``, ``"University"``,
    ``"Company"`` -- subtypes finer than the broad ``EntityType`` returned by
    ``_expected_answer_type``. Returns ``""`` when no cue is present.

    The detection is purely linguistic: it inspects sub-question text for
    explicit type cues such as ``"which city"`` or ``"what year"``. It does NOT
    consult any entity database or benchmark-specific list. The same cue
    inventory works across datasets because the cues are how natural-language
    questions request a specific granularity, not what entities qualify.
    """
    q = (subquestion_text or "").lower()
    if not q:
        return ""
    for label, cues in _GRANULARITY_QUESTION_CUES:
        for cue in cues:
            if cue in q:
                return label
    return ""


# Linguistic markers indicating an answer is a country-level (not city/region)
# location. Used only to detect coarse-vs-fine mismatches when the question
# asks for a fine granularity (e.g. "which city"). These are language tokens
# that appear *inside* country names ("United States" contains "states") --
# not entity names themselves.
_COUNTRY_MARKERS: tuple[str, ...] = (
    "country", "nation", "republic", "kingdom", "federation", "states",
    "emirates", "principality", "sultanate", "commonwealth",
)


def _looks_like_country(text: str) -> bool:
    """Best-effort linguistic check for country-level surface forms.

    Returns True when the surface form contains a country-level word marker
    (e.g. "States" in "United States", "Kingdom" in "United Kingdom"). This
    is a *language* heuristic over the surface tokens, not an entity lookup.
    Single-token surface forms ("France", "Japan") are NOT classified as
    countries by this function -- they fall through and the verifier remains
    permissive about them, consistent with the rule that we only reject when
    we can prove a mismatch.
    """
    lowered = (text or "").lower()
    if not lowered:
        return False
    tokens = re.findall(r"[a-z]+", lowered)
    if not tokens:
        return False
    return any(marker in tokens for marker in _COUNTRY_MARKERS)


def _infer_entity_type(surface: str) -> EntityType:
    """Very light heuristic entity typing used as a safety check.

    Pipelines with a stronger NER/linker can replace this with model-backed typing.
    """
    text = (surface or "").strip()
    lowered = text.lower()

    if not text:
        return "Unknown"

    org_markers = (
        "university",
        "college",
        "institute",
        "school",
        "company",
        "corporation",
        "corp",
        "inc",
        "ltd",
        "laboratory",
        "lab",
        "agency",
        "committee",
    )
    if any(marker in lowered for marker in org_markers):
        return "Organization"

    location_markers = (
        "city",
        "country",
        "state",
        "province",
        "kingdom",
        "republic",
        "island",
        "county",
    )
    if any(marker in lowered for marker in location_markers):
        return "Location"

    # "<Place>, <Place>" surface forms (e.g. a city followed by a state or
    # country abbreviation, separated by a comma) are *structurally* place
    # names, not people. Personal names virtually never carry a comma in
    # answer spans -- person spans with commas are usually role-suffixed
    # ("X, MP") which we still catch via the org markers above.
    if "," in text:
        comma_parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(comma_parts) >= 2 and all(
            p[:1].isupper() for p in comma_parts if p[0].isalpha()
        ):
            return "Location"

    # Country-marker heuristic: surface forms containing words like
    # "Kingdom", "States", "Federation" are countries (Locations).
    if _looks_like_country(text):
        return "Location"

    # Multi-token title-cased spans are usually entities; this fallback keeps the
    # verifier permissive while still catching explicit location/org markers above.
    tokens = [t for t in text.split() if t]
    if len(tokens) >= 2 and all(t[:1].isupper() for t in tokens if t[0].isalpha()):
        return "Person"

    return "Unknown"


# ---------------------------------------------------------------------------
# Public helpers used by the LangGraph pipeline
# ---------------------------------------------------------------------------


_VALID_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "Person",
        "Location",
        "Organization",
        "Work",
        "Event",
        "Date",
        "Number",
        "Other",
        "Unknown",
    }
)


def expected_answer_type(relation: str, subquestion_text: str) -> EntityType:
    """Public wrapper around the relation/wording-based type inference.

    Exposed so the critic and reasoner can derive the expected hop output type
    without depending on an internal helper.
    """
    return _expected_answer_type(relation, subquestion_text)


def expected_answer_granularity(relation: str, subquestion_text: str) -> str:
    """Public wrapper for fine-grained answer-type detection from question wording.

    Returns labels like ``"City"``, ``"Country"``, ``"Year"``, ``"University"``,
    ``"Company"``. ``""`` if no clear cue is present in the sub-question. The
    classification is language-only -- no entity database or dataset patterns.
    """
    return _expected_answer_granularity(relation, subquestion_text)


def looks_like_country(text: str) -> bool:
    """Public wrapper for the country-marker linguistic check.

    Returns True when the surface form contains a country-level token
    (e.g. ``"States"`` in ``"United States"``). Single-token country names
    (``"France"``) are NOT flagged -- the heuristic only fires on
    confident multi-token surface forms.
    """
    return _looks_like_country(text)


def infer_entity_type(surface: str) -> EntityType:
    """Public wrapper around the heuristic surface-form type inference."""
    return _infer_entity_type(surface)


def normalize_entity(text: str) -> str:
    """Public alphanumeric-folded entity normaliser for equality checks."""
    return _normalize_entity(text)


def hop_memory_to_dict(memory: HopMemory) -> dict[str, Any]:
    """Serialise a ``HopMemory`` to a plain-dict structure for graph state.

    The dict round-trips through ``hop_memory_from_dict``. Use this on every
    pipeline-node return so LangGraph can persist hop progress between nodes.
    """
    return {
        "original_question": memory.original_question,
        "hops": [
            {
                "hop_index": h.hop_index,
                "subquestion_text": h.subquestion_text,
                "main_entity": {
                    "text": h.main_entity.text,
                    "entity_type": h.main_entity.entity_type,
                    "role": h.main_entity.role,
                },
                "relation": h.relation,
                "answer": h.answer,
                "confidence": h.confidence,
            }
            for h in memory.hops
        ],
    }


def hop_memory_from_dict(payload: dict[str, Any] | None) -> HopMemory | None:
    """Reconstruct a ``HopMemory`` from a serialised state-dict."""
    if not payload:
        return None
    memory = HopMemory(original_question=str(payload.get("original_question") or ""))
    for raw in payload.get("hops") or []:
        ent = raw.get("main_entity") or {}
        entity_type_raw = str(ent.get("entity_type") or "Unknown")
        entity_type: EntityType = (
            entity_type_raw if entity_type_raw in _VALID_ENTITY_TYPES else "Unknown"
        )  # type: ignore[assignment]
        record = HopRecord(
            hop_index=int(raw.get("hop_index") or 0),
            subquestion_text=str(raw.get("subquestion_text") or ""),
            main_entity=EntityState(
                text=str(ent.get("text") or ""),
                entity_type=entity_type,
                role=str(ent.get("role") or "subject"),
            ),
            relation=str(raw.get("relation") or "related_to"),
            answer=raw.get("answer"),
            confidence=raw.get("confidence"),
        )
        memory.hops.append(record)
    return memory


def build_reasoner_guard_prompt(
    current_hop: HopRecord,
    memory: HopMemory,
) -> str:
    """Return a short guard prompt enforcing active-sub-question discipline.

    The prompt:
    - Names the active sub-question, subject and relation so the reasoner is
      anchored to the current hop, not the original multi-hop question.
    - States the expected answer type derived from the relation/wording.
    - States the *fine-grained* answer granularity (City vs Country, Year,
      University, ...) when the sub-question text gives a clear cue. This is
      what stops "born in which city?" from being answered with a country.
    - Enumerates prior-hop subjects and answers as forbidden outputs to make
      regression to an earlier hop impossible-by-instruction. (The verifier
      enforces this independently — the prompt is a soft preventative.)
    """
    expected_type = _expected_answer_type(current_hop.relation, current_hop.subquestion_text)
    granularity = _expected_answer_granularity(
        current_hop.relation, current_hop.subquestion_text
    )
    forbidden: list[str] = []
    for prior in memory.previous_hops(current_hop.hop_index):
        if prior.main_entity.text:
            forbidden.append(prior.main_entity.text)
        if prior.answer:
            forbidden.append(prior.answer)
    forbidden = list(dict.fromkeys(s.strip() for s in forbidden if s and s.strip()))
    forbidden_clause = (
        "Forbidden answers (regress to a previous hop): " + ", ".join(forbidden)
        if forbidden
        else "Forbidden answers: (none — this is hop 1)"
    )

    # When the sub-question demands a fine granularity, the reasoner is
    # required to return that *specific* unit. The phrasing avoids any
    # entity-name guidance (no "United States", no "Seattle") -- it works for
    # any city/country/year/etc. in any dataset because it relies entirely on
    # the linguistic cue already present in the sub-question.
    granularity_clause = ""
    if granularity:
        granularity_clause = (
            f"Required answer granularity: {granularity}.\n"
            f"  - The answer MUST be a specific {granularity.lower()} that is "
            "literally written in the retrieved passages.\n"
            f"  - Do NOT generalise upward to a coarser unit (e.g. country "
            f"when a city is asked, decade when a year is asked).\n"
            f"  - Do NOT derive a {granularity.lower()} from a nationality "
            "adjective; that is a different relation.\n"
        )

    return (
        "ACTIVE SUB-QUESTION LOCK — answer ONLY this sub-question, ignore the "
        "broader multi-hop question framing.\n"
        f"Active sub-question: {current_hop.subquestion_text}\n"
        f"Active subject: {current_hop.main_entity.text} "
        f"({current_hop.main_entity.entity_type}, role={current_hop.main_entity.role})\n"
        f"Active relation: {current_hop.relation}\n"
        f"Expected answer type: {expected_type}\n"
        f"{granularity_clause}"
        f"{forbidden_clause}\n"
        "Most-specific-span rule: when several entities of the expected type "
        "appear in the evidence, return the MOST SPECIFIC one explicitly named "
        "in the passages, not a broader category derived from world knowledge.\n"
        "If the evidence does not support a NEW entity (one not in the forbidden "
        "list) that fits the active relation, expected type, and granularity, "
        "return EXACTLY: 'Not enough information in retrieved passages.'"
    )
