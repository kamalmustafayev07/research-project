"""Tests for the hop-state-locked multi-hop QA pipeline fix.

These tests verify that the architectural fix (persistent ``HopMemory`` +
relation-conditioned, regression-aware verification + active-hop locking in
the reasoner) generalises across multi-hop QA shapes without depending on
any specific entity, dataset, or example.

Test matrix (from the task brief):

    A. writer of X        -> which university did writer attend?
    B. inventor of Y      -> what country was inventor born in?
    C. CEO of Z           -> what university did CEO attend?
    D. 2-hop, second answer is a short entity name.
    E. Person -> Organization with explicit regression attempt.

The tests intentionally do NOT call any LLM. They exercise:

- ``HopMemory`` state transitions across hops (no regression from k -> k-1).
- ``verify_hop_answer`` typed/regression checks for each scenario above.
- ``CriticAgent._active_hop_hard_checks`` against an active-hop payload.
- The pipeline graph router's behaviour on approve/reject (advance vs retry).
- Generality: every scenario uses different entities, no hard-coding.

Run:
    .venv/Scripts/python.exe _test_hop_state.py
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from src.agents.critic import CriticAgent
from src.agents.hop_safety import (
    EntityState,
    HopMemory,
    HopRecord,
    build_reasoner_guard_prompt,
    expected_answer_type,
    hop_memory_from_dict,
    hop_memory_to_dict,
    infer_entity_type,
    verify_hop_answer,
)


SEPARATOR = "=" * 72


class TestFailure(AssertionError):
    """Raised when a test assertion fails."""


# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


_passed: list[str] = []
_failed: list[tuple[str, str]] = []


def section(title: str) -> None:
    print(f"\n{SEPARATOR}\n  {title}\n{SEPARATOR}")


def expect(name: str, condition: bool, detail: str = "") -> None:
    """Check an assertion. Records pass/fail without aborting."""
    if condition:
        _passed.append(name)
        print(f"  [PASS] {name}")
    else:
        msg = f"{name} -- {detail}" if detail else name
        _failed.append((name, detail))
        print(f"  [FAIL] {msg}")


def build_two_hop_memory(
    question: str,
    hop1: tuple[str, str, str, str],
    hop2: tuple[str, str, str, str],
) -> HopMemory:
    """Build a 2-hop HopMemory.

    Each tuple is ``(subquestion_text, subject_text, subject_type, relation)``.
    """
    memory = HopMemory(original_question=question)
    memory.add_hop(
        subquestion_text=hop1[0],
        main_entity_text=hop1[1],
        main_entity_type=hop1[2],  # type: ignore[arg-type]
        relation=hop1[3],
    )
    memory.add_hop(
        subquestion_text=hop2[0],
        main_entity_text=hop2[1],
        main_entity_type=hop2[2],  # type: ignore[arg-type]
        relation=hop2[3],
    )
    return memory


# ---------------------------------------------------------------------------
# Test A — writer of X -> which university did writer attend?
#
# Scenarios tested:
#  A.1  Hop 1 approves a person; hop 2 must NOT accept the hop-1 person again
#       (regression rejection).
#  A.2  Hop 2 accepts a valid Organization answer.
#  A.3  Hop 2 accepts a valid SHORT one-token answer (no length-based reject).
# ---------------------------------------------------------------------------


def test_a_writer_university() -> None:
    section("Test A — writer of X -> which university did writer attend?")

    question = "Which university did the writer of <WORK_X> attend?"
    memory = build_two_hop_memory(
        question,
        hop1=("Who is the writer of <WORK_X>?", "<WORK_X>", "Work", "author_of"),
        hop2=(
            "Which university did <answer of hop 1> attend?",
            "<answer of hop 1>",
            "Unknown",
            "studied_at",
        ),
    )

    # Hop 1 resolves to a person -- generic placeholder name (no real entity).
    memory.record_answer(1, "PERSON_ALPHA", confidence=0.9)
    # Now lock hop 2 with the person as its concrete subject.
    memory.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=memory.hops[1].subquestion_text,
        main_entity=EntityState(text="PERSON_ALPHA", entity_type="Person"),
        relation="studied_at",
    )

    # A.1: the reasoner regresses -- proposes hop-1 person again. REJECT.
    # The verifier may reject this either because the candidate equals the
    # current hop's (locked) subject (the prior-hop answer became the new
    # subject) OR because it equals a prior-hop answer. Both are valid
    # rejections — both indicate "this is a regression, not a hop answer".
    v_regress = verify_hop_answer(
        proposed_answer="PERSON_ALPHA",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "A.1 hop2 regression to hop1 person rejected",
        not v_regress.approved
        and (
            "regress" in v_regress.explanation.lower()
            or "subject" in v_regress.explanation.lower()
        ),
        v_regress.explanation,
    )

    # A.2: a valid Organization answer for studied_at. APPROVE.
    v_org = verify_hop_answer(
        proposed_answer="University of Northgate",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "A.2 hop2 valid organization answer approved",
        v_org.approved,
        v_org.explanation,
    )

    # A.3: a valid short answer (one-token). Inferred type is Unknown which
    # means the conservative type rule SKIPS the type check -> APPROVE.
    v_short = verify_hop_answer(
        proposed_answer="Northgate",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "A.3 hop2 short single-token answer approved (no length-based reject)",
        v_short.approved,
        v_short.explanation,
    )


# ---------------------------------------------------------------------------
# Test B — inventor of Y -> what country was inventor born in?
# ---------------------------------------------------------------------------


def test_b_inventor_country() -> None:
    section("Test B — inventor of Y -> what country was inventor born in?")

    question = "What country was the inventor of <DEVICE_Y> born in?"
    memory = build_two_hop_memory(
        question,
        hop1=("Who invented <DEVICE_Y>?", "<DEVICE_Y>", "Work", "invented_by"),
        hop2=(
            "In which country was <answer of hop 1> born?",
            "<answer of hop 1>",
            "Unknown",
            "born_in",
        ),
    )

    memory.record_answer(1, "PERSON_BETA", confidence=0.85)
    memory.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=memory.hops[1].subquestion_text,
        main_entity=EntityState(text="PERSON_BETA", entity_type="Person"),
        relation="born_in",
    )

    # B.1: regression to inventor (a Person) on a born_in (Location) hop.
    v_regress = verify_hop_answer(
        proposed_answer="PERSON_BETA",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "B.1 hop2 regression to inventor rejected",
        not v_regress.approved,
        v_regress.explanation,
    )

    # B.2: a valid country answer with a country marker -> Location -> APPROVE.
    v_loc = verify_hop_answer(
        proposed_answer="Republic of Aurelia",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "B.2 hop2 valid country (with marker) approved",
        v_loc.approved,
        v_loc.explanation,
    )

    # B.3: short country surface form (no marker). Type infers Unknown ->
    # check is skipped -> approved as long as no regression and not abstain.
    v_short = verify_hop_answer(
        proposed_answer="Aurelia",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "B.3 hop2 short country approved (Unknown -> skip type check)",
        v_short.approved,
        v_short.explanation,
    )

    # B.4: hard wrong answer for a born_in question -- e.g. an obviously
    # mistyped Person answer ("John Smith") to a born_in hop. The verifier
    # should reject because expected=Location, inferred=Person (multi-token
    # capitalised surface form).
    v_typed = verify_hop_answer(
        proposed_answer="John Smith",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "B.4 hop2 person-typed answer rejected on born_in (Location-typed) hop",
        not v_typed.approved
        and ("type mismatch" in v_typed.explanation.lower()
             or "regress" in v_typed.explanation.lower()),
        v_typed.explanation,
    )


# ---------------------------------------------------------------------------
# Test C — CEO of company -> what university did CEO attend?
# ---------------------------------------------------------------------------


def test_c_ceo_university() -> None:
    section("Test C — CEO of company -> what university did CEO attend?")

    question = "Which university did the CEO of <COMPANY_Z> attend?"
    memory = build_two_hop_memory(
        question,
        hop1=("Who is the CEO of <COMPANY_Z>?", "<COMPANY_Z>", "Organization", "founded_by"),
        hop2=(
            "Which university did <answer of hop 1> attend?",
            "<answer of hop 1>",
            "Unknown",
            "studied_at",
        ),
    )
    memory.record_answer(1, "PERSON_GAMMA", confidence=0.88)
    memory.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=memory.hops[1].subquestion_text,
        main_entity=EntityState(text="PERSON_GAMMA", entity_type="Person"),
        relation="studied_at",
    )

    # C.1: regression to the CEO (Person) -- REJECT.
    v_regress = verify_hop_answer(
        proposed_answer="PERSON_GAMMA",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "C.1 hop2 regression to CEO person rejected",
        not v_regress.approved,
        v_regress.explanation,
    )

    # C.2: regression to the COMPANY (the hop-1 *subject*) -- REJECT.
    v_subj = verify_hop_answer(
        proposed_answer="<COMPANY_Z>",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "C.2 hop2 regression to hop1 subject (company) rejected",
        not v_subj.approved,
        v_subj.explanation,
    )

    # C.3: valid org answer for studied_at -- APPROVE.
    v_ok = verify_hop_answer(
        proposed_answer="Plumstead University",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "C.3 hop2 valid university answer approved",
        v_ok.approved,
        v_ok.explanation,
    )


# ---------------------------------------------------------------------------
# Test D — 2-hop question where second answer is a short entity name.
# ---------------------------------------------------------------------------


def test_d_short_final_answer() -> None:
    section("Test D — 2-hop with short final answer")

    question = "In which city does the founder of <STARTUP_W> work?"
    memory = build_two_hop_memory(
        question,
        hop1=("Who founded <STARTUP_W>?", "<STARTUP_W>", "Organization", "founded_by"),
        hop2=(
            "In which city does <answer of hop 1> work?",
            "<answer of hop 1>",
            "Unknown",
            "located_in",
        ),
    )
    memory.record_answer(1, "PERSON_DELTA", confidence=0.9)
    memory.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=memory.hops[1].subquestion_text,
        main_entity=EntityState(text="PERSON_DELTA", entity_type="Person"),
        relation="located_in",
    )

    # D.1: short, single-token city name should approve. Inferred=Unknown,
    # expected=Location -> conservative skip -> approve.
    v_short = verify_hop_answer(
        proposed_answer="Tilburg",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "D.1 short single-token city answer approved",
        v_short.approved,
        v_short.explanation,
    )

    # D.2: high reasoner confidence does not change the verifier's decision —
    # we only check that the verifier is not flipping on length. We model this
    # by also feeding the same answer through the critic agent's hard-check
    # helper directly with reasoner_confidence=0.9; it must not reject.
    hop2 = memory.hops[1]
    active_hop_payload: dict[str, Any] = {
        "hop_index": 2,
        "subquestion_text": hop2.subquestion_text,
        "resolved_subquestion": hop2.subquestion_text,
        "subject_text": hop2.main_entity.text,
        "subject_type": hop2.main_entity.entity_type,
        "relation": hop2.relation,
        "expected_answer_type": expected_answer_type(hop2.relation, hop2.subquestion_text),
        "prior_subjects": [memory.hops[0].main_entity.text],
        "prior_answers": [memory.hops[0].answer or ""],
    }
    failures = CriticAgent._active_hop_hard_checks(
        candidate_answer="Tilburg",
        reasoner_confidence=0.9,
        active_hop=active_hop_payload,
    )
    expect(
        "D.2 critic hard-checks accept short answer with high confidence",
        failures == [],
        f"unexpected failures: {failures}",
    )


# ---------------------------------------------------------------------------
# Test E — Previous hop is a Person, next hop expects an Organization.
#          Ensure no regression to the Person on the second hop.
# ---------------------------------------------------------------------------


def test_e_person_then_org_no_regression() -> None:
    section("Test E — Person -> Organization regression prevention")

    question = "What organisation did the founder of <STARTUP_E> later join?"
    memory = build_two_hop_memory(
        question,
        hop1=("Who founded <STARTUP_E>?", "<STARTUP_E>", "Organization", "founded_by"),
        hop2=(
            "What organisation did <answer of hop 1> later join?",
            "<answer of hop 1>",
            "Unknown",
            "associated_with",
        ),
    )
    memory.record_answer(1, "PERSON_EPSILON", confidence=0.88)
    memory.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=memory.hops[1].subquestion_text,
        main_entity=EntityState(text="PERSON_EPSILON", entity_type="Person"),
        relation="associated_with",
    )

    # E.1: Critic hard-checks reject regression to the Person on the next hop.
    hop2 = memory.hops[1]
    active_hop_payload: dict[str, Any] = {
        "hop_index": 2,
        "subquestion_text": hop2.subquestion_text,
        "resolved_subquestion": hop2.subquestion_text,
        "subject_text": hop2.main_entity.text,
        "subject_type": hop2.main_entity.entity_type,
        "relation": hop2.relation,
        "expected_answer_type": expected_answer_type(hop2.relation, hop2.subquestion_text),
        "prior_subjects": [memory.hops[0].main_entity.text],
        "prior_answers": [memory.hops[0].answer or ""],
    }
    regress_fail = CriticAgent._active_hop_hard_checks(
        candidate_answer="PERSON_EPSILON",
        reasoner_confidence=0.85,
        active_hop=active_hop_payload,
    )
    # Accept either rejection mode — "subject equals answer" (because the
    # prior-hop person became the next-hop subject) or explicit prior-hop
    # regression detection. Both indicate the regression was caught.
    expect(
        "E.1 critic hard-checks reject regression (Person) for active Org-relation hop",
        any(
            "regress" in m.lower() or "equals the hop subject" in m.lower()
            for m in regress_fail
        ),
        f"failures={regress_fail}",
    )

    # E.2: Valid organisation answer is accepted by the critic hard-checks.
    org_fail = CriticAgent._active_hop_hard_checks(
        candidate_answer="Sigma Research Institute",
        reasoner_confidence=0.85,
        active_hop=active_hop_payload,
    )
    expect(
        "E.2 critic hard-checks accept valid organisation answer",
        org_fail == [],
        f"unexpected failures: {org_fail}",
    )

    # E.3: The verifier from hop_safety also rejects the regression directly.
    v = verify_hop_answer(
        proposed_answer="PERSON_EPSILON",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "E.3 hop_safety.verify_hop_answer rejects regression (Person) on Org hop",
        not v.approved
        and (
            "regress" in v.explanation.lower()
            or "subject" in v.explanation.lower()
        ),
        v.explanation,
    )

    # E.4 — Three-hop scenario where hop 3's regression target is *not* the
    # current subject but an earlier hop's answer. This exercises the
    # ``prior_answers`` regression rule independently of subject equality.
    memory3 = HopMemory(original_question="3-hop test")
    memory3.add_hop(
        subquestion_text="Hop1",
        main_entity_text="<WORK_X>",
        main_entity_type="Work",
        relation="author_of",
    )
    memory3.add_hop(
        subquestion_text="Hop2",
        main_entity_text="PERSON_X",
        main_entity_type="Person",
        relation="founded_by",
    )
    memory3.add_hop(
        subquestion_text="Hop3",
        main_entity_text="ORG_X",
        main_entity_type="Organization",
        relation="located_in",
    )
    memory3.record_answer(1, "PERSON_X", confidence=0.9)
    memory3.record_answer(2, "ORG_X", confidence=0.85)

    # Reasoner attempts to regress hop 3 to the hop-1 answer (PERSON_X).
    # That answer is neither the current hop's subject nor the prior hop's
    # answer (which is ORG_X), so the regression must be caught via the
    # explicit "prior-hop answer equality" rule.
    v3 = verify_hop_answer(
        proposed_answer="PERSON_X",
        current_subquestion="Hop3",
        hop_memory=memory3,
    )
    expect(
        "E.4 hop3 regression to hop1 answer rejected via prior-hop rule",
        not v3.approved and "regress" in v3.explanation.lower(),
        v3.explanation,
    )


# ---------------------------------------------------------------------------
# State-machine tests — HopMemory serialisation, guard prompt, advancement.
# ---------------------------------------------------------------------------


def test_state_machine() -> None:
    section("State machine — serialisation, guard prompts, advancement")

    memory = build_two_hop_memory(
        "Original multi-hop question.",
        hop1=("Who founded <ORG_K>?", "<ORG_K>", "Organization", "founded_by"),
        hop2=(
            "Which university did <answer of hop 1> attend?",
            "<answer of hop 1>",
            "Unknown",
            "studied_at",
        ),
    )

    # S.1: hop_memory_to_dict -> hop_memory_from_dict round-trips cleanly.
    payload = hop_memory_to_dict(memory)
    restored = hop_memory_from_dict(payload)
    expect(
        "S.1 HopMemory round-trip preserves hop count and subjects",
        restored is not None
        and len(restored.hops) == 2
        and restored.hops[0].main_entity.text == "<ORG_K>"
        and restored.hops[1].main_entity.text == "<answer of hop 1>",
    )

    # S.2: build_reasoner_guard_prompt enumerates prior subjects + answers
    # as forbidden outputs (after hop 1 has been recorded).
    memory.record_answer(1, "PERSON_KAPPA", confidence=0.9)
    memory.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=memory.hops[1].subquestion_text,
        main_entity=EntityState(text="PERSON_KAPPA", entity_type="Person"),
        relation="studied_at",
    )
    guard = build_reasoner_guard_prompt(memory.hops[1], memory)
    expect(
        "S.2 guard prompt names active sub-question",
        memory.hops[1].subquestion_text in guard,
    )
    expect(
        "S.2 guard prompt forbids prior-hop subject (the org)",
        "<ORG_K>" in guard,
    )
    expect(
        "S.2 guard prompt forbids prior-hop answer (the person)",
        "PERSON_KAPPA" in guard,
    )
    expect(
        "S.2 guard prompt names active relation",
        "studied_at" in guard,
    )
    expect(
        "S.2 guard prompt names expected answer type (Organization)",
        "Organization" in guard,
    )

    # S.3: expected_answer_type for born_in is Location.
    expect(
        "S.3 expected_answer_type(born_in, ...) == Location",
        expected_answer_type("born_in", "Where was X born?") == "Location",
    )
    # S.4: expected_answer_type for studied_at is Organization.
    expect(
        "S.4 expected_answer_type(studied_at, ...) == Organization",
        expected_answer_type("studied_at", "Which university did X attend?") == "Organization",
    )


# ---------------------------------------------------------------------------
# Pipeline routing test — drives the LangGraph router with synthetic state to
# verify approve -> advance and reject -> retry-same-hop semantics, with no
# regression possible across iterations.
# ---------------------------------------------------------------------------


def test_pipeline_router() -> None:
    section("Pipeline router — advance / retry-same-hop / done semantics")

    from src.pipeline import AgentEnhancedGraphRAG

    pipeline = AgentEnhancedGraphRAG.__new__(AgentEnhancedGraphRAG)

    # Two-hop memory, currently at hop 1.
    memory = build_two_hop_memory(
        "Q",
        hop1=("Who founded <ORG_M>?", "<ORG_M>", "Organization", "founded_by"),
        hop2=(
            "Which university did <answer of hop 1> attend?",
            "<answer of hop 1>",
            "Unknown",
            "studied_at",
        ),
    )

    base_state: dict[str, Any] = {
        "hop_memory": hop_memory_to_dict(memory),
        "current_hop_index": 1,
        "hop_attempts": 0,
        "max_hop_attempts": 2,
        "retrieval_loops": 1,
        "max_retrieval_loops": 6,
        "answer": "PERSON_MU",
        "confidence": 0.85,
    }

    # R.1: hop 1 approved AND not last hop -> "advance".
    state_a = dict(base_state, approved=True)
    expect(
        "R.1 approved hop1 -> advance",
        pipeline._route_after_critic(state_a) == "advance",
    )

    # R.2: hop 1 rejected, attempts remaining -> "retry".
    state_b = dict(base_state, approved=False)
    expect(
        "R.2 rejected hop1 with budget -> retry",
        pipeline._route_after_critic(state_b) == "retry",
    )

    # R.3: hop 1 rejected, attempts exhausted -> "done".
    state_c = dict(base_state, approved=False, hop_attempts=2)
    expect(
        "R.3 rejected hop1 with attempts exhausted -> done",
        pipeline._route_after_critic(state_c) == "done",
    )

    # Now advance to hop 2 and check behaviour on the LAST hop.
    advance_payload = pipeline._advance_hop(state_a)
    expect(
        "R.4 advance_hop increments current_hop_index to 2",
        advance_payload.get("current_hop_index") == 2,
    )
    expect(
        "R.4 advance_hop resets hop_attempts to 0",
        advance_payload.get("hop_attempts") == 0,
    )

    state_d_base: dict[str, Any] = {
        "hop_memory": advance_payload.get("hop_memory"),
        "current_hop_index": 2,
        "hop_attempts": 0,
        "max_hop_attempts": 2,
        "retrieval_loops": 2,
        "max_retrieval_loops": 6,
        "answer": "Northgate University",
        "confidence": 0.88,
    }

    # R.5: approved on the LAST hop -> "done".
    state_d = dict(state_d_base, approved=True)
    expect(
        "R.5 approved last hop -> done",
        pipeline._route_after_critic(state_d) == "done",
    )

    # R.6: rejected on the LAST hop with budget -> "retry" (NOT advance,
    # NOT done -- this is the central fix: a failed last hop must retry, not
    # bail out and fall back to the previous hop's answer).
    state_e = dict(state_d_base, approved=False)
    expect(
        "R.6 rejected last hop with budget -> retry (no regression possible)",
        pipeline._route_after_critic(state_e) == "retry",
    )

    # R.7: rejected on the LAST hop, attempts exhausted -> "done"
    # (graceful termination, NOT a regression to a previous hop).
    state_f = dict(state_d_base, approved=False, hop_attempts=2)
    expect(
        "R.7 rejected last hop with budget exhausted -> done (no regression)",
        pipeline._route_after_critic(state_f) == "done",
    )


# ---------------------------------------------------------------------------
# Anti-overfitting check — the verifier and routing logic must produce the
# same decisions regardless of the *content* of entities, as long as the
# *structural* properties (relation, prior-hop subject, prior-hop answer)
# are equivalent. We replay Test A with completely different placeholder
# names and confirm decisions are identical.
# ---------------------------------------------------------------------------


def test_no_overfitting() -> None:
    section("Anti-overfitting — entity-content invariance")

    cases: list[tuple[str, str, str, str, str]] = [
        # (label, hop1_subject, hop1_answer, hop2_relation, hop2_correct_answer)
        ("Run-1", "<WORK_A>", "PERSON_A", "studied_at", "University Alpha"),
        ("Run-2", "<WORK_B>", "PERSON_B", "studied_at", "University Beta"),
        ("Run-3", "<WORK_C>", "PERSON_C", "studied_at", "University Gamma"),
        ("Run-4", "<INVENTION_D>", "PERSON_D", "born_in", "Country Delta"),
    ]

    decisions: list[tuple[bool, bool, bool]] = []
    for label, h1_subj, h1_ans, h2_rel, h2_correct in cases:
        memory = build_two_hop_memory(
            f"{label}: 2-hop test",
            hop1=("Hop1 subq", h1_subj, "Work", "author_of"),
            hop2=("Hop2 subq", "<answer of hop 1>", "Unknown", h2_rel),
        )
        memory.record_answer(1, h1_ans, confidence=0.9)
        memory.hops[1] = HopRecord(
            hop_index=2,
            subquestion_text=memory.hops[1].subquestion_text,
            main_entity=EntityState(text=h1_ans, entity_type="Person"),
            relation=h2_rel,
        )
        # 1) regression to hop-1 answer is rejected
        v_regress = verify_hop_answer(h1_ans, memory.hops[1].subquestion_text, memory)
        # 2) correct answer is approved
        v_correct = verify_hop_answer(h2_correct, memory.hops[1].subquestion_text, memory)
        # 3) regression to hop-1 SUBJECT is rejected
        v_subj = verify_hop_answer(h1_subj, memory.hops[1].subquestion_text, memory)
        decisions.append((not v_regress.approved, v_correct.approved, not v_subj.approved))
        print(
            f"  {label}: regress_rejected={not v_regress.approved}, "
            f"correct_approved={v_correct.approved}, "
            f"subj_rejected={not v_subj.approved}"
        )

    expect(
        "Anti-overfitting: regression-to-prior-answer ALWAYS rejected",
        all(d[0] for d in decisions),
    )
    expect(
        "Anti-overfitting: correct answer ALWAYS approved",
        all(d[1] for d in decisions),
    )
    expect(
        "Anti-overfitting: regression-to-prior-subject ALWAYS rejected",
        all(d[2] for d in decisions),
    )


# ---------------------------------------------------------------------------
# Test G — fine-grained answer granularity (city vs country, year, ...).
#
# Verifies that when a sub-question explicitly asks for a CITY, a country-level
# answer is rejected with a recovery action that asks for a more specific span.
# The cues used by the granularity machinery are linguistic only -- there is
# no entity database or hard-coded mapping.
# ---------------------------------------------------------------------------


def test_g_answer_granularity() -> None:
    section("Test G — answer-granularity rejects coarse country for 'city' question")

    from src.agents.hop_safety import (
        expected_answer_granularity,
        looks_like_country,
    )

    # G.1: question explicitly asks for a city -> granularity is "City".
    expect(
        "G.1 'In which city was X born?' -> granularity=City",
        expected_answer_granularity("born_in", "In which city was X born?") == "City",
    )
    # G.2: question asks for a country -> granularity is "Country".
    expect(
        "G.2 'What country was X born in?' -> granularity=Country",
        expected_answer_granularity("born_in", "What country was X born in?") == "Country",
    )
    # G.3: question asks "which year" -> granularity is "Year".
    expect(
        "G.3 'In which year was X founded?' -> granularity=Year",
        expected_answer_granularity("founded_by", "In which year was X founded?") == "Year",
    )
    # G.4: surface-form heuristic detects a country marker.
    expect(
        "G.4 looks_like_country('United States') == True",
        looks_like_country("United States") is True,
    )
    expect(
        "G.4b looks_like_country('Seattle, Washington') == False",
        looks_like_country("Seattle, Washington") is False,
    )

    # G.5: build a 'city' question hop and verify a country-level answer
    # is REJECTED with a specific-span recovery hint.
    memory = build_two_hop_memory(
        "The founder of <ORG_X> was born in which city?",
        hop1=("Who is the founder of <ORG_X>?", "<ORG_X>", "Organization", "founded_by"),
        hop2=(
            "In which city was <answer of hop 1> born?",
            "<answer of hop 1>",
            "Unknown",
            "born_in",
        ),
    )
    memory.record_answer(1, "PERSON_OMEGA", confidence=0.9)
    memory.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=memory.hops[1].subquestion_text,
        main_entity=EntityState(text="PERSON_OMEGA", entity_type="Person"),
        relation="born_in",
    )

    v_country = verify_hop_answer(
        proposed_answer="United States",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "G.5 hop2 rejects 'United States' for a 'city' question",
        not v_country.approved,
        v_country.explanation,
    )
    expect(
        "G.5b rejection mentions city / specific / country granularity context",
        not v_country.approved
        and (
            "city" in v_country.explanation.lower()
            or "country" in v_country.explanation.lower()
            or (v_country.recovery_action or "").lower().find("specific") >= 0
        ),
        f"explanation={v_country.explanation!r} action={v_country.recovery_action!r}",
    )

    # G.6: same hop, named city is APPROVED.
    v_city = verify_hop_answer(
        proposed_answer="Seattle, Washington",
        current_subquestion=memory.hops[1].subquestion_text,
        hop_memory=memory,
    )
    expect(
        "G.6 hop2 approves a named city for a 'city' question",
        v_city.approved,
        v_city.explanation,
    )


# ---------------------------------------------------------------------------
# Test H — relation prototype scorer (LLM-driven cue scoring).
#
# Verifies the prototype scorer's two key properties:
#   - cue_phrases substrings yield high scores; off-topic passages score low;
#   - empty / missing prototypes return None so that the keyword-table
#     fallback in entity_scoring.py keeps working.
# We do NOT exercise the LLM here; we synthesise a prototype directly.
# ---------------------------------------------------------------------------


def test_h_relation_prototype_scoring() -> None:
    section("Test H — LLM-driven relation prototype scoring")

    from src.agents.relation_scorer import (
        RelationPrototype,
        prototype_from_dict,
        prototype_to_dict,
        relation_prototype_score,
    )
    from src.graph.entity_scoring import relation_consistency_score

    proto = RelationPrototype(
        description="Biographical passage stating where the subject was born.",
        cue_phrases=["born in", "born and raised", "birthplace", "raised in"],
        negative_cue_phrases=["died in", "founded in"],
        answer_pattern="city name",
        focus_query="<subject> birthplace city",
    )

    answer_bearing = (
        "Born and raised in Northgate, the subject co-founded the company in "
        "another city in 1975."
    )
    off_topic = (
        "The subject founded the company in 1975 and later died in another country."
    )

    s_hi = relation_prototype_score(answer_bearing, proto)
    s_lo = relation_prototype_score(off_topic, proto)
    expect(
        "H.1 prototype scores answer-bearing passage > off-topic passage",
        s_hi > s_lo,
        f"answer={s_hi:.3f} off_topic={s_lo:.3f}",
    )
    expect(
        "H.1b answer-bearing passage scores > 0.0",
        s_hi > 0.0,
        f"answer={s_hi:.3f}",
    )

    # H.2: prototype dict round-trip preserves cue inventory.
    p2 = prototype_from_dict(prototype_to_dict(proto))
    expect(
        "H.2 prototype dict round-trip preserves cue_phrases",
        p2.cue_phrases == proto.cue_phrases
        and p2.negative_cue_phrases == proto.negative_cue_phrases,
    )

    # H.3: relation_consistency_score prefers prototype cues over keywords
    # when a prototype is supplied. Off-topic passages should score near 0
    # under the prototype path.
    rel_hi = relation_consistency_score(
        {"title": "X", "text": answer_bearing},
        relation_hints=["born_in"],
        prototype=prototype_to_dict(proto),
    )
    rel_lo = relation_consistency_score(
        {"title": "X", "text": off_topic},
        relation_hints=["born_in"],
        prototype=prototype_to_dict(proto),
    )
    expect(
        "H.3 relation_consistency_score with prototype: answer > off_topic",
        rel_hi > rel_lo,
        f"answer={rel_hi:.3f} off_topic={rel_lo:.3f}",
    )

    # H.4: missing prototype falls back to keyword table -- non-zero for an
    # answer-bearing born_in passage that mentions the keyword "born".
    rel_kw = relation_consistency_score(
        {"title": "X", "text": "She was born in Northgate."},
        relation_hints=["born_in"],
        prototype=None,
    )
    expect(
        "H.4 keyword fallback non-zero when no prototype is supplied",
        rel_kw > 0.0,
        f"score={rel_kw:.3f}",
    )

    # H.5: empty cue list returns the neutral signal so the caller can fall
    # back to the keyword table without bias.
    empty_proto = prototype_to_dict(RelationPrototype())
    rel_empty = relation_consistency_score(
        {"title": "X", "text": "She was born in Northgate."},
        relation_hints=["born_in"],
        prototype=empty_proto,
    )
    expect(
        "H.5 empty prototype -> falls through to keyword table (>0 here)",
        rel_empty > 0.0,
        f"score={rel_empty:.3f}",
    )


# ---------------------------------------------------------------------------
# Test I — reasoner focus-sentence extraction.
#
# Ensures that when a passage is too long to fit in the per-passage budget,
# answer-bearing sentences from the truncated tail are surfaced via cue-
# anchored sentence extraction. This covers the "Seattle appears past
# char 320" failure mode from the original Bill Gates trace.
# ---------------------------------------------------------------------------


def test_i_focus_sentence_extraction() -> None:
    section("Test I — reasoner focus-sentence extraction surfaces tail evidence")

    from src.agents.react_reasoner import _focus_sentences_for, _split_sentences

    # Deliberately long passage. The "Born and raised in <CITY>" sentence is
    # placed AFTER the first 320-char window to mimic the Bill Gates failure.
    head_pad = (
        "X is described as an entrepreneur and philanthropist. The biography "
        "spans many years and discusses career milestones, board roles, and "
        "later philanthropic work, including foundations and cross-industry "
        "collaborations spanning multiple decades and countries. " * 4
    )
    tail = "Born and raised in Northgate, the subject later co-founded a company."
    text = head_pad + tail
    assert len(head_pad) > 320, "test setup: head must exceed truncation budget"

    # Caller passes anchor cues from prototype.cue_phrases.
    anchors = ["born and raised", "birthplace", "raised in"]
    sentences = _focus_sentences_for(text, anchors, max_sentences=3)
    expect(
        "I.1 focus-sentence extractor finds answer-bearing tail sentence",
        any("Northgate" in s for s in sentences),
        f"sentences={sentences}",
    )

    # I.2: extractor returns at most max_sentences.
    short = "Born in A. Born in B. Born in C. Born in D. Born in E."
    sents = _focus_sentences_for(short, ["born in"], max_sentences=3)
    expect(
        "I.2 focus-sentence extractor caps output at max_sentences",
        len(sents) <= 3,
        f"len={len(sents)}",
    )

    # I.3: sentence splitter is non-empty for normal prose.
    expect(
        "I.3 sentence splitter returns >= 2 sentences for compound prose",
        len(_split_sentences("First sentence. Second sentence!")) >= 2,
    )


# ---------------------------------------------------------------------------
# Execute everything
# ---------------------------------------------------------------------------


def main() -> int:
    tests = [
        test_a_writer_university,
        test_b_inventor_country,
        test_c_ceo_university,
        test_d_short_final_answer,
        test_e_person_then_org_no_regression,
        test_state_machine,
        test_pipeline_router,
        test_no_overfitting,
        test_g_answer_granularity,
        test_h_relation_prototype_scoring,
        test_i_focus_sentence_extraction,
    ]
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"  [ERROR] {t.__name__}: {exc}")
            traceback.print_exc()
            _failed.append((t.__name__, str(exc)))

    print(f"\n{SEPARATOR}")
    print(f"  Summary: {len(_passed)} passed, {len(_failed)} failed")
    print(SEPARATOR)
    if _failed:
        for name, detail in _failed:
            print(f"  FAILED: {name} -- {detail}")
        return 1
    print("  All tests PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
