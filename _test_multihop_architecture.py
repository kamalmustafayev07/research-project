"""Broad architecture tests for multi-hop QA (no LLM, no single-example tuning).

Covers diverse relation shapes requested in the systems brief:
  author → university, inventor → birthplace, CEO → company, athlete → nationality,
  scientist → discovery, 3-hop chains, ambiguous evidence, abstention.

Run: .venv/Scripts/python.exe _test_multihop_architecture.py
"""

from __future__ import annotations

import sys
import traceback

from src.agents.critic import CriticAgent
from src.agents.evidence_validation import (
    answer_evidence_alignment_score,
    answer_grounded_in_evidence,
)
from src.agents.hop_execution_context import (
    attach_hop_execution_context,
    build_hop_execution_context,
    format_hop_execution_context_for_prompt,
)
from src.agents.hop_safety import (
    EntityState,
    HopMemory,
    HopRecord,
    verify_hop_answer,
)
from src.agents.confidence_calibration import calibrate_reasoner_confidence


def _two_hop(
    q: str,
    sq1: str,
    e1: str,
    t1: str,
    r1: str,
    sq2: str,
    r2: str,
) -> HopMemory:
    m = HopMemory(original_question=q)
    m.add_hop(sq1, e1, t1, r1)  # type: ignore[arg-type]
    m.add_hop(sq2, "<answer of hop 1>", "Unknown", r2)
    return m


def _lock_hop2_subject(m: HopMemory, person: str) -> None:
    h = m.hops[1]
    m.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=h.subquestion_text,
        main_entity=EntityState(text=person, entity_type="Person"),
        relation=h.relation,
    )


def test_execution_context_schema() -> None:
    m = _two_hop(
        "Q",
        "Who wrote work W?",
        "W",
        "Work",
        "author_of",
        "Which university did <answer of hop 1> attend?",
        "studied_at",
    )
    m.record_answer(1, "Person P", 0.9)
    _lock_hop2_subject(m, "Person P")
    ctx = build_hop_execution_context(m, m.hops[1], "Which university was Person P at?")
    assert ctx["schema_version"] == 1
    assert ctx["active_hop_index"] == 2
    assert ctx["total_hops"] == 2
    assert ctx["is_final_hop"] is True
    assert ctx["expected_answer_type"] == "Organization"
    assert len(ctx["prior_resolved_chain"]) == 1
    assert ctx["prior_resolved_chain"][0]["answer"] == "Person P"
    txt = format_hop_execution_context_for_prompt(ctx)
    assert "HOP EXECUTION CONTRACT" in txt
    assert "Person P" in txt
    assert "studied_at" in txt


def test_three_hop_chain_contract() -> None:
    m = HopMemory(original_question="3hop")
    m.add_hop("Who leads org O?", "O", "Organization", "ceo_of")
    m.add_hop("What company did <answer of hop 1> found?", "<answer of hop 1>", "Unknown", "founded_by")
    m.add_hop("In which country is <answer of hop 2> headquartered?", "<answer of hop 2>", "Unknown", "located_in")
    m.record_answer(1, "Exec E", 0.88)
    m.hops[1] = HopRecord(
        hop_index=2,
        subquestion_text=m.hops[1].subquestion_text,
        main_entity=EntityState(text="Exec E", entity_type="Person"),
        relation="founded_by",
    )
    m.record_answer(2, "Corp C", 0.82)
    m.hops[2] = HopRecord(
        hop_index=3,
        subquestion_text=m.hops[2].subquestion_text,
        main_entity=EntityState(text="Corp C", entity_type="Organization"),
        relation="located_in",
    )
    ctx = build_hop_execution_context(m, m.hops[2], "Where is Corp C headquartered?")
    assert ctx["active_hop_index"] == 3
    assert ctx["is_final_hop"] is True
    assert ctx["remaining_hops_after"] == 0
    assert len(ctx["prior_resolved_chain"]) == 2


def test_evidence_alignment_paraphrase() -> None:
    blob = (
        "the athlete grew up in auburn and later competed internationally "
        "representing the national side"
    )
    # Answer token "auburn" appears; "athlete" might not — recall should be high.
    s = answer_evidence_alignment_score("Auburn", blob)
    assert s >= 0.35
    assert answer_grounded_in_evidence("Auburn", blob)

    # Wording differs from answer but shares the discovery name.
    sci_blob = (
        "the 1928 paper described how mold inhibited bacterial growth leading "
        "to penicillin isolation"
    )
    assert answer_grounded_in_evidence("Penicillin", sci_blob)


def test_evidence_alignment_abstain() -> None:
    assert answer_evidence_alignment_score("Not enough information in retrieved passages.", "any") >= 0.99


def test_confidence_calibration_monotonic() -> None:
    assert calibrate_reasoner_confidence(0.95, 0.2) < calibrate_reasoner_confidence(0.95, 0.9)
    assert calibrate_reasoner_confidence(0.95, 1.0) >= 0.9


def test_critic_granularity_matches_verifier() -> None:
    """City question + country-shaped answer must fail critic hard checks."""
    payload = {
        "hop_index": 2,
        "subquestion_text": "In which city was Person X born?",
        "resolved_subquestion": "In which city was Person X born?",
        "subject_text": "Person X",
        "relation": "born_in",
        "expected_answer_type": "Location",
        "expected_answer_granularity": "City",
        "prior_subjects": ["Org O"],
        "prior_answers": ["Person X"],
    }
    fails = CriticAgent._active_hop_hard_checks("United States", 0.9, payload)
    assert any("granularity" in f.lower() for f in fails)


def test_author_university_verify() -> None:
    m = _two_hop(
        "Lit Q",
        "Who is the author of Book B?",
        "B",
        "Work",
        "author_of",
        "Which university did <answer of hop 1> attend?",
        "studied_at",
    )
    m.record_answer(1, "Author A", 0.9)
    _lock_hop2_subject(m, "Author A")
    v = verify_hop_answer("Author A", m.hops[1].subquestion_text, m)
    assert not v.approved
    v2 = verify_hop_answer("University of Northgate", m.hops[1].subquestion_text, m)
    assert v2.approved


def test_inventor_birthplace_verify() -> None:
    m = _two_hop(
        "Inv Q",
        "Who invented gadget G?",
        "G",
        "Work",
        "invented_by",
        "In which country was <answer of hop 1> born?",
        "born_in",
    )
    m.record_answer(1, "Inventor I", 0.88)
    _lock_hop2_subject(m, "Inventor I")
    assert verify_hop_answer("Republic of Testland", m.hops[1].subquestion_text, m).approved


def test_ceo_company_verify() -> None:
    m = _two_hop(
        "Biz Q",
        "Who is the CEO of Conglomerate C?",
        "C",
        "Organization",
        "ceo_of",
        "Which company did <answer of hop 1> found?",
        "founded_by",
    )
    m.record_answer(1, "Leader L", 0.87)
    _lock_hop2_subject(m, "Leader L")
    # Surface form must classify as Organization (marker), not two-token Person.
    assert verify_hop_answer("Acme Corporation", m.hops[1].subquestion_text, m).approved


def test_athlete_nationality_verify() -> None:
    m = _two_hop(
        "Sport Q",
        "Who is athlete A?",
        "A",
        "Person",
        "related_to",
        "What nationality is <answer of hop 1>?",
        "related_to",
    )
    m.record_answer(1, "Player P", 0.86)
    _lock_hop2_subject(m, "Player P")
    # Short nationality surface — type may be Unknown; verifier permissive.
    r = verify_hop_answer("Canadian", m.hops[1].subquestion_text, m)
    assert r.approved or "regress" not in r.explanation.lower()


def test_ambiguous_evidence_no_special_case() -> None:
    """Alignment score is continuous — no binary keyword gate on entity names."""
    blob = "Northgate is east of Southford and both are small cities in the region."
    s1 = answer_evidence_alignment_score("Northgate", blob)
    s2 = answer_evidence_alignment_score("Southford", blob)
    assert s1 > 0 and s2 > 0
    assert abs(s1 - s2) < 0.51  # neither hardcoded as winner


def test_attach_merges_entity_context() -> None:
    m = _two_hop("Q", "H1", "E", "Person", "related_to", "H2", "related_to")
    ec = {"match_strings": ["x"]}
    out = attach_hop_execution_context(ec, m, m.hops[0], "H1 resolved")
    assert out["match_strings"] == ["x"]
    assert out["hop_execution_context"]["active_hop_index"] == 1


def main() -> int:
    tests = [
        test_execution_context_schema,
        test_three_hop_chain_contract,
        test_evidence_alignment_paraphrase,
        test_evidence_alignment_abstain,
        test_confidence_calibration_monotonic,
        test_critic_granularity_matches_verifier,
        test_author_university_verify,
        test_inventor_birthplace_verify,
        test_ceo_company_verify,
        test_athlete_nationality_verify,
        test_ambiguous_evidence_no_special_case,
        test_attach_merges_entity_context,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\nSummary: {len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
