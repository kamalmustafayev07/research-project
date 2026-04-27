"""Entity linking and referential span detection before retrieval.

Referential anchor extraction (the most fragile part of the old regex approach)
is now delegated to a small, focused LLM call.  The LLM understands natural
language and verb tenses universally, so it handles informal phrasings like
"director of Dunkirk studied where?" without a static stop-word list.

The regex implementation is retained as a silent fallback for the rare cases
where the LLM is unavailable or returns malformed output.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

logger = logging.getLogger(__name__)


def fold_for_overlap(s: str) -> str:
    """Lowercase and strip non-alphanumerics for robust substring tests."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _surname_anchors(name: str) -> list[str]:
    """Return last-name and full-name forms for passage overlap checks."""
    toks = re.findall(r"[A-Za-z][A-Za-z\-']+", name or "")
    if not toks:
        return []
    last = toks[-1]
    if len(last) < 3:
        return []
    return list(dict.fromkeys([last, (name or "").strip()]))


def _dedupe_canonical_entities(
    work_edges: list[tuple[str, str]],
    mention_to_canonical: dict[str, str],
    required: list[str],
    relation_hints: list[str],
) -> list[str]:
    """Prefer author/person entities when relation hints indicate a people-first question."""
    has_person_relation = bool(
        set(relation_hints or [])
        & {
            "studied_at",
            "born_in",
            "author_of",
            "directed_by",
            "composed_by",
            "invented_by",
            "founded_by",
            "created_by",
        }
    )
    out: list[str] = []
    if work_edges and has_person_relation:
        for _w, auth in work_edges:
            if auth and auth not in out:
                out.append(auth)
        if out:
            return out
    for _w, auth in work_edges:
        if auth and auth not in out:
            out.append(auth)
    for _k, v in mention_to_canonical.items():
        if v and v not in out:
            out.append(v)
    if not out:
        for r in required:
            if r and r not in out:
                out.append(r)
    return out


# ---------------------------------------------------------------------------
# Referential expression detection — no hard-coded entity KB
# ---------------------------------------------------------------------------

# Role words that signal a referential expression like "the writer of X".
# Ordered longest-first so the regex alternation matches the most specific form.
_REFERENTIAL_ROLES: tuple[str, ...] = (
    "screenwriter",
    "screenwriters",
    "playwright",
    "playwrights",
    "journalist",
    "journalists",
    "illustrator",
    "illustrators",
    "cartoonist",
    "cartoonists",
    "photographer",
    "photographers",
    "choreographer",
    "choreographers",
    "lyricist",
    "lyricists",
    "librettist",
    "biographer",
    "biographers",
    "novelists",
    "novelist",
    "directors",
    "director",
    "producers",
    "producer",
    "composers",
    "composer",
    "inventors",
    "inventor",
    "founders",
    "founder",
    "designers",
    "designer",
    "creators",
    "creator",
    "authors",
    "author",
    "writers",
    "writer",
    "painters",
    "painter",
    "sculptors",
    "sculptor",
    "poets",
    "poet",
)

# Role surface form → canonical relation label used downstream.
_ROLE_TO_RELATION: dict[str, str] = {
    "writer": "author_of",
    "writers": "author_of",
    "author": "author_of",
    "authors": "author_of",
    "novelist": "author_of",
    "novelists": "author_of",
    "playwright": "author_of",
    "playwrights": "author_of",
    "screenwriter": "author_of",
    "screenwriters": "author_of",
    "poet": "author_of",
    "poets": "author_of",
    "lyricist": "author_of",
    "librettist": "author_of",
    "biographer": "author_of",
    "biographers": "author_of",
    "journalist": "author_of",
    "journalists": "author_of",
    "creator": "author_of",
    "creators": "author_of",
    "illustrator": "author_of",
    "illustrators": "author_of",
    "cartoonist": "author_of",
    "cartoonists": "author_of",
    "director": "directed_by",
    "directors": "directed_by",
    "producer": "produced_by",
    "producers": "produced_by",
    "composer": "composed_by",
    "composers": "composed_by",
    "inventor": "invented_by",
    "inventors": "invented_by",
    "founder": "founded_by",
    "founders": "founded_by",
    "designer": "designed_by",
    "designers": "designed_by",
    "painter": "created_by",
    "painters": "created_by",
    "sculptor": "created_by",
    "sculptors": "created_by",
    "photographer": "created_by",
    "photographers": "created_by",
    "choreographer": "created_by",
    "choreographers": "created_by",
}

_REFERENTIAL_ROLE_PATTERN = re.compile(
    r"\b(?:the\s+)?("
    + "|".join(re.escape(r) for r in _REFERENTIAL_ROLES)
    + r")\s+of\s+",
    re.IGNORECASE,
)

# Words that terminate a work title extracted after "of <TITLE> ..."
_TITLE_STOP_WORDS = re.compile(
    r"\s+(?:who|when|where|which|that|was|is|are|did|had|has|have"
    # infinitive + past-tense / past-participle / gerund forms so that
    # questions like "director of Dunkirk studied where?" correctly stop
    # the title at "studied" rather than rolling it into the work name.
    r"|attend|attended|attending"
    r"|live|lived|living"
    r"|born"
    r"|go|went|going"
    r"|graduate|graduated|graduating"
    r"|study|studied|studying"
    r"|work|worked|working"
    r"|get|got|gotten|getting"
    r"|become|became|becoming"
    r"|come|came|coming"
    r"|do|done|doing"
    r"|make|made|making"
    r"|see|saw|seen|seeing"
    r"|know|knew|known|knowing"
    r"|use|used|using"
    r"|find|found|finding"
    r"|give|gave|given|giving"
    r"|tell|told|telling"
    r"|show|showed|shown|showing"
    r"|hear|heard|hearing"
    r"|try|tried|trying"
    r"|ask|asked|asking"
    r"|seem|seemed|seeming"
    r"|feel|felt|feeling"
    r"|leave|left|leaving"
    r"|call|called|calling"
    r"|keep|kept|keeping"
    r"|let|letting"
    r"|begin|began|begun|beginning"
    r"|appear|appeared|appearing"
    r"|bring|brought|bringing"
    r"|speak|spoke|spoken|speaking"
    r"|stand|stood|standing"
    r"|lose|lost|losing"
    r"|pay|paid|paying"
    r"|meet|met|meeting"
    r"|run|ran|running"
    r"|believe|believed|believing"
    r"|hold|held|holding"
    r"|write|wrote|written|writing"
    r"|provide|provided|providing"
    r"|sit|sat|sitting"
    r"|read|reading"
    r"|continue|continued|continuing"
    r"|set|setting"
    r"|learn|learned|learnt|learning"
    r"|change|changed|changing"
    r"|move|moved|moving"
    r"|play|played|playing"
    r"|follow|followed|following"
    r"|stop|stopped|stopping"
    r"|create|created|creating"
    r"|raise|raised|raising"
    r"|pass|passed|passing"
    r"|build|built|building"
    r"|spend|spent|spending"
    r"|cut|cutting"
    r"|kill|killed|killing"
    r"|win|won|winning"
    r"|fall|fell|fallen|falling"
    r"|reach|reached|reaching"
    r"|catch|caught|catching"
    r"|drive|drove|driven|driving"
    r"|serve|served|serving"
    r"|throw|threw|thrown|throwing"
    r"|stay|stayed|staying"
    r"|draw|drew|drawn|drawing"
    r"|open|opened|opening"
    r"|send|sent|sending"
    r"|put|putting"
    r"|remember|remembered|remembering"
    r"|grow|grew|grown|growing"
    r"|add|added|adding"
    r"|look|looked|looking"
    r"|care|cared|caring"
    r"|carry|carried|carrying"
    r"|buy|bought|buying)\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class EntityLinkingResult:
    """Resolved entities and referential anchors for downstream retrieval and reasoning."""

    question: str
    anchor_mentions: list[str]
    mention_to_canonical: dict[str, str]
    work_to_author_edges: list[tuple[str, str]]
    required_surface_forms: list[str]
    match_strings: list[str]
    relation_hints: list[str]
    # Unresolved referential spans: {"role": str, "work": str, "relation": str, "surface": str}
    referential_anchors: list[dict[str, str]]
    # Primary canonical entities for propagation and critic checks.
    canonical_entities: list[str]

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["work_to_author_edges"] = [list(t) for t in self.work_to_author_edges]
        return d


def entity_linking_result_from_dict(payload: dict[str, Any] | None) -> EntityLinkingResult | None:
    if not payload:
        return None
    raw_edges = payload.get("work_to_author_edges") or []
    edges: list[tuple[str, str]] = []
    for e in raw_edges:
        if isinstance(e, (list, tuple)) and len(e) >= 2:
            edges.append((str(e[0]), str(e[1])))
    raw_anchors = payload.get("referential_anchors") or []
    referential_anchors: list[dict[str, str]] = [
        {k: str(v) for k, v in a.items()} for a in raw_anchors if isinstance(a, dict)
    ]
    return EntityLinkingResult(
        question=str(payload.get("question", "")),
        anchor_mentions=list(payload.get("anchor_mentions") or []),
        mention_to_canonical=dict(payload.get("mention_to_canonical") or {}),
        work_to_author_edges=edges,
        required_surface_forms=list(payload.get("required_surface_forms") or []),
        match_strings=list(payload.get("match_strings") or []),
        relation_hints=list(payload.get("relation_hints") or []),
        referential_anchors=referential_anchors,
        canonical_entities=list(payload.get("canonical_entities") or []),
    )


# Matches <answer of hop N> and <hop N> placeholder strings — must never enter match_strings.
_PLACEHOLDER_RE = re.compile(r"<\s*(?:answer\s+of\s+hop|hop)\s*\d+\s*>", re.I)


def build_retrieval_entity_context(
    link: EntityLinkingResult,
    sub_question_entities: list[dict[str, Any]],
    relation_sequence: list[str],
) -> dict[str, Any]:
    """Merge linker output with decomposer plans for retrieval + reranking.

    Filters out <answer of hop N> placeholder strings that the LLM may emit in
    entity_map keys so they never become literal retrieval queries or
    entity-overlap match strings.

    NOTE: enriched authorship phrases ("wrote X", "author of X") are generated
    as retrieval *query* variants in build_retrieval_query_pack, NOT here.
    match_strings are only used for entity-overlap re-ranking, so they must
    remain precise; flooding them with broad phrases degrades ranking quality.
    """
    match_strings: list[str] = list(link.match_strings)

    for row in sub_question_entities:
        for e in row.get("resolved_entities") or []:
            s = str(e).strip()
            # Skip LLM-generated placeholder strings
            if s and not _PLACEHOLDER_RE.search(s):
                match_strings.append(s)
        em = row.get("entity_map") or {}
        for k, v in em.items():
            # Filter out placeholder keys/values before adding to match_strings
            if not _PLACEHOLDER_RE.search(str(k)):
                match_strings.append(str(k))
            if not _PLACEHOLDER_RE.search(str(v)):
                match_strings.append(str(v))

    rel_hints = list(link.relation_hints)
    for rel in relation_sequence:
        if isinstance(rel, str) and rel and rel not in rel_hints:
            rel_hints.append(rel)

    seen: set[str] = set()
    deduped: list[str] = []
    for m in match_strings:
        key = m.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(m.strip())

    from src.config import SETTINGS

    return {
        "match_strings": deduped,
        "relation_hints": rel_hints,
        "mention_to_canonical": dict(link.mention_to_canonical),
        "work_to_author_edges": list(link.work_to_author_edges),
        "sub_question_entities": sub_question_entities,
        "canonical_entities": list(link.canonical_entities),
        "referential_anchors": list(link.referential_anchors),
        "semantic_blend": float(getattr(SETTINGS.retrieval, "retrieval_semantic_blend", 0.08)),
    }


class EntityLinker:
    """Entity linker with LLM-based referential expression detection.

    Referential anchor extraction is performed by a focused LLM prompt that
    understands natural language universally (handles informal phrasing, any
    verb tense, multi-word titles, etc.).  The legacy regex path is kept as a
    silent fallback for cases where the LLM call fails.
    """

    def __init__(self) -> None:
        # Lazy import to avoid circular dependency at module load time.
        from src.utils.llm import LLMClient
        from src.agents.prompts import ANCHOR_EXTRACTION_PROMPT

        self._llm = LLMClient()
        self._anchor_prompt_template = ANCHOR_EXTRACTION_PROMPT
        self._work_verb_pattern = re.compile(
            r"""\b(?:created|wrote|penned|composed|directed|founded|invented)\s+['"]?([^'"\n?]{2,64})['"]?""",
            re.I,
        )
        self._quoted = re.compile(r"""['"]([^'"\n]{2,64})['"]""")

    # ------------------------------------------------------------------
    # Referential anchor extraction — LLM primary, regex fallback
    # ------------------------------------------------------------------

    def _extract_referential_anchors(self, question: str) -> list[dict[str, str]]:
        """Detect '[role] of [WORK]' patterns using an LLM call.

        The LLM understands natural-language phrasing universally (any verb tense,
        informal word order, multi-word titles) without needing a static stop-word
        list.  If the LLM is unreachable or returns invalid output the call silently
        falls back to the regex implementation.

        Each returned record: {"role": str, "work": str, "relation": str, "surface": str}
        """
        try:
            return self._extract_anchors_llm(question)
        except Exception as exc:
            logger.debug("LLM anchor extraction failed (%s); using regex fallback.", exc)
            return self._extract_anchors_regex(question)

    def _extract_anchors_llm(self, question: str) -> list[dict[str, str]]:
        """Ask the LLM to extract referential anchors from the question."""
        prompt = self._anchor_prompt_template.replace("{question}", question)
        response = self._llm.generate(prompt, max_new_tokens=256)
        parsed = self._llm.extract_json(response.text)

        raw_anchors = parsed.get("anchors") if isinstance(parsed, dict) else None
        if not isinstance(raw_anchors, list):
            raise ValueError(f"LLM returned unexpected structure: {parsed!r}")

        valid: list[dict[str, str]] = []
        seen_works: set[str] = set()
        for item in raw_anchors:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            work = str(item.get("work") or "").strip()
            relation = str(item.get("relation") or "related_to").strip()
            surface = str(item.get("surface") or "").strip()
            if not role or not work or len(work) < 2:
                continue
            work_key = fold_for_overlap(work)
            if work_key in seen_works:
                continue
            seen_works.add(work_key)
            valid.append({"role": role, "work": work, "relation": relation, "surface": surface})
        return valid

    def _extract_anchors_regex(self, question: str) -> list[dict[str, str]]:
        """Regex fallback: detect '[role] of [WORK]' via pattern matching."""
        anchors: list[dict[str, str]] = []
        seen_works: set[str] = set()
        for m in _REFERENTIAL_ROLE_PATTERN.finditer(question):
            role = m.group(1).lower()
            tail = question[m.end():]
            work = self._extract_title_from_tail(tail)
            if not work or len(work) < 2:
                continue
            work_key = fold_for_overlap(work)
            if work_key in seen_works:
                continue
            seen_works.add(work_key)
            relation = _ROLE_TO_RELATION.get(role, "related_to")
            anchors.append(
                {
                    "role": role,
                    "work": work,
                    "relation": relation,
                    "surface": (m.group(0) + work).strip(),
                }
            )
        return anchors

    @staticmethod
    def _extract_title_from_tail(tail: str) -> str:
        """Regex fallback: extract work title from text following 'of '."""
        stripped = tail.lstrip()
        cap_match = re.match(
            r"([A-Z][^\s,?!.]*(?:\s+[A-Z][^\s,?!.]*){0,5})",
            stripped,
        )
        if cap_match:
            title = cap_match.group(1).strip()
        else:
            gen_match = re.match(r"([^,?!.\n]{2,60}?)(?=[,?!.]|$)", stripped)
            if gen_match:
                title = gen_match.group(1).strip()
            else:
                return ""
        stop_match = _TITLE_STOP_WORDS.search(title)
        if stop_match:
            title = title[: stop_match.start()].strip()
        return title.rstrip(" .,;:")

    # ------------------------------------------------------------------
    # Named entity surface-form extraction
    # ------------------------------------------------------------------

    def _collect_explicit_work_anchors(self, question: str) -> list[str]:
        """Collect work/entity titles from verb patterns and quoted text."""
        found: list[str] = []
        for m in self._work_verb_pattern.finditer(question):
            title = m.group(1).strip(" .,;")
            if len(title) > 2:
                found.append(title)
        for m in self._quoted.finditer(question):
            t = m.group(1).strip()
            if len(t) > 2:
                found.append(t)
        return list(dict.fromkeys(found))

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def link(self, question: str) -> EntityLinkingResult:
        """Analyse the question and return an EntityLinkingResult.

        No entity is resolved to a canonical name via a static KB.  Instead:
        - Referential expressions ("the writer of X") are detected and stored in
          referential_anchors so that the decomposer produces a first-hop
          sub-question and the retriever generates work→creator queries.
        - Named proper-noun mentions are collected as anchor_mentions and
          match_strings for entity-overlap scoring in the retriever.
        - Relation hints are derived from question vocabulary for query expansion.
        """
        q = (question or "").strip()
        mention_to_canonical: dict[str, str] = {}
        anchors: list[str] = []
        required: list[str] = []

        # 1. Detect referential anchors ("the writer of X", "the creator of Y", …)
        referential_anchors = self._extract_referential_anchors(q)

        # Work titles from referential anchors become primary pivot anchors.
        for ra in referential_anchors:
            work = ra["work"]
            if work and work not in anchors:
                anchors.append(work)
            if work and work not in required:
                required.append(work)
            mention_to_canonical.setdefault(work, work)

        # 2. Collect proper-noun mentions from the question (NE-like heuristic).
        wh_stop = frozenset({"what", "who", "where", "when", "which", "how", "why"})
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", q):
            span = m.group(1).strip()
            if span.split()[0].lower() in wh_stop:
                continue
            if span and span not in anchors and len(span) > 2:
                anchors.append(span)

        # 3. Collect work titles from explicit verb/quote patterns.
        for w in self._collect_explicit_work_anchors(q):
            wkey = w.strip()
            if wkey not in anchors:
                anchors.append(wkey)
            if wkey not in required:
                required.append(wkey)

        if not required and anchors:
            required = list(anchors[:4])

        # 4. Build match_strings (used for entity-overlap scoring in the retriever).
        base_strings: list[str] = []
        for x in [*required, *mention_to_canonical.keys(), *mention_to_canonical.values()]:
            s = str(x).strip()
            if s and s not in base_strings:
                base_strings.append(s)
        for ra in referential_anchors:
            if ra.get("work") and ra["work"] not in base_strings:
                base_strings.append(ra["work"])
        match_strings = list(dict.fromkeys(base_strings))

        # 5. Derive relation hints from question vocabulary.
        relation_hints: list[str] = []
        low = q.lower()
        if any(x in low for x in ("university", "college", "educated", "attend", "studied", "school")):
            relation_hints.append("studied_at")
        if "born" in low or "birthplace" in low or "birth" in low:
            relation_hints.append("born_in")
        if "director" in low or "directed" in low:
            relation_hints.append("directed_by")
        if any(x in low for x in ("author", "writer", "wrote", "written", "penned", "creator", "created")):
            relation_hints.append("author_of")
        if "capital" in low:
            relation_hints.append("capital_of")
        if "country" in low or "nation" in low:
            relation_hints.append("located_in")
        if "founded" in low or "founder" in low:
            relation_hints.append("founded_by")
        if "compos" in low or "composer" in low:
            relation_hints.append("composed_by")
        if "invent" in low or "inventor" in low:
            relation_hints.append("invented_by")
        # Relations declared by referential anchors take precedence (deduplicated).
        for ra in referential_anchors:
            rel = ra.get("relation", "")
            if rel and rel not in relation_hints:
                relation_hints.append(rel)

        return EntityLinkingResult(
            question=q,
            anchor_mentions=anchors,
            mention_to_canonical=mention_to_canonical,
            work_to_author_edges=[],  # Resolution via retrieval, not static KB
            required_surface_forms=required,
            match_strings=match_strings,
            relation_hints=list(dict.fromkeys(relation_hints)),
            referential_anchors=referential_anchors,
            canonical_entities=_dedupe_canonical_entities(
                [], mention_to_canonical, required, relation_hints
            ),
        )
