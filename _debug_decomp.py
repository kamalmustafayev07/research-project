"""Debug the raw decomposer LLM output. Writes to _debug_output.txt (UTF-8)."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from src.agents.entity_linker import EntityLinker
from src.agents.prompts import DECOMPOSER_PROMPT_TEMPLATE
from src.utils.llm import LLMClient

Q = "Which university did the writer of The Hobbit attend?"

linker = EntityLinker()
link = linker.link(Q)

ref_anchors = link.referential_anchors
hints = (
    f"anchors={link.anchor_mentions}; "
    f"canonical_map={link.mention_to_canonical}; "
    f"required_spans={link.required_surface_forms}; "
    f"referential_anchors={ref_anchors}"
)
prompt = DECOMPOSER_PROMPT_TEMPLATE.format(question=Q, entity_hints=hints)

OUT = open("_debug_output.txt", "w", encoding="utf-8")

OUT.write("=== PROMPT SENT TO LLM ===\n")
OUT.write(prompt + "\n\n")

llm = LLMClient()
resp = llm.generate(prompt)

OUT.write("=== RAW LLM RESPONSE ===\n")
OUT.write(resp.text + "\n\n")

parsed = llm.extract_json(resp.text)
OUT.write("=== PARSED JSON ===\n")
OUT.write(json.dumps(parsed, indent=2) + "\n")
OUT.close()

# Also print summary to console (ASCII-safe)
print("sub_questions:", parsed.get("sub_questions"))
print("relation_sequence:", parsed.get("relation_sequence"))
print("Full output written to _debug_output.txt")
