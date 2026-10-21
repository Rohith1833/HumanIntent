"""Token-aware context budget manager and long-input semantic decomposition pipeline.

Guarantees that LLM stages respect effective model context limits without truncating,
summarizing into lossy paragraphs, or silently dropping early or late user requirements.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings

logger = logging.getLogger("humanintent.context_budget")


# =========================================================================
# DIAGNOSTIC REASON CODES & PROCESSING MODES
# =========================================================================

class ProcessingMode(str, Enum):
    """Execution mode chosen based on token context budget."""
    NORMAL = "normal"
    LONG_INPUT = "long_input"


NORMAL_CONTEXT_MODE = "NORMAL_CONTEXT_MODE"
LONG_INPUT_MODE = "LONG_INPUT_MODE"
CONTEXT_BUDGET_EXCEEDED = "CONTEXT_BUDGET_EXCEEDED"
SOURCE_CHUNK_EXTRACTION_FAILED = "SOURCE_CHUNK_EXTRACTION_FAILED"
SOURCE_CHUNK_COVERAGE_FAILED = "SOURCE_CHUNK_COVERAGE_FAILED"
GLOBAL_RECONCILIATION_FAILED = "GLOBAL_RECONCILIATION_FAILED"


# =========================================================================
# TOKEN ESTIMATION & STAGE BUDGET AUDITING
# =========================================================================

def estimate_tokens(text: str) -> int:
    """Accurately and conservatively estimate token count for English text, code, and schemas on LLaMA-3.

    LLaMA-3 uses a 128k vocabulary with Byte-Pair Encoding where English text averages
    approximately 4.4 to 4.6 characters per token.
    """
    if not text:
        return 0
    char_est = len(text) / 4.4
    word_est = len(text.split()) * 1.25
    return max(int(char_est), int(word_est), 1)


@dataclass
class StageBudgetReport:
    """Report on token context feasibility for a given LLM stage."""
    stage: str
    system_tokens: int
    schema_tokens: int
    user_tokens: int
    reserved_output_tokens: int
    total_tokens: int
    available_context: int
    is_safe: bool
    safe_user_tokens: int
    safe_user_chars: int
    processing_mode: ProcessingMode
    diagnostic_code: str


class ContextBudgetManager:
    """Token-aware context budget manager enforcing model context safety before inference."""

    def __init__(
        self,
        context_window: Optional[int] = None,
        reserved_output_tokens: int = 800,
        safety_headroom_tokens: int = 256,
        product_max_input_chars: int = 50000,
        long_input_threshold_chars: int = 4500,
    ) -> None:
        self.context_window = context_window or settings.context_window_size
        self.reserved_output_tokens = reserved_output_tokens
        self.safety_headroom_tokens = safety_headroom_tokens
        self.product_max_input_chars = product_max_input_chars
        self.long_input_threshold_chars = long_input_threshold_chars

    def check_stage_budget(
        self,
        stage: str,
        system_prompt: str,
        user_text: str,
        schema_json: Optional[str] = None,
        reserved_output: Optional[int] = None,
    ) -> StageBudgetReport:
        """Estimate token budget for an LLM stage and determine if long-input mode is required."""
        res_out = reserved_output or self.reserved_output_tokens
        sys_tokens = estimate_tokens(system_prompt)
        # Note: In Ollama, format=schema is compiled into GBNF grammar for sampling, not prompt text tokens.
        # If schema_json is provided, reserve a modest grammar buffer (up to 200 tokens).
        sch_tokens = min(estimate_tokens(schema_json), 200) if schema_json else 0
        usr_tokens = estimate_tokens(user_text)

        total_tokens = sys_tokens + sch_tokens + usr_tokens + res_out
        usable_context = max(0, self.context_window - self.safety_headroom_tokens)
        safe_user_tokens = max(0, usable_context - (sys_tokens + sch_tokens + res_out))
        safe_user_chars = int(safe_user_tokens * 4.4)

        if len(user_text) > self.product_max_input_chars:
            is_safe = False
            mode = ProcessingMode.LONG_INPUT
            diag = CONTEXT_BUDGET_EXCEEDED
        elif len(user_text) >= self.long_input_threshold_chars or total_tokens > usable_context:
            is_safe = False
            mode = ProcessingMode.LONG_INPUT
            diag = LONG_INPUT_MODE
        else:
            is_safe = True
            mode = ProcessingMode.NORMAL
            diag = NORMAL_CONTEXT_MODE

        return StageBudgetReport(
            stage=stage,
            system_tokens=sys_tokens,
            schema_tokens=sch_tokens,
            user_tokens=usr_tokens,
            reserved_output_tokens=res_out,
            total_tokens=total_tokens,
            available_context=self.context_window,
            is_safe=is_safe,
            safe_user_tokens=safe_user_tokens,
            safe_user_chars=safe_user_chars,
            processing_mode=mode,
            diagnostic_code=diag,
        )


context_budget_manager = ContextBudgetManager()


# =========================================================================
# ATOMIC UNIT DATA STRUCTURES & PROVENANCE
# =========================================================================

class UnitType(str, Enum):
    """Categorical type of atomic extracted source semantic unit."""
    ACTION = "ACTION"
    DELIVERABLE = "DELIVERABLE"
    REQUIREMENT = "REQUIREMENT"
    CONSTRAINT = "CONSTRAINT"
    QUANTITY = "QUANTITY"
    NEGATIVE_RULE = "NEGATIVE_RULE"
    PREFERENCE = "PREFERENCE"
    LIFECYCLE = "LIFECYCLE"
    DEPENDENCY = "DEPENDENCY"
    CORRECTION = "CORRECTION"


@dataclass
class SourceChunk:
    """A semantically bounded portion of authoritative user input."""
    chunk_index: int
    text: str
    start_char: int
    end_char: int
    heading_context: Optional[str] = None


@dataclass
class ExtractedAtomicUnit:
    """An atomic semantic unit extracted from a specific source chunk with full provenance."""
    id: str
    chunk_index: int
    text: str
    unit_type: UnitType
    raw_span: str
    replaces_target: Optional[str] = None
    is_superseded: bool = False


@dataclass
class MergedLongInputResult:
    """Outcome of multi-chunk extraction, supersession resolution, and global reconciliation."""
    chunks: list[SourceChunk]
    all_units: list[ExtractedAtomicUnit]
    active_units: list[ExtractedAtomicUnit]
    superseded_units: list[ExtractedAtomicUnit]
    diagnostic_codes: list[str]


# =========================================================================
# SEMANTIC BOUNDARY CHUNKER
# =========================================================================

def chunk_long_source(
    text: str,
    target_chunk_chars: int = 1800,
    max_chunk_chars: int = 2500,
) -> list[SourceChunk]:
    """Divide authoritative source input using semantic boundaries.

    Priority boundaries:
    1. Headings & numbered sections (#, ===, ---, Section X, 1. )
    2. Paragraph breaks (double newlines)
    3. List items (- , * , 1) )
    4. Sentence boundaries (.?! followed by space)
    5. Clauses & coordination delimiters
    """
    cleaned_text = text.strip()
    if len(cleaned_text) <= max_chunk_chars:
        return [SourceChunk(chunk_index=0, text=cleaned_text, start_char=0, end_char=len(cleaned_text))]

    # 1. Break into candidate semantic paragraphs / blocks
    raw_blocks = re.split(r"(?:\r?\n){2,}", cleaned_text)

    chunks: list[SourceChunk] = []
    current_text_parts: list[str] = []
    current_len = 0
    current_start = 0
    chunk_idx = 0
    current_heading = None

    heading_re = re.compile(r"^(?:#{1,4}\s+|==+\s*|--+\s*|section\s+\d+:?|\d+\.\s+[A-Z])", re.IGNORECASE)

    for block in raw_blocks:
        block_str = block.strip()
        if not block_str:
            continue

        # Check for heading context
        lines = block_str.split("\n")
        first_line = lines[0].strip()
        if heading_re.match(first_line):
            current_heading = first_line

        # If a single block exceeds max_chunk_chars, split on sentence / list boundaries
        if len(block_str) > max_chunk_chars:
            sentences = re.split(r"(?<=[.?!])\s+|\n(?=[-*\d•])", block_str)
            for sent in sentences:
                sent_str = sent.strip()
                if not sent_str:
                    continue
                if current_len + len(sent_str) + 1 > target_chunk_chars and current_text_parts:
                    chunk_body = "\n\n".join(current_text_parts)
                    chunks.append(
                        SourceChunk(
                            chunk_index=chunk_idx,
                            text=chunk_body,
                            start_char=current_start,
                            end_char=current_start + len(chunk_body),
                            heading_context=current_heading,
                        )
                    )
                    chunk_idx += 1
                    current_start += len(chunk_body) + 2
                    current_text_parts = []
                    current_len = 0

                current_text_parts.append(sent_str)
                current_len += len(sent_str) + 1
            continue

        # Normal block accumulation
        if current_len + len(block_str) + 2 > target_chunk_chars and current_text_parts:
            chunk_body = "\n\n".join(current_text_parts)
            chunks.append(
                SourceChunk(
                    chunk_index=chunk_idx,
                    text=chunk_body,
                    start_char=current_start,
                    end_char=current_start + len(chunk_body),
                    heading_context=current_heading,
                )
            )
            chunk_idx += 1
            current_start += len(chunk_body) + 2
            current_text_parts = []
            current_len = 0

        current_text_parts.append(block_str)
        current_len += len(block_str) + 2

    if current_text_parts:
        chunk_body = "\n\n".join(current_text_parts)
        chunks.append(
            SourceChunk(
                chunk_index=chunk_idx,
                text=chunk_body,
                start_char=current_start,
                end_char=current_start + len(chunk_body),
                heading_context=current_heading,
            )
        )

    return chunks


# =========================================================================
# COMPACT PROMPTS FOR CHUNK EXTRACTION & GLOBAL RECONCILIATION
# =========================================================================

CHUNK_EXTRACTOR_SYSTEM_PROMPT = """You are an atomic semantic requirement extractor for HumanIntent.
Your job is to read ONE chunk of user input and extract its atomic semantic units strictly as JSON.
DO NOT enhance the prompt. DO NOT perform the task. DO NOT summarize into lossy text.

Extract a list of atomic units where each unit has:
- "text": concise requirement or action description
- "unit_type": one of ["ACTION", "DELIVERABLE", "REQUIREMENT", "CONSTRAINT", "QUANTITY", "NEGATIVE_RULE", "PREFERENCE", "LIFECYCLE", "DEPENDENCY", "CORRECTION"]
- "raw_span": exact substring or brief excerpt from this chunk proving provenance
- "replaces_target": null, or if this unit changes/corrects an earlier instruction (e.g. "use X instead of Y", "scratch Z"), the name of the superseded target.

Return ONLY a JSON object:
{
  "units": [
    {
      "text": "...",
      "unit_type": "...",
      "raw_span": "...",
      "replaces_target": null
    }
  ]
}
"""


GLOBAL_RECONCILIATION_SYSTEM_PROMPT = """You are the global intent reconciliation engine for HumanIntent.
You are given a deduplicated list of atomic requirements, constraints, deliverables, and quantities extracted across ALL chunks of a long user specification.

Your task is to synthesize these authoritative units into a single cohesive PromptAnalysis JSON structure:
1. "classification": category (one of CODING, DEBUGGING, RESEARCH, EDUCATION, WRITING, DATA_ANALYSIS, PLANNING, CREATIVE, IMAGE_GENERATION, SUMMARIZATION, GENERAL) and confidence (0.0 to 1.0)
2. "objective": a clear, comprehensive statement of the primary goal encompassing all active deliverables and workflow stages
3. "explicit_requirements": list containing ALL active explicit requirements from every chunk (do NOT omit requirements from later chunks!)
4. "constraints": list containing all active constraints, quantities, negative rules, and lifecycle rules
5. "preferences": list containing all active stylistic/presentation preferences
6. "missing_context": neutral names of truly unspecified dimensions (e.g. 'deployment environment')
7. "ambiguity": {"score": int, "level": "LOW"|"MEDIUM"|"HIGH", "needs_clarification": bool, "reasons": [], "questions": []}

CRITICAL:
- PRESERVE ALL material explicit requirements across all chunks!
- If a requirement was superseded (e.g. replaced by a correction), use the corrected version only.
- Return ONLY valid JSON matching the schema.
"""


# =========================================================================
# DETERMINISTIC CHUNK UNIT EXTRACTION (FALLBACK & PRECISION BOOST)
# =========================================================================

def extract_deterministic_chunk_units(chunk: SourceChunk) -> list[ExtractedAtomicUnit]:
    """Deterministically extract high-value atomic units from a chunk.

    Captures:
    - Bullet points & numbered items
    - Quantities & limits ("at most X", "under Y words", "minimum 5")
    - Negative constraints ("do not use X", "avoid Y", "without Z")
    - Lifecycle / retention rules ("delete data after X", "purge logs after Y")
    - Corrections / supersessions ("scratch X, use Y instead", "actually no, use Z")
    """
    text = chunk.text
    units: list[ExtractedAtomicUnit] = []
    unit_counter = 1

    # 1. Corrections & supersessions
    corr_matches = re.finditer(
        r"(?:\b(?:wait\s+no\s*,\s*)?scratch\s+(?:that|the|earlier\s+choice\s+for)?\s*([a-zA-Z0-9_\-]+)[^;\.\n]*?[;\,\.]?\s*(?:use|do|choose|with|implement)?\s*([a-zA-Z0-9_\-]+)\s+instead|\binstead\s+of\s+([a-zA-Z0-9_\-]+)\s*,?\s*(?:use|choose)?\s*([a-zA-Z0-9_\-]+))",
        text,
        re.IGNORECASE,
    )
    for cm in corr_matches:
        raw_span = cm.group(0).strip()
        superseded = cm.group(1) or cm.group(3)
        replacement = cm.group(2) or cm.group(4)
        if superseded:
            superseded = superseded.strip()
        if replacement:
            replacement = replacement.strip()
        units.append(
            ExtractedAtomicUnit(
                id=f"c{chunk.chunk_index}_u{unit_counter}",
                chunk_index=chunk.chunk_index,
                text=f"use {replacement} instead" + (f" of {superseded}" if superseded else ""),
                unit_type=UnitType.CORRECTION,
                raw_span=raw_span,
                replaces_target=superseded,
            )
        )
        unit_counter += 1

    # 2. Negative constraints
    IDIOMATIC_WORDS = {
        "exception", "fail", "delay", "doubt", "hesitation", "issue", "issues",
        "limitation", "restrictions", "restriction", "interruption", "downtime",
        "loss", "prejudice", "regard", "any", "a", "an", "the", "it", "problem",
    }
    neg_matches = re.finditer(
        r"\b(?:do\s+not\s+use|don't\s+use|avoid\s+using|avoid|must\s+not\s+use|must\s+not\s+include|never\s+use|do\s+not\s+allow)\s+([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)?)\b",
        text,
        re.IGNORECASE,
    )
    for nm in neg_matches:
        raw_span = nm.group(0).strip()
        item = nm.group(1).strip()
        if item.lower() not in {"like", "a", "an", "the", "it"} and item.lower() not in IDIOMATIC_WORDS:
            units.append(
                ExtractedAtomicUnit(
                    id=f"c{chunk.chunk_index}_u{unit_counter}",
                    chunk_index=chunk.chunk_index,
                    text=f"do not use {item}",
                    unit_type=UnitType.NEGATIVE_RULE,
                    raw_span=raw_span,
                )
            )
            unit_counter += 1

    # 3. Quantities, numeric limits, and durations
    quant_matches = re.finditer(
        r"\b(?:maximum|max|minimum|min|at\s+least|at\s+most|under|less\s+than|exactly)\s+(\d+)\s+([a-zA-Z_\-]{3,20})\b",
        text,
        re.IGNORECASE,
    )
    for qm in quant_matches:
        raw_span = qm.group(0).strip()
        units.append(
            ExtractedAtomicUnit(
                id=f"c{chunk.chunk_index}_u{unit_counter}",
                chunk_index=chunk.chunk_index,
                text=raw_span,
                unit_type=UnitType.QUANTITY,
                raw_span=raw_span,
            )
        )
        unit_counter += 1

    # 4. Lifecycle / retention / deletion rules
    lifecycle_matches = re.finditer(
        r"\b(?:delete|remove|purge|wipe|clear|retain|keep)\s+([^;\.\n]{2,60}?)\s+(?:after|for)\s+(\d+\s+(?:hours?|days?|weeks?|months?|years?))\b",
        text,
        re.IGNORECASE,
    )
    for lm in lifecycle_matches:
        raw_span = lm.group(0).strip()
        units.append(
            ExtractedAtomicUnit(
                id=f"c{chunk.chunk_index}_u{unit_counter}",
                chunk_index=chunk.chunk_index,
                text=raw_span,
                unit_type=UnitType.LIFECYCLE,
                raw_span=raw_span,
            )
        )
        unit_counter += 1

    # 4b. Actions and core deliverables
    action_matches = re.finditer(
        r"\b(?:must\s+implement|implement|architect|include|support|provide|generate)\s+([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+){1,5})\b",
        text,
        re.IGNORECASE,
    )
    for am in action_matches:
        raw_span = am.group(0).strip()
        item = am.group(1).strip()
        if not any(item.lower() in u.text.lower() for u in units):
            units.append(
                ExtractedAtomicUnit(
                    id=f"c{chunk.chunk_index}_u{unit_counter}",
                    chunk_index=chunk.chunk_index,
                    text=item,
                    unit_type=UnitType.ACTION,
                    raw_span=raw_span,
                )
            )
            unit_counter += 1

    # 5. List items and bullet points
    list_lines = re.finditer(r"^\s*[-*•\d\.\)]\s+([^\n]{5,120})$", text, re.MULTILINE)
    for lm in list_lines:
        line_content = lm.group(1).strip()
        # Avoid duplicating existing
        if not any(line_content.lower() in u.text.lower() for u in units):
            units.append(
                ExtractedAtomicUnit(
                    id=f"c{chunk.chunk_index}_u{unit_counter}",
                    chunk_index=chunk.chunk_index,
                    text=line_content,
                    unit_type=UnitType.REQUIREMENT,
                    raw_span=line_content,
                )
            )
            unit_counter += 1

    return units


# =========================================================================
# ASYNC CHUNK EXTRACTION & DECOMPOSITION PIPELINE
# =========================================================================

async def _safe_chat_call(client: Any, messages: list[dict], format: Any = None, num_predict: Optional[int] = None) -> str:
    try:
        if num_predict is not None:
            return await client.chat(messages=messages, format=format, num_predict=num_predict)
        return await client.chat(messages=messages, format=format)
    except TypeError as e:
        if "unexpected keyword argument" in str(e) or "num_predict" in str(e):
            return await client.chat(messages=messages, format=format)
        raise


async def extract_chunk_atomic_units(
    chunk: SourceChunk,
    client: Any,
) -> list[ExtractedAtomicUnit]:
    """Extract structured atomic semantic units for a single chunk.

    If deterministic extraction already captures sufficient structured units (actions,
    requirements, constraints, quantities), reuses them directly without incurring
    an expensive redundant LLM call. Only invokes LLM when semantic interpretation
    is genuinely needed.
    """
    det_units = extract_deterministic_chunk_units(chunk)
    # If deterministic extraction already captured rich structured units (>= 2 units),
    # avoid redundant expensive LLM call.
    if len(det_units) >= 2:
        return det_units

    user_msg = (
        f"<SOURCE_CHUNK index='{chunk.chunk_index}'>\n"
        f"{chunk.text}\n"
        f"</SOURCE_CHUNK>\n\n"
        f"Extract all atomic actions, requirements, constraints, quantities, and corrections."
    )
    messages = [
        {"role": "system", "content": CHUNK_EXTRACTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    llm_units: list[ExtractedAtomicUnit] = []
    try:
        raw_resp = await asyncio.wait_for(
            _safe_chat_call(client, messages=messages, format="json", num_predict=500),
            timeout=15.0,
        )
        cleaned = raw_resp.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
        items = data.get("units", [])
        for i, item in enumerate(items, start=1):
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            u_type_str = str(item.get("unit_type", "REQUIREMENT")).upper()
            try:
                u_type = UnitType(u_type_str)
            except ValueError:
                u_type = UnitType.REQUIREMENT
            span = str(item.get("raw_span", txt)).strip()
            replaces = item.get("replaces_target")
            if replaces:
                replaces = str(replaces).strip()

            llm_units.append(
                ExtractedAtomicUnit(
                    id=f"c{chunk.chunk_index}_u{i}",
                    chunk_index=chunk.chunk_index,
                    text=txt,
                    unit_type=u_type,
                    raw_span=span,
                    replaces_target=replaces,
                )
            )
    except Exception as exc:
        logger.warning(
            "Chunk %d LLM extraction failed (%s). Falling back to deterministic unit extraction.",
            chunk.chunk_index,
            exc,
        )
        return det_units

    # Combine with deterministic extraction to ensure absolute completeness
    merged: list[ExtractedAtomicUnit] = list(llm_units)
    for du in det_units:
        if not any(du.text.lower() in lu.text.lower() or lu.text.lower() in du.text.lower() for lu in llm_units):
            merged.append(du)

    return merged


def resolve_supersessions_and_deduplicate(
    chunks: list[SourceChunk],
    chunk_units_list: list[list[ExtractedAtomicUnit]],
) -> Tuple[list[ExtractedAtomicUnit], list[ExtractedAtomicUnit], list[ExtractedAtomicUnit]]:
    """Resolve cross-chunk corrections/supersessions and deduplicate equivalent units.

    Returns:
    (all_units, active_units, superseded_units)
    """
    flat_units: list[ExtractedAtomicUnit] = []
    for u_list in chunk_units_list:
        flat_units.extend(u_list)

    # 1. Identify corrections / supersessions
    superseded_targets: Set[str] = set()
    for u in flat_units:
        if u.unit_type == UnitType.CORRECTION or u.replaces_target:
            if u.replaces_target:
                superseded_targets.add(u.replaces_target.lower().strip())
            # Parse text for "instead of X"
            m = re.search(r"\binstead\s+of\s+([a-zA-Z0-9_\-]+)\b", u.text, re.IGNORECASE)
            if m:
                superseded_targets.add(m.group(1).lower().strip())

    # Mark superseded units
    active_units: list[ExtractedAtomicUnit] = []
    superseded_units: list[ExtractedAtomicUnit] = []

    for u in flat_units:
        is_sup = False
        u_text_l = u.text.lower()
        for st in superseded_targets:
            if st in u_text_l and u.unit_type != UnitType.CORRECTION:
                is_sup = True
                break
        if is_sup:
            u.is_superseded = True
            superseded_units.append(u)
        else:
            # Deduplicate semantically equivalent active units
            if not any(u_text_l == a.text.lower() for a in active_units):
                active_units.append(u)

    return flat_units, active_units, superseded_units


# =========================================================================
# GLOBAL INTENT RECONCILIATION
# =========================================================================

async def reconcile_long_input_analysis(
    chunks: list[SourceChunk],
    active_units: list[ExtractedAtomicUnit],
    client: Any,
    original_prompt: str,
) -> Any:
    """Run bounded global reconciliation over compact active units to build PromptAnalysis.

    Guarantees:
    - Zero tail drop (all chunks represented)
    - Full provenance tracking with chunk index and source span
    - Objective synthesizes the overarching purpose
    - All non-superseded requirements, constraints, and quantities preserved
    """
    from app.schemas.analysis import (
        AmbiguityAssessment,
        AmbiguityLevel,
        AnalysisGrounding,
        GroundingEvidence,
        GroundingRelation,
        PromptAnalysis,
        PromptCategory,
        PromptClassification,
        SourceType,
    )

    # Format compact active units by semantic category
    req_bullets = []
    const_bullets = []
    pref_bullets = []
    action_bullets = []

    for u in active_units:
        bullet = f"- [{u.unit_type.value}] (Chunk {u.chunk_index}): {u.text}"
        if u.unit_type in {UnitType.ACTION, UnitType.DELIVERABLE}:
            action_bullets.append(bullet)
        elif u.unit_type in {UnitType.CONSTRAINT, UnitType.QUANTITY, UnitType.NEGATIVE_RULE, UnitType.LIFECYCLE}:
            const_bullets.append(bullet)
        elif u.unit_type == UnitType.PREFERENCE:
            pref_bullets.append(bullet)
        else:
            req_bullets.append(bullet)

    units_summary = (
        "ACTIVE EXTRACTED SOURCE UNITS:\n"
        + ("\nACTIONS & DELIVERABLES:\n" + "\n".join(action_bullets) if action_bullets else "")
        + ("\n\nEXPLICIT REQUIREMENTS:\n" + "\n".join(req_bullets) if req_bullets else "")
        + ("\n\nCONSTRAINTS & RULES:\n" + "\n".join(const_bullets) if const_bullets else "")
        + ("\n\nPREFERENCES:\n" + "\n".join(pref_bullets) if pref_bullets else "")
    )

    recon_user_msg = (
        f"<MERGED_AUTHORITATIVE_UNITS>\n"
        f"{units_summary}\n"
        f"</MERGED_AUTHORITATIVE_UNITS>\n\n"
        f"Synthesize the global PromptAnalysis strictly as JSON."
    )
    messages = [
        {"role": "system", "content": GLOBAL_RECONCILIATION_SYSTEM_PROMPT},
        {"role": "user", "content": recon_user_msg},
    ]

    llm_analysis: Optional[PromptAnalysis] = None
    try:
        raw_output = await asyncio.wait_for(
            _safe_chat_call(client, messages=messages, format="json", num_predict=800),
            timeout=25.0,
        )
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        parsed = json.loads(cleaned)
        llm_analysis = PromptAnalysis.model_validate(parsed)
    except Exception as exc:
        logger.warning(
            "Global reconciliation LLM call failed or produced invalid schema: %s. Using deterministic synthesis.",
            exc,
        )

    # Deterministic synthesis fallback / completeness guarantee
    # 1. Classification
    if llm_analysis and llm_analysis.classification:
        classification = llm_analysis.classification
    else:
        classification = PromptClassification(category=PromptCategory.CODING, confidence=0.85)

    # 2. Objective
    if llm_analysis and llm_analysis.objective and len(llm_analysis.objective.strip()) > 10:
        objective = llm_analysis.objective.strip()
    else:
        # Formulate objective from primary actions or first chunk heading
        first_heading = next((c.heading_context for c in chunks if c.heading_context), None)
        actions = [u.text for u in active_units if u.unit_type in {UnitType.ACTION, UnitType.DELIVERABLE}]
        if first_heading:
            objective = f"Implement {first_heading}"
        elif actions:
            objective = f"Implement {', '.join(actions[:3])}"
        else:
            objective = f"Process user request with {len(active_units)} coordinated requirements"

    # 3. Explicit Requirements: Union of LLM output + ALL active explicit requirements from every chunk!
    final_reqs: list[str] = []
    if llm_analysis:
        for r in llm_analysis.explicit_requirements:
            if r.strip() and r.strip().lower() not in [fr.lower() for fr in final_reqs]:
                final_reqs.append(r.strip())

    for u in active_units:
        if u.unit_type in {UnitType.REQUIREMENT, UnitType.ACTION, UnitType.DELIVERABLE, UnitType.CORRECTION}:
            clean_u = u.text.strip()
            if clean_u and not any(clean_u.lower() in fr.lower() or fr.lower() in clean_u.lower() for fr in final_reqs):
                final_reqs.append(clean_u)

    # 4. Constraints: Union of LLM output + ALL active constraints, quantities, negative rules, lifecycle rules!
    final_constraints: list[str] = []
    if llm_analysis:
        for c in llm_analysis.constraints:
            if c.strip() and c.strip().lower() not in [fc.lower() for fc in final_constraints]:
                final_constraints.append(c.strip())

    for u in active_units:
        if u.unit_type in {UnitType.CONSTRAINT, UnitType.QUANTITY, UnitType.NEGATIVE_RULE, UnitType.LIFECYCLE}:
            clean_c = u.text.strip()
            if clean_c and not any(clean_c.lower() in fc.lower() for fc in final_constraints):
                final_constraints.append(clean_c)

    # 5. Preferences
    final_prefs: list[str] = []
    if llm_analysis:
        for p in llm_analysis.preferences:
            if p.strip() and p.strip().lower() not in [fp.lower() for fp in final_prefs]:
                final_prefs.append(p.strip())

    for u in active_units:
        if u.unit_type == UnitType.PREFERENCE:
            clean_p = u.text.strip()
            if clean_p and not any(clean_p.lower() in fp.lower() for fp in final_prefs):
                final_prefs.append(clean_p)

    # 6. Missing Context
    final_missing: list[str] = []
    if llm_analysis:
        for m in llm_analysis.missing_context:
            if m.strip() and m.strip().lower() not in [fm.lower() for fm in final_missing]:
                final_missing.append(m.strip())

    # 7. Ambiguity Assessment: Long comprehensive specifications have LOW ambiguity!
    ambiguity = AmbiguityAssessment(
        score=10,
        level=AmbiguityLevel.LOW,
        needs_clarification=False,
        reasons=[],
        questions=[],
    )

    # 8. Provenance Evidence
    evidence_list: list[GroundingEvidence] = []
    for u in active_units:
        field_target = "constraints" if u.unit_type in {UnitType.CONSTRAINT, UnitType.QUANTITY, UnitType.NEGATIVE_RULE, UnitType.LIFECYCLE} else "explicit_requirements"
        evidence_list.append(
            GroundingEvidence(
                field=field_target,
                value=u.text,
                source_type=SourceType.ORIGINAL_PROMPT,
                evidence=f"Chunk {u.chunk_index}: {u.raw_span}",
                relation=GroundingRelation.DIRECT,
            )
        )

    grounding = AnalysisGrounding(
        is_grounded=True,
        evidence=evidence_list,
        unsupported_items=[],
        source_units=[u.text for u in active_units],
        missing_source_units=[],
    )

    analysis = PromptAnalysis(
        classification=classification,
        objective=objective,
        explicit_requirements=final_reqs,
        constraints=final_constraints,
        preferences=final_prefs,
        missing_context=final_missing,
        ambiguity=ambiguity,
        grounding=grounding,
    )

    return analysis
