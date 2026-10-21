import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from dataset_factory.schemas import (
    CurationIssueType,
    DatasetCurationIssue,
    DatasetCurationReport,
    SeedIntent,
)
from app.core.intent_anchor import (
    ACTION_FAMILY_TO_CATEGORY,
    CATEGORY_TO_ACTION_FAMILY,
    extract_intent_anchor,
)
from app.core.source_coverage import (
    TECH_ALIASES,
    STOP_WORDS,
    GENERIC_FILLER_WORDS,
    GENERIC_ACTION_VERBS,
    token_matches,
)
from app.core.semantic_matcher import semantic_matcher
from app.core.nli_matcher import nli_matcher
from app.schemas.analysis import ActionFamily, PromptCategory

logger = logging.getLogger("dataset_factory.curation")

# Disallowed self-referential / meta commentary patterns in enhanced training targets
META_COMMENTARY_PATTERNS = [
    r"(?i)\bhumanintent\b",
    r"(?i)\bi have enhanced this prompt\b",
    r"(?i)\bi strictly adhered to\b",
    r"(?i)\bi have removed the invented\b",
    r"(?i)\bhere is your (?:enhanced|optimized) prompt\b",
    r"(?i)\bprompt is now concise, direct\b",
    r"(?i)\bfree of artificial complexity\b",
    r"(?i)\bdo not assume these details without confirmation\b",
    r"(?i)\bask for clarification if needed\b",
    r"(?i)\bas an ai (?:language model|assistant)\b",
]

# Patterns for obvious malformed or placeholder artifacts in teacher targets
MALFORMED_TARGET_PATTERNS = [
    (r"(?im)^\s*[-*•]\s*$", "blank bullet item"),
    (r"(?im)^\s*\d+\.\s*$", "blank numbered list item"),
    (r"(?i)[-*•]\s*\?", "question mark bullet placeholder"),
    (r"\?\s*\?\s*\?", "unresolved question mark sequence"),
    (r"(?i)<insert[^>]*>", "unresolved angle-bracket placeholder"),
    (r"(?i)\[insert[^\]]*\]", "unresolved bracketed placeholder"),
    (r"\[Name\]", "unresolved name placeholder"),
    (r"\[Company\]", "unresolved company placeholder"),
    (r"\[Your Name\]", "unresolved signature placeholder"),
    (r"(?im)^\s*[-*•]\s*(?:modality|interface\s+type)\s*$", "hallucinated software template parameter"),
    (r"(?im)^\s*[-*•]\s*missing\s+(?:architecture/storage/interface\s+details)\s*$", "hallucinated software template parameter"),
]

# Source-dependent action keywords and objects that necessitate external input text
SOURCE_DEPENDENT_ACTIONS = [
    # 1. Summarization / condensation of an artifact (abstract, paper, article, report, document, memo, transcript)
    re.compile(
        r"(?i)\b(?:summarize|summarise|condense|abstract|give\s+(?:me\s+)?a\s+summary\s+of|extract\s+(?:the\s+)?(?:key\s+)?takeaways\s+from)\s+"
        r"(?:the\s+|an?\s+|this\s+|the\s+following\s+|the\s+provided\s+|the\s+attached\s+|the\s+supplied\s+)?"
        r"(?:abstract|executive\s+summary)\s+of\s+(?:the\s+|an?\s+)?(?:research\s+|scientific\s+|academic\s+)?(?:paper|article|study|report|document|publication)\b"
    ),
    re.compile(
        r"(?i)\b(?:summarize|summarise|condense|abstract|give\s+(?:me\s+)?a\s+summary\s+of|extract\s+(?:the\s+)?(?:key\s+)?takeaways\s+from)\s+"
        r"(?:this|the\s+following|the\s+provided|the\s+attached|the\s+supplied|the\s+below|the|an?)\s+"
        r"(?:research\s+|scientific\s+|academic\s+)?(?:report|paper|article|document|text|study|memo|transcript|passage|contract|agreement|rfc|notes|email)\b"
        r"(?!\s+(?:about|on|regarding)\s+how\b)"
    ),
    # 2. Code review, debugging, inspection, profiling of a code artifact
    re.compile(
        r"(?i)\b(?:debug|diagnose|fix|review|refactor|audit|profile|optimize\s+(?:the\s+)?performance\s+of|find\s+(?:bugs|issues|flaws|vulnerabilities|race\s+conditions)\s+in|check\s+(?:for\s+bugs|for\s+issues|for\s+race\s+conditions)\s+in)\s+"
        r"(?:this|the\s+following|the\s+provided|the\s+attached|the\s+supplied|the\s+below|my|our|the|an?|a\s+given)\s+"
        r"(?:python\s+|javascript\s+|typescript\s+|c\+\+\s+|java\s+|go\s+|rust\s+|sql\s+)?(?:code|script|function|snippet|program|loop|crash|method|class|implementation|query|algorithm)\b"
    ),
    # 3. Data analysis, parsing, calculation of metrics on a dataset / sample / numbers / logs
    re.compile(
        r"(?i)\b(?:calculate|compute|find|determine|analyze|analyse|parse|inspect|examine|audit|extract\s+(?:trends|insights|metrics)\s+from)\s+"
        r"(?:the\s+|an?\s+|this\s+|my\s+|our\s+)?(?:median|mean|average|p\d+|percentiles?|latency|standard\s+deviation|variance|outliers?|statistics|metrics)\s+"
        r"(?:of|from|for|in)\s+(?:the\s+|this\s+|a\s+|our\s+)?(?:sample|dataset|data|response\s+times|latencies|logs?|numbers?|telemetry|metrics)\b"
    ),
    re.compile(
        r"(?i)\b(?:here\s*'?s|here\s+is|given|using|from)\s+(?:a\s+|the\s+|this\s+)?(?:sample|dataset|data|table|csv|json|log|logs|metrics|telemetry)\s+of\b"
    ),
    re.compile(
        r"(?i)\b(?:analyze|analyse|parse|inspect|examine|audit|extract\s+(?:trends|insights|metrics)\s+from)\s+"
        r"(?:this|the\s+following|the\s+provided|the\s+attached|the\s+supplied|the\s+below|my|our|the|an?|a\s+given)\s+"
        r"(?:dataset|data|csv|json|table|sample|numbers|metrics|log|logs|telemetry|spreadsheet|dataframe|df)\b"
    ),
    # 4. Translation, proofreading, copyediting of a document / text passage
    re.compile(
        r"(?i)\b(?:translate|rewrite|proofread|copyedit|edit|check\s+(?:the\s+)?(?:grammar|spelling)\s+(?:of|in))\s+"
        r"(?:this|the\s+following|the\s+provided|the\s+attached|the\s+supplied|the\s+below|my|our|the|an?)\s+"
        r"(?:paragraph|text|sentence|passage|document|manuscript|draft|essay|letter|email)\b"
    ),
]

# Quantities that must be recalled if specified
QUANTITY_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty)\s+"
    r"(?:[a-z0-9_\-]+\s+){0,3}"
    r"(bullets?|takeaways?|milestones?|ideas?|concepts?|days?|weeks?|words?|options?|examples?|loglines?)\b",
    re.IGNORECASE,
)


def _check_source_dependency(original_prompt: str) -> Tuple[bool, bool]:
    """
    Determine if task is logically source-dependent and whether source material was supplied.
    Returns: (source_required, source_present)
    """
    prompt_lower = original_prompt.lower()

    source_required = any(
        bool(pat.search(prompt_lower) if hasattr(pat, "search") else re.search(pat, prompt_lower))
        for pat in SOURCE_DEPENDENT_ACTIONS
    )

    if not source_required:
        return False, True

    # Check if prompt actually contains the referenced source material:
    # 1. Code presence: fenced code blocks, multi-line indented code, def/function/class syntax
    has_fenced_code = "```" in original_prompt
    has_indented_code = bool(re.search(r"\n(?:    |\t)[a-zA-Z0-9_#]", original_prompt))
    has_code_syntax = bool(re.search(r"\b(?:def|class|function|async def|const|let|var)\s+[a-zA-Z0-9_]+\s*[\(:=]", original_prompt))
    has_code = has_fenced_code or has_indented_code or has_code_syntax

    # 2. Data presence: markdown table pipes, json structure, csv lines with multiple comma columns
    has_table = "|" in original_prompt and "\n" in original_prompt and original_prompt.count("|") >= 4
    has_json = bool(re.search(r"\{\s*\"[^\"]+\"\s*:", original_prompt)) or bool(re.search(r"\[\s*\{", original_prompt))
    has_csv_rows = bool(re.search(r"(?:^[^\n,]+,[^\n,]+,[^\n,]+(?:\n|\r\n?)){2,}", original_prompt, flags=re.MULTILINE))
    has_data_numbers = bool(re.search(r"\d+[\s,]+\d+[\s,]+\d+", original_prompt))
    has_data = has_table or has_json or has_csv_rows or has_data_numbers

    # 3. Document/text presence: blockquotes, substantial multi-sentence body text, or long prompt
    has_blockquote = "\n>" in original_prompt or original_prompt.startswith(">")
    has_double_quoted_doc = bool(re.search(r'"[^"]{120,}"', original_prompt, flags=re.DOTALL))
    has_single_quoted_doc = bool(re.search(r"(?:^|[\s:\n])'([^'\n]{120,})'(?:$|[\s,\.!\?\n])", original_prompt))
    has_quoted_doc = has_double_quoted_doc or has_single_quoted_doc
    has_long_doc = len(original_prompt.split()) > 90

    source_present = has_code or has_data or has_blockquote or has_quoted_doc or has_long_doc
    return True, source_present


def _extract_negative_constraints(original_prompt: str, seed: Optional[SeedIntent] = None) -> List[str]:
    """Extract negative or restrictive constraints from prompt and seed.
    The generated original_prompt is authoritative. Constraints from seed are only enforced
    if their concept or key keywords are actually reflected in original_prompt.
    """
    constraints: List[str] = []
    prompt_lower = original_prompt.lower()

    if seed and seed.optional_constraints:
        for c in seed.optional_constraints:
            c_low = c.lower()
            if any(neg in c_low for neg in ["avoid", "no ", "without", "only", "under", "never", "do not"]):
                c_key_tokens = [
                    w for w in re.findall(r"\b[a-z0-9_\-]+\b", c_low)
                    if len(w) > 3 and w not in ["avoid", "standard", "generic", "without", "never", "using", "under"]
                ]
                # Only enforce seed constraint if reflected in generated prompt
                if c_key_tokens and any(t in prompt_lower for t in c_key_tokens):
                    constraints.append(c)

    # Regex extractions from raw prompt
    neg_patterns = [
        r"(?i)\b(?:avoid|avoiding|do not use|don't use|without|no|do\s+not\s+feature)\s+([a-z0-9_\-,\s]{3,45})(?:[,.]|$)",
        r"(?i)\b(only\s+standard\s+library|standard\s+library\s+only)\b",
        r"(?i)\b(no\s+external\s+dependencies)\b",
        r"(?i)\b(under\s+\d+\s+words)\b",
        r"(?i)\b(without\s+mathematical\s+formulas)\b",
        r"(?i)\b(avoid\s+external\s+dependencies|avoiding\s+external\s+dependencies)\b",
        r"(?i)\b(?:must\s+run\s+offline|run\s+offline|offline\s+only)\b",
        r"(?i)\b(?:without\s+authentication|no\s+authentication)\b",
    ]
    for pat in neg_patterns:
        for match in re.finditer(pat, original_prompt):
            constraints.append(match.group(0).strip())

    return list(set(constraints))


def _extract_compound_deliverables(original_prompt: str) -> List[str]:
    """Detect compound deliverables (e.g. 'A, B, and C' or 'outline, code, and tests')."""
    deliverables: List[str] = []
    # Pattern for "provide/include/generate X, Y, and Z"
    match = re.search(
        r"(?i)\b(?:include|provide|generate|give\s+me|create)\s+([a-z0-9\s,]+(?:and|\&)\s+[a-z0-9\s]+?)(?:\.|\n|$)",
        original_prompt,
    )
    if match:
        phrase = match.group(1)
        # Split on commas and 'and'
        parts = re.split(r",|\band\b|\b\&\b", phrase)
        for p in parts:
            clean = p.strip()
            # Strip trailing prepositional phrases (e.g. "unit tests for a priority queue in Python" -> "unit tests")
            clean = re.sub(r"\s+(?:for|in|with|using|of)\b.*$", "", clean, flags=re.IGNORECASE).strip()
            if len(clean) > 3 and clean.lower() not in ["both", "all", "the following"]:
                deliverables.append(clean)
    return deliverables


def _extract_quantities(original_prompt: str, seed: Optional[SeedIntent] = None) -> List[str]:
    """Extract quantitative bounds from original_prompt (authoritative).
    Seed quantities are only included if reflected in the prompt.
    """
    quantities: List[str] = []
    prompt_lower = original_prompt.lower()
    for match in QUANTITY_PATTERN.finditer(original_prompt):
        quantities.append(match.group(0).lower().strip())

    if seed:
        for req in seed.core_requirements + seed.optional_constraints:
            for match in QUANTITY_PATTERN.finditer(req):
                q_text = match.group(0).lower().strip()
                q_words = q_text.split()
                # Only include seed quantity if the prompt also mentions the number or unit
                if any(w in prompt_lower for w in q_words):
                    quantities.append(q_text)

    return list(set(quantities))


def validate_preference_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    category: str,
    seed: Optional[SeedIntent] = None,
    initial_rejection_reason: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate that a preference pair represents genuine improvement:
    1. Chosen passes dataset curation.
    2. Rejected has a verifiable flaw (either curation failure or initial validator rejection).
    3. Chosen fixes the specific flaw and introduces no new major issues.
    """
    if not chosen or not rejected or chosen.strip() == rejected.strip():
        return False, "Chosen and rejected outputs are identical or empty."

    # Validate chosen target
    chosen_report = curate_candidate(
        original_prompt=prompt,
        target_prompt=chosen,
        category=category,
        seed=seed,
    )
    if not chosen_report.is_acceptable:
        return False, f"Chosen candidate failed curation: {[i.message for i in chosen_report.issues]}"

    # Validate rejected target
    rejected_report = curate_candidate(
        original_prompt=prompt,
        target_prompt=rejected,
        category=category,
        seed=seed,
    )

    has_verifiable_rejection = not rejected_report.is_acceptable or bool(initial_rejection_reason)
    if not has_verifiable_rejection:
        return False, "Rejected candidate exhibits no verifiable flaw under curation or validation."

    return True, None


CANONICAL_GROUNDING_SENSITIVE_SECTIONS = {
    "requirements",
    "constraints",
    "questions to answer",
    "metrics",
    "methodology",
}

CANONICAL_PRESENTATIONAL_SECTIONS = {
    "role",
    "preferences",
    "output format",
    "unspecified details",
}

PRESENTATIONAL_SECTION_KEYWORDS = {
    "role", "persona", "output structure", "output format", "response format",
    "format", "formatting", "structure", "style", "tone", "unspecified details",
    "presentation", "presentation format", "layout", "missing context",
    "expectations", "output expectations",
} | CANONICAL_PRESENTATIONAL_SECTIONS

GROUNDING_SENSITIVE_SECTION_KEYWORDS = {
    "core requirements", "explicit requirements", "requirements", "constraints",
    "questions to answer", "required metrics", "metrics", "required methodology",
    "methodology", "deliverables", "key deliverables", "functional requirements",
    "technical requirements", "business rules", "specifications", "tasks", "instructions",
    "steps",
} | CANONICAL_GROUNDING_SENSITIVE_SECTIONS

AMBIGUOUS_TECH_WORDS = {"go", "next", "spring", "ruby"}


def normalize_section_header(line: str) -> Optional[str]:
    """Normalize equivalent section headers into canonical semantic section types.

    Formatting-insensitive: strips Markdown decorations (#, **, __), colons, and punctuation.
    Treats '/', '&', '+', '_', '-' as separators and classifies based on semantic tokens.
    Canonical section types returned:
      - 'requirements' (e.g. 'CORE / EXPLICIT REQUIREMENTS', '### CORE & EXPLICIT REQUIREMENTS:', 'DELIVERABLES')
      - 'constraints' (e.g. 'CONSTRAINTS:', 'EXPLICIT CONSTRAINTS', 'RESTRICTIONS')
      - 'questions to answer' (e.g. 'QUESTIONS TO ANSWER:', 'REQUIRED QUESTIONS')
      - 'metrics' (e.g. 'REQUIRED METRICS', 'METRICS:')
      - 'methodology' (e.g. 'REQUIRED METHODOLOGY', 'METHODOLOGY:')
      - 'preferences' (e.g. 'PREFERENCES', 'STYLE PREFERENCES:')
      - 'output format' (e.g. 'OUTPUT FORMAT:', 'RESPONSE FORMAT', 'LAYOUT', 'OUTPUT EXPECTATIONS')
      - 'role' (e.g. 'ROLE:', 'PERSONA')
      - 'unspecified details' (e.g. 'UNSPECIFIED DETAILS:', 'ASSUMPTIONS', 'MISSING CONTEXT')
    Returns None for bullet list items, body text, or non-header lines.
    """
    stripped = line.strip()
    if not stripped:
        return None

    # Reject bullet list items or numbered items
    if re.match(r"^(?:[-•]|\*\s+|\+\s+|\d+[\.)]\s+)", stripped):
        return None

    is_markdown_header = bool(re.match(r"^#{1,6}\s+", stripped))
    is_bold_header = (
        (stripped.startswith("**") and ("**" in stripped[2:])) or
        (stripped.startswith("__") and ("__" in stripped[2:]))
    )
    ends_with_colon = stripped.endswith(":")

    clean = re.sub(r"^[\s#\*_`]+|[\s#\*_`:]+$", "", stripped).strip()
    if not clean or len(clean) > 80:
        return None

    # Disqualify body sentences with punctuation unless explicitly structured as markdown header
    if not is_markdown_header:
        if any(p in clean for p in (".", "!", "?", ";")):
            return None

    # If line has no markdown/bold/colon, require short title or uppercase
    if not (is_markdown_header or is_bold_header or ends_with_colon):
        words = clean.split()
        if len(words) > 6:
            return None
        if not (clean.isupper() or clean.istitle()):
            return None

    norm = re.sub(r"[/&+,_\-]+", " ", clean.lower())
    tokens = re.findall(r"\b[a-z0-9]+\b", norm)
    if not tokens:
        return None
    token_set = set(tokens)

    # Semantic classification into canonical section types
    if token_set & {
        "requirements", "requirement", "deliverables", "deliverable",
        "tasks", "task", "obligations", "obligation", "specifications",
        "specification", "instructions", "instruction", "rules", "criteria",
    }:
        return "requirements"

    if token_set & {
        "constraints", "constraint", "restrictions", "restriction",
        "limitations", "limitation",
    }:
        return "constraints"

    if token_set & {"questions", "question"}:
        return "questions to answer"

    if token_set & {"metrics", "metric"}:
        return "metrics"

    if token_set & {"methodology", "methodologies", "methods", "method"}:
        return "methodology"

    if token_set & {"preferences", "preference"} or ("style" in token_set and "preferences" in token_set):
        return "preferences"

    if token_set & {"format", "formatting", "layout", "presentation"} or ("output" in token_set and ("structure" in token_set or "expectations" in token_set or "format" in token_set)):
        return "output format"

    if token_set & {"role", "persona", "identity"}:
        return "role"

    if token_set & {"unspecified", "assumptions"} or ("missing" in token_set and "context" in token_set) or token_set == {"details"}:
        return "unspecified details"

    # Fallback: if explicitly marked header, return cleaned tokens
    if is_markdown_header or is_bold_header or ends_with_colon or clean.isupper():
        return " ".join(tokens)

    return None


def is_grounding_sensitive_section(section_name: str) -> bool:
    """Check if a section is grounding-sensitive (requirements, constraints, metrics, etc.)."""
    sec = section_name.strip().lower()
    if is_presentational_section(sec):
        return False
    if sec in CANONICAL_GROUNDING_SENSITIVE_SECTIONS:
        return True
    return any(k in sec for k in GROUNDING_SENSITIVE_SECTION_KEYWORDS)


def is_presentational_section(section_name: str) -> bool:
    """Check if a section is presentational (role, output format, style preferences, etc.)."""
    sec = section_name.strip().lower()
    if sec in CANONICAL_PRESENTATIONAL_SECTIONS:
        return True
    return any(k in sec for k in PRESENTATIONAL_SECTION_KEYWORDS)


def _extract_substantive_target_obligations(target_text: str) -> List[str]:
    """Extract substantive obligation clauses from grounding-sensitive target sections.

    Excludes presentational structure (role, formatting headers, output layout)
    while capturing functional obligations, metrics, methodologies, and constraints.
    """
    lines = target_text.splitlines()
    sections: Dict[str, List[str]] = defaultdict(list)
    current_section = "GENERAL"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        sec_header = normalize_section_header(line)
        if sec_header:
            current_section = sec_header
            continue

        sections[current_section].append(stripped)

    obligations: List[str] = []

    presentational_filter_pats = [
        r"(?i)^(?:act\s+as|you\s+are|role:)",
        r"(?i)^(?:format|output|provide|structure|present)\s+(?:the\s+)?(?:response|answer|output|solution|results?)\s+(?:as|in|with|clearly|concisely|using)\b",
        r"(?i)^(?:use|include)\s+(?:clear\s+)?(?:markdown|code\s+blocks?|bullet\s+points?|headings?|docstrings?|comments?)\b",
        r"(?i)^(?:use|provide|include)\s+(?:everyday\s+|simple\s+|intuitive\s+|concrete\s+)?(?:analogies|examples|metaphors|illustrations)\b",
        r"(?i)^(?:keep|ensure)\s+(?:it|the\s+response|the\s+output)\s+(?:concise|brief|clear|well-structured|professional)\b",
        r"(?i)^(?:structure|organize)\s+(?:the\s+)?(?:response|output|solution)\s+with\b",
        r"(?i)^explain\s+(?:your\s+)?(?:reasoning|approach|steps?)\s+(?:step\s+by\s+step|clearly)\b",
        r"(?i)^(?:return|yield|produce|output)\s+(?:the\s+)?(?:new\s+|resulting\s+|final\s+)?(?:output|result|value|head|list|response|data|answer|solution)\b",
    ]

    has_grounding_sensitive = any(
        is_grounding_sensitive_section(s)
        for s in sections
    )

    for section_name, sec_lines in sections.items():
        # Presentational sections are never interpreted as user requirements
        if is_presentational_section(section_name):
            continue

        # If the target has explicit grounding-sensitive sections, ONLY extract from them
        if has_grounding_sensitive:
            if not is_grounding_sensitive_section(section_name):
                continue
        else:
            # Skip explicitly presentational sections
            if is_presentational_section(section_name):
                continue

        for line in sec_lines:
            bullet_match = re.match(r"^\s*(?:[-*•]|\d+\.)\s*(.*)$", line)
            # Only extract bullet items or lines within explicit grounding-sensitive sections
            if not bullet_match and not has_grounding_sensitive:
                continue

            content = bullet_match.group(1).strip() if bullet_match else line

            if any(re.search(pat, content) for pat in presentational_filter_pats):
                continue

            words = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", content) if len(w) > 2]
            if len(words) < 2:
                continue

            obligations.append(content)

    return obligations


def check_target_unsupported_requirements(
    original_prompt: str,
    target_prompt: str,
    seed: Optional[SeedIntent] = None,
    normalized_intent: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, List[str]]:
    """Verify that every substantive obligation in the enhanced target is grounded in authoritative sources.

    Layered verification:
    1. Deterministic invariant: Named technology detection
    2. Deterministic invariant: Token match ratio against authoritative corpus
    3. Semantic similarity: Paraphrase recognition using SentenceTransformers
    4. NLI verification: Natural Language Inference entailment check on ambiguous band
    """
    ungrounded_issues: List[str] = []

    # Assemble authoritative text sources
    authoritative_texts: List[str] = [original_prompt]
    if seed:
        if seed.intent:
            authoritative_texts.append(seed.intent)
        if seed.core_requirements:
            authoritative_texts.extend(seed.core_requirements)
        if seed.optional_constraints:
            authoritative_texts.extend(seed.optional_constraints)

    if normalized_intent:
        if isinstance(normalized_intent, dict):
            obj = normalized_intent.get("objective")
            if obj:
                authoritative_texts.append(obj)
            reqs = normalized_intent.get("explicit_requirements") or []
            authoritative_texts.extend(reqs)
            constraints = normalized_intent.get("constraints") or []
            authoritative_texts.extend(constraints)
        elif hasattr(normalized_intent, "objective"):
            if normalized_intent.objective:
                authoritative_texts.append(normalized_intent.objective)
            reqs = getattr(normalized_intent, "explicit_requirements", []) or []
            authoritative_texts.extend(reqs)
            constraints = getattr(normalized_intent, "constraints", []) or []
            authoritative_texts.extend(constraints)

    authoritative_texts = [t.strip() for t in authoritative_texts if t and t.strip()]
    authoritative_corpus = " ".join(authoritative_texts).lower()

    # 1. Deterministic Technology Invariant
    authoritative_techs: Set[str] = set()
    for word in re.findall(r"\b[a-zA-Z0-9_\-+#.]+\b", authoritative_corpus):
        w_low = word.lower()
        if w_low in TECH_ALIASES:
            if w_low in AMBIGUOUS_TECH_WORDS:
                if w_low == "go" and not re.search(r"\b(?:golang|in\s+go|go\s+(?:language|code|cli|tool|program|developer|systems|standard\s+library))\b", authoritative_corpus):
                    continue
                if w_low == "next" and not re.search(r"\bnext\.?js\b", authoritative_corpus):
                    continue
                if w_low == "spring" and not re.search(r"\bspring\s*(?:boot|framework)\b", authoritative_corpus):
                    continue
                if w_low == "ruby" and not re.search(r"\b(?:ruby\s+on\s+rails|in\s+ruby|ruby\s+language)\b", authoritative_corpus):
                    continue
            authoritative_techs.add(TECH_ALIASES[w_low])

    # Check for ungrounded technologies anywhere in the target
    target_all_techs: Set[str] = set()
    for word in re.findall(r"\b[a-zA-Z0-9_\-+#.]+\b", target_prompt.lower()):
        w_low = word.lower()
        if w_low in TECH_ALIASES:
            if w_low in AMBIGUOUS_TECH_WORDS:
                if w_low == "go" and not re.search(r"\b(?:golang|in\s+go|go\s+(?:language|code|cli|tool|program|developer|systems|standard\s+library))\b", target_prompt.lower()):
                    continue
                if w_low == "next" and not re.search(r"\bnext\.?js\b", target_prompt.lower()):
                    continue
                if w_low == "spring" and not re.search(r"\bspring\s*(?:boot|framework)\b", target_prompt.lower()):
                    continue
                if w_low == "ruby" and not re.search(r"\b(?:ruby\s+on\s+rails|in\s+ruby|ruby\s+language)\b", target_prompt.lower()):
                    continue
            target_all_techs.add(TECH_ALIASES[w_low])

    unsupported_techs = target_all_techs - authoritative_techs
    if unsupported_techs:
        for tech in sorted(unsupported_techs):
            ungrounded_issues.append(
                f"Target invents unrequested technology or framework: '{tech}' (not requested in source prompt or seed)"
            )

    # Check for deliverable escalation (explanatory conceptual inquiry escalated to mandatory code deliverable)
    is_conceptual_inquiry = bool(re.search(
        r"(?i)^(?:why\b|how\s+(?:does|do|can|is|are)\b|what\s+(?:is|are|causes|leads\s+to)\b|explain\b)",
        original_prompt.strip()
    ))
    asks_for_code = bool(re.search(
        r"(?i)\b(?:code|script|snippet|implementation|program|write|create|build)\b",
        original_prompt
    ))
    if is_conceptual_inquiry and not asks_for_code:
        code_escalation_match = re.search(
            r"(?i)\b(?:followed\s+by\s+code|provide\s+(?:an?\s+)?(?:implementation\s+approach\s+followed\s+by\s+)?code|deliverable:\s*code|write\s+(?:the\s+)?code|include\s+(?:the\s+)?(?:full\s+)?code)\b",
            target_prompt
        )
        if code_escalation_match:
            ungrounded_issues.append(
                f"Target escalates explanatory conceptual inquiry into mandatory code deliverable: '{code_escalation_match.group(0)}'"
            )

    # 2. Extract and check substantive obligations
    obligations = _extract_substantive_target_obligations(target_prompt)

    # Break authoritative texts into individual clauses
    authoritative_clauses: List[str] = []
    for txt in authoritative_texts:
        parts = re.split(r"[.\n;!]", txt)
        for p in parts:
            p_strip = p.strip()
            if len(p_strip.split()) >= 2:
                authoritative_clauses.append(p_strip)
    if not authoritative_clauses:
        authoritative_clauses = authoritative_texts

    for U in obligations:
        # A. Deterministic Quantity Grounding Invariant
        u_quantities = _extract_quantities(U)
        if u_quantities and any(q in authoritative_corpus for q in u_quantities):
            continue

        # B. Deterministic Technology Grounding Invariant for short tech requirements (e.g. "Use Python")
        u_techs = [TECH_ALIASES[w.lower()] for w in re.findall(r"\b[a-zA-Z0-9_\-+#.]+\b", U) if w.lower() in TECH_ALIASES]
        if u_techs and all(t in authoritative_techs for t in u_techs):
            u_clean_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", U) if w.lower() not in STOP_WORDS and w.lower() not in GENERIC_ACTION_VERBS]
            if len(u_clean_words) <= 1:
                continue

        # C. Deterministic Negative Constraint Grounding Invariant (e.g. "No Redis")
        m_neg = re.search(r"(?i)\b(?:no|avoid|avoiding|without|never|do\s+not\s+use)\s+([a-z0-9_\-]+)\b", U)
        if m_neg:
            neg_target = m_neg.group(1).lower()
            if neg_target in authoritative_corpus or (neg_target in TECH_ALIASES and TECH_ALIASES[neg_target] in authoritative_techs):
                continue

        # Check token grounding
        u_tokens = [
            w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", U)
            if w.lower() not in STOP_WORDS and w.lower() not in GENERIC_FILLER_WORDS and w.lower() not in GENERIC_ACTION_VERBS
        ]
        if not u_tokens:
            continue

        match_count = sum(1 for tok in u_tokens if token_matches(tok, authoritative_corpus))
        token_ratio = match_count / len(u_tokens)

        # 1. Deterministic / strong token match grounding
        if token_ratio >= 0.75 or (len(u_tokens) <= 2 and match_count == len(u_tokens)):
            continue

        # 2. Semantic similarity against authoritative clauses and full prompt
        max_sim = 0.0
        if semantic_matcher.is_available:
            for auth_c in authoritative_clauses:
                sim = semantic_matcher.similarity(U, auth_c)
                if sim > max_sim:
                    max_sim = sim
            sim_full = semantic_matcher.similarity(U, original_prompt)
            max_sim = max(max_sim, sim_full)

        # If completely unrelated semantically
        if max_sim < 0.35:
            ungrounded_issues.append(
                f"Target introduces ungrounded substantive requirement: '{U.strip()}' (token_ratio={token_ratio:.2f}, max_sim={max_sim:.2f})"
            )
            continue

        # 3. Layered verification: NLI entailment for ambiguous / new vocabulary
        # Since token_ratio < 0.75, new obligations or terms were introduced.
        # Verify that authoritative source logically supports/entails this obligation.
        is_entailed = False
        if nli_matcher.is_available:
            for auth_c in authoritative_clauses:
                label, probs = nli_matcher.predict(auth_c, U)
                if probs.get("entailment", 0.0) >= 0.40 or label == "entailment":
                    is_entailed = True
                    break
            if not is_entailed:
                label, probs = nli_matcher.predict(original_prompt, U)
                if probs.get("entailment", 0.0) >= 0.40 or label == "entailment":
                    is_entailed = True

            if not is_entailed:
                ungrounded_issues.append(
                    f"Target introduces ungrounded substantive requirement: '{U.strip()}' (token_ratio={token_ratio:.2f}, max_sim={max_sim:.2f})"
                )
                continue
        else:
            # Fallback when NLI is not available
            if not (max_sim >= 0.85 and token_ratio >= 0.50):
                ungrounded_issues.append(
                    f"Target introduces ungrounded substantive requirement: '{U.strip()}' (token_ratio={token_ratio:.2f}, max_sim={max_sim:.2f})"
                )
                continue

    no_unsupported = (len(ungrounded_issues) == 0)
    return no_unsupported, ungrounded_issues


DISCOURSE_CONVERSATIONAL_PATTERNS = [
    # Interjections, agreement, conversational acknowledgments
    r"(?i)^(?:right|yeah|yep|nope|okay|ok|alright|cool|sure|exactly|definitely)[.!?]?\s*$",
    r"(?i)^(?:hey|hello|hi|oh|ah|uh|um|oops|whoa)[.!?]?\s*$",
    # Conversational hedging and filler thoughts
    r"(?i)^(?:oh\s+yeah|you\s+know|you\s+see|i\s+mean|i'm\s+thinking|i\s+think|i\s+guess|i\s+suppose|i\s+believe)[.!?]?\s*$",
    # Generic colloquial bundles
    r"(?i)^(?:all\s+that(?:\s+jazz|\s+stuff)?|and\s+all\s+that|and\s+stuff(?:\s+like\s+that)?|etc\s+etc|and\s+so\s+on)[.!?]?\s*$",
    # Dangling conversational fragments
    r"(?i)^(?:all\s+that\.\s+so|and\s+so|we're\s+looking\s+at|we\s+are\s+looking\s+at|like\s+i\s+said|as\s+i\s+said|as\s+mentioned)[.!?]?\s*$",
    # Bare evaluative / adjectival statements lacking entity or actionable task
    r"(?i)^(?:super\s+|very\s+|really\s+|pretty\s+|totally\s+)?(?:important|critical|crucial|essential|vital|good|nice|cool|awesome|helpful|useful)[.!?]?\s*$",
    # Fragments ending with dangling prepositions or connectives
    r"(?i)^[a-z0-9\s']+\b(?:at|so|and|but|or|with|of|to|because)\s*[.!?]?$",
]

DISCOURSE_MARKERS = {
    "right", "yeah", "yep", "nope", "okay", "ok", "alright", "hey", "hello",
    "hi", "oh", "ah", "um", "uh", "thinking", "thought", "guess", "suppose",
    "know", "see", "super", "very", "jazz", "stuff", "thing", "things",
    "basically", "literally", "honestly", "actually", "like", "just", "well",
    "pretty", "somewhat", "maybe", "perhaps", "looking", "gotta", "gonna",
    "wanna", "important", "critical", "essential", "crucial",
}


def check_structured_target_substantive_quality(target_prompt: str) -> Tuple[bool, List[str]]:
    """Validate that structured requirement sections contain substantive semantic obligations rather than conversational filler.

    Rejects target prompts where sections like CORE REQUIREMENTS or CONSTRAINTS promote conversational
    discourse fragments (e.g. 'Right', 'You know', 'I'm thinking', 'Super important', 'All that jazz')
    into requirement bullets. Preserves short valid requirements like 'Use Python', 'No Redis', '3 bullets'.
    """
    issues: List[str] = []
    lines = target_prompt.splitlines()
    sections: Dict[str, List[str]] = defaultdict(list)
    current_section = "GENERAL"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        sec_header = normalize_section_header(line)
        if sec_header:
            current_section = sec_header
            continue
        sections[current_section].append(stripped)

    for section_name, sec_lines in sections.items():
        # Only inspect grounding-sensitive / requirements / constraints sections
        if not is_grounding_sensitive_section(section_name):
            continue

        sec_norm = normalize_section_header(section_name) or section_name.lower()
        is_constraint_section = ("constraint" in sec_norm or "constraint" in section_name.lower())
        is_requirements_section = ("requirement" in sec_norm or "deliverable" in sec_norm or "requirement" in section_name.lower())

        for line in sec_lines:
            bullet_match = re.match(r"^\s*(?:[-*•]|\d+\.)\s*(.*)$", line)
            item = bullet_match.group(1).strip() if bullet_match else line.strip()
            if not item:
                continue

            # 1. Match against pure discourse patterns
            if any(re.match(pat, item) for pat in DISCOURSE_CONVERSATIONAL_PATTERNS):
                issues.append(
                    f"Section '{section_name}' contains non-substantive conversational fragment: '{item}'"
                )
                continue

            # 2. Check for bare topic keyword phrases in CONSTRAINTS section:
            # A valid constraint must represent a boundary or restriction, not isolated single topic words like "hybrid" or "equilibrium"
            if is_constraint_section:
                constraint_markers = [
                    "avoid", "no ", "without", "never", "only", "must", "do not", "don't",
                    "under", "limit", "max", "min", "restrict", "exclude", "ensure",
                    "keep", "require", "cannot", "less than", "more than", "within", "prohibit",
                    "factor", "factors", "parameter", "parameters", "metric", "metrics", "tradeoff", "tradeoffs"
                ]
                has_constraint_marker = any(m in item.lower() for m in constraint_markers)
                if not has_constraint_marker:
                    item_words = [w for w in re.findall(r"\b[a-zA-Z0-9_\-]+\b", item) if w.lower() not in STOP_WORDS]
                    if len(item_words) <= 2:
                        issues.append(
                            f"Section '{section_name}' treats bare topic keyword as constraint without restrictive directive: '{item}'"
                        )
                        continue

            # 4. Check if item has recognized substantive anchors (technologies, quantities, negative constraints)
            item_words = [w.lower() for w in re.findall(r"\b[a-zA-Z0-9_\-+#.]+\b", item)]
            if any(w in TECH_ALIASES for w in item_words):
                continue

            if QUANTITY_PATTERN.search(item) or re.search(r"\b\d+\s+(?:days?|weeks?|months?|items?|bullets?|lines?|words?|steps?)\b", item, re.IGNORECASE):
                continue

            if re.search(r"(?i)\b(?:avoid|avoiding|without|never|no|do\s+not\s+use)\s+[a-z0-9_\-]{3,}\b", item):
                continue

            # 5. Substantive content-token analysis
            substantive_tokens = [
                w for w in item_words
                if w not in STOP_WORDS
                and w not in GENERIC_ACTION_VERBS
                and w not in DISCOURSE_MARKERS
                and len(w) > 1
            ]

            # If an item in Core Requirements has zero substantive domain words
            if len(substantive_tokens) == 0:
                issues.append(
                    f"Section '{section_name}' contains non-substantive requirement with zero domain content: '{item}'"
                )
                continue

    return len(issues) == 0, issues


def curate_candidate(
    original_prompt: str,
    target_prompt: Optional[str],
    category: str,
    normalized_intent: Optional[Dict[str, Any]] = None,
    seed: Optional[SeedIntent] = None,
    clarification_required: bool = False,
    clarification_questions: Optional[List[str]] = None,
    already_good: bool = False,
) -> DatasetCurationReport:
    """
    Independent Dataset Curation Gate.
    Determines whether a candidate prompt-target pair meets strict training-data standards.
    """
    issues: List[DatasetCurationIssue] = []

    # 1. Ambiguity / Clarification Validation
    if clarification_required:
        if not clarification_questions:
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.OTHER,
                    severity="ERROR",
                    message="Candidate marked as clarification_required but missing clarification questions.",
                )
            )
        else:
            for q in clarification_questions:
                if any(re.search(pat, q) for pat in META_COMMENTARY_PATTERNS):
                    issues.append(
                        DatasetCurationIssue(
                            type=CurationIssueType.META_CONTENT,
                            severity="ERROR",
                            message=f"Clarification question contains internal meta commentary: '{q}'",
                        )
                    )

        report = DatasetCurationReport(
            is_acceptable=len(issues) == 0,
            issues=issues,
        )
        return report

    # 2. Positive SFT Enhancement Target Validation
    if not target_prompt or not target_prompt.strip():
        issues.append(
            DatasetCurationIssue(
                type=CurationIssueType.MALFORMED_TARGET,
                severity="ERROR",
                message="Positive enhancement target is empty or null.",
            )
        )
        return DatasetCurationReport(
            is_acceptable=False,
            target_well_formed=False,
            issues=issues,
        )

    target_text = target_prompt.strip()
    target_lower = target_text.lower()
    prompt_lower = original_prompt.lower()

    # Gate A: Structural & Malformed Target Checker
    target_well_formed = True
    no_placeholder_content = True
    for pat, desc in MALFORMED_TARGET_PATTERNS:
        if re.search(pat, target_text):
            target_well_formed = False
            no_placeholder_content = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MALFORMED_TARGET,
                    severity="ERROR",
                    message=f"Target contains malformed artifact: {desc} matching '{pat}'.",
                )
            )

    # Structured Empty Section Header Detection
    target_lines = target_text.splitlines()
    detected_headers: List[Tuple[int, str, str]] = []  # (line_idx, raw_header, norm_header)
    header_content_counts: Dict[int, int] = defaultdict(int)
    current_hdr_idx: Optional[int] = None

    for idx, l in enumerate(target_lines):
        norm_hdr = normalize_section_header(l)
        if norm_hdr:
            current_hdr_idx = idx
            detected_headers.append((idx, l.strip(), norm_hdr))
        elif l.strip() and current_hdr_idx is not None:
            header_content_counts[current_hdr_idx] += 1

    for idx, raw_hdr, norm_hdr in detected_headers:
        if header_content_counts[idx] == 0:
            target_well_formed = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MALFORMED_TARGET,
                    severity="ERROR",
                    message=f"Target contains an empty section header '{raw_hdr}' with no content.",
                )
            )

    # Useless placeholder values for unspecified details (e.g. "None", "N/A" as sole bullet text)
    placeholder_val_pat = r"(?im)^\s*(?:[-*•]|\d+\.)\s*(?:None|N/A|\?)\s*$"
    if re.search(placeholder_val_pat, target_text):
        no_placeholder_content = False
        issues.append(
            DatasetCurationIssue(
                type=CurationIssueType.PLACEHOLDER_CONTENT,
                severity="ERROR",
                message="Target teaches empty placeholder bullet value ('None' / 'N/A').",
            )
        )

    # Gate B: Meta-Commentary / HumanIntent Self-Referential Leakage
    no_internal_meta = True
    for pat in META_COMMENTARY_PATTERNS:
        if re.search(pat, target_text):
            no_internal_meta = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.META_CONTENT,
                    severity="ERROR",
                    message=f"Target contains forbidden internal meta-commentary matching '{pat}'.",
                )
            )

    # Gate C: Source Dependency Check
    source_required, source_present = _check_source_dependency(original_prompt)
    source_dependency_ok = True
    if source_required and not source_present:
        # Prompt requires source text/data/code that was not supplied
        source_dependency_ok = False
        issues.append(
            DatasetCurationIssue(
                type=CurationIssueType.SOURCE_MISSING,
                severity="ERROR",
                message="Task requires source artifact (code/data/text) not provided in input; target pretended source exists instead of clarifying.",
            )
        )

    # Gate D: Constraint Recall Gate
    constraint_recall_ok = True
    negative_constraints = _extract_negative_constraints(original_prompt, seed)
    for constr in negative_constraints:
        c_low = constr.lower()
        # Key tokens in the constraint
        key_tokens = [w for w in re.findall(r"\b[a-z0-9_\-]+\b", c_low) if w not in ["the", "a", "an", "and", "or", "in", "to", "of", "with", "avoid", "do", "not", "use", "don't", "without", "no", "only", "must"]]
        if not key_tokens:
            continue

        # Concept-specific check
        if "external dependencies" in c_low and not any(k in target_lower for k in ["external dependenc", "no external", "avoid external", "standard library"]):
            constraint_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_CONSTRAINT,
                    severity="ERROR",
                    message=f"Target omitted explicit constraint: '{constr}'.",
                )
            )
        elif "standard library" in c_low and not any(k in target_lower for k in ["standard library", "standard-library"]):
            constraint_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_CONSTRAINT,
                    severity="ERROR",
                    message=f"Target omitted explicit constraint: '{constr}'.",
                )
            )
        elif "mathematical formulas" in c_low and not any(k in target_lower for k in ["mathematical formula", "math formula", "formulas"]):
            constraint_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_CONSTRAINT,
                    severity="ERROR",
                    message=f"Target omitted explicit constraint: '{constr}'.",
                )
            )
        elif "medieval fantasy" in c_low and not any(k in target_lower for k in ["medieval fantasy", "generic sci-fi", "fantasy"]):
            constraint_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_CONSTRAINT,
                    severity="ERROR",
                    message=f"Target omitted explicit constraint: '{constr}'.",
                )
            )
        elif "offline" in c_low and "offline" not in target_lower:
            constraint_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_CONSTRAINT,
                    severity="ERROR",
                    message=f"Target omitted explicit offline constraint: '{constr}'.",
                )
            )
        elif "authentication" in c_low and not any(k in target_lower for k in ["without authentication", "no authentication", "unauthenticated", "no auth"]):
            constraint_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_CONSTRAINT,
                    severity="ERROR",
                    message=f"Target omitted explicit authentication constraint: '{constr}'.",
                )
            )
        else:
            # General constraint check: at least the most specific non-stopword token must appear
            # e.g. "do not use redis" -> "redis" must be mentioned in target with restriction
            substantive_tokens = [t for t in key_tokens if len(t) > 3]
            if substantive_tokens and not any(t in target_lower for t in substantive_tokens):
                constraint_recall_ok = False
                issues.append(
                    DatasetCurationIssue(
                        type=CurationIssueType.MISSING_CONSTRAINT,
                        severity="ERROR",
                        message=f"Target omitted explicit restrictive constraint: '{constr}'.",
                    )
                )

    # Gate E: Quantity Recall Gate
    WORD_TO_DIGIT = {
        "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "twenty": "20", "thirty": "30",
    }
    DIGIT_TO_WORD = {v: k for k, v in WORD_TO_DIGIT.items()}

    quantity_recall_ok = True
    quantities = _extract_quantities(original_prompt, seed)
    for q_spec in quantities:
        num_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty)\b", q_spec)
        unit_match = re.search(r"\b(bullets?|takeaways?|milestones?|ideas?|concepts?|days?|weeks?|words?|options?|examples?|loglines?)\b", q_spec)
        if num_match:
            num_val = num_match.group(0).lower()
            equiv_num = WORD_TO_DIGIT.get(num_val) or DIGIT_TO_WORD.get(num_val) or num_val
            has_num = (num_val in target_lower) or (equiv_num in target_lower)
            if not has_num:
                quantity_recall_ok = False
                issues.append(
                    DatasetCurationIssue(
                        type=CurationIssueType.QUANTITY_MISMATCH,
                        severity="ERROR",
                        message=f"Target omitted explicit quantity requirement: '{q_spec}'.",
                    )
                )

    # Gate F: Explicit Core Requirement Recall Gate
    requirement_recall_ok = True
    if seed and seed.core_requirements:
        missing_core_reqs: List[str] = []
        for req in seed.core_requirements:
            r_tokens = [w for w in re.findall(r"\b[a-z0-9_\-]+\b", req.lower()) if len(w) > 3 and w not in ["with", "from", "that", "this", "need", "make", "help"]]
            if not r_tokens:
                continue
            # If not reflected in original prompt, prompt is authoritative
            if not any(tok in prompt_lower for tok in r_tokens):
                continue
            # If none of the substantial tokens appear in target, check semantic equivalence
            if not any(tok in target_lower for tok in r_tokens):
                is_sem_match = False
                if semantic_matcher.is_available:
                    sim = semantic_matcher.similarity(req, target_text)
                    if sim >= 0.70:
                        is_sem_match = True
                if not is_sem_match:
                    missing_core_reqs.append(req)

        if missing_core_reqs:
            requirement_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_REQUIREMENT,
                    severity="ERROR",
                    message=f"Target omitted core requirements from prompt/seed: {missing_core_reqs}.",
                )
            )

    # Gate F2: Compound Deliverable Recall Gate
    compound_deliverables = _extract_compound_deliverables(original_prompt)
    if compound_deliverables:
        missing_deliverables = []
        for deliv in compound_deliverables:
            deliv_tokens = [w for w in re.findall(r"\b[a-z0-9_\-]+\b", deliv.lower()) if len(w) > 3 and w not in ["with", "from", "that", "this", "provide", "include"]]
            if deliv_tokens and not any(t in target_lower for t in deliv_tokens):
                missing_deliverables.append(deliv)
        if missing_deliverables:
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.COMPOUND_TASK_LOSS,
                    severity="ERROR",
                    message=f"Target omitted compound deliverables requested in prompt: {missing_deliverables}.",
                )
            )

    # Gate G: Task-Family & Role Drift Gate
    task_family_ok = True
    anchor = extract_intent_anchor(original_prompt)
    if anchor.action_family and anchor.is_strong:
        # Extract role framing from target: e.g. "Act as an expert data analyst"
        role_match = re.search(r"(?im)^\s*Act as an? (?:expert|experienced|senior)?\s*([a-z0-9_\-\s]+?)(?:\.|\n|$)", target_text)
        if role_match:
            role_title = role_match.group(1).lower().strip()
            # If user asked for RESEARCH and role says "data analyst", or user asked for PLANNING and role says "debugger"
            if anchor.action_family == ActionFamily.RESEARCH and any(w in role_title for w in ["data analyst", "data scientist"]):
                task_family_ok = False
                issues.append(
                    DatasetCurationIssue(
                        type=CurationIssueType.TASK_FAMILY_DRIFT,
                        severity="ERROR",
                        message=f"Target role line '{role_title}' drifts from requested RESEARCH task into DATA_ANALYSIS.",
                    )
                )
            elif anchor.action_family == ActionFamily.PLANNING and any(w in role_title for w in ["debugger", "troubleshooter"]):
                task_family_ok = False
                issues.append(
                    DatasetCurationIssue(
                        type=CurationIssueType.TASK_FAMILY_DRIFT,
                        severity="ERROR",
                        message=f"Target role line '{role_title}' drifts from requested PLANNING task into DEBUGGING.",
                    )
                )
            elif anchor.action_family == ActionFamily.WRITING and any(w in role_title for w in ["software developer", "debugger", "programmer"]):
                task_family_ok = False
                issues.append(
                    DatasetCurationIssue(
                        type=CurationIssueType.TASK_FAMILY_DRIFT,
                        severity="ERROR",
                        message=f"Target role line '{role_title}' drifts from requested WRITING task into SOFTWARE.",
                    )
                )

    # Gate H: Already-Good Input Policy
    if already_good:
        # If input was already good, target expansion should be restrained
        expansion = len(target_text) / max(len(original_prompt), 1)
        if expansion > 3.0:
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.EXCESSIVE_INFLATION,
                    severity="ERROR",
                    message=f"Already-good prompt underwent excessive rewriting/expansion ({expansion:.1f}x) without semantic necessity.",
                )
            )

    # Gate I: Unsupported-Requirement / Target-Invention Gate
    no_unsupported_requirement, ungrounded_issues = check_target_unsupported_requirements(
        original_prompt=original_prompt,
        target_prompt=target_text,
        seed=seed,
        normalized_intent=normalized_intent,
    )
    for issue_msg in ungrounded_issues:
        issues.append(
            DatasetCurationIssue(
                type=CurationIssueType.UNSUPPORTED_REQUIREMENT,
                severity="ERROR",
                message=issue_msg,
            )
        )

    # Gate J: Structured Target Substantive Quality Gate
    # Structured requirements (Core Requirements, Constraints, etc.) must not promote conversational filler
    no_non_substantive_requirement = True
    if target_text:
        substantive_ok, non_substantive_issues = check_structured_target_substantive_quality(target_text)
        if not substantive_ok:
            no_non_substantive_requirement = False
            for iss_msg in non_substantive_issues:
                issues.append(
                    DatasetCurationIssue(
                        type=CurationIssueType.NON_SUBSTANTIVE_REQUIREMENT,
                        severity="ERROR",
                        message=iss_msg,
                    )
                )

    # Compile final admission report
    is_acceptable = (
        target_well_formed
        and no_placeholder_content
        and no_internal_meta
        and source_dependency_ok
        and constraint_recall_ok
        and quantity_recall_ok
        and requirement_recall_ok
        and task_family_ok
        and no_unsupported_requirement
        and no_non_substantive_requirement
        and not any(iss.severity == "ERROR" for iss in issues)
    )

    return DatasetCurationReport(
        is_acceptable=is_acceptable,
        requirement_recall_ok=requirement_recall_ok,
        constraint_recall_ok=constraint_recall_ok,
        quantity_recall_ok=quantity_recall_ok,
        source_dependency_ok=source_dependency_ok,
        task_family_ok=task_family_ok,
        target_well_formed=target_well_formed,
        no_placeholder_content=no_placeholder_content,
        no_internal_meta_content=no_internal_meta,
        no_unsupported_requirement=no_unsupported_requirement,
        no_non_substantive_requirement=no_non_substantive_requirement,
        issues=issues,
    )
