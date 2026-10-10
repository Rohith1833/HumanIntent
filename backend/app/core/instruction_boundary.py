"""Internal instruction and control-text boundary enforcement for HumanIntent.

Core Invariant:
INTERNAL CONTROL INSTRUCTIONS ARE NEVER USER SEMANTIC DATA.

This module provides centralized control-plane vs data-plane separation:
1. Dynamic harvesting of internal control headings, step names, and instruction snippets
   from prompt modules (analyzer, enhancer, validator, scorer).
2. General pattern detection for instruction-like markers, workflow steps, schema terms,
   and rubric text.
3. Source-based false-positive protection: if suspicious wording was genuinely authored
   by the user in the authoritative request/clarifications, it is permitted.
4. Comprehensive contamination scanning for all PromptAnalysis fields and enhanced prompts.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from app.schemas.analysis import ClarificationAnswer, PromptAnalysis

logger = logging.getLogger("humanintent.instruction_boundary")


def normalize_text(text: str) -> str:
    """Normalize text for fingerprint comparison by stripping punctuation and collapsing whitespace."""
    if not text:
        return ""
    # Lowercase, replace non-alphanumeric with spaces, collapse spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
    return " ".join(cleaned.split())


# Static core control fingerprints (headings, step names, rubric names, schema concepts)
CORE_CONTROL_PHRASES: Set[str] = {
    # Analyzer workflow & step labels
    "step 1 task classification",
    "step 2 extract structured information",
    "step 3 handling human self corrections",
    "step 4 ambiguity high impact clarification assessment",
    "step 5 re analyzing with user clarifications",
    "step 6 source grounding provenance",
    "task classification",
    "extract structured information",
    "source grounding provenance",
    "source grounding",
    "perform source grounding",
    "assess ambiguity",
    "high impact clarification assessment",
    "ambiguity assessment",
    "clarification assessment",
    "empty array contract",
    "field overlap guard",
    "tie breaking rules",
    "confidence guidelines",
    "json schema contract",
    "required output format",
    "system instructions",
    "system prompt",
    "analyzer system prompt",
    "rule of neutrality",
    "modality rule",

    # Enhancer & template concepts
    "humanintent",
    "intent preserving adaptive prompt enhancement system",
    "intent preserving prompt optimizer",
    "intent preservation",
    "core product principle",
    "strict bans on unrequested expansions",
    "strict output contract",
    "adaptive structure category templates",
    "category template guidance",
    "global invariants",
    "direct downstream language",
    "length discipline",
    "enhancement directive",
    "reasonable assumptions directive",

    # Validator concepts
    "validator system prompt",
    "intent preservation validator",
    "source of truth hierarchy",
    "critical authority rules",
    "dual evaluation",
    "original prompt wins",
    "clarifications win",
    "mutual hallucination trap",
    "core intent preservation invariants",
    "validation errors to fix",
    "repair directive",
    "validation issue",
    "validation error",
    "intent preservation score",
    "according to the validator",
    "objective drift",
    "missing requirement",
    "invented technology",
    "invented feature",
    "rejected option reintroduced",
    "missing context promoted",
    "meta commentary",
    "semantic grounding mismatch",

    # Scorer rubrics & dimensions
    "prompt quality scorer",
    "scorer system prompt",
    "explicit dimension rubrics",
    "semantic dimensions output",
    "inflation ratio",
    "inflation risk",
    "output definition rubric",
    "clarity rubric",
    "specificity rubric",
    "concision rubric",
    "structure rubric",

    # Schema entity names
    "promptanalysis",
    "promptvalidation",
    "promptqualityreport",
    "promptclassification",
    "clarificationanswer",
    "clarificationquestion",
    "groundingevidence",
    "analysisgrounding",
    "needs clarification",
    "target dimension",
    "is grounded",
    "unsupported items",
}

# Compiled general instruction patterns
GENERAL_INSTRUCTION_PATTERNS: List[Tuple[re.Pattern, str]] = [
    # Step markers: "STEP 1:", "STEP 2 -", "STEP 4", etc.
    (re.compile(r"\bstep\s*\d+\b", re.IGNORECASE), "numbered step label"),
    (re.compile(r"\bphase\s*\d+\b", re.IGNORECASE), "numbered phase label"),

    # System/instruction labels: "SYSTEM:", "INSTRUCTIONS:", "SYSTEM PROMPT"
    (re.compile(r"\b(?:system|instructions?|directive|rubric)\s*:", re.IGNORECASE), "system instruction label"),
    (re.compile(r"\b(?:system\s+prompt|analyzer\s+instructions?|validator\s+instructions?|system\s+instructions?)\b", re.IGNORECASE), "system prompt reference"),

    # Procedural workflow commands
    (re.compile(r"\b(?:perform|check|verify)\s+source\s+grounding\b", re.IGNORECASE), "workflow procedure"),
    (re.compile(r"\b(?:assess|evaluate|recalculate)\s+(?:ambiguity|clarification)\b", re.IGNORECASE), "workflow procedure"),
    (re.compile(r"\b(?:field[\s-]overlap\s+guard|empty\s+array\s+contract)\b", re.IGNORECASE), "internal pipeline contract"),
    (re.compile(r"\bmutual\s+hallucination\s+trap\b", re.IGNORECASE), "internal validator concept"),

    # Schema & output formatting instructions
    (re.compile(r"\b(?:output\s+must\s+be\s+valid\s+json|return\s+(?:only\s+)?(?:the\s+)?(?:raw\s+)?valid\s+json)\b", re.IGNORECASE), "schema instruction"),
    (re.compile(r"\b(?:return|output)\s+(?:only\s+)?(?:the\s+)?(?:raw\s+)?(?:valid\s+)?json\s+(?:object|schema|contract)\b", re.IGNORECASE), "schema instruction"),
    (re.compile(r"\b(?:json\s+schema|schema\s+contract|required\s+schema)\b", re.IGNORECASE), "schema instruction"),
    (re.compile(r"\bprovided\s+by\s+the\s+schema\b", re.IGNORECASE), "schema instruction"),
    (re.compile(r"\brequired\s+by\s+the\s+schema\b", re.IGNORECASE), "schema instruction"),

    # Validator repair references
    (re.compile(r"\b(?:according\s+to|as\s+stated\s+by)\s+(?:the\s+)?validator\b", re.IGNORECASE), "validator meta-reference"),
    (re.compile(r"\bfix\s+validation\s+(?:issue|error)s?\b", re.IGNORECASE), "validator meta-reference"),
    (re.compile(r"\bensure\s+intent\s+preservation\b", re.IGNORECASE), "system meta-reference"),
    (re.compile(r"\bintent[\s-]preserving\s+prompt\s+optimizer\b", re.IGNORECASE), "system branding"),
    (re.compile(r"\bhumanintent\b", re.IGNORECASE), "system name"),
]


def is_in_authoritative_source(candidate: str, user_corpus: str) -> bool:
    """
    Check if candidate text was genuinely present in authoritative user input.
    Provides source-based false-positive protection: if the user explicitly wrote
    'Step 1' or 'Step 2', that content is authoritative and permitted.
    """
    if not user_corpus or not candidate:
        return False
    cand_norm = normalize_text(candidate)
    corpus_norm = normalize_text(user_corpus)
    if not cand_norm:
        return False

    # Exact normalized substring match
    if cand_norm in corpus_norm:
        return True

    # If multi-token, check whether all substantive tokens appear in user corpus in proximity
    tokens = [t for t in cand_norm.split() if len(t) > 2]
    if len(tokens) >= 2 and all(tok in corpus_norm for tok in tokens):
        # Verify tokens appear in sequence or close proximity in user corpus
        cand_pat = r"\b" + r"\b.*?\b".join(re.escape(t) for t in tokens) + r"\b"
        if re.search(cand_pat, corpus_norm):
            return True

    return False


def detect_control_text(
    text: str,
    user_corpus: str = "",
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate whether a text string contains internal system instructions, step labels,
    schema terminology, or rubric language that is NOT grounded in authoritative user input.

    Returns:
        (is_contaminated, reason_description)
    """
    if not text or not text.strip():
        return False, None

    text_stripped = text.strip()
    norm_text = normalize_text(text_stripped)

    # 1. Check general regex instruction patterns
    for pat, desc in GENERAL_INSTRUCTION_PATTERNS:
        match = pat.search(text_stripped)
        if match:
            matched_snippet = match.group(0)
            # If the user themselves authored this snippet, allow it
            if is_in_authoritative_source(matched_snippet, user_corpus):
                continue
            return True, f"contains {desc}: '{matched_snippet}'"

    # 2. Check exact normalized phrase matches against known control fingerprints
    for phrase in CORE_CONTROL_PHRASES:
        if phrase in norm_text:
            if is_in_authoritative_source(phrase, user_corpus):
                continue
            return True, f"matches internal control phrase: '{phrase}'"

    return False, None


def build_user_corpus(
    raw_prompt: str,
    clarification_answers: Optional[List[ClarificationAnswer]] = None,
) -> str:
    """Build a unified corpus of all authoritative user text (original prompt + clarification answers)."""
    corpus = raw_prompt or ""
    if clarification_answers:
        for ca in clarification_answers:
            if ca.answer:
                corpus += " " + ca.answer
    return corpus


def scan_analysis_contamination(
    analysis: PromptAnalysis,
    raw_prompt: str,
    clarification_answers: Optional[List[ClarificationAnswer]] = None,
) -> List[str]:
    """
    Exhaustively scan all fields of a PromptAnalysis for internal control-text contamination.

    Fields protected:
    - objective
    - explicit_requirements
    - constraints
    - preferences
    - missing_context
    - ambiguity.reasons
    - ambiguity.questions (question, reason, target_dimension)

    Returns a list of human-readable contamination descriptions.
    """
    user_corpus = build_user_corpus(raw_prompt, clarification_answers)
    contamination_issues: List[str] = []

    # 1. Objective
    if analysis.objective:
        is_bad, reason = detect_control_text(analysis.objective, user_corpus)
        if is_bad:
            contamination_issues.append(f"objective {reason}")

    # 2. Explicit requirements
    for req in analysis.explicit_requirements:
        is_bad, reason = detect_control_text(req, user_corpus)
        if is_bad:
            contamination_issues.append(f"explicit requirement {reason}")

    # 3. Constraints
    for c in analysis.constraints:
        is_bad, reason = detect_control_text(c, user_corpus)
        if is_bad:
            contamination_issues.append(f"constraint {reason}")

    # 4. Preferences
    for p in analysis.preferences:
        is_bad, reason = detect_control_text(p, user_corpus)
        if is_bad:
            contamination_issues.append(f"preference {reason}")

    # 5. Missing context
    # missing_context represents unanswered dimensions; it must NEVER contain internal instructions
    for mc in analysis.missing_context:
        is_bad, reason = detect_control_text(mc, user_corpus)
        if is_bad:
            contamination_issues.append(f"missing context {reason}")

    # 6. Ambiguity reasons
    if analysis.ambiguity and analysis.ambiguity.reasons:
        for r in analysis.ambiguity.reasons:
            is_bad, reason = detect_control_text(r, user_corpus)
            if is_bad:
                contamination_issues.append(f"ambiguity reason {reason}")

    # 7. Clarification questions
    if analysis.ambiguity and analysis.ambiguity.questions:
        for q in analysis.ambiguity.questions:
            if q.question:
                is_bad, reason = detect_control_text(q.question, user_corpus)
                if is_bad:
                    contamination_issues.append(f"clarification question {reason}")
            if q.reason:
                is_bad, reason = detect_control_text(q.reason, user_corpus)
                if is_bad:
                    contamination_issues.append(f"clarification reason {reason}")
            if q.target_dimension:
                is_bad, reason = detect_control_text(q.target_dimension, user_corpus)
                if is_bad:
                    contamination_issues.append(f"clarification target dimension {reason}")

    return contamination_issues


def scan_enhanced_prompt_contamination(
    enhanced_prompt: str,
    raw_prompt: str,
    clarification_answers: Optional[List[ClarificationAnswer]] = None,
) -> List[str]:
    """
    Scan an enhanced prompt for internal system instructions, validator concepts,
    template directives, rubric language, or HumanIntent implementation references
    that were NOT present in the authoritative user input.
    """
    user_corpus = build_user_corpus(raw_prompt, clarification_answers)
    lines = enhanced_prompt.split("\n")
    issues: List[str] = []

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        is_bad, reason = detect_control_text(line_clean, user_corpus)
        if is_bad:
            issues.append(f"enhanced prompt line '{line_clean[:80]}' {reason}")

    return issues
