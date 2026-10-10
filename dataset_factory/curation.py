import logging
import re
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
]

# Source-dependent action keywords and objects that necessitate external input text
SOURCE_DEPENDENT_ACTIONS = [
    r"(?i)\b(?:summarize|summarise|condense|abstract)\s+(?:this|the\s+following|the\s+provided)\s+(?:report|paper|article|document|text|study|memo)\b",
    r"(?i)\b(?:debug|diagnose|fix|review|refactor)\s+(?:this|the\s+following|the\s+provided)\s+(?:code|script|function|snippet|program|loop|crash)\b",
    r"(?i)\b(?:analyze|analyse|parse|inspect)\s+(?:this|the\s+following|the\s+provided)\s+(?:dataset|data|csv|json|table|sample|numbers|metrics|log)\b",
    r"(?i)\b(?:translate|rewrite|proofread|edit)\s+(?:this|the\s+following|the\s+provided)\s+(?:paragraph|text|sentence|passage)\b",
]

# Quantities that must be recalled if specified
QUANTITY_PATTERN = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty)\s+"
    r"(bullets?|takeaways?|milestones?|ideas?|concepts?|days?|weeks?|words?|options?|examples?)\b",
    re.IGNORECASE,
)


def _check_source_dependency(original_prompt: str) -> Tuple[bool, bool]:
    """
    Determine if task is logically source-dependent and whether source material was supplied.
    Returns: (source_required, source_present)
    """
    prompt_lower = original_prompt.lower()

    source_required = False
    for pat in SOURCE_DEPENDENT_ACTIONS:
        if re.search(pat, prompt_lower):
            source_required = True
            break

    if not source_required:
        return False, True

    # Check if prompt actually contains the referenced source material
    # Code presence: fenced code blocks, multi-line indented code, def/function keywords
    has_code = "```" in original_prompt or "\n    " in original_prompt or "def " in original_prompt or "async def " in original_prompt
    # Data presence: csv commas in numbers, table pipes, json brackets
    has_data = ("|" in original_prompt and "\n" in original_prompt) or ("[" in original_prompt and "]" in original_prompt and "," in original_prompt) or bool(re.search(r"\d+[\s,]+\d+[\s,]+\d+", original_prompt))
    # Document presence: substantial substantive text (> 120 words)
    has_long_doc = len(original_prompt.split()) > 100

    source_present = has_code or has_data or has_long_doc
    return True, source_present


def _extract_negative_constraints(original_prompt: str, seed: Optional[SeedIntent] = None) -> List[str]:
    """Extract negative or restrictive constraints from prompt and seed."""
    constraints: List[str] = []

    if seed and seed.optional_constraints:
        for c in seed.optional_constraints:
            c_low = c.lower()
            if any(neg in c_low for neg in ["avoid", "no ", "without", "only", "under", "never", "do not"]):
                constraints.append(c)

    # Regex extractions from raw prompt
    neg_patterns = [
        r"(?i)\b(?:avoid|avoiding|do not use|don't use|without|no)\s+([a-z0-9_\-\s]{3,35})(?:[,.]|$)",
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
    """Extract quantitative bounds (e.g., '3 bullets', 'under 250 words', '10 concepts')."""
    quantities: List[str] = []
    for match in QUANTITY_PATTERN.finditer(original_prompt):
        quantities.append(match.group(0).lower().strip())

    if seed:
        for req in seed.core_requirements + seed.optional_constraints:
            for match in QUANTITY_PATTERN.finditer(req):
                quantities.append(match.group(0).lower().strip())

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

    # Empty Section Headers (e.g. "UNSPECIFIED DETAILS:\n\nOUTPUT")
    empty_section_pat = r"(?im)^\s*(?:\*{1,2}|#{1,4}\s*)?(?:UNSPECIFIED DETAILS|CORE REQUIREMENTS|OUTPUT STRUCTURE)(?:\*{1,2})?:?\s*\n+(?=(?:(?:\*{1,2}|#{1,4}\s*)?[A-Z]{3,}|\Z))"
    if re.search(empty_section_pat, target_text):
        target_well_formed = False
        issues.append(
            DatasetCurationIssue(
                type=CurationIssueType.MALFORMED_TARGET,
                severity="ERROR",
                message="Target contains an empty section header with no content.",
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
    quantity_recall_ok = True
    quantities = _extract_quantities(original_prompt, seed)
    for q_spec in quantities:
        # e.g., '3 bullets', 'under 250 words', '10 concepts'
        num_match = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|twenty|thirty)\b", q_spec)
        unit_match = re.search(r"\b(bullets?|takeaways?|milestones?|ideas?|concepts?|days?|weeks?|words?|options?|examples?)\b", q_spec)
        if num_match and unit_match:
            num_val = num_match.group(0)
            unit_val = unit_match.group(0)
            # Verify that either the number or the unit with restriction is preserved
            if num_val not in target_lower and unit_val not in target_lower:
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
            # If none of the substantial tokens from the core requirement appear in target
            if not any(tok in target_lower for tok in r_tokens):
                missing_core_reqs.append(req)

        if missing_core_reqs:
            requirement_recall_ok = False
            issues.append(
                DatasetCurationIssue(
                    type=CurationIssueType.MISSING_REQUIREMENT,
                    severity="ERROR",
                    message=f"Target omitted core requirements from seed: {missing_core_reqs}.",
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
        no_unsupported_requirement=True,
        issues=issues,
    )
