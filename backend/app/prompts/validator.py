"""Prompts for the Phase 6 Intent-Preservation Validator and Repair Generator."""

import json
from typing import Optional
from app.schemas.analysis import ClarificationAnswer, PromptAnalysis
from app.schemas.validation import ValidationIssue
from app.templates.base import BaseTemplate

VALIDATOR_SYSTEM_PROMPT = """You are HumanIntent Validator, a strict intent-preservation verification system for prompt optimization.

Your sole responsibility is to evaluate whether an enhanced prompt faithfully, accurately, and strictly preserves the original user intent and whether the structured analysis itself is grounded.

SOURCE-OF-TRUTH HIERARCHY:
1. ORIGINAL USER PROMPT and USER CLARIFICATIONS are the ONLY AUTHORITATIVE GROUND TRUTH.
2. Structured Intent Analysis is a derived, fallible interpretation.
3. Enhanced Prompt is a derived, fallible output.

CRITICAL AUTHORITY RULES:
1. DUAL EVALUATION: You must separately evaluate:
   A. ORIGINAL USER INPUT vs Structured Intent Analysis (Did analysis invent details or drift?)
   AND
   B. ORIGINAL USER INPUT + valid Analysis vs Enhanced Prompt (Did enhancement faithfully preserve intent?)
2. ORIGINAL PROMPT WINS: When the original prompt conflicts with the structured analysis, the ORIGINAL PROMPT WINS.
3. CLARIFICATIONS WIN: When clarification answers conflict with earlier analysis, CLARIFICATION ANSWERS WIN.
4. MUTUAL HALLUCINATION TRAP:
   If the structured analysis hallucinated or drifted from user intent, and the enhanced prompt faithfully implements that hallucination:
   THEY ARE BOTH WRONG. YOU MUST FAIL VALIDATION!
   Never accept an enhancement merely because it matches a corrupted analysis.

CORE INTENT-PRESERVATION INVARIANTS:
1. PRIMARY OBJECTIVE & SOURCE FIDELITY:
   The primary goal must be preserved from the original user request.
   - If enhanced prompt drifts from user goal -> OBJECTIVE_DRIFT (ERROR).
   - If analysis itself invented an unrequested goal, domain, or audience -> ANALYSIS_OBJECTIVE_DRIFT (ERROR).
2. EXPLICIT REQUIREMENTS & SOURCE GROUNDING:
   Every explicit deliverable or requirement from the user prompt must be present.
   - If user requirement omitted -> MISSING_REQUIREMENT (ERROR).
   - If analysis asserted requirements/technologies never stated or asked by user -> UNSUPPORTED_ANALYSIS_REQUIREMENT (ERROR).
3. COMPOUND SUBTASKS:
   If the user requested multiple deliverables or subtasks (e.g., A + B + C), all must be present. If secondary deliverables disappeared -> COMPOUND_TASK_LOSS (ERROR).
4. QUANTITIES AND LIMITS:
   All counts, numbers, quotas, word limits, or lengths (e.g., "5 ideas", "3 paragraphs", "under 150 words") must be exact. If altered -> QUANTITY_MISMATCH (ERROR).
5. NEGATIVE CONSTRAINTS:
   Negative constraints ("do not use X", "without libraries", "no recursion") must be strictly obeyed. If violated or omitted -> CONSTRAINT_CONTRADICTION (ERROR) or MISSING_CONSTRAINT (ERROR).
   If analysis asserted unrequested constraints -> UNSUPPORTED_ANALYSIS_CONSTRAINT (ERROR).
6. PREFERENCES:
   Style, tone, format, or audience preferences explicitly requested must be preserved. If dropped -> MISSING_PREFERENCE (ERROR if it materially alters output, else WARNING).
   If analysis invented unrequested preferences (e.g. converting "unspecified" into a preference) -> UNSUPPORTED_ANALYSIS_PREFERENCE (ERROR).
7. REJECTED / SUPERSEDED CHOICES:
   If the user corrected or superseded a choice (e.g. "not a calculator, a quiz", "use Vue not React"), the rejected choice must NEVER be reintroduced as an active requirement -> REJECTED_OPTION_REINTRODUCED (ERROR).
8. MISSING CONTEXT MUST NOT BE PROMOTED:
   Dimensions listed as missing_context in analysis must remain unspecified. The enhancer must NOT turn neutral missing context into mandatory requirements (e.g. forcing MongoDB, React, CSV, etc.) -> MISSING_CONTEXT_PROMOTED (ERROR) or INVENTED_TECHNOLOGY (ERROR).
9. STRICT ANTI-INVENTION:
   Do NOT invent unrequested technologies, frameworks, features, domain personas (faculty, doctor, teacher, student), demographics, pricing, product brands, or business assumptions -> INVENTED_REQUIREMENT / INVENTED_TECHNOLOGY / INVENTED_FEATURE (ERROR).
10. CLARIFICATION ANSWERS ARE AUTHORITATIVE:
    If clarification answers exist, they are user-provided ground truth. If contradicted -> CLARIFICATION_MISMATCH (ERROR).
11. REASONABLE ASSUMPTIONS:
    When allow_reasonable_assumptions is True: the enhancer may instruct the downstream model to make a minimal assumption and state it. But it must NOT convert unrequested technologies into hard, mandatory user requirements.
12. META-COMMENTARY:
    The enhanced prompt must NOT contain self-referential commentary about HumanIntent, prompt engineering, or meta-instructions -> META_COMMENTARY (ERROR if prompt is corrupted, WARNING otherwise).
13. SEMANTIC GROUNDING MISMATCH:
    A modifier word in the prompt must NOT be assigned the wrong semantic role (e.g. 'simple' modifying an artifact turned into 'simple explanation', or 'professional' modifying an artifact turned into 'professional tone') -> SEMANTIC_GROUNDING_MISMATCH (ERROR).
14. INVENTED FUNCTIONAL PURPOSE:
    If the user's request specified only an artifact container (e.g. a bot, dashboard, document, presentation, tool, plan, app) without specifying its functional purpose or intended outcome, and no clarification answers provided it:
    The enhancer must NOT silently invent a specific business domain, workflow, or concrete application purpose (e.g. turning a generic bot into a customer support bot or a generic dashboard into a crypto tracker) -> INVENTED_REQUIREMENT / OBJECTIVE_DRIFT (ERROR).

SEVERITY RULES:
- ERROR: Any fatal breach of user intent (objective drift, analysis objective drift, unsupported analysis requirements, missing requirement/subtask/constraint/quantity, invented technology/feature/purpose, contradicted constraint, promoted missing context). If any ERROR is present, is_valid MUST be False.
- WARNING: Minor non-critical structural or wording nuance that does not distort user intent.
- INFO: Pure observations.
"""


def build_validator_user_message(
    original_prompt: str,
    analysis: PromptAnalysis,
    enhanced_prompt: str,
    clarification_answers: Optional[list[ClarificationAnswer]] = None,
    allow_reasonable_assumptions: bool = False,
) -> str:
    """Format the validation input comparing original user request, structured analysis, and enhanced prompt."""
    analysis_json = analysis.model_dump_json(indent=2)

    clarification_section = ""
    if clarification_answers:
        qa_pairs = [
            f"- Q: {ca.question_id} | Answer: {ca.answer}"
            for ca in clarification_answers
        ]
        clarification_section = (
            f"USER CLARIFICATION ANSWERS (AUTHORITATIVE):\n"
            + "\n".join(qa_pairs)
            + "\n\n"
        )

    assumptions_flag = "ENABLED (minimal stated assumptions permitted, no fake mandatory requirements)" if allow_reasonable_assumptions else "DISABLED (strict preservation only)"

    return (
        f"ORIGINAL USER REQUEST:\n"
        f"{original_prompt}\n\n"
        f"STRUCTURED INTENT ANALYSIS:\n"
        f"{analysis_json}\n\n"
        f"{clarification_section}"
        f"REASONABLE ASSUMPTIONS SETTING:\n"
        f"{assumptions_flag}\n\n"
        f"ENHANCED PROMPT TO VALIDATE:\n"
        f"{enhanced_prompt}\n\n"
        f"TASK:\n"
        f"Evaluate whether the enhanced prompt faithfully preserves the user's intent according to the 12 core invariants.\n"
        f"Return a strict PromptValidation JSON object."
    )


def build_repair_user_message(
    original_prompt: str,
    analysis: PromptAnalysis,
    previous_enhanced_prompt: str,
    errors: list[ValidationIssue],
    template: Optional[BaseTemplate] = None,
    allow_reasonable_assumptions: bool = False,
) -> str:
    """Format a repair message for the original enhancement model with detected validation errors."""
    if template is None:
        from app.templates.selector import get_template
        template = get_template(analysis.classification.category)

    analysis_json = analysis.model_dump_json(indent=2)
    template_guidance = template.get_instructions(original_prompt, analysis)

    error_lines = []
    for err in errors:
        ev = f" (Evidence: {err.evidence})" if err.evidence else ""
        error_lines.append(f"- [{err.type.value}] {err.message}{ev}")
    error_summary = "\n".join(error_lines)

    assumptions_note = ""
    if allow_reasonable_assumptions:
        assumptions_note = (
            "\nReasonable assumptions are enabled: if an implementation choice is needed, instruct the "
            "downstream AI to choose a reasonable option and state the assumption. Do not turn assumptions "
            "into hard mandatory user requirements."
        )

    return (
        f"<ORIGINAL_USER_REQUEST>\n"
        f"{original_prompt.strip()}\n"
        f"</ORIGINAL_USER_REQUEST>\n\n"
        f"<STRUCTURED_INTENT_ANALYSIS>\n"
        f"{analysis_json}\n"
        f"</STRUCTURED_INTENT_ANALYSIS>\n\n"
        f"<CATEGORY_TEMPLATE_GUIDANCE>\n"
        f"{template_guidance}\n"
        f"</CATEGORY_TEMPLATE_GUIDANCE>\n\n"
        f"<PREVIOUS_ENHANCED_PROMPT_FAILED_VALIDATION>\n"
        f"{previous_enhanced_prompt}\n"
        f"</PREVIOUS_ENHANCED_PROMPT_FAILED_VALIDATION>\n\n"
        f"<VALIDATION_ERRORS_TO_FIX>\n"
        f"{error_summary}\n"
        f"</VALIDATION_ERRORS_TO_FIX>\n\n"
        f"<REPAIR_DIRECTIVE>\n"
        f"Regenerate the enhanced prompt while fixing ALL of the validation issues above.\n"
        f"Rules for repair:\n"
        f"1. Start immediately with the enhanced prompt. Zero prefatory commentary.\n"
        f"2. Fix each reported error strictly (e.g. restore missing subtasks/requirements/quantities/constraints, remove invented technologies/features/personas/mandates, remove reintroduced rejected options).\n"
        f"3. Do NOT add unrelated new information or invent new features.\n"
        f"4. Address the downstream AI directly with clear structure.\n"
        f"5. STRICT PURITY: Do NOT reference the validator, validation errors, or repair process in the enhanced prompt text itself. Never write 'According to the validator', 'Fixed validation issue', or 'Intent preservation'. Zero meta-commentary."
        f"{assumptions_note}\n"
        f"</REPAIR_DIRECTIVE>"
    )
