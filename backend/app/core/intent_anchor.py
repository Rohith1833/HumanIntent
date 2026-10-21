import logging
import re
from typing import Optional

from app.schemas.analysis import (
    ActionFamily,
    ClarificationAnswer,
    IntentAnchor,
    PromptAnalysis,
    PromptCategory,
)

logger = logging.getLogger("humanintent.intent_anchor")

# Central 1-to-1 canonical mapping between ActionFamily and PromptCategory
ACTION_FAMILY_TO_CATEGORY: dict[ActionFamily, PromptCategory] = {
    ActionFamily.SOFTWARE_CREATION: PromptCategory.CODING,
    ActionFamily.SOFTWARE_DEBUGGING: PromptCategory.DEBUGGING,
    ActionFamily.WRITING: PromptCategory.WRITING,
    ActionFamily.EDUCATION_EXPLANATION: PromptCategory.EDUCATION,
    ActionFamily.RESEARCH: PromptCategory.RESEARCH,
    ActionFamily.PLANNING: PromptCategory.PLANNING,
    ActionFamily.DATA_ANALYSIS: PromptCategory.DATA_ANALYSIS,
    ActionFamily.CREATIVE_IDEATION: PromptCategory.CREATIVE,
    ActionFamily.IMAGE_CREATION: PromptCategory.IMAGE_GENERATION,
    ActionFamily.SUMMARIZATION: PromptCategory.SUMMARIZATION,
    ActionFamily.GENERAL: PromptCategory.GENERAL,
}

CATEGORY_TO_ACTION_FAMILY: dict[PromptCategory, ActionFamily] = {
    cat: fam for fam, cat in ACTION_FAMILY_TO_CATEGORY.items()
}

# Software-specific technical dimension terms that are incoherent for non-software tasks
SOFTWARE_ONLY_DIMENSIONS = {
    "development platform or framework",
    "development platform",
    "framework",
}

# Academic/research-specific dimension terms that are incoherent for standard coding/writing tasks
RESEARCH_ONLY_DIMENSIONS = {
    "citation style",
    "research methodology",
    "academic sources",
    "peer review standards",
}


def extract_intent_anchor(
    raw_prompt: str,
    clarification_answers: Optional[list[ClarificationAnswer]] = None,
) -> IntentAnchor:
    """
    Extract a lightweight internal representation of the primary requested action.
    Determines action family from what the user is asking the AI to do, NOT the topic vocabulary.
    Distinguishes STRONG (unambiguous imperative verb + target) vs WEAK anchors.
    """
    prompt_lower = raw_prompt.lower().strip()
    corpus = prompt_lower
    if clarification_answers:
        corpus += " " + " ".join(ca.answer.lower() for ca in clarification_answers if ca.answer)

    # 1. IMAGE CREATION (High specificity: visual generation deliverable)
    if (
        re.search(r"\b(?:generate|create|draw|make|render)\s+(?:an?\s+)?(?:[a-z\-]+\s+){0,3}(?:image|illustration|picture|photo|graphic|portrait|artwork)\b", prompt_lower)
        or re.search(r"\b(?:midjourney|dall-e|stable\s+diffusion)\s+prompt\b", prompt_lower)
        or re.search(r"^(?:a\s+|an\s+)?(?:photorealistic|oil\s+on\s+canvas|vector\s+minimalist|hyperdetailed|watercolor|charcoal|pop\s+art|stained\s+glass|gouache|low\s+poly|macro\s+photo|isometric\s+3d|vintage\s+botanical|pencil\s+sketch|architectural\s+exterior|cinematic\s+wide|lush\s+oil|high-fashion|dark\s+fantasy|medieval\s+illuminated|abstract\s+acrylic|national\s+geographic|retro\s+1970s|black\s+and\s+white\s+street)\b", prompt_lower)
    ):
        return IntentAnchor(
            action_family=ActionFamily.IMAGE_CREATION,
            primary_action="create image or image prompt",
            target_or_deliverable="image deliverable",
            confidence=0.95,
            is_strong=True,
        )

    # 2. SUMMARIZATION (Explicit summarization or information extraction from provided material)
    is_summarization_action = bool(
        re.search(r"^(?:please\s+)?(?:summarize|condense|digest|synopsis|tldr|give\s+me\s+a\s+(?:brief\s+|short\s+|one-paragraph\s+|concise\s+)?summary|provide\s+(?:a\s+)?(?:[a-z0-9\-]+\s+){0,3}(?:tldr|summary|synopsis|recap))\b", prompt_lower)
        or re.search(r"\b(?:extract\s+(?:all\s+|the\s+)?[a-z0-9_\-\s]+\s+from\s+(?:this|these|the)\s+)\b", prompt_lower)
        or re.search(r"\b(?:extract\s+the\s+key\s+action\s+items|extract\s+decisions\s+made|tldr\s+this)\b", prompt_lower)
        or (
            re.search(r"\b(?:summarize|condense|digest|synopsis|tldr)\b", prompt_lower)
            and not re.search(r"\bsummarize\s+(?:them|it|the\s+results|the\s+findings)?\s+in\s+a\s+table\b", prompt_lower)
        )
    )
    if is_summarization_action:
        return IntentAnchor(
            action_family=ActionFamily.SUMMARIZATION,
            primary_action="summarize or condense material",
            target_or_deliverable="summary deliverable",
            confidence=0.95,
            is_strong=True,
        )

    # 3. WRITING (Composing communication or prose content: email, letter, message, memo, post, article)
    is_text_writing = bool(
        re.search(r"\b(?:draft|compose|write|rewrite)\s+(?:an?\s+|the\s+|\d+\s+)?(?:[a-z\-]+\s+){0,3}(?:email|letter|message|memo|note|announcement|newsletter|blog\s+post|post|article|essay|speech|press\s+release|cover\s+letter|copy|blurb|statement\s+of\s+work)\b", prompt_lower)
        or re.search(r"\b(?:write|draft)\s+(?:a\s+)?(?:polite|formal|casual|friendly|professional)\s+(?:message|reply|response|note|email|letter)\b", prompt_lower)
    )
    if is_text_writing:
        return IntentAnchor(
            action_family=ActionFamily.WRITING,
            primary_action="draft communication or content",
            target_or_deliverable="written communication deliverable",
            confidence=0.95,
            is_strong=True,
        )

    # 4. SOFTWARE DEBUGGING (High specificity: fixing broken software/errors)
    if (
        re.search(r"\b(?:fix|debug|troubleshoot|diagnose|resolve)\s+(?:the\s+|this\s+|my\s+|an?\s+)?(?:bug|error|issue|crash|exception|deadlock|failing\s+test|problem\s+in\s+(?:the\s+)?code)\b", prompt_lower)
        or re.search(r"\bwhy\s+(?:is|does)\s+(?:this|my)\s+(?:[a-z\-]+\s+){0,3}(?:code|function|app|query|test|script|build|worker|process|job|pipeline)\s+(?:failing|crashing|broken|throwing|not\s+working|deadlock|deadlocking|timeout|freeze|freezes|hang|hangs)\b", prompt_lower)
        or (re.search(r"\b(?:fix|debug|troubleshoot|diagnose)\b", prompt_lower) and any(w in prompt_lower for w in ["crash", "error", "bug", "assertion", "exception", "deadlock", "fail", "fails", "failing"]))
        or re.search(r"\b(?:stack\s*trace|traceback|syntaxerror|typeerror|nullpointerexception|panic:)\b", prompt_lower)
    ):
        return IntentAnchor(
            action_family=ActionFamily.SOFTWARE_DEBUGGING,
            primary_action="fix or diagnose broken software",
            target_or_deliverable="code fix or diagnosis",
            confidence=0.95,
            is_strong=True,
        )

    # 5. RESEARCH (Investigation, literature survey, evidence synthesis, comparative approaches)
    is_research = bool(
        re.search(r"\b(?:research|investigate|literature\s+review|synthesize\s+(?:the\s+)?evidence|survey\s+of|find\s+studies\s+on)\b", prompt_lower)
        or re.search(r"\bcompare\s+(?:the\s+)?(?:main\s+)?(?:approaches|methods|techniques|options)\b", prompt_lower)
    )
    if is_research and not re.search(r"\b(?:fix|debug|crash)\b", prompt_lower):
        return IntentAnchor(
            action_family=ActionFamily.RESEARCH,
            primary_action="investigate and synthesize research evidence",
            target_or_deliverable="research synthesis",
            confidence=0.95,
            is_strong=True,
        )

    # 6. SOFTWARE CREATION (Executable software, functions, scripts, web apps, endpoints, queries)
    # Check verbs: create, build, implement, code, develop, write, make function/script/app/webpage
    is_creative_script = bool(re.search(r"\b(?:scene\s+script|screenplay|movie\s+script|theatrical\s+script|dialogue\s+script)\b", prompt_lower))
    is_software_code = bool(
        not is_creative_script
        and (
            re.search(r"\b(?:create|build|implement|code|develop|write|generate|make)\s+(?:an?\s+|the\s+)?(?:[a-z\-]+\s+){0,5}(?:function|script|class|module|endpoint|api|algorithm|program|app|application|web\s+app|website|webpage|page|component|service|microservice|server|database\s+schema|sql\s+query|regex|parser|utility|cli\s+tool|tool|limiter|handler|middleware)\b", prompt_lower)
            or re.search(r"\b(?:write|create|implement|make|build)\s+(?:a\s+|an\s+)?(?:go|python|javascript|typescript|rust|java|c\+\+|c#|ruby|sql|bash|html)\s+(?:function|script|program|code|routine|snippet|endpoint|microservice|server|utility|tool|app|limiter|handler)\b", prompt_lower)
            or re.search(r"\b(?:implement|build|code)\s+.*?\bin\s+(?:python|javascript|typescript|go|rust|java|c\+\+|c#|ruby|sql|bash|html)\b", prompt_lower)
            or re.search(r"\b(?:function|routine|script)\s+that\s+(?:takes|returns|calculates|handles|converts|processes|parses)\b", prompt_lower)
            or re.search(r"\b(?:build|create|code|make)\s+(?:an?\s+)?(?:interactive\s+)?(?:calculator|dashboard|quiz|converter|tool|portal)\b", prompt_lower)
            or re.search(r"\b(?:quiz|calculator|tool|app)\s+(?:for|on)\s+(?:my\s+)?(?:website|site|page)\b", prompt_lower)
            or re.search(r"\b(?:thing\s+for\s+my\s+website|page\s+on\s+my\s+website)\b", prompt_lower)
        )
    )
    if is_software_code:
        return IntentAnchor(
            action_family=ActionFamily.SOFTWARE_CREATION,
            primary_action="implement executable software",
            target_or_deliverable="software deliverable",
            confidence=0.95,
            is_strong=True,
        )
    is_text_writing = bool(
        re.search(r"\b(?:draft|compose|write|rewrite)\s+(?:an?\s+|the\s+|\d+\s+)?(?:[a-z\-]+\s+){0,3}(?:email|letter|message|memo|note|announcement|newsletter|blog\s+post|post|article|essay|speech|press\s+release|cover\s+letter|copy|blurb)\b", prompt_lower)
        or re.search(r"\b(?:write|draft)\s+(?:a\s+)?(?:polite|formal|casual|friendly|professional)\s+(?:message|reply|response|note|email|letter)\b", prompt_lower)
    )
    if is_text_writing:
        return IntentAnchor(
            action_family=ActionFamily.WRITING,
            primary_action="draft communication or content",
            target_or_deliverable="written communication deliverable",
            confidence=0.95,
            is_strong=True,
        )

    # 7. EDUCATION / EXPLANATION (Conceptual questions, explaining how/why, differences, teaching)
    is_education = bool(
        re.search(r"\b(?:explain|teach|help\s+me\s+understand|break\s+down)\s+(?:to\s+me\s+)?(?:the\s+concept\s+of\s+|how\s+|why\s+|what\s+)?", prompt_lower)
        and re.search(r"\b(?:explain|teach|difference\s+between|how\s+does|why\s+does|what\s+is\s+the\s+difference)\b", prompt_lower)
    ) or bool(
        re.search(r"\bwhat\s+(?:is\s+the\s+difference\s+between|makes\s+[a-zA-Z0-9_\-\s]+\s+different\s+from)\b", prompt_lower)
        or re.search(r"\bhow\s+does\s+[a-zA-Z0-9_\-\s]+\s+work\??", prompt_lower)
        or re.search(r"\bwhy\s+does\s+[a-zA-Z0-9_\-\s]+\s+(?:happen|occur|behave)\??", prompt_lower)
        or re.search(r"\bexplain\s+(?:how|why|what|the\s+difference|the\s+concept)\b", prompt_lower)
        or re.search(r"\bexplain\s+[a-zA-Z0-9_\-]+\s+simply\b", prompt_lower)
        or re.search(r"\bhelp\s+me\s+understand\b", prompt_lower)
    )
    if is_education:
        return IntentAnchor(
            action_family=ActionFamily.EDUCATION_EXPLANATION,
            primary_action="explain conceptual topic or differences",
            target_or_deliverable="conceptual explanation",
            confidence=0.95,
            is_strong=True,
        )

    # 8. PLANNING (Roadmaps, schedules, multi-day plans, study plans, curricula)
    is_planning = bool(
        re.search(r"\b(?:plan|schedule|roadmap|curriculum|timeline|itinerary)\b", prompt_lower)
        and any(w in prompt_lower for w in ["plan", "schedule", "roadmap", "curriculum", "timeline", "rollout", "week", "day", "month", "milestone", "phases"])
        and any(w in prompt_lower for w in ["create", "draft", "make", "develop", "provide", "outline", "organize"])
    ) or bool(
        re.search(r"\b(?:plan\s+a\s+|create\s+a\s+(?:study\s+)?schedule|rollout\s+plan|launch\s+plan|project\s+plan)\b", prompt_lower)
    )
    if is_planning:
        return IntentAnchor(
            action_family=ActionFamily.PLANNING,
            primary_action="plan or schedule roadmap",
            target_or_deliverable="plan or schedule deliverable",
            confidence=0.90,
            is_strong=True,
        )

    # 9. DATA ANALYSIS (Analyzing supplied or referenced datasets/metrics)
    is_data_analysis = bool(
        re.search(r"\b(?:analyze|analyse)\s+(?:the\s+|this\s+|my\s+)?(?:[a-z\-]+\s+){0,3}(?:dataset|data|metrics|numbers|table|statistics|csv|records|measurements)\b", prompt_lower)
        or re.search(r"\b(?:find\s+trends\s+in|statistical\s+analysis\s+of|correlation\s+between)\b", prompt_lower)
        or re.search(r"\b(?:calculate|compute)\s+(?:the\s+)?(?:mean|median|mode|average|standard\s+deviation|variance|p95|p99|latency|statistics|metrics)\b", prompt_lower)
    )
    if is_data_analysis:
        return IntentAnchor(
            action_family=ActionFamily.DATA_ANALYSIS,
            primary_action="analyze supplied or referenced data",
            target_or_deliverable="data analysis deliverable",
            confidence=0.90,
            is_strong=True,
        )

    # 10. CREATIVE IDEATION & FICTION (Brainstorming concepts, names, creative stories)
    is_brainstorm = bool(
        re.search(r"\b(?:brainstorm|ideate|come\s+up\s+with\s+(?:\d+\s+)?ideas|generate\s+(?:\d+\s+)?(?:creative\s+)?ideas|creative\s+names|suggest\s+names|title\s+ideas|slogans?|taglines?|plot\s+twists?)\b", prompt_lower)
        or re.search(r"\b(?:write|compose|create)\s+(?:an?\s+)?(?:short\s+|flash\s+)?(?:story|fiction|poem|poetry|song|screenplay|scene\s+script|dialogue|narrative|haiku|ballad|monologue|riddle)\b", prompt_lower)
        or re.search(r"\b(?:scene\s+script|screenplay|flash\s+fiction)\b", prompt_lower)
    )
    if is_brainstorm:
        return IntentAnchor(
            action_family=ActionFamily.CREATIVE_IDEATION,
            primary_action="brainstorm and ideate concepts",
            target_or_deliverable="brainstormed ideas",
            confidence=0.90,
            is_strong=True,
        )

    # WEAK FALLBACK: Multiple interpretations remain plausible
    return IntentAnchor(
        action_family=ActionFamily.GENERAL,
        primary_action="general request",
        target_or_deliverable="general response",
        confidence=0.50,
        is_strong=False,
    )


def is_self_contained_operation(raw_prompt: str, action_family: ActionFamily) -> bool:
    """
    Determine if the prompt is a self-contained operation with a well-defined target/outcome.
    Self-contained operations (calculations, transformations, explanations, comparisons,
    debugging a supplied error, explicit drafting tasks) already possess functional purpose.
    """
    prompt_lower = raw_prompt.lower().strip()

    # Explanations and conceptual questions are self-contained
    if action_family == ActionFamily.EDUCATION_EXPLANATION:
        return True

    # Summarizations are self-contained
    if action_family == ActionFamily.SUMMARIZATION:
        return True

    # Software debugging with error/behavior specified is self-contained
    if action_family == ActionFamily.SOFTWARE_DEBUGGING:
        return True

    # Concrete function/routine transformations (e.g., function that calculates X, takes Y and returns Z)
    if action_family == ActionFamily.SOFTWARE_CREATION:
        if re.search(r"\b(?:function|routine|script)\s+that\s+(?:takes|calculates|computes|returns|converts|formats|parses)\b", prompt_lower) or \
           re.search(r"\b(?:calculate|convert|format|parse|find)\s+[a-zA-Z0-9_\-\s]+\b", prompt_lower):
            return True

    # Explicit drafting tasks where the message topic and recipient/context are given
    if action_family == ActionFamily.WRITING:
        if any(w in prompt_lower for w in ["about", "regarding", "for", "to", "asking", "explaining", "notifying", "requesting", "following up"]):
            return True

    # Comparisons and evaluations
    if re.search(r"\b(?:compare|difference\s+between|versus|vs\.?)\b", prompt_lower):
        return True

    return False


def normalize_missing_context_dimension(
    dim: str,
    action_family: ActionFamily,
    raw_prompt: str,
) -> Optional[str]:
    """
    Normalize a missing context entry.
    Ensures:
    1. Interrogative questions are converted to neutral dimension names if safely possible,
       or rejected/flagged.
    2. Cross-domain incoherent dimensions (e.g., software framework in writing) are rejected.
    3. Self-contained operations do not report 'functional purpose'.
    """
    dim_str = dim.strip()
    if not dim_str:
        return None

    dim_lower = dim_str.lower()
    prompt_lower = raw_prompt.lower()

    # Rule 1: Self-contained operations must NOT report 'functional purpose' or 'purpose'
    if is_self_contained_operation(raw_prompt, action_family):
        if dim_lower in {"functional purpose", "purpose", "application purpose", "intended purpose", "goal"}:
            return None

    # Rule 2: Interrogative sentence handling
    # If the entry is phrased as a question (ends with '?' or starts with question words)
    is_question = dim_str.endswith("?") or bool(
        re.match(r"^(?:what|which|how|who|where|when|why|is\s+there|are\s+there|do\s+you|can\s+you)\b", dim_lower)
    )
    if is_question:
        # Attempt safe normalization to neutral dimension name
        normalized = None
        # Pattern: "What/Which platform/framework/language should be used?"
        platform_match = re.search(r"\b(platform|framework|programming\s+language|database|format|audience|style|tone|length)\b", dim_lower)
        if platform_match:
            term = platform_match.group(1)
            normalized = f"preferred {term}" if term not in {"audience", "tone"} else f"target {term}"
        # Pattern: "What specific aspects of <noun> need planning?" -> "<noun> scope"
        elif "aspects" in dim_lower and "planning" in dim_lower:
            noun_match = re.search(r"\b(?:aspects\s+of\s+(?:the\s+)?([a-zA-Z\s]{3,30})\s+need\s+planning)\b", dim_lower)
            if noun_match:
                normalized = f"{noun_match.group(1).strip()} scope"
            else:
                normalized = "scope of planning"
        # Pattern: "What is the intended audience / who is the audience?" -> "target audience"
        elif "audience" in dim_lower:
            normalized = "target audience"
        # Pattern: "What specific features/items should be included?" -> "scope of requested content"
        elif any(w in dim_lower for w in ["features", "items", "details", "content"]):
            normalized = "content scope"
        # If safely normalized:
        if normalized:
            dim_str = normalized
            dim_lower = dim_str.lower()
        else:
            # Cannot safely normalize interrogative sentence into a neutral dimension
            return None

    # Rule 3: Cross-domain coherence check
    # Check if a software-only dimension is injected into a non-software task
    is_software_task = action_family in {ActionFamily.SOFTWARE_CREATION, ActionFamily.SOFTWARE_DEBUGGING}
    prompt_has_software_keywords = bool(
        re.search(r"\b(?:app|application|software|website|code|program|database|script|api|tool)\b", prompt_lower)
    )
    if not is_software_task and not prompt_has_software_keywords:
        if any(term in dim_lower for term in SOFTWARE_ONLY_DIMENSIONS):
            return None

    # Check if research-only dimension is injected into non-research task
    if action_family != ActionFamily.RESEARCH and not re.search(r"\b(?:research|paper|study|academic|literature)\b", prompt_lower):
        if any(term in dim_lower for term in RESEARCH_ONLY_DIMENSIONS):
            return None

    return dim_str


def check_analysis_coherence(
    authoritative_input: str,
    analysis: PromptAnalysis,
    intent_anchor: Optional[IntentAnchor] = None,
) -> list[str]:
    """
    Centralized cross-field semantic coherence gate.
    Validates that:
    - classification matches strong action anchor
    - objective agrees with requested action family
    - deliverable semantics agrees with objective
    - missing context is relevant to the requested task family
    - missing context does not contain unresolved interrogative questions
    - self-contained operations are not marked as lacking functional purpose
    Returns a list of detected violation strings. Empty list indicates full coherence.
    """
    violations: list[str] = []
    anchor = intent_anchor or extract_intent_anchor(authoritative_input)
    prompt_lower = authoritative_input.lower().strip()
    obj_lower = (analysis.objective or "").lower().strip()

    # 1. Classification vs Strong Action Anchor Consistency
    expected_category = ACTION_FAMILY_TO_CATEGORY.get(anchor.action_family, PromptCategory.GENERAL)
    if anchor.is_strong:
        if analysis.classification.category != expected_category:
            violations.append(
                f"Classification category '{analysis.classification.category.value}' contradicts strong requested action family '{anchor.action_family.value}' (expected '{expected_category.value}')."
            )

    # 2. Objective vs Requested Action Consistency
    if anchor.is_strong:
        if anchor.action_family == ActionFamily.SOFTWARE_CREATION:
            if any(w in obj_lower for w in ["draft an email", "draft a message", "debug the error", "summarize the article", "conduct research on"]):
                violations.append(
                    f"Objective '{analysis.objective}' contradicts software creation task family."
                )
        elif anchor.action_family == ActionFamily.WRITING:
            if any(w in obj_lower for w in ["implement", "code a function", "debug the error", "build an application", "analyze dataset"]):
                violations.append(
                    f"Objective '{analysis.objective}' contradicts text writing task family."
                )
        elif anchor.action_family == ActionFamily.EDUCATION_EXPLANATION:
            if any(w in obj_lower for w in ["implement", "build an app", "draft an email", "debug code", "summarize text"]):
                violations.append(
                    f"Objective '{analysis.objective}' contradicts educational explanation task family."
                )
        elif anchor.action_family == ActionFamily.SOFTWARE_DEBUGGING:
            if any(w in obj_lower for w in ["draft an email", "conduct research", "summarize article"]):
                violations.append(
                    f"Objective '{analysis.objective}' contradicts software debugging task family."
                )
        elif anchor.action_family == ActionFamily.PLANNING:
            if any(w in obj_lower for w in ["implement software", "code a function", "debug error"]):
                violations.append(
                    f"Objective '{analysis.objective}' contradicts planning task family."
                )

    # 3. Deliverable Semantics Consistency
    if hasattr(analysis, "deliverable_semantics") and analysis.deliverable_semantics:
        deliv = analysis.deliverable_semantics
        is_self_contained = is_self_contained_operation(authoritative_input, anchor.action_family)
        if is_self_contained and not deliv.is_functionally_complete:
            violations.append(
                f"Deliverable semantics reports lacks functional completeness on self-contained {anchor.action_family.value} task."
            )

    # 4. Missing Context Coherence & Question Separation
    is_software_task = anchor.action_family in {ActionFamily.SOFTWARE_CREATION, ActionFamily.SOFTWARE_DEBUGGING}
    prompt_has_software = bool(re.search(r"\b(?:app|application|software|website|code|program|database|script|api|tool)\b", prompt_lower))

    for item in analysis.missing_context:
        item_stripped = item.strip()
        item_lower = item_stripped.lower()

        # Interrogative check
        if item_stripped.endswith("?") or bool(re.match(r"^(?:what|which|how|who|where|when|why)\b", item_lower)):
            violations.append(
                f"Missing context contains raw clarification question '{item_stripped}' rather than a neutral dimension name."
            )

        # Cross-domain software leakage in non-software task
        if not is_software_task and not prompt_has_software:
            if any(dim in item_lower for dim in SOFTWARE_ONLY_DIMENSIONS):
                violations.append(
                    f"Missing context contains software dimension '{item_stripped}' irrelevant to {anchor.action_family.value} task."
                )

        # Contradictory functional purpose on self-contained operations
        if is_self_contained_operation(authoritative_input, anchor.action_family):
            if item_lower in {"functional purpose", "purpose", "intended purpose", "application purpose"}:
                violations.append(
                    f"Missing context asserts missing 'functional purpose' on self-contained {anchor.action_family.value} task."
                )

    return violations
