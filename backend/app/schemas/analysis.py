from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class PromptCategory(str, Enum):
    """Locked set of supported task categories for HumanIntent."""

    CODING = "CODING"
    DEBUGGING = "DEBUGGING"
    RESEARCH = "RESEARCH"
    EDUCATION = "EDUCATION"
    WRITING = "WRITING"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    PLANNING = "PLANNING"
    CREATIVE = "CREATIVE"
    IMAGE_GENERATION = "IMAGE_GENERATION"
    SUMMARIZATION = "SUMMARIZATION"
    GENERAL = "GENERAL"


class ActionFamily(str, Enum):
    """Canonical action families defining what the user is asking the AI to do."""

    SOFTWARE_CREATION = "SOFTWARE_CREATION"
    SOFTWARE_DEBUGGING = "SOFTWARE_DEBUGGING"
    WRITING = "WRITING"
    EDUCATION_EXPLANATION = "EDUCATION_EXPLANATION"
    RESEARCH = "RESEARCH"
    PLANNING = "PLANNING"
    DATA_ANALYSIS = "DATA_ANALYSIS"
    CREATIVE_IDEATION = "CREATIVE_IDEATION"
    IMAGE_CREATION = "IMAGE_CREATION"
    SUMMARIZATION = "SUMMARIZATION"
    GENERAL = "GENERAL"


class IntentAnchor(BaseModel):
    """Lightweight internal representation of the primary requested user action."""

    action_family: ActionFamily = Field(
        ...,
        description="The primary action family inferred from what the user is asking the AI to do.",
    )
    primary_action: str = Field(
        ...,
        description="The canonical action verb or phrase (e.g., 'create function', 'draft message').",
    )
    target_or_deliverable: str = Field(
        ...,
        description="The primary target or deliverable container.",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score for the action anchor.",
    )
    is_strong: bool = Field(
        default=True,
        description="Whether the requested operation unambiguously indicates this task family.",
    )


class PromptClassification(BaseModel):
    """Model-estimated task category and classification confidence."""

    category: PromptCategory = Field(
        ...,
        description="The primary task category identifying what type of task the user requested.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model-estimated confidence score for the classification (0.0 to 1.0).",
    )

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            return cleaned
        return v

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 4)


EMPTY_SENTINELS = {
    "none",
    "none specified",
    "n/a",
    "na",
    "not applicable",
    "unknown",
    "no requirements",
    "no constraints",
    "no preferences",
    "no missing context",
    "null",
    "undefined",
    "nil",
    "...",
    "…",
    ".",
    "-",
    "--",
    "",
}


class AmbiguityLevel(str, Enum):
    """Categorical ambiguity level derived from the ambiguity heuristic score."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class ClarificationQuestion(BaseModel):
    """A single high-impact, neutral clarification question."""

    id: str = Field(
        ...,
        description="Deterministic identifier for this question (e.g., 'q1', 'q2', 'q3').",
    )
    question: str = Field(
        ...,
        description="Concise, neutral, non-leading clarification question.",
    )
    reason: str = Field(
        ...,
        description="Internal reason why this unknown materially affects the prompt execution.",
    )
    target_dimension: str = Field(
        ...,
        description="The unanswered dimension this question targets.",
    )


class ClarificationAnswer(BaseModel):
    """User response to a previously asked clarification question."""

    question_id: str = Field(
        ...,
        description="Deterministic ID of the question answered (e.g., 'q1').",
    )
    question: str = Field(
        ...,
        description="The question text that was asked.",
    )
    answer: str = Field(
        ...,
        description="The user's response or chosen specification.",
    )
    target_dimension: Optional[str] = Field(
        default=None,
        description="The unanswered dimension this question targeted.",
    )


class SourceType(str, Enum):
    """Authoritative source category for grounding evidence."""

    ORIGINAL_PROMPT = "ORIGINAL_PROMPT"
    CLARIFICATION = "CLARIFICATION"


class GroundingRelation(str, Enum):
    """Semantic relation explaining why user source text supports an extracted fact."""

    DIRECT = "DIRECT"
    MODIFIES_DELIVERABLE = "MODIFIES_DELIVERABLE"
    MODIFIES_OUTPUT_STYLE = "MODIFIES_OUTPUT_STYLE"
    CONSTRAINT = "CONSTRAINT"
    CLARIFICATION = "CLARIFICATION"
    PARAPHRASE = "PARAPHRASE"


class GroundingEvidence(BaseModel):
    """Specific user-provided evidence grounding an analysis item."""

    field: str = Field(
        ...,
        description="The field name (e.g., 'objective', 'explicit_requirements', 'constraints', 'preferences').",
    )
    value: str = Field(
        ...,
        description="The specific value or assertion grounded by this evidence.",
    )
    source_type: SourceType = Field(
        default=SourceType.ORIGINAL_PROMPT,
        description="Authoritative source origin: ORIGINAL_PROMPT or CLARIFICATION.",
    )
    evidence: str = Field(
        default="",
        description="User-provided text or excerpt directly supporting this assertion.",
    )
    relation: GroundingRelation = Field(
        default=GroundingRelation.DIRECT,
        description="Semantic relation explaining how the evidence supports the assertion.",
    )


class DeliverableSemantics(BaseModel):
    """Semantic decomposition of the requested deliverable container and its functional purpose."""

    deliverable_type: Optional[str] = Field(
        default=None,
        description="The container, medium, or artifact type requested (e.g., 'application', 'document', 'bot').",
    )
    intended_outcome: Optional[str] = Field(
        default=None,
        description="The functional purpose, behavior, domain problem, or topic the deliverable addresses.",
    )
    is_functionally_complete: bool = Field(
        default=True,
        description="True if the deliverable's functional purpose is known or the operation is inherently self-contained.",
    )


class AnalysisGrounding(BaseModel):
    """Provenance metadata verifying that every asserted fact in PromptAnalysis is grounded in authoritative user sources."""

    is_grounded: bool = Field(
        default=True,
        description="Whether all asserted facts are grounded in user-provided input.",
    )
    evidence: list[GroundingEvidence] = Field(
        default_factory=list,
        description="List of provenance evidence records for asserted items.",
    )
    unsupported_items: list[str] = Field(
        default_factory=list,
        description="Items asserted in analysis that lack grounding in authoritative user input.",
    )


class AmbiguityAssessment(BaseModel):
    """Product heuristic assessment of whether prompt contains enough information to enhance safely."""

    score: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Heuristic ambiguity score from 0 (fully specified) to 100 (critical unknowns).",
    )
    level: AmbiguityLevel = Field(
        default=AmbiguityLevel.LOW,
        description="Categorical ambiguity level: LOW (0–29), MEDIUM (30–64), HIGH (65–100).",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Whether a critical unresolved ambiguity requires clarification before safe enhancement.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Key factors contributing to the ambiguity evaluation.",
    )
    questions: list[ClarificationQuestion] = Field(
        default_factory=list,
        description="At most 3 high-impact clarification questions when clarification is required.",
    )

    @model_validator(mode="before")
    @classmethod
    def defensive_normalize_ambiguity(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        # 1. Clamp score 0 to 100
        raw_score = data.get("score", 0)
        try:
            score = int(raw_score)
        except (ValueError, TypeError):
            score = 0
        clamped_score = max(0, min(100, score))
        data["score"] = clamped_score

        # 2. Derive / normalize level based on clamped score
        # LOW: 0-29, MEDIUM: 30-64, HIGH: 65-100
        if clamped_score < 30:
            canonical_level = AmbiguityLevel.LOW
        elif clamped_score < 65:
            canonical_level = AmbiguityLevel.MEDIUM
        else:
            canonical_level = AmbiguityLevel.HIGH
        data["level"] = canonical_level

        # 3. Defensive questions normalization (max 3, deterministic IDs)
        raw_questions = data.get("questions", [])
        if isinstance(raw_questions, list):
            cleaned_questions = []
            for idx, q in enumerate(raw_questions[:3], start=1):
                if isinstance(q, dict):
                    q_id = f"q{idx}"
                    q_text = str(q.get("question", "")).strip()
                    q_reason = str(q.get("reason", "")).strip() or "Materially changes implementation."
                    q_target = str(q.get("target_dimension", "")).strip() or "task specification"
                    if q_text:
                        cleaned_questions.append({
                            "id": q_id,
                            "question": q_text,
                            "reason": q_reason,
                            "target_dimension": q_target,
                        })
                elif isinstance(q, ClarificationQuestion):
                    cleaned_questions.append(q)
            data["questions"] = cleaned_questions
        else:
            data["questions"] = []

        # 4. Consistency: LOW ambiguity -> needs_clarification must be False, questions []
        raw_needs = data.get("needs_clarification", False)
        if canonical_level == AmbiguityLevel.LOW:
            data["needs_clarification"] = False
            data["questions"] = []
        elif not raw_needs:
            data["questions"] = []
        elif raw_needs and not data["questions"]:
            if canonical_level == AmbiguityLevel.HIGH and data.get("reasons"):
                data["questions"] = [{
                    "id": "q1",
                    "question": f"Could you provide more detail regarding {data['reasons'][0]}?",
                    "reason": "Clarification required to resolve primary intent.",
                    "target_dimension": "primary objective",
                }]
            else:
                data["needs_clarification"] = False

        return data


class PromptAnalysis(BaseModel):
    """Structured understanding of what the user actually provided vs. what was left unstated."""

    classification: PromptClassification = Field(
        ...,
        description="Task classification identifying the primary category and confidence.",
    )
    objective: str = Field(
        ...,
        description="The user's primary goal or task.",
    )
    explicit_requirements: list[str] = Field(
        default_factory=list,
        description="Specific requirements, technologies, languages, or formats directly and explicitly stated in the user's wording.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Explicit boundaries or limitations directly imposed by the user.",
    )
    preferences: list[str] = Field(
        default_factory=list,
        description="Explicit stylistic or behavioral preferences directly stated by the user.",
    )
    missing_context: list[str] = Field(
        default_factory=list,
        description="Neutral descriptions of information absent from the request. Must only identify unanswered dimensions without suggesting candidate technologies, options, or examples.",
    )
    ambiguity: AmbiguityAssessment = Field(
        default_factory=lambda: AmbiguityAssessment(
            score=10,
            level=AmbiguityLevel.LOW,
            needs_clarification=False,
            reasons=[],
            questions=[],
        ),
        description="Ambiguity assessment and optional high-impact clarification questions.",
    )
    grounding: AnalysisGrounding = Field(
        default_factory=AnalysisGrounding,
        description="Provenance metadata verifying every asserted fact is grounded in authoritative user input.",
    )
    deliverable_semantics: DeliverableSemantics = Field(
        default_factory=DeliverableSemantics,
        description="Semantic breakdown of deliverable identity vs functional purpose.",
    )
    intent_anchor: Optional[IntentAnchor] = Field(
        default=None,
        description="Internal consistency anchor tracking the primary action family.",
    )
    raw_category: Optional[str] = Field(
        default=None,
        description="Raw model classification before consistency gates.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_keys(cls, data: Any) -> Any:
        """Defensive normalization: Reconstruct classification/objective if model mapped keys to task/category."""
        if isinstance(data, dict):
            if "objective" not in data or not data.get("objective"):
                for alt_key in ("task", "goal", "primary_goal", "intent"):
                    if alt_key in data and data[alt_key]:
                        data["objective"] = str(data[alt_key])
                        break
            if "classification" not in data:
                cat = data.get("category", "CODING")
                conf = data.get("confidence", 0.95)
                data["classification"] = {"category": cat, "confidence": conf}
            elif isinstance(data["classification"], dict):
                if "category" not in data["classification"]:
                    data["classification"]["category"] = data.get("category", "CODING")
                if "confidence" not in data["classification"]:
                    data["classification"]["confidence"] = data.get("confidence", 0.95)
        return data

    @field_validator("explicit_requirements", "constraints", "preferences", "missing_context", mode="before")
    @classmethod
    def normalize_list_sentinels(cls, v: Any) -> list[str]:
        """Defensive normalization: Strip common empty sentinel values like 'None specified', 'N/A', etc."""
        if v is None:
            return []
        if isinstance(v, str):
            cleaned_str = v.strip()
            if cleaned_str.lower() in EMPTY_SENTINELS:
                return []
            return [cleaned_str]
        if isinstance(v, list):
            filtered: list[str] = []
            for item in v:
                if isinstance(item, str):
                    cleaned_item = item.strip()
                    if cleaned_item.lower() not in EMPTY_SENTINELS:
                        filtered.append(cleaned_item)
                elif item is not None:
                    cleaned_item = str(item).strip()
                    if cleaned_item.lower() not in EMPTY_SENTINELS:
                        filtered.append(cleaned_item)
            return filtered
        return v



class AnalyzeRequest(BaseModel):
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The rough human prompt to analyze.",
    )

    @field_validator("prompt")
    @classmethod
    def validate_not_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Prompt cannot be empty or contain only whitespace.")
        return stripped
