"""General semantic category router for HumanIntent.

Core Invariants:
- Combines strong deterministic action-family anchors with lightweight semantic prototype embeddings.
- Classifies based on primary requested action and deliverable, NOT topic nouns.
- Calibrated confidence and margin gating:
  * Strong deterministic anchor -> immediate deterministic classification.
  * High-confidence semantic classification with clear margin -> semantic category anchor.
  * Close / low-confidence classification -> fallback to LLM analyzer.
- Authoritative source boundary preserved: NLP never silently overrides explicit user intent.
"""

import logging
import re
import warnings
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

# Filter out harmless HF Hub unauthenticated request warning on import
warnings.filterwarnings("ignore", message=".*You are sending unauthenticated requests to the HF Hub.*")

from app.core.intent_anchor import (
    ACTION_FAMILY_TO_CATEGORY,
    IntentAnchor,
    extract_intent_anchor,
)
from app.core.semantic_matcher import semantic_matcher
from app.schemas.analysis import ActionFamily, ClarificationAnswer, PromptCategory

logger = logging.getLogger("humanintent.category_router")


class CategoryRoutingResult(BaseModel):
    """Diagnostic output from the Category Router."""

    category: PromptCategory
    confidence: float = Field(..., ge=0.0, le=1.0)
    margin: float = Field(..., ge=0.0, le=1.0)
    evidence_type: str = Field(..., description="deterministic_anchor | semantic_prototype | disagreement_fallback | llm_fallback")
    is_high_confidence: bool
    can_fast_path: bool = False
    tier: str = Field("tier_c_uncertain", description="tier_a_deterministic | tier_b_semantic | tier_c_uncertain")
    disagreement_detected: bool = False
    second_best_category: Optional[PromptCategory] = None
    scores: Dict[str, float] = Field(default_factory=dict)


# Canonical task-semantic prototype expressions per category (action + deliverable)
# Completely domain-agnostic, free of technology or subject-matter topic nouns
CATEGORY_EXPRESSIONS: Dict[PromptCategory, List[str]] = {
    PromptCategory.CODING: [
        "implement a software function, routine, script, or component",
        "build a web application, website, backend service, or API endpoint",
        "write software code, algorithms, utilities, or database schemas",
        "create an executable software tool or program",
        "implement an in-memory rate limiter, cache, or algorithm",
        "data structure or backend software module implementation",
        "write a script or program to compute, process, or format data",
    ],
    PromptCategory.DEBUGGING: [
        "diagnose, troubleshoot, and fix software errors or bugs",
        "investigate crashes, exceptions, tracebacks, or failing tests",
        "resolve broken code, syntax errors, or malfunctioning scripts",
        "fix deadlocks, memory leaks, timeouts, or execution failures in code",
    ],
    PromptCategory.WRITING: [
        "draft, compose, or write an email, letter, or interpersonal message",
        "write an essay, article, memo, announcement, speech, or newsletter",
        "rewrite, edit, or polish written prose communication",
        "draft a polite, professional, or formal written response or decline",
        "compose a statement of work, proposal, or formal project overview",
    ],
    PromptCategory.EDUCATION: [
        "explain how or why something works conceptually",
        "teach a concept, theoretical principle, or mechanism for learning",
        "explain the difference between two concepts, theories, or methods",
        "break down an educational topic simply for understanding",
        "what makes two technologies or concepts different from each other",
    ],
    PromptCategory.RESEARCH: [
        "investigate and synthesize academic or technical research literature",
        "conduct a comparative survey of published methods, approaches, and studies",
        "gather empirical evidence and review scientific findings on a topic",
        "systematic inquiry, comparative study, and literature research",
    ],
    PromptCategory.PLANNING: [
        "create a project plan, timeline, roadmap, or milestone schedule",
        "organize a study schedule, curriculum, progression, or rollout plan",
        "plan a multi-step sequence, itinerary, agenda, or phased strategy",
        "outline phases, deadlines, and milestones for completing an initiative",
    ],
    PromptCategory.DATA_ANALYSIS: [
        "analyze a dataset, spreadsheet, metrics, numbers, or data table",
        "calculate statistical metrics, averages, percentiles, or trends",
        "interpret numerical distributions, measurements, and quantitative tables",
        "quantitative data analysis, computation, and statistical evaluation",
    ],
    PromptCategory.CREATIVE: [
        "brainstorm creative ideas, novel angles, or inventive concepts",
        "suggest catchy names, titles, or marketing slogans",
        "ideate imaginative suggestions and inventive themes",
        "creative brainstorming of options without drafting a full document",
        "write a creative story, poem, or fictional narrative",
        "science fiction, fantasy, or imaginative creative writing",
    ],
    PromptCategory.IMAGE_GENERATION: [
        "generate an image, visual scene, picture, or illustration",
        "create a prompt for text-to-image models or render visual artwork",
        "photorealistic portrait, drawing, graphic, or digital art deliverable",
        "produce visual imagery or descriptive scene for image generation",
    ],
    PromptCategory.SUMMARIZATION: [
        "summarize, condense, or digest provided text, articles, or documents",
        "extract key takeaways, bullet points, or executive summary from text",
        "create a concise synopsis of an article, transcript, or report",
        "shorten long supplied material into a brief summary",
        "provide a brief recap, synopsis, or tldr of provided material",
        "extract action items, decisions, or key details from text",
    ],
    PromptCategory.GENERAL: [
        "general inquiry, broad advice, or miscellaneous open assistance",
        "multidisciplinary question not fitting a specific single domain",
        "open-ended general guidance or broad conversational help",
    ],
}


class SemanticCategoryRouter:
    """Classifies user requests using deterministic action anchors, semantic prototypes, and calibrated margins."""

    def __init__(self) -> None:
        self._prototype_embeddings: Optional[Dict[PromptCategory, List[any]]] = None

    def _ensure_prototypes_embedded(self) -> None:
        if self._prototype_embeddings is not None:
            return

        embeddings: Dict[PromptCategory, List[any]] = {}
        for cat, exprs in CATEGORY_EXPRESSIONS.items():
            embeddings[cat] = [semantic_matcher.encode(expr) for expr in exprs]

        self._prototype_embeddings = embeddings

    def _extract_intent_phrase(self, raw_prompt: str) -> str:
        """Extract the imperative or core action clause to focus semantic embedding on action over nouns."""
        cleaned = raw_prompt.strip()
        lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
        first_line = lines[0] if lines else cleaned
        # Extract imperative action clause before subordinate clauses (e.g., 'that', 'which', 'where')
        action_match = re.split(r"\b(?:that|which|where|when|in\s+order\s+to)\b", first_line, flags=re.IGNORECASE)
        if action_match and len(action_match[0].strip().split()) >= 2:
            return action_match[0].strip()
        return first_line

    def route(
        self,
        raw_prompt: str,
        clarification_answers: Optional[List[ClarificationAnswer]] = None,
        anchor: Optional[IntentAnchor] = None,
    ) -> CategoryRoutingResult:
        """Route user prompt to category using calibrated Three-Tier routing architecture with Disagreement Safety.

        Policy:
        - Tier A (Strong Deterministic):
          Strong action-family anchor AND semantic router agree (concurrence).
          May safely bypass analyzer (can_fast_path=True).
        - Tier B (High-Confidence Semantic):
          No strong anchor, but semantic prototype has high similarity (>= 0.36) and wide margin (>= 0.08)
          with zero contradictory weak anchor.
          May safely bypass analyzer (can_fast_path=True).
        - Tier C (Uncertain / Disagreement):
          Anchor and semantic router disagree, or top two semantic categories are close,
          or task family is weak/general.
          CANNOT bypass analyzer (can_fast_path=False). Defers to full Llama analyzer.
        """
        # 1. Compute Semantic Prototype Similarities
        self._ensure_prototypes_embedded()
        intent_phrase = self._extract_intent_phrase(raw_prompt)

        prompt_emb = semantic_matcher.encode(raw_prompt)
        phrase_emb = semantic_matcher.encode(intent_phrase)

        scores: Dict[PromptCategory, float] = {}
        for cat, expr_embs in self._prototype_embeddings.items():
            sims = []
            for emb in expr_embs:
                s_full = float(prompt_emb.dot(emb))
                s_phrase = float(phrase_emb.dot(emb))
                sims.append(0.60 * s_full + 0.40 * s_phrase)

            max_s = max(sims)
            mean_s = sum(sims) / len(sims)
            cat_score = 0.70 * max_s + 0.30 * mean_s
            scores[cat] = max(0.0, cat_score)

        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_cats[0]
        second_cat, second_score = sorted_cats[1]
        margin = max(0.0, top_score - second_score)
        score_dict = {cat.value: round(score, 4) for cat, score in sorted_cats}

        # 2. Extract Intent Anchor
        active_anchor = anchor or extract_intent_anchor(raw_prompt, clarification_answers)
        has_strong_anchor = active_anchor.is_strong and active_anchor.action_family != ActionFamily.GENERAL

        # 3. Three-Tier Policy with Disagreement Safety Check
        if has_strong_anchor:
            anchor_cat = ACTION_FAMILY_TO_CATEGORY.get(active_anchor.action_family, PromptCategory.GENERAL)
            if anchor_cat == top_cat:
                # Tier A: Deterministic anchor and semantic prototype CONCUR
                return CategoryRoutingResult(
                    category=anchor_cat,
                    confidence=max(active_anchor.confidence, 0.94),
                    margin=round(margin, 4),
                    evidence_type="deterministic_anchor",
                    is_high_confidence=True,
                    can_fast_path=True,
                    tier="tier_a_deterministic",
                    disagreement_detected=False,
                    second_best_category=second_cat,
                    scores=score_dict,
                )
            else:
                # DISAGREEMENT SAFETY CHECK: Anchor and Semantic Router disagree!
                # DO NOT FAST PATH. Fallback to full LLM analyzer.
                logger.info(
                    "Routing disagreement detected: anchor=%s vs semantic=%s (margin=%.4f). Fast path disabled.",
                    anchor_cat.value,
                    top_cat.value,
                    margin,
                )
                return CategoryRoutingResult(
                    category=anchor_cat,
                    confidence=0.60,
                    margin=round(margin, 4),
                    evidence_type="disagreement_fallback",
                    is_high_confidence=False,
                    can_fast_path=False,
                    tier="tier_c_uncertain",
                    disagreement_detected=True,
                    second_best_category=second_cat,
                    scores=score_dict,
                )

        # Cases without a strong anchor: Check Tier B (High-Confidence Semantic)
        weak_cat = (
            ACTION_FAMILY_TO_CATEGORY.get(active_anchor.action_family)
            if active_anchor.action_family != ActionFamily.GENERAL
            else None
        )
        is_tier_b = (
            top_score >= 0.28
            and margin >= 0.05
            and (weak_cat is None or weak_cat == top_cat)
        )

        if is_tier_b:
            calibrated_conf = min(0.95, max(0.85, top_score + (margin * 0.8)))
            # Strict calibrated fast path qualification: domain-specific (non-GENERAL) with wide margin and high similarity
            can_fp = (top_cat != PromptCategory.GENERAL and margin >= 0.14 and top_score >= 0.38)
            return CategoryRoutingResult(
                category=top_cat,
                confidence=round(calibrated_conf, 4),
                margin=round(margin, 4),
                evidence_type="semantic_prototype",
                is_high_confidence=True,
                can_fast_path=can_fp,
                tier="tier_b_semantic" if can_fp else "tier_c_uncertain",
                disagreement_detected=False,
                second_best_category=second_cat,
                scores=score_dict,
            )

        # Tier C: Uncertain / Close top categories / General task
        return CategoryRoutingResult(
            category=top_cat,
            confidence=round(top_score, 4),
            margin=round(margin, 4),
            evidence_type="llm_fallback",
            is_high_confidence=False,
            can_fast_path=False,
            tier="tier_c_uncertain",
            disagreement_detected=False,
            second_best_category=second_cat,
            scores=score_dict,
        )


# Singleton category router
category_router = SemanticCategoryRouter()
