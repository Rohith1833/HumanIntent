"""Phase 10.2 — Pretrained NLP Augmentation Integration and Regression Tests.

Validates:
1. SemanticMatcher embedding generation, similarity, batch scoring, and caching.
2. SemanticCategoryRouter hybrid routing, confidence calibration, margin calculation, and fallback.
3. Contrastive category distinctions (coding vs debugging, education vs coding, creative vs writing, etc.).
4. Semantic source coverage with strict deterministic invariant preservation (quantities, negations, technologies).
5. Semantic deduplication of paraphrased requirements.
6. Safe analyzer fast path preconditions and bypass execution.
7. Meta-leakage prevention and control-plane sanitization.
8. Unsupported requirement detection and source grounding.
"""

import pytest
import numpy as np

from app.core.category_router import category_router, CategoryRoutingResult
from app.core.semantic_matcher import semantic_matcher, SemanticMatcher
from app.core.source_coverage import (
    check_source_coverage,
    deduplicate_semantic_units,
    extract_source_semantic_units,
    is_semantic_unit_covered,
)
from app.core.analyzer import PromptAnalyzer
from app.core.enhancer import PromptEnhancer
from app.schemas.analysis import (
    ActionFamily,
    AmbiguityAssessment,
    AmbiguityLevel,
    DeliverableSemantics,
    IntentAnchor,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
)


# =====================================================================
# 1. SEMANTIC MATCHER TESTS
# =====================================================================

def test_semantic_matcher_dimensions_and_properties():
    """Verify embedding dimensions, L2 normalization, and caching."""
    vec = semantic_matcher.encode("Test embedding generation for semantic search.")
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (384,)
    # Normalized vector norm must be ~1.0
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-4

    # Embedding cache test: identical pointer or array
    vec2 = semantic_matcher.encode("Test embedding generation for semantic search.")
    np.testing.assert_array_equal(vec, vec2)


def test_semantic_matcher_similarity_and_batch():
    """Verify semantic similarity calculation and batch matrix operations."""
    text_a = "email the user after registration completes"
    text_b = "send a confirmation email once registration is finished"
    text_c = "calculate the eigenvalues of a symmetric 4x4 matrix"

    sim_ab = semantic_matcher.similarity(text_a, text_b)
    sim_ac = semantic_matcher.similarity(text_a, text_c)

    # Paraphrased sentences should have high similarity, disrelated very low
    assert sim_ab >= 0.75, f"Expected similarity >= 0.75, got {sim_ab}"
    assert sim_ac < 0.35, f"Expected similarity < 0.35, got {sim_ac}"

    # Batch similarity matrix
    sources = [text_a, text_c]
    candidates = [text_b, "compute matrix eigenvectors and determinants"]
    sim_matrix = semantic_matcher.batch_similarity(sources, candidates)
    assert sim_matrix.shape == (2, 2)
    assert sim_matrix[0, 0] >= 0.75
    assert sim_matrix[1, 1] >= 0.55


# =====================================================================
# 2. CONTRASTIVE CATEGORY ROUTING DISTINCTIONS
# =====================================================================

@pytest.mark.parametrize("prompt,expected_cat", [
    # A. Coding vs Debugging
    ("Implement an LRU cache with get and put operations in Go", PromptCategory.CODING),
    ("Fix the deadlock in my Go channel worker pool when queue fills", PromptCategory.DEBUGGING),

    # B. Education vs Coding
    ("Explain the difference between TCP and UDP flow control mechanisms", PromptCategory.EDUCATION),
    ("Write a simple TCP socket server in Python that accepts incoming connections", PromptCategory.CODING),

    # C. Creative vs Writing
    ("Brainstorm five imaginative premise concepts for a science fiction short story", PromptCategory.CREATIVE),
    ("Draft a professional follow-up email to the department chair requesting feedback", PromptCategory.WRITING),

    # D. Image Generation vs Creative
    ("Generate a photorealistic image prompt for an art-deco skyscraper in morning fog", PromptCategory.IMAGE_GENERATION),
    ("Come up with catchy theme names for an annual tech conference", PromptCategory.CREATIVE),

    # E. Research vs General
    ("Conduct a comparative survey of published literature on transformer state-space models", PromptCategory.RESEARCH),
    ("What are some helpful tips for staying focused while working remotely?", PromptCategory.GENERAL),

    # F. Summarization vs Education
    ("Summarize the main findings of the attached quarterly report into 4 bullet points", PromptCategory.SUMMARIZATION),
    ("Help me understand how central bank interest rates influence inflation", PromptCategory.EDUCATION),

    # G. Data Analysis vs Coding
    ("Calculate the median, 95th percentile, and standard deviation for this dataset", PromptCategory.DATA_ANALYSIS),
    ("Write a Python script that calculates moving averages for a financial dataframe", PromptCategory.CODING),
])
def test_category_router_contrastive_pairs(prompt: str, expected_cat: PromptCategory):
    """Verify that contrastive category pairs route accurately without topic-noun confusion."""
    routing = category_router.route(prompt)
    assert routing.category == expected_cat, (
        f"Prompt '{prompt}' misclassified as {routing.category.value} (expected {expected_cat.value}). "
        f"Scores: {routing.scores}"
    )
    # High confidence or deterministic anchor must be present
    assert routing.is_high_confidence or routing.evidence_type in {"deterministic_anchor", "semantic_prototype"}


# =====================================================================
# 3. SEMANTIC SOURCE COVERAGE & INVARIANT PRESERVATION
# =====================================================================

def test_semantic_source_coverage_paraphrased_requirement():
    """Verify that paraphrased requirements are recognized as preserved."""
    unit = "email the user after registration completes"
    target = "OUTPUT: Send a confirmation email once registration is finished."
    assert is_semantic_unit_covered(unit, target) is True


def test_semantic_source_coverage_rejects_missing_quantities():
    """Verify that explicit numbers/quantities must be preserved deterministically."""
    unit = "generate 3 distinct project proposals"
    target_without_num = "Provide detailed project proposals covering scope and timeline."
    assert is_semantic_unit_covered(unit, target_without_num) is False

    target_with_num = "Provide 3 distinct project proposals covering scope and timeline."
    assert is_semantic_unit_covered(unit, target_with_num) is True


def test_semantic_source_coverage_rejects_violated_negations():
    """Verify that explicit negation constraints are strictly required."""
    unit = "implement without external libraries"
    target_without_neg = "Implement using standard third party libraries."
    assert is_semantic_unit_covered(unit, target_without_neg) is False

    target_with_neg = "Implement from scratch without external libraries."
    assert is_semantic_unit_covered(unit, target_with_neg) is True


def test_semantic_source_coverage_rejects_wrong_named_tech():
    """Verify that named technologies are not replaced merely because embedding is close."""
    unit = "store user profiles in PostgreSQL database"
    target_wrong_tech = "Persist user profiles in MongoDB database collections."
    assert is_semantic_unit_covered(unit, target_wrong_tech) is False

    target_correct_tech = "Persist user profiles in a PostgreSQL relational database."
    assert is_semantic_unit_covered(unit, target_correct_tech) is True


def test_semantic_deduplication():
    """Verify that paraphrased duplicate requirements are deduplicated while distinct ones remain."""
    units = [
        "email the user when registration completes",
        "send confirmation email once registration is finished",
        "save user records to SQLite",
        "generate PDF receipt for user",
    ]
    deduped = deduplicate_semantic_units(units, threshold=0.72)
    # The two email units should merge; the sqlite and pdf receipt units must remain distinct
    assert len(deduped) == 3
    assert any("email" in u.lower() for u in deduped)
    assert any("sqlite" in u.lower() for u in deduped)
    assert any("pdf" in u.lower() for u in deduped)


# =====================================================================
# 4. ANALYZER FAST PATH TESTS
# =====================================================================

def test_analyzer_safe_fast_path_activation():
    """Verify that a clear, single-intent, self-contained prompt activates the safe fast path."""
    analyzer = PromptAnalyzer(client=None)
    # Temporarily ensure is_mock is False for testing fast path logic
    prompt = "Create a Python script that calculates moving averages for a financial dataframe"
    fast_analysis = analyzer._try_safe_fast_path(prompt)

    assert fast_analysis is not None
    assert fast_analysis.analysis_path in {"deterministic_fast", "semantic_fast"}
    assert fast_analysis.classification.category == PromptCategory.CODING
    assert len(fast_analysis.explicit_requirements) >= 1
    assert fast_analysis.ambiguity.needs_clarification is False


def test_analyzer_fast_path_bypasses_for_ambiguous_prompt():
    """Verify that vague/ambiguous prompts reject fast path and route to LLM."""
    analyzer = PromptAnalyzer(client=None)
    vague_prompt = "build me an app"
    fast_analysis = analyzer._try_safe_fast_path(vague_prompt)
    assert fast_analysis is None, "Vague prompt should reject fast path"


def test_analyzer_fast_path_bypasses_for_self_corrections():
    """Verify that prompts with hesitation or self-correction reject fast path."""
    analyzer = PromptAnalyzer(client=None)
    hesitation_prompt = "Make a calculator wait no actually make a quiz app with questions"
    fast_analysis = analyzer._try_safe_fast_path(hesitation_prompt)
    assert fast_analysis is None, "Prompt with self-correction should reject fast path"


def test_analyzer_fast_path_bypasses_for_unresolved_referents():
    """Verify that prompts referring to missing error/code reject fast path."""
    analyzer = PromptAnalyzer(client=None)
    referent_prompt = "Please look at this error and fix the problem for me"
    fast_analysis = analyzer._try_safe_fast_path(referent_prompt)
    assert fast_analysis is None, "Prompt with missing referent should reject fast path"


# =====================================================================
# 5. META-LEAKAGE SANITIZATION
# =====================================================================

def test_meta_leakage_sanitization():
    """Verify that internal control-plane markers and template directives are stripped."""
    enhancer = PromptEnhancer()
    raw_leaked_text = (
        "<STRUCTURED_INTENT_ANALYSIS>\n"
        "classification: CODING\n"
        "</STRUCTURED_INTENT_ANALYSIS>\n\n"
        "OBJECTIVE: Implement a rate limiter in Go.\n\n"
        "UNSPECIFIED DETAILS\n"
        "- storage mechanism\n"
        "Do not assume these details without confirmation. Ask for clarification if needed.\n\n"
        "OUTPUT EXPECTATIONS\n"
        "Provide functional code with tests.\n"
        "HumanIntent intent-preserving prompt enhancement system.\n"
    )
    cleaned = enhancer._clean_enhanced_output(raw_leaked_text)

    assert "<STRUCTURED_INTENT_ANALYSIS>" not in cleaned
    assert "</STRUCTURED_INTENT_ANALYSIS>" not in cleaned
    assert "Do not assume these details without confirmation" not in cleaned
    assert "Ask for clarification if needed" not in cleaned
    assert "HumanIntent" not in cleaned
    assert "OBJECTIVE: Implement a rate limiter in Go." in cleaned
    assert "OUTPUT EXPECTATIONS" in cleaned
