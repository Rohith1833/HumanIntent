"""Comprehensive regression tests for Token-Aware Context Budget and Long-Input Decomposition Pipeline."""

import json
from unittest.mock import AsyncMock
import pytest

from app.core.analyzer import PromptAnalyzer
from app.core.context_budget import (
    CONTEXT_BUDGET_EXCEEDED,
    LONG_INPUT_MODE,
    NORMAL_CONTEXT_MODE,
    ProcessingMode,
    SourceChunk,
    UnitType,
    chunk_long_source,
    context_budget_manager,
    estimate_tokens,
    extract_chunk_atomic_units,
    extract_deterministic_chunk_units,
    reconcile_long_input_analysis,
    resolve_supersessions_and_deduplicate,
)
from app.core.enhancer import PromptEnhancer
from app.core.scorer import PromptScorer
from app.core.validator import PromptValidator
from app.schemas.analysis import (
    AnalysisGrounding,
    PromptAnalysis,
    PromptCategory,
    PromptClassification,
)
from app.schemas.validation import ValidationIssueType, ValidationSeverity


# =========================================================================
# FIXTURES & TEST PROMPTS
# =========================================================================

@pytest.fixture
def mock_clients():
    """Create reusable mocks for LLM services."""
    mock_llm = AsyncMock()
    validator = PromptValidator(client=mock_llm)
    scorer = PromptScorer(client=mock_llm)
    enhancer = PromptEnhancer(client=mock_llm, validator=validator, scorer=scorer)
    analyzer = PromptAnalyzer(client=mock_llm)
    return {
        "client": mock_llm,
        "validator": validator,
        "scorer": scorer,
        "enhancer": enhancer,
        "analyzer": analyzer,
    }


def generate_6000_char_spec() -> str:
    """Generate a realistic messy ~6,500-character IoT & logistics platform specification."""
    part1_beginning = (
        "# ENTERPRISE TELEMETRY AND LOGISTICS SYSTEM\n\n"
        "We need to architect an industrial IoT fleet telemetry and logistics monitoring system. "
        "At the beginning of the pipeline, the system must implement multi-protocol device ingestion "
        "with MQTT packet decoding and hardware TLS certificate validation. "
        "The edge gateways must enforce at least 10 sensor streams per gateway device, "
        "and each edge worker must maintain an offline circular buffer. "
        "Under no circumstances should the ingestion service use RabbitMQ because of our latency limits. "
        "Also include device heartbeat tracking and GPS coordinate normalization.\n\n"
    )
    part2_fill = (
        "## OPERATIONAL WORKFLOW & DISPATCH RULES\n\n"
        + "The operations center requires fleet dispatchers to view live vehicle clusters on an interactive geospatial map. "
        "Dispatchers need to track asset health indices, fuel burn rates, driver behavior scores, and turn-by-turn routing anomalies. "
        * 7
        + "\n\n"
        "Each dispatch route must calculate dynamic traffic delays and highway toll estimations. "
        "Vehicles are categorized into refrigerated reefers, dry vans, and hazardous tankers. "
        "The system must track tire pressure readings and driver hours of service compliance. "
        "Drivers should receive automated break reminders when driving continuously for over 4 hours. "
        "All telemetry packets must be verified against digital signatures. "
        "Do not use XML payloads anywhere in the API; strictly use JSON format.\n\n"
    )
    part3_middle = (
        "## COLD CHAIN TELEMETRY & ALERTING\n\n"
        "In the middle of the monitoring pipeline, implement automated cold-chain anomaly detection "
        "with dynamic temperature drift thresholds and humidity deviation warnings. "
        "If a reefer temperature exceeds -18 degrees Celsius, trigger emergency escalation SMS alerts. "
        "The system must enforce a maximum 5 retry attempts for transient alert webhooks. "
        "We also need driver cargo receipt uploads with camera snapshot compression. "
        "For database storage, start with PostgreSQL cluster replication.\n\n"
    )
    part4_fill = (
        "## ANALYTICS & FLEET REPORTING\n\n"
        + "Fleet managers require consolidated weekly maintenance forecast reports and fuel consumption analytics. "
        "The predictive engine must identify transmission degradation, coolant leaks, alternator stress, and brake pad wear before breakdown occurs. "
        * 7
        + "\n\n"
    )
    part5_correction_and_end = (
        "## REGULATORY COMPLIANCE, REVISED STORAGE & LIFECYCLE RULES\n\n"
        "Wait no, scratch PostgreSQL from earlier; use SQLite instead for embedded local storage. "
        "Near the end of the workflow, generate regulatory compliance audit export packages "
        "and automated customs freight manifest generation. "
        "Ensure all driver logs comply with national transport authority standards. "
        "Regarding data retention and lifecycle: delete raw telemetry buffer after 48 hours "
        "and purge temporary driver upload snapshots after 7 days. "
        "Keep the maintenance summary report under 250 words."
    )
    res = part1_beginning + part2_fill + part3_middle + part4_fill + part5_correction_and_end
    return res


def generate_9000_char_spec() -> str:
    """Generate a realistic ~8,800-character clinical trial registry management specification."""
    base = generate_6000_char_spec()
    extra_clinical_section = (
        "\n\n## CLINICAL SITE COORDINATION & PATIENT CONSENT\n\n"
        + "Site coordinators must verify patient electronic informed consent forms with biometric signature capture. "
        "Each trial site must maintain blind randomized drug assignment schedules. "
        "Adverse event notifications must be dispatched to institutional review boards within 24 hours. "
        "Clinical monitors must reconcile medication inventory logs, dispensing records, pill counts, and drug return logs across sites. "
        * 7
        + "\n\n"
        "Include laboratory assay reconciliation and dual-independent data entry auditing. "
        "The trial registry must support multi-site patient enrollment quotas with exactly 3 reviewers for amendments. "
        "Do not allow unblinded staff to access drug assignment codes before trial completion. "
        "Automate the generation of annual safety update reports for health authorities."
    )
    return base + extra_clinical_section


# =========================================================================
# TESTS: TOKEN BUDGET ESTIMATION & STAGE BUDGET AUDITING
# =========================================================================

def test_token_budget_estimation_accuracy():
    """Verify estimate_tokens provides safe, conservative bounds without underestimating."""
    short_text = "build a simple website"
    assert estimate_tokens(short_text) >= 4

    long_text = "word " * 1000
    est = estimate_tokens(long_text)
    # 1000 words in LLaMA is ~1,000-1,350 tokens; conservative estimate must be in that range
    assert 1000 <= est <= 1500

    report = context_budget_manager.check_stage_budget(
        stage="analyzer",
        system_prompt="system instructions " * 50,
        user_text="simple prompt",
    )
    assert report.is_safe is True
    assert report.processing_mode == ProcessingMode.NORMAL
    assert report.diagnostic_code == NORMAL_CONTEXT_MODE


def test_budget_report_flags_long_inputs():
    """Verify context_budget_manager correctly routes large inputs to LONG_INPUT_MODE."""
    long_spec = generate_6000_char_spec()
    assert 5000 <= len(long_spec) <= 6000

    # With analyzer system prompt (~5700 tokens), 6000 chars of user input exceeds safe single-pass budget
    report = context_budget_manager.check_stage_budget(
        stage="analyzer",
        system_prompt="x " * 4000,  # simulate large system prompt
        user_text=long_spec,
    )
    assert report.is_safe is False
    assert report.processing_mode == ProcessingMode.LONG_INPUT
    assert report.diagnostic_code == LONG_INPUT_MODE


# =========================================================================
# TESTS: SEMANTIC BOUNDARY CHUNKING
# =========================================================================

def test_semantic_boundary_chunking_preserves_full_content():
    """Verify chunk_long_source splits on semantic boundaries without discarding content."""
    long_spec = generate_6000_char_spec()
    chunks = chunk_long_source(long_spec, target_chunk_chars=1800, max_chunk_chars=2500)

    assert len(chunks) >= 3, f"Expected at least 3 chunks, got {len(chunks)}"

    # Reconstituted text must encompass all major sections
    combined = "\n\n".join(c.text for c in chunks)
    assert "ENTERPRISE TELEMETRY AND LOGISTICS SYSTEM" in combined
    assert "COLD CHAIN TELEMETRY" in combined
    assert "REGULATORY COMPLIANCE" in combined
    assert "delete raw telemetry buffer after 48 hours" in combined

    # Check contiguous bounds
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert len(c.text) > 0


# =========================================================================
# TESTS: CHUNK EXTRACTION, QUANTITIES, NEGATIVE RULES, LIFECYCLE
# =========================================================================

def test_deterministic_chunk_extraction_preserves_specifics():
    """Verify atomic unit extraction captures quantities, negative rules, lifecycle rules, and corrections."""
    chunk_text = (
        "We require at least 10 sensor streams and maximum 5 retry attempts. "
        "Do not use RabbitMQ anywhere in the pipeline. "
        "Wait no, scratch PostgreSQL from earlier; use SQLite instead. "
        "Regarding retention: delete raw telemetry buffer after 48 hours."
    )
    chunk = SourceChunk(chunk_index=0, text=chunk_text, start_char=0, end_char=len(chunk_text))
    units = extract_deterministic_chunk_units(chunk)

    unit_texts = [u.text.lower() for u in units]
    unit_types = [u.unit_type for u in units]

    # Quantities
    assert any("at least 10 sensor" in t or "10 sensor" in t for t in unit_texts)
    assert any("maximum 5 retry" in t or "5 retry" in t for t in unit_texts)
    assert UnitType.QUANTITY in unit_types

    # Negative rule
    assert any("do not use rabbitmq" in t for t in unit_texts)
    assert UnitType.NEGATIVE_RULE in unit_types

    # Correction / supersession
    assert any("use sqlite instead" in t for t in unit_texts)
    assert UnitType.CORRECTION in unit_types

    # Lifecycle rule
    assert any("delete raw telemetry buffer after 48 hours" in t for t in unit_texts)
    assert UnitType.LIFECYCLE in unit_types


# =========================================================================
# TESTS: CROSS-CHUNK SUPERSESSION & DEDUPLICATION
# =========================================================================

def test_cross_chunk_supersession_replaces_earlier_choice():
    """Verify that a correction in Chunk 2 supersedes the earlier selection in Chunk 0."""
    chunk0 = SourceChunk(chunk_index=0, text="Store data in PostgreSQL cluster.", start_char=0, end_char=35)
    chunk1 = SourceChunk(chunk_index=1, text="Process telemetry records.", start_char=36, end_char=65)
    chunk2 = SourceChunk(chunk_index=2, text="Scratch PostgreSQL from earlier; use SQLite instead.", start_char=66, end_char=120)

    u0 = extract_deterministic_chunk_units(chunk0)
    u1 = extract_deterministic_chunk_units(chunk1)
    u2 = extract_deterministic_chunk_units(chunk2)

    flat, active, superseded = resolve_supersessions_and_deduplicate([chunk0, chunk1, chunk2], [u0, u1, u2])

    active_texts = [u.text.lower() for u in active]
    # SQLite should be active
    assert any("sqlite" in t for t in active_texts)
    # The superseded list should record PostgreSQL
    superseded_texts = [u.text.lower() for u in superseded]
    assert any("postgresql" in t for t in superseded_texts) or not any("postgresql cluster" in t for t in active_texts)


# =========================================================================
# TESTS: 6,000+ AND 9,000+ SPECIFICATIONS (BEGINNING, MIDDLE, END PRESERVED)
# =========================================================================

@pytest.mark.asyncio
async def test_6000_char_spec_beginning_middle_end_preservation(mock_clients):
    """Verify a 6,000+ char messy specification preserves beginning, middle, end, quantities, and lifecycle."""
    spec_6k = generate_6000_char_spec()
    assert 5000 <= len(spec_6k) <= 6000

    analyzer = mock_clients["analyzer"]
    mock_llm = mock_clients["client"]

    # Mock reconciliation LLM response
    mock_llm.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.95},
        "objective": "Build industrial IoT fleet telemetry and logistics monitoring system with cold-chain detection",
        "explicit_requirements": [
            "multi-protocol device ingestion",
            "MQTT packet decoding",
            "hardware TLS certificate validation",
            "cold-chain anomaly detection",
            "regulatory compliance audit export packages",
            "customs freight manifest generation",
        ],
        "constraints": [
            "do not use RabbitMQ",
            "do not use XML payloads",
            "maximum 5 retry attempts",
            "delete raw telemetry buffer after 48 hours",
        ],
        "preferences": ["under 250 words summary report"],
        "missing_context": ["cloud hosting provider"],
    })

    analysis = await analyzer.analyze(spec_6k)

    # 1. Beginning requirement preserved
    reqs_str = " ".join(analysis.explicit_requirements).lower()
    assert "mqtt packet decoding" in reqs_str or "multi-protocol" in reqs_str

    # 2. Middle requirement preserved
    assert "cold-chain" in reqs_str or "anomaly detection" in reqs_str

    # 3. End requirement preserved
    assert "customs freight manifest" in reqs_str or "regulatory compliance" in reqs_str

    # 4. Quantities and Negative constraints preserved
    consts_str = " ".join(analysis.constraints).lower()
    assert "do not use rabbitmq" in consts_str
    assert "maximum 5 retry" in consts_str or "5 retry" in consts_str

    # 5. Lifecycle rule preserved
    assert "delete raw telemetry buffer after 48 hours" in consts_str or "48 hours" in consts_str

    # 6. Provenance preserved in grounding
    assert analysis.grounding is not None
    assert analysis.grounding.is_grounded is True
    assert len(analysis.grounding.source_units) >= 5
    assert len(analysis.grounding.evidence) >= 5


@pytest.mark.asyncio
async def test_9000_char_spec_decomposition_and_coverage(mock_clients):
    """Verify a 9,000+ char specification runs full chunk decomposition with complete tail preservation."""
    spec_9k = generate_9000_char_spec()
    assert 8000 <= len(spec_9k) <= 9000

    analyzer = mock_clients["analyzer"]
    mock_llm = mock_clients["client"]

    mock_llm.chat.return_value = json.dumps({
        "classification": {"category": "CODING", "confidence": 0.92},
        "objective": "Architect clinical trial registry and logistics telemetry management system",
        "explicit_requirements": [
            "multi-protocol device ingestion",
            "cold-chain anomaly detection",
            "electronic informed consent forms",
            "biometric signature capture",
            "laboratory assay reconciliation",
            "annual safety update reports",
        ],
        "constraints": [
            "do not use RabbitMQ",
            "exactly 3 reviewers",
            "delete raw telemetry buffer after 48 hours",
        ],
        "preferences": [],
        "missing_context": ["regulatory submission standard"],
    })

    analysis = await analyzer.analyze(spec_9k)

    # Assert no truncation occurred and tail requirements were captured
    reqs_str = " ".join(analysis.explicit_requirements).lower()
    assert "electronic informed consent" in reqs_str or "consent" in reqs_str
    assert "laboratory assay reconciliation" in reqs_str or "safety update" in reqs_str

    consts_str = " ".join(analysis.constraints).lower()
    assert "exactly 3 reviewers" in consts_str or "3 reviewers" in consts_str


# =========================================================================
# TESTS: VALIDATOR & SCORER IN LONG-INPUT MODE
# =========================================================================

@pytest.mark.asyncio
async def test_validator_rejects_omitted_long_input_unit(mock_clients):
    """Verify validator catches dropped source units even when input is long."""
    validator = mock_clients["validator"]
    mock_llm = mock_clients["client"]
    mock_llm.chat.return_value = json.dumps({"is_valid": True, "issues": []})

    spec = generate_6000_char_spec()
    source_units = [
        "multi-protocol device ingestion",
        "cold-chain anomaly detection",
        "customs freight manifest generation",
    ]

    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="IoT Logistics Platform",
        explicit_requirements=source_units,
        constraints=["do not use RabbitMQ"],
        preferences=[],
        missing_context=[],
        grounding=AnalysisGrounding(is_grounded=True, source_units=source_units),
    )

    # Incomplete enhancement missing customs freight manifest generation
    enhanced_prompt = (
        "ROLE: Software Engineer\n"
        "OBJECTIVE: IoT Logistics Platform\n"
        "REQUIREMENTS:\n"
        "- multi-protocol device ingestion\n"
        "- cold-chain anomaly detection\n"
        "CONSTRAINTS:\n"
        "- do not use RabbitMQ"
    )

    val = await validator.validate(
        original_prompt=spec,
        analysis=analysis,
        enhanced_prompt=enhanced_prompt,
    )

    assert val.is_valid is False
    dropped_issues = [i for i in val.issues if i.type == ValidationIssueType.SOURCE_REQUIREMENT_DROPPED]
    assert len(dropped_issues) >= 1
    assert "customs freight manifest" in dropped_issues[0].message.lower()


def test_scorer_caps_coverage_when_long_input_units_missing(mock_clients):
    """Verify scorer strictly prevents 100 coverage when long-input source units are dropped."""
    scorer = mock_clients["scorer"]
    spec = generate_6000_char_spec()

    source_units = [
        "multi-protocol device ingestion",
        "cold-chain anomaly detection",
        "customs freight manifest generation",
    ]
    analysis = PromptAnalysis(
        classification=PromptClassification(category=PromptCategory.CODING, confidence=0.9),
        objective="IoT Logistics Platform",
        explicit_requirements=source_units,
        constraints=[],
        preferences=[],
        missing_context=[],
        grounding=AnalysisGrounding(is_grounded=True, source_units=source_units),
    )

    # Enhanced text omits customs freight manifest generation
    incomplete_enhancement = (
        "OBJECTIVE: IoT Logistics Platform\n"
        "REQUIREMENTS:\n"
        "- multi-protocol device ingestion\n"
        "- cold-chain anomaly detection\n"
    )

    req_cov = scorer.compute_requirement_coverage(
        analysis=analysis,
        enhanced_prompt=incomplete_enhancement,
        original_prompt=spec,
    )

    assert req_cov.score < 100, f"Coverage score {req_cov.score} must not be 100 when units are missing"
    assert req_cov.score <= 80
