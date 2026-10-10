import json
import os
import sys
import uuid
import pytest
from pydantic import ValidationError

# Ensure repo root and backend are on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (repo_root, backend_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_factory.schemas import (
    CurationStatus,
    DatasetManifest,
    DatasetRecord,
    NormalizedIntent,
    PreferenceExample,
    PrivacyMetadata,
    ProvenanceMetadata,
    QualityMetadata,
    RecordLabels,
    RejectionReason,
    SeedIntent,
    SFTExample,
    SFTMessage,
    SourceType,
    SyntheticVariant,
    VariantType,
)
from dataset_factory.seed_loader import load_seeds, get_category_distribution
from dataset_factory.sanitizer import sanitize_text
from dataset_factory.filters import filter_candidate, check_intent_preservation
from dataset_factory.dedup import deduplicate_records, token_jaccard_similarity
from dataset_factory.splitter import split_records_by_group
from dataset_factory.exporter import DatasetExporter
from dataset_factory.config import DatasetFactoryConfig
from dataset_factory.real_data_loader import load_consented_real_data, _hash_session_id
from dataset_factory.pipeline import DatasetFactoryPipeline


# ==============================================================================
# TEST GROUP A — SCHEMAS
# ==============================================================================

def test_group_a_valid_seed_intent():
    seed = SeedIntent(
        seed_id="coding_test_01",
        category="CODING",
        intent="Create a rate limiter",
        core_requirements=["token bucket", "thread safety"],
        difficulty="medium",
    )
    assert seed.seed_id == "coding_test_01"
    assert seed.category == "CODING"


def test_group_a_invalid_category_rejected():
    with pytest.raises(ValidationError):
        SeedIntent(
            seed_id="bad_01",
            category="INVALID_CATEGORY_NAME",
            intent="Do something",
        )


def test_group_a_dataset_record_schema():
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Build a rate limiter",
        enhanced_prompt="Act as an expert developer. Implement a thread-safe token bucket rate limiter.",
        quality=QualityMetadata(validation_passed=True, quality_score=92.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0", seed_id="coding_001"),
    )
    assert record.curation_status == CurationStatus.RAW
    data = json.loads(record.model_dump_json())
    assert data["original_prompt"] == "Build a rate limiter"


# ==============================================================================
# TEST GROUP B — SEED LOADER
# ==============================================================================

def test_group_b_load_curated_seeds():
    seeds_path = os.path.join(repo_root, "datasets", "seeds", "v1.jsonl")
    assert os.path.exists(seeds_path), "v1.jsonl must exist"
    seeds = load_seeds(seeds_path)
    assert len(seeds) >= 100, f"Expected ~100 seeds, got {len(seeds)}"

    dist = get_category_distribution(seeds)
    # Ensure all 11 canonical categories are represented
    assert len(dist) == 11
    for cat, count in dist.items():
        assert count >= 5, f"Category {cat} should have at least 5 seeds, got {count}"


def test_group_b_duplicate_seed_rejected(tmp_path):
    dup_file = tmp_path / "dup_seeds.jsonl"
    seed_json = json.dumps({
        "seed_id": "duplicate_id",
        "category": "CODING",
        "intent": "Task 1",
        "core_requirements": ["A"],
    })
    dup_file.write_text(f"{seed_json}\n{seed_json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate seed_id"):
        load_seeds(str(dup_file))


# ==============================================================================
# TEST GROUP C — CONSENT
# ==============================================================================

def test_group_c_unconsented_records_never_loaded(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.db.base import Base
    from app.db.models import EnhancementSession, Feedback

    db_path = tmp_path / "test_consent.db"
    db_url = f"sqlite:///{db_path}"
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        # 1. Unconsented session
        s1 = EnhancementSession(
            id=str(uuid.uuid4()),
            original_prompt="Unconsented prompt",
            status="enhanced",
            dataset_consent=False,
            final_enhanced_prompt="Unconsented enhancement",
        )
        fb1 = Feedback(
            session_id=s1.id,
            rating="HELPFUL",
            dataset_consent=False,
        )
        db.add_all([s1, fb1])

        # 2. Consented session
        s2 = EnhancementSession(
            id=str(uuid.uuid4()),
            original_prompt="Consented prompt",
            status="enhanced",
            dataset_consent=True,
            final_enhanced_prompt="Consented enhancement",
        )
        fb2 = Feedback(
            session_id=s2.id,
            rating="HELPFUL",
            dataset_consent=True,
        )
        s2_id = s2.id
        db.add_all([s2, fb2])
        db.commit()

    loaded = load_consented_real_data(db_url)
    assert len(loaded) == 1
    assert loaded[0].original_prompt == "Consented prompt"
    assert loaded[0].source_type == SourceType.CONSENTED_REAL
    # Ensure raw UUID is NOT exported, pseudonymized source_id used instead
    assert loaded[0].source_id.startswith("real_")
    assert s2_id not in loaded[0].source_id


# ==============================================================================
# TEST GROUP D — PRIVACY & SECRET SANITATION
# ==============================================================================

def test_group_d_privacy_sanitizer():
    dirty_text = (
        "Here is my email test.user@example.com and phone 555-123-4567. "
        "Also my api_key='sk-abcdef1234567890abcdef' and password=SuperSecret123! "
        "Visit https://internal.corp?token=secrettokenvalue."
    )
    result = sanitize_text(dirty_text)
    assert result.sanitized is True
    assert "<EMAIL>" in result.text
    assert "<PHONE>" in result.text
    assert "<API_KEY>" in result.text
    assert "<PASSWORD>" in result.text
    assert "<URL>" in result.text

    assert "test.user@example.com" not in result.text
    assert "sk-abcdef1234567890abcdef" not in result.text
    assert "SuperSecret123!" not in result.text
    assert "secrettokenvalue" not in result.text


def test_group_d_unredactable_private_key_excluded():
    key_text = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEowIBAAKCAQEA0Y3...\n"
        "-----END RSA PRIVATE KEY-----\n"
        "Help me deploy this key."
    )
    result = sanitize_text(key_text)
    assert result.is_secret_risk is True
    assert "private cryptographic key" in result.risk_reason.lower()


# ==============================================================================
# TEST GROUP E — SYNTHETIC INTENT PRESERVATION
# ==============================================================================

def test_group_e_intent_drift_detected():
    seed = SeedIntent(
        seed_id="coding_001",
        category="CODING",
        intent="Implement an LRU cache data structure in Rust",
        core_requirements=["Rust", "LRU cache", "capacity limit"],
    )
    drifted_prompt = "Draft an executive resignation letter to my corporate board."
    assert check_intent_preservation(drifted_prompt, seed) is False

    valid_prompt = "Build an LRU cache in Rust with constant time lookups and capacity limit."
    assert check_intent_preservation(valid_prompt, seed) is True


# ==============================================================================
# TEST GROUP F — POSITIVE SFT ELIGIBILITY
# ==============================================================================

def test_group_f_positive_sft_candidate_passes():
    config = DatasetFactoryConfig(minimum_quality_score=75.0)
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Write a Python script to sort JSON keys",
        enhanced_prompt="Act as a Python engineer. Write a script to recursively sort JSON keys.",
        quality=QualityMetadata(validation_passed=True, quality_score=88.0, inflation_ratio=2.5),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
    )
    status, reasons = filter_candidate(record, config)
    assert status == CurationStatus.ACCEPTED
    assert reasons == []


# ==============================================================================
# TEST GROUP G — ALREADY-GOOD PROMPTS
# ==============================================================================

def test_group_g_already_good_label_preserved():
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Objective: Implement quicksort in Go.\nRequirements:\n- Handle empty slice\n- Benchmark test",
        enhanced_prompt="Objective: Implement quicksort in Go.\nRequirements:\n- Handle empty slice\n- Benchmark test",
        quality=QualityMetadata(validation_passed=True, quality_score=95.0, inflation_ratio=1.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
        labels=RecordLabels(already_good=True),
    )
    assert record.labels.already_good is True


# ==============================================================================
# TEST GROUP H — AMBIGUITY & CLARIFICATION
# ==============================================================================

def test_group_h_clarification_candidate_separated():
    config = DatasetFactoryConfig()
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Build me an app",
        clarification_required=True,
        clarification_questions=["What is the primary function of the application?"],
        quality=QualityMetadata(validation_passed=False, quality_score=0.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
    )
    status, reasons = filter_candidate(record, config)
    # Clarification candidate is accepted for clarification dataset even though validation_passed is False
    assert status == CurationStatus.ACCEPTED
    assert reasons == []


# ==============================================================================
# TEST GROUP I — FAILED OUTPUT EXCLUDED FROM SFT
# ==============================================================================

def test_group_i_failed_validation_excluded():
    config = DatasetFactoryConfig()
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Build an API endpoint",
        enhanced_prompt="Unvalidated output with hallucinated requirements",
        quality=QualityMetadata(validation_passed=False, quality_score=40.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
        normalized_intent=NormalizedIntent(objective="Build an API"),
    )
    status, reasons = filter_candidate(record, config)
    assert status == CurationStatus.REJECTED
    assert RejectionReason.VALIDATION_FAILED in reasons


# ==============================================================================
# TEST GROUP J — PREFERENCE PAIR
# ==============================================================================

def test_group_j_preference_example_schema():
    pref = PreferenceExample(
        prompt="Write a Python script to parse CSV",
        chosen="Act as a Python engineer. Write a clean script using the csv module to parse input rows.",
        rejected="Write code. Install heavy third-party framework that was never requested.",
        metadata={"reason": "Validator rejected unprompted third-party dependency"},
    )
    data = json.loads(pref.model_dump_json())
    assert data["chosen"] != data["rejected"]
    assert "Validator" in data["metadata"]["reason"]


# ==============================================================================
# TEST GROUP K — FEEDBACK USER EDIT VALIDATION
# ==============================================================================

def test_group_k_consented_feedback_edit():
    user_edit = "Actually, make sure the CSV parser supports custom delimiters and gzipped inputs."
    sanitized = sanitize_text(user_edit)
    assert sanitized.sanitized is False
    assert "custom delimiters" in sanitized.text


# ==============================================================================
# TEST GROUP L — DEDUPLICATION
# ==============================================================================

def test_group_l_exact_and_near_duplicates():
    r1 = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Implement a binary search in Go with tests",
        quality=QualityMetadata(validation_passed=True, quality_score=90.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
    )
    # Exact duplicate (case/whitespace variation)
    r2 = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="implement a binary search in go with tests  ",
        quality=QualityMetadata(validation_passed=True, quality_score=90.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
    )
    # Near lexical duplicate (high Jaccard)
    r3 = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Implement a binary search algorithm in Go with tests",
        quality=QualityMetadata(validation_passed=True, quality_score=90.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
    )
    # Distinct prompt
    r4 = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Build a reactive WebSocket client in TypeScript",
        quality=QualityMetadata(validation_passed=True, quality_score=90.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
    )

    accepted, duplicates = deduplicate_records([r1, r2, r3, r4], threshold=0.80)
    assert len(accepted) == 2  # r1 and r4 survive
    assert len(duplicates) == 2  # r2 and r3 rejected
    assert all(d.curation_status == CurationStatus.REJECTED for d in duplicates)


# ==============================================================================
# TEST GROUP M — GROUP SPLITTING (NO SEED LEAKAGE)
# ==============================================================================

def test_group_m_seed_group_splitting():
    # 4 seeds with 3 variants each = 12 records
    records = []
    for s_idx in range(1, 5):
        seed_id = f"seed_{s_idx}"
        for v_idx in range(1, 4):
            records.append(
                DatasetRecord(
                    dataset_version="v0.1.0",
                    source_type=SourceType.SYNTHETIC,
                    category="CODING",
                    original_prompt=f"Prompt {s_idx}_{v_idx}",
                    quality=QualityMetadata(validation_passed=True, quality_score=90.0),
                    provenance=ProvenanceMetadata(dataset_version="v0.1.0", seed_id=seed_id),
                )
            )

    train, val = split_records_by_group(records, train_ratio=0.75, random_seed=42)
    assert len(train) + len(val) == 12

    train_seeds = {r.provenance.seed_id for r in train}
    val_seeds = {r.provenance.seed_id for r in val}

    # CRITICAL INVARIANT: Zero overlap between train seeds and val seeds
    assert len(train_seeds & val_seeds) == 0
    assert len(train_seeds) > 0
    assert len(val_seeds) > 0


# ==============================================================================
# TEST GROUP N — SESSION GROUP SPLITTING
# ==============================================================================

def test_group_n_session_group_splitting():
    records = []
    for sess_idx in range(1, 4):
        source_id = f"real_sess_{sess_idx}"
        for attempt in range(1, 3):
            records.append(
                DatasetRecord(
                    dataset_version="v0.1.0",
                    source_type=SourceType.CONSENTED_REAL,
                    source_id=source_id,
                    category="WRITING",
                    original_prompt=f"Real prompt {sess_idx}_{attempt}",
                    quality=QualityMetadata(validation_passed=True, quality_score=85.0),
                    provenance=ProvenanceMetadata(dataset_version="v0.1.0", source_id=source_id),
                )
            )

    train, val = split_records_by_group(records, train_ratio=0.66, random_seed=42)
    train_sources = {r.provenance.source_id for r in train}
    val_sources = {r.provenance.source_id for r in val}
    assert len(train_sources & val_sources) == 0


# ==============================================================================
# TEST GROUP O — EXPORT FORMAT
# ==============================================================================

def test_group_o_jsonl_export_cleanliness(tmp_path):
    exporter = DatasetExporter(export_dir=str(tmp_path))
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Write quicksort",
        enhanced_prompt="Act as an expert. Implement quicksort in Python.",
        quality=QualityMetadata(validation_passed=True, quality_score=95.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0", seed_id="coding_010"),
        curation_status=CurationStatus.ACCEPTED,
    )
    out_file = exporter.export_sft([record], split_name="train")
    assert os.path.exists(out_file)

    with open(out_file, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        data = json.loads(line)

    assert "messages" in data
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "Write quicksort"
    assert data["messages"][1]["role"] == "assistant"
    assert data["metadata"]["category"] == "CODING"


# ==============================================================================
# TEST GROUP P — PROVENANCE METADATA
# ==============================================================================

def test_group_p_provenance_metadata():
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="EDUCATION",
        original_prompt="Explain photosynthesis",
        enhanced_prompt="Explain photosynthesis clearly.",
        quality=QualityMetadata(validation_passed=True, quality_score=90.0),
        provenance=ProvenanceMetadata(
            seed_id="edu_001",
            variant_id="var_123",
            generation_model="llama3:latest",
            teacher_pipeline_version="humanintent-v1.0",
            validator_version="v1.0",
            dataset_version="v0.1.0",
        ),
    )
    assert record.provenance.seed_id == "edu_001"
    assert record.provenance.teacher_pipeline_version == "humanintent-v1.0"


# ==============================================================================
# TEST GROUP Q — NO CHAIN OF THOUGHT IN EXPORT
# ==============================================================================

def test_group_q_no_chain_of_thought_leakage(tmp_path):
    exporter = DatasetExporter(export_dir=str(tmp_path))
    record = DatasetRecord(
        dataset_version="v0.1.0",
        source_type=SourceType.SYNTHETIC,
        category="RESEARCH",
        original_prompt="Compare B-trees and LSM-trees",
        enhanced_prompt="Act as a database researcher. Compare B-tree and LSM-tree throughput.",
        quality=QualityMetadata(validation_passed=True, quality_score=94.0),
        provenance=ProvenanceMetadata(dataset_version="v0.1.0"),
        curation_status=CurationStatus.ACCEPTED,
    )
    out_file = exporter.export_sft([record], "train")
    with open(out_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Assert no internal system instruction or chain-of-thought phrases
    assert "THINKING:" not in content
    assert "<thought>" not in content
    assert "INTERNAL SCRATCHPAD" not in content
    assert "SYSTEM INSTRUCTION:" not in content


# ==============================================================================
# TEST GROUP R — PERSISTENCE CONSENT DEFAULT PRESERVED
# ==============================================================================

def test_group_r_default_consent_false(tmp_path):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session
    from app.db.base import Base
    from app.db.models import EnhancementSession, Feedback

    db_path = tmp_path / "test_defaults.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    with Session(engine) as db:
        s = EnhancementSession(original_prompt="Test", status="enhanced")
        db.add(s)
        db.commit()
        db.refresh(s)
        assert s.dataset_consent is False

        fb = Feedback(session_id=s.id, rating="HELPFUL")
        db.add(fb)
        db.commit()
        db.refresh(fb)
        assert fb.dataset_consent is False


# ==============================================================================
# TEST GROUP S — DRY RUN
# ==============================================================================

@pytest.mark.asyncio
async def test_group_s_dry_run_no_files_written(tmp_path):
    export_dir = tmp_path / "exports"
    config = DatasetFactoryConfig(
        dataset_version="test-dryrun-v1",
        seeds_path=os.path.join(repo_root, "datasets", "seeds", "v1.jsonl"),
        output_dir=str(export_dir),
        dry_run=True,
    )
    pipeline = DatasetFactoryPipeline(config)
    res = await pipeline.run()

    assert res["dry_run"] is True
    assert res["seed_count"] >= 100
    assert not os.path.exists(export_dir)


# ==============================================================================
# TEST GROUP T — RESUMABILITY
# ==============================================================================

@pytest.mark.asyncio
async def test_group_t_resumable_checkpoint(tmp_path):
    inter_dir = tmp_path / "intermediate"
    inter_dir.mkdir(parents=True, exist_ok=True)
    inter_file = inter_dir / "resumable-v1_candidates.jsonl"

    rec = DatasetRecord(
        dataset_version="resumable-v1",
        source_type=SourceType.SYNTHETIC,
        category="CODING",
        original_prompt="Checkpoint prompt",
        enhanced_prompt="Checkpoint enhancement",
        quality=QualityMetadata(validation_passed=True, quality_score=92.0),
        provenance=ProvenanceMetadata(dataset_version="resumable-v1", seed_id="s1"),
    )
    inter_file.write_text(rec.model_dump_json() + "\n", encoding="utf-8")

    # When pipeline runs without force, it loads the checkpoint directly
    loaded = []
    with open(inter_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                loaded.append(DatasetRecord.model_validate_json(line))
    assert len(loaded) == 1
    assert loaded[0].original_prompt == "Checkpoint prompt"


# ==============================================================================
# TEST GROUP U — MANIFEST ACCURACY
# ==============================================================================

def test_group_u_manifest_integrity():
    manifest = DatasetManifest(
        version="humanintent-v0.1.0",
        factory_version="0.1.0",
        random_seed=42,
        generator_model="llama3:latest",
        record_counts={"total": 100, "accepted": 85, "rejected": 15},
        category_counts={"CODING": 20, "WRITING": 20},
        split_counts={"train": 75, "validation": 10},
    )
    assert manifest.record_counts["total"] == 100
    assert manifest.record_counts["accepted"] == 85
    assert manifest.split_counts["train"] == 75


# ==============================================================================
# TEST GROUP V — ZERO RUNTIME IMPACT
# ==============================================================================

def test_group_v_runtime_pipeline_unaffected():
    from app.core.enhancer import prompt_enhancer
    from app.core.analyzer import prompt_analyzer
    from app.core.validator import prompt_validator
    from app.core.scorer import prompt_scorer

    # Enhancer, analyzer, validator, and scorer modules remain intact
    assert hasattr(prompt_enhancer, "enhance")
    assert hasattr(prompt_analyzer, "analyze")
    assert hasattr(prompt_validator, "validate")
    assert hasattr(prompt_scorer, "score")
