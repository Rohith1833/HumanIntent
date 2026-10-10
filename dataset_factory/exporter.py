import json
import os
import random
from typing import Any, Dict, List, Optional
from dataset_factory.schemas import (
    ClarificationExample,
    CurationStatus,
    DatasetManifest,
    DatasetRecord,
    PreferenceExample,
    SFTExample,
    SFTMessage,
)


class DatasetExporter:
    """Exports curated HumanIntent datasets to versioned JSONL files and documentation."""

    def __init__(self, export_dir: str):
        self.export_dir = export_dir

    def _ensure_dir(self):
        os.makedirs(self.export_dir, exist_ok=True)

    def export_sft(
        self,
        records: List[DatasetRecord],
        split_name: str = "train",
    ) -> str:
        """Export positive SFT examples (excluding clarification-only prompts)."""
        self._ensure_dir()
        path = os.path.join(self.export_dir, f"sft_{split_name}.jsonl")
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                if r.curation_status != CurationStatus.ACCEPTED:
                    continue
                if r.clarification_required or not r.enhanced_prompt:
                    continue

                example = SFTExample(
                    messages=[
                        SFTMessage(role="user", content=r.original_prompt),
                        SFTMessage(role="assistant", content=r.enhanced_prompt),
                    ],
                    metadata={
                        "record_id": r.record_id,
                        "category": r.category,
                        "quality_score": r.quality.quality_score,
                        "source_type": r.source_type.value,
                        "provenance": r.provenance.model_dump(),
                        "labels": r.labels.model_dump(),
                    },
                )
                f.write(example.model_dump_json() + "\n")
                count += 1
        return path

    def export_clarification(
        self,
        records: List[DatasetRecord],
        split_name: str = "train",
    ) -> str:
        """Export ambiguity clarification pairs."""
        self._ensure_dir()
        path = os.path.join(self.export_dir, f"clarification_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                if r.curation_status != CurationStatus.ACCEPTED:
                    continue
                if not r.clarification_required or not r.clarification_questions:
                    continue

                # Format clarification assistant message
                questions_formatted = "\n".join(f"- {q}" for q in r.clarification_questions)
                content = (
                    "To provide the best enhancement, could you please clarify the following:\n"
                    f"{questions_formatted}"
                )

                example = ClarificationExample(
                    messages=[
                        SFTMessage(role="user", content=r.original_prompt),
                        SFTMessage(role="assistant", content=content),
                    ],
                    metadata={
                        "record_id": r.record_id,
                        "category": r.category,
                        "source_type": r.source_type.value,
                        "provenance": r.provenance.model_dump(),
                    },
                )
                f.write(example.model_dump_json() + "\n")
        return path

    def export_preference(
        self,
        records: List[DatasetRecord],
        preference_pairs: List[Dict[str, Any]],
        split_name: str = "train",
    ) -> str:
        """Export chosen vs. rejected pairs where traceable evidence exists."""
        self._ensure_dir()
        path = os.path.join(self.export_dir, f"preference_{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for pair in preference_pairs:
                example = PreferenceExample(
                    prompt=pair["prompt"],
                    chosen=pair["chosen"],
                    rejected=pair["rejected"],
                    metadata=pair.get("metadata", {}),
                )
                f.write(example.model_dump_json() + "\n")
        return path

    def export_review_sample(
        self,
        accepted_records: List[DatasetRecord],
        sample_size: int = 50,
        random_seed: int = 42,
    ) -> str:
        """Export a stratified random sample of accepted records for human review."""
        self._ensure_dir()
        path = os.path.join(self.export_dir, "review_sample.jsonl")
        rng = random.Random(random_seed)

        # Stratify by category
        by_category: Dict[str, List[DatasetRecord]] = {}
        for r in accepted_records:
            by_category.setdefault(r.category, []).append(r)

        sample: List[DatasetRecord] = []
        per_cat = max(1, sample_size // max(1, len(by_category)))
        for cat, recs in by_category.items():
            k = min(len(recs), per_cat)
            sample.extend(rng.sample(recs, k))

        # Fill remaining slots up to sample_size if needed
        remaining_pool = [r for r in accepted_records if r not in sample]
        if len(sample) < sample_size and remaining_pool:
            needed = min(len(remaining_pool), sample_size - len(sample))
            sample.extend(rng.sample(remaining_pool, needed))

        with open(path, "w", encoding="utf-8") as f:
            for r in sample:
                summary = {
                    "record_id": r.record_id,
                    "category": r.category,
                    "source_type": r.source_type.value,
                    "original_prompt": r.original_prompt,
                    "enhanced_prompt": r.enhanced_prompt,
                    "clarification_required": r.clarification_required,
                    "quality_score": r.quality.quality_score,
                    "inflation_ratio": r.quality.inflation_ratio,
                    "labels": r.labels.model_dump(),
                    "sanitized": r.privacy.sanitized,
                }
                f.write(json.dumps(summary) + "\n")
        return path

    def export_manifest(self, manifest: DatasetManifest) -> str:
        """Write the authoritative build manifest.json."""
        self._ensure_dir()
        path = os.path.join(self.export_dir, "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))
        return path

    def export_dataset_card(
        self,
        manifest: DatasetManifest,
        stats: Dict[str, Any],
    ) -> str:
        """Generate human-readable DATASET_CARD.md."""
        self._ensure_dir()
        path = os.path.join(self.export_dir, "DATASET_CARD.md")
        card_content = (
            f"# HumanIntent Dataset Card: {manifest.version}\n\n"
            f"**Created At:** {manifest.created_at}  \n"
            f"**Factory Version:** {manifest.factory_version}  \n"
            f"**Generator Model:** {manifest.generator_model}  \n"
            f"**Random Seed:** {manifest.random_seed}  \n\n"
            f"## Purpose & Philosophy\n"
            f"This dataset is constructed by the HumanIntent Dataset Factory to train future intent-preserving "
            f"prompt optimization models via Supervised Fine-Tuning (SFT) and Preference Optimization.\n\n"
            f"### Training-Data Acceptable != Runtime Acceptable\n"
            f"Teacher outputs are NEVER automatically trusted for training-data admission. Even when a teacher output "
            f"achieves a high runtime quality score (>= 75) and passes runtime validation, it must independently pass "
            f"the **Dataset Curation Gate**. The curation gate strictly enforces requirement recall, negative constraint "
            f"preservation, quantity bounds, task-family consistency, structural well-formedness, and source dependency checks.\n\n"
            f"## Consent & Privacy Policy\n"
            f"- **Consented Real Data**: Sourced exclusively from local interaction history where `dataset_consent == True`. "
            f"Unconsented interactions are strictly filtered at the SQL query level.\n"
            f"- **Privacy Sanitation**: All prompts undergo automated regex redaction of emails, phone numbers, API keys, "
            f"bearer tokens, passwords, and private URLs. High-risk unredactable credentials trigger immediate record rejection.\n\n"
            f"## Dataset Statistics\n"
            f"- **Total Candidates Processed:** {stats.get('total_candidates', 0)}\n"
            f"- **Accepted Records:** {stats.get('accepted_count', 0)}\n"
            f"- **Rejected Records:** {stats.get('rejected_count', 0)}\n"
            f"- **Acceptance Rate:** {stats.get('acceptance_rate', 0.0):.1%}\n"
            f"- **Sanitized Records Count:** {stats.get('sanitization_count', 0)}\n\n"
            f"### Curation Gate Breakdown\n"
            f"- **Target Integrity Pass Rate:** {manifest.target_integrity_pass_rate:.1%}\n"
            f"- **Source Dependency Rejections:** {manifest.source_dependency_rejections}\n"
            f"- **Constraint Recall Failures:** {manifest.constraint_recall_failures}\n"
            f"- **Malformed Target Rejections:** {manifest.malformed_target_rejections}\n"
            f"- **Detailed Rejection Counts:**\n"
            + "\n".join(f"  - `{k}`: {v}" for k, v in manifest.curation_rejection_counts.items())
            + f"\n\n### Category Coverage\n"
            + "\n".join(f"- **{cat}:** {cnt}" for cat, cnt in stats.get("category_counts", {}).items())
            + f"\n\n### Quality & Expansion\n"
            f"- **Average Quality Score:** {stats.get('quality_distribution', {}).get('mean', 0.0)}/100\n"
            f"- **Median Prompt Expansion:** {stats.get('expansion_metrics', {}).get('median_expansion', 1.0)}x\n"
            f"- **P90 Prompt Expansion:** {stats.get('expansion_metrics', {}).get('p90_expansion', 1.0)}x\n\n"
            f"### Splits\n"
            f"- **Train Groups:** {manifest.split_counts.get('train', 0)}\n"
            f"- **Validation Groups:** {manifest.split_counts.get('validation', 0)}\n\n"
            f"## Known Limitations & Self-Distillation Risks\n"
            f"- **Self-Distillation Bias**: Training on teacher outputs risks inheriting latent stylistic biases and template phrasing from the teacher model.\n"
            f"- **Synthetic Stereotypes**: Synthetic variations can exhibit modal linguistic patterns; independent invariant gates reduce but do not eliminate this.\n"
            f"- **No Online Runtime Impact**: Curation is strictly an offline process; the runtime `/api/enhance` path remains untouched and lightweight.\n"
            f"- **No Training Conducted**: This phase constructs and hardens training data only; no model fine-tuning, LoRA, or embeddings have been applied.\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(card_content)
        return path
