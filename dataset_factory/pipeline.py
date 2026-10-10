import json
import logging
import os
from typing import Any, Dict, List, Optional

from dataset_factory.candidate_builder import build_synthetic_candidate
from dataset_factory.config import DatasetFactoryConfig
from dataset_factory.curation import validate_preference_pair
from dataset_factory.dedup import deduplicate_records
from dataset_factory.evaluator import PipelineEvaluator, TeacherEvaluation
from dataset_factory.exporter import DatasetExporter
from dataset_factory.filters import filter_candidate
from dataset_factory.real_data_loader import load_consented_real_data
from dataset_factory.schemas import (
    CurationStatus,
    DatasetManifest,
    DatasetRecord,
    SeedIntent,
    SourceType,
    SyntheticVariant,
)
from dataset_factory.seed_loader import get_category_distribution, load_seeds
from dataset_factory.splitter import split_records_by_group
from dataset_factory.stats import compute_dataset_statistics
from dataset_factory.synthetic_generator import SyntheticGenerator

logger = logging.getLogger("dataset_factory.pipeline")


class DatasetFactoryPipeline:
    """End-to-end dataset factory pipeline."""

    def __init__(self, config: DatasetFactoryConfig):
        self.config = config
        self.generator = SyntheticGenerator(
            model=config.generator_model,
            temperature=config.generator_temperature,
            cache_dir=config.cache_dir,
            random_seed=config.random_seed,
        )
        self.evaluator = PipelineEvaluator(cache_dir=config.cache_dir)
        version_dir = os.path.join(config.output_dir, config.dataset_version)
        self.exporter = DatasetExporter(export_dir=version_dir)

    def _get_intermediate_path(self) -> str:
        inter_dir = os.path.join("datasets", "intermediate")
        os.makedirs(inter_dir, exist_ok=True)
        return os.path.join(inter_dir, f"{self.config.dataset_version}_candidates.jsonl")

    async def run(self) -> Dict[str, Any]:
        """Execute the dataset factory build."""
        logger.info("Initializing Dataset Factory build: %s", self.config.dataset_version)

        # 1. Load Seed Intents
        seeds: List[SeedIntent] = []
        if not self.config.real_only:
            seeds = load_seeds(self.config.seeds_path)
            cat_dist = get_category_distribution(seeds)
            logger.info("Loaded %d seed intents across %d categories.", len(seeds), len(cat_dist))

        # 2. Dry Run Mode
        if self.config.dry_run:
            planned_variants = len(seeds) * self.config.synthetic_per_seed
            logger.info("[DRY RUN] Planned synthetic variants: %d", planned_variants)
            return {
                "dry_run": True,
                "dataset_version": self.config.dataset_version,
                "seed_count": len(seeds),
                "planned_synthetic_candidates": planned_variants,
                "category_distribution": get_category_distribution(seeds) if seeds else {},
                "config": self.config.model_dump(),
            }

        # 3. Check for Resumable Intermediate Checkpoints
        intermediate_path = self._get_intermediate_path()
        all_candidates: List[DatasetRecord] = []
        preference_pairs: List[Dict[str, Any]] = []

        if os.path.exists(intermediate_path) and not self.config.force:
            logger.info("Resuming from existing intermediate checkpoint: %s", intermediate_path)
            with open(intermediate_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        all_candidates.append(DatasetRecord.model_validate_json(line))
            logger.info("Loaded %d candidate records from checkpoint.", len(all_candidates))

        else:
            # 4. Generate Synthetic Candidates
            if not self.config.real_only:
                for idx, seed in enumerate(seeds, 1):
                    logger.info("Processing seed %d/%d: %s (%s)", idx, len(seeds), seed.seed_id, seed.category)
                    try:
                        variants = await self.generator.generate_variants_for_seed(
                            seed=seed,
                            target_count=self.config.synthetic_per_seed,
                            use_llm=True,
                        )
                        for var in variants:
                            eval_res = await self.evaluator.evaluate_prompt(var.prompt, force=self.config.force)
                            candidate = build_synthetic_candidate(
                                seed=seed,
                                variant=var,
                                evaluation=eval_res,
                                dataset_version=self.config.dataset_version,
                                model=self.config.generator_model,
                            )
                            all_candidates.append(candidate)

                            # Check for potential preference pair (repaired enhancement vs initial rejected)
                            if eval_res.repaired and eval_res.is_valid and eval_res.initial_rejected_prompt:
                                is_valid_pair, pair_issue = validate_preference_pair(
                                    prompt=var.prompt,
                                    chosen=eval_res.enhanced_prompt,
                                    rejected=eval_res.initial_rejected_prompt,
                                    category=seed.category,
                                    seed=seed,
                                    initial_rejection_reason=eval_res.initial_rejection_reason,
                                )
                                if is_valid_pair:
                                    preference_pairs.append({
                                        "prompt": var.prompt,
                                        "chosen": eval_res.enhanced_prompt,
                                        "rejected": eval_res.initial_rejected_prompt,
                                        "metadata": {
                                            "reason": eval_res.initial_rejection_reason or "Validator rejection repair",
                                            "category": seed.category,
                                        },
                                    })
                                else:
                                    logger.debug("Rejected preference pair: %s", pair_issue)

                    except Exception as e:
                        logger.error("Error generating variants for seed %s: %s", seed.seed_id, e)
                        continue

            # 5. Ingest Consented Real Data
            if self.config.include_consented_real and not self.config.synthetic_only:
                db_url = self.config.database_url or "sqlite:///./data/humanintent.db"
                try:
                    real_records = load_consented_real_data(
                        database_url=db_url,
                        dataset_version=self.config.dataset_version,
                    )
                    all_candidates.extend(real_records)
                except Exception as e:
                    logger.warning("Could not load consented real history: %s", e)

            # Save checkpoint
            with open(intermediate_path, "w", encoding="utf-8") as f:
                for c in all_candidates:
                    f.write(c.model_dump_json() + "\n")
            logger.info("Saved %d candidates to intermediate checkpoint.", len(all_candidates))

        # 6. Apply Staged Filtering Pipeline
        logger.info("Applying quality and safety filters to %d candidates...", len(all_candidates))
        seed_map = {s.seed_id: s for s in seeds}

        filtered_candidates: List[DatasetRecord] = []
        for candidate in all_candidates:
            seed = seed_map.get(candidate.provenance.seed_id) if candidate.provenance.seed_id else None
            status, reasons = filter_candidate(candidate, self.config, seed)
            candidate.curation_status = status
            candidate.rejection_reasons = reasons
            filtered_candidates.append(candidate)

        # 7. Deduplicate Accepted Records
        accepted_pool = [c for c in filtered_candidates if c.curation_status == CurationStatus.ACCEPTED]
        rejected_pool = [c for c in filtered_candidates if c.curation_status == CurationStatus.REJECTED]

        deduped_accepted, duplicate_records = deduplicate_records(
            accepted_pool,
            threshold=self.config.near_duplicate_threshold,
        )
        rejected_pool.extend(duplicate_records)
        all_processed = deduped_accepted + rejected_pool

        # 8. Split by Seed / Session Group
        train_records, val_records = split_records_by_group(
            deduped_accepted,
            train_ratio=self.config.train_split_ratio,
            random_seed=self.config.random_seed,
        )
        logger.info(
            "Split %d accepted records into Train: %d, Validation: %d",
            len(deduped_accepted),
            len(train_records),
            len(val_records),
        )

        # 9. Compute Statistics & Report
        stats = compute_dataset_statistics(all_processed)

        # Compute curation-specific counters
        curation_rejections: Dict[str, int] = {}
        source_dependency_count = 0
        constraint_recall_failures = 0
        malformed_target_rejections = 0

        for r in rejected_pool:
            for reason in r.rejection_reasons:
                curation_rejections[reason.value] = curation_rejections.get(reason.value, 0) + 1
                if reason.value == "SOURCE_MISSING":
                    source_dependency_count += 1
                elif reason.value == "MISSING_CONSTRAINT":
                    constraint_recall_failures += 1
                elif reason.value in ("MALFORMED_TARGET", "PLACEHOLDER_CONTENT"):
                    malformed_target_rejections += 1

        target_integrity_pass_rate = round(len(deduped_accepted) / max(len(all_processed), 1), 4)

        # 10. Build Manifest
        manifest = DatasetManifest(
            version=self.config.dataset_version,
            factory_version="0.1.0",
            random_seed=self.config.random_seed,
            generator_model=self.config.generator_model,
            record_counts={
                "total": len(all_processed),
                "accepted": len(deduped_accepted),
                "rejected": len(rejected_pool),
            },
            category_counts=stats.get("category_counts", {}),
            source_counts=stats.get("source_counts", {}),
            excluded_counts=stats.get("rejection_reasons", {}),
            filter_reasons=stats.get("rejection_reasons", {}),
            quality_distribution=stats.get("quality_distribution", {}),
            consent_count=sum(1 for r in all_processed if r.source_type == SourceType.CONSENTED_REAL),
            sanitization_count=stats.get("sanitization_count", 0),
            dedup_count=len(duplicate_records),
            split_counts={"train": len(train_records), "validation": len(val_records)},
            generation_parameters={
                "temperature": self.config.generator_temperature,
                "synthetic_per_seed": self.config.synthetic_per_seed,
            },
            filter_configuration={
                "minimum_quality_score": self.config.minimum_quality_score,
                "max_inflation_ratio": self.config.max_inflation_ratio,
                "near_duplicate_threshold": self.config.near_duplicate_threshold,
            },
            curation_rejection_counts=curation_rejections,
            target_integrity_pass_rate=target_integrity_pass_rate,
            source_dependency_rejections=source_dependency_count,
            constraint_recall_failures=constraint_recall_failures,
            malformed_target_rejections=malformed_target_rejections,
        )

        # 11. Export All Artifacts
        logger.info("Exporting dataset artifacts...")
        sft_tr_path = self.exporter.export_sft(train_records, "train")
        sft_va_path = self.exporter.export_sft(val_records, "validation")
        clar_tr_path = self.exporter.export_clarification(train_records, "train")
        clar_va_path = self.exporter.export_clarification(val_records, "validation")
        pref_tr_path = self.exporter.export_preference(train_records, preference_pairs, "train")
        pref_va_path = self.exporter.export_preference(val_records, [], "validation")
        review_path = self.exporter.export_review_sample(
            deduped_accepted,
            sample_size=self.config.review_sample_size,
            random_seed=self.config.random_seed,
        )
        manifest_path = self.exporter.export_manifest(manifest)
        card_path = self.exporter.export_dataset_card(manifest, stats)

        logger.info("Dataset Factory build complete. Exports saved to: %s", self.exporter.export_dir)

        return {
            "dataset_version": self.config.dataset_version,
            "export_dir": self.exporter.export_dir,
            "manifest_path": manifest_path,
            "dataset_card_path": card_path,
            "review_sample_path": review_path,
            "sft_train_path": sft_tr_path,
            "sft_val_path": sft_va_path,
            "clarification_train_path": clar_tr_path,
            "clarification_val_path": clar_va_path,
            "preference_train_path": pref_tr_path,
            "stats": stats,
            "manifest": manifest.model_dump(),
        }
