import hashlib
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

from dataset_factory.schemas import SeedIntent, SyntheticVariant, VariantType

logger = logging.getLogger("dataset_factory.synthetic_generator")


def _cache_key(seed_id: str, variant_type: str, model: str, salt: str = "v1") -> str:
    content = f"{seed_id}:{variant_type}:{model}:{salt}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


class SyntheticGenerator:
    """Generates varied, realistic human prompt expressions from seed intents."""

    def __init__(
        self,
        model: str = "llama3:latest",
        temperature: float = 0.7,
        cache_dir: str = "datasets/cache",
        random_seed: int = 42,
    ):
        self.model = model
        self.temperature = temperature
        self.cache_dir = cache_dir
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
        os.makedirs(cache_dir, exist_ok=True)

    def _get_cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"gen_{key}.json")

    def _read_cache(self, key: str) -> Optional[SyntheticVariant]:
        path = self._get_cache_path(key)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return SyntheticVariant.model_validate(data)
            except Exception as e:
                logger.debug("Cache read error for %s: %s", key, e)
        return None

    def _write_cache(self, key: str, variant: SyntheticVariant) -> None:
        path = self._get_cache_path(key)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(variant.model_dump_json(indent=2))
        except Exception as e:
            logger.debug("Cache write error for %s: %s", key, e)

    def generate_deterministic_variant(
        self,
        seed: SeedIntent,
        variant_type: VariantType,
    ) -> SyntheticVariant:
        """High-fidelity deterministic template generator for testing and offline builds."""
        reqs = seed.core_requirements
        req_str = ", ".join(reqs) if reqs else "the core functionality"
        constraints_str = ", ".join(seed.optional_constraints) if seed.optional_constraints else ""

        intent = seed.intent
        # Strip trailing punctuation
        intent_clean = intent.rstrip(".!?")

        if variant_type == VariantType.CLEAN:
            prompt = f"Please {intent_clean}. Core requirements: {req_str}."
            if constraints_str:
                prompt += f" Ensure {constraints_str}."

        elif variant_type == VariantType.CASUAL:
            prompt = f"hey can you help me {intent_clean.lower()}? I need {req_str}."
            if constraints_str:
                prompt += f" also {constraints_str} if possible thanks"

        elif variant_type == VariantType.MESSY:
            prompt = (
                f"So I've been thinking about this... basically I want to {intent_clean.lower()}. "
                f"Not really sure about the best way but it definitely has to include {req_str}. "
                f"Oh and also {constraints_str or 'make sure it works well'}, let me know what you think!"
            )

        elif variant_type == VariantType.SHORT:
            prompt = f"{intent_clean}: {req_str}"

        elif variant_type == VariantType.COMPOUND:
            prompt = (
                f"First, {intent_clean} with {req_str}. "
                f"Additionally, provide a structured summary of the design choices and key tradeoffs."
            )

        elif variant_type == VariantType.SELF_CORRECTING:
            prompt = (
                f"I need to {intent_clean.lower()} using MySQL... wait actually scratch that, "
                f"let's use SQLite instead. Make sure it satisfies {req_str}."
            )

        elif variant_type == VariantType.CONSTRAINED:
            neg_constraint = "avoid any heavy external dependencies or frameworks"
            prompt = (
                f"{intent_clean}. Explicit constraints: must include {req_str}. "
                f"Do NOT use third-party runtime dependencies, {neg_constraint}."
            )

        elif variant_type == VariantType.ALREADY_GOOD:
            if seed.already_good_reference:
                prompt = seed.already_good_reference
            else:
                prompt = (
                    f"Objective: {intent_clean}\n\n"
                    f"Core Requirements:\n"
                    + "\n".join(f"- {r}" for r in reqs)
                    + (f"\n\nConstraints:\n- {constraints_str}" if constraints_str else "")
                    + "\n\nPlease structure the solution cleanly with clear documentation."
                )

        elif variant_type == VariantType.AMBIGUOUS:
            # Intentionally omit core requirements to create genuine ambiguity
            words = intent_clean.split()
            short_noun = words[-1] if words else "it"
            prompt = f"Can you build me a {short_noun} for my project?"

        else:
            prompt = f"{intent_clean} with {req_str}."

        return SyntheticVariant(
            seed_id=seed.seed_id,
            prompt=prompt.strip(),
            variant_type=variant_type,
            intended_requirements=reqs,
            intent_preserved=True,
        )

    async def generate_llm_variant(
        self,
        seed: SeedIntent,
        variant_type: VariantType,
    ) -> SyntheticVariant:
        """Generate human-like variant via local Ollama LLM with caching."""
        key = _cache_key(seed.seed_id, variant_type.value, self.model)
        cached = self._read_cache(key)
        if cached:
            return cached

        # Prepare generation prompt
        system_instruction = (
            "You are a synthetic prompt generator creating realistic, human-authored prompts "
            "for training an AI prompt optimization assistant.\n"
            "Given a target seed intent, generate a SINGLE realistic user prompt reflecting the requested style.\n"
            "Do NOT include conversational preamble. Return ONLY the prompt text."
        )

        style_descriptions = {
            VariantType.CLEAN: "A clear, well-phrased, direct user prompt.",
            VariantType.CASUAL: "An informal, conversational prompt with casual phrasing and lowercase text.",
            VariantType.MESSY: "A rambling, poorly structured prompt with conversational filler and stream-of-consciousness thoughts.",
            VariantType.SHORT: "A terse, minimal prompt with minimal punctuation and telegraphic phrasing.",
            VariantType.COMPOUND: "A prompt asking for multiple distinct but related deliverables in one request.",
            VariantType.SELF_CORRECTING: "A prompt where the user changes their mind or corrects a choice mid-sentence (e.g. 'actually no, use X instead').",
            VariantType.CONSTRAINED: "A prompt containing negative constraints ('do NOT use X') or specific quantitative requirements.",
            VariantType.ALREADY_GOOD: "A thorough, already well-structured prompt with clear bullet points and explicit constraints.",
            VariantType.AMBIGUOUS: "A prompt where critical functional details are intentionally left out, requiring clarification.",
        }

        user_content = (
            f"SEED INTENT:\n"
            f"- Category: {seed.category}\n"
            f"- Intent: {seed.intent}\n"
            f"- Requirements: {', '.join(seed.core_requirements)}\n"
            f"- Optional constraints: {', '.join(seed.optional_constraints)}\n\n"
            f"TARGET STYLE: {variant_type.value}\n"
            f"STYLE GUIDANCE: {style_descriptions.get(variant_type, 'Natural prompt.')}\n\n"
            f"Generate the user prompt now:"
        )

        try:
            from app.llm.ollama_client import OllamaClient
            client = OllamaClient()
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content},
            ]
            response = await client.chat(
                messages=messages,
                model_override=self.model,
            )
            raw_text = response.strip().strip('"').strip("'")
            if not raw_text:
                raise ValueError("Empty response from Ollama")

            variant = SyntheticVariant(
                seed_id=seed.seed_id,
                prompt=raw_text,
                variant_type=variant_type,
                intended_requirements=seed.core_requirements,
                intent_preserved=True,
            )
            self._write_cache(key, variant)
            return variant

        except Exception as e:
            logger.warning(
                "Ollama variant generation failed for seed %s (%s): %s. Falling back to deterministic variant.",
                seed.seed_id,
                variant_type.value,
                e,
            )
            fallback = self.generate_deterministic_variant(seed, variant_type)
            self._write_cache(key, fallback)
            return fallback

    async def generate_variants_for_seed(
        self,
        seed: SeedIntent,
        target_count: int = 3,
        use_llm: bool = True,
    ) -> List[SyntheticVariant]:
        """Generate a diverse set of prompt variants for a single seed."""
        # Define available styles
        all_styles = [
            VariantType.CLEAN,
            VariantType.CASUAL,
            VariantType.MESSY,
            VariantType.SHORT,
            VariantType.COMPOUND,
            VariantType.SELF_CORRECTING,
            VariantType.CONSTRAINED,
            VariantType.ALREADY_GOOD,
            VariantType.AMBIGUOUS,
        ]

        # Select target_count distinct styles
        k = min(target_count, len(all_styles))
        selected_styles = self.rng.sample(all_styles, k)

        variants: List[SyntheticVariant] = []
        for style in selected_styles:
            if use_llm:
                v = await self.generate_llm_variant(seed, style)
            else:
                v = self.generate_deterministic_variant(seed, style)
            variants.append(v)

        return variants
