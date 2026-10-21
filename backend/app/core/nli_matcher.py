"""Lightweight Natural Language Inference (NLI) service for semantic source-coverage verification.

Architecture:
- Model: 'cross-encoder/nli-deberta-v3-xsmall' (~22M parameters, ~84MB, fast CPU inference).
- Classifies premise-hypothesis pairs into: ENTAILMENT, CONTRADICTION, NEUTRAL.
- Thread-safe lazy initialization with local_files_only priority for zero network dependency.
- Used strictly as a targeted fallback on ambiguous similarity bands (never called on obvious matches).
"""

import logging
import threading
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("humanintent.nli_matcher")

DEFAULT_NLI_MODEL = "cross-encoder/nli-deberta-v3-xsmall"


class NLIMatcher:
    """Thread-safe local NLI inference engine for requirement fulfillment and constraint verification."""

    _instance: Optional["NLIMatcher"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = DEFAULT_NLI_MODEL) -> None:
        self.model_name = model_name
        self._model = None
        self._initialized = False

    def _ensure_model_loaded(self) -> None:
        if self._initialized and self._model is not None:
            return

        with self._lock:
            if self._initialized and self._model is not None:
                return
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading lightweight NLI CrossEncoder model '%s' locally...", self.model_name)
                # Load from local cache first for instant offline startup
                try:
                    self._model = CrossEncoder(self.model_name, device="cpu", local_files_only=True)
                except Exception:
                    self._model = CrossEncoder(self.model_name, device="cpu")
                self._initialized = True
                logger.info("NLI CrossEncoder model '%s' loaded successfully.", self.model_name)
            except Exception as e:
                logger.error("Failed to load NLI model '%s': %s", self.model_name, e)
                self._model = None
                self._initialized = True

    @property
    def is_available(self) -> bool:
        """Check if local NLI model is ready."""
        self._ensure_model_loaded()
        return self._model is not None

    def predict(self, premise: str, hypothesis: str) -> Tuple[str, Dict[str, float]]:
        """Classify relationship between premise (candidate delivery) and hypothesis (requirement/constraint).

        Returns:
            label: "entailment" | "contradiction" | "neutral"
            probabilities: {"entailment": float, "contradiction": float, "neutral": float}
        """
        self._ensure_model_loaded()
        if self._model is None:
            return "neutral", {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

        try:
            scores = self._model.predict([(premise, hypothesis)])
            logits = scores[0]

            # Softmax normalization
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / exp_logits.sum()

            id2label = getattr(self._model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
            prob_dict = {
                id2label[i].lower(): float(probs[i])
                for i in range(len(probs))
            }

            best_idx = int(np.argmax(probs))
            best_label = id2label[best_idx].lower()
            return best_label, prob_dict
        except Exception as e:
            logger.debug("NLI prediction error: %s", e)
            return "neutral", {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}

    def predict_batch(self, pairs: List[Tuple[str, str]]) -> List[Tuple[str, Dict[str, float]]]:
        """Batch predict relationships for efficiency."""
        self._ensure_model_loaded()
        if self._model is None or not pairs:
            return [("neutral", {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}) for _ in pairs]

        try:
            scores = self._model.predict(pairs)
            id2label = getattr(self._model.config, "id2label", {0: "contradiction", 1: "entailment", 2: "neutral"})
            results = []
            for logits in scores:
                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / exp_logits.sum()
                prob_dict = {
                    id2label[i].lower(): float(probs[i])
                    for i in range(len(probs))
                }
                best_idx = int(np.argmax(probs))
                results.append((id2label[best_idx].lower(), prob_dict))
            return results
        except Exception as e:
            logger.debug("NLI batch prediction error: %s", e)
            return [("neutral", {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0}) for _ in pairs]


# Singleton instance
nli_matcher = NLIMatcher()
