"""Per-stage telemetry and timing diagnostics schema and tracker."""

import time
from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class StageTimingDiagnostic(BaseModel):
    """Structured telemetry record for a single stage in the enhancement pipeline."""

    stage: str = Field(..., description="Stage name (e.g. analyzer, enhancer, validator).")
    started_at: str = Field(..., description="ISO UTC timestamp when the stage started.")
    ended_at: str = Field(..., description="ISO UTC timestamp when the stage completed.")
    elapsed_ms: float = Field(..., description="Execution time in milliseconds.")
    llm_called: bool = Field(default=False, description="Whether this stage invoked an LLM inference call.")
    prompt_tokens: Optional[int] = Field(default=None, description="Model prompt token count if available.")
    generated_tokens: Optional[int] = Field(default=None, description="Model generated token count if available.")
    repair_reason: Optional[str] = Field(default=None, description="Reason if a repair attempt or retry was triggered.")
    status_code: str = Field(default="OK", description="Outcome code: OK, ANALYZER_TIMEOUT, ENHANCER_TIMEOUT, etc.")


class StageTimer:
    """Helper to accurately measure stage durations and capture Ollama token metadata."""

    def __init__(self, stage_name: str, client: Optional[Any] = None) -> None:
        self.stage_name = stage_name
        self.client = client
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.t0 = time.perf_counter()

    def complete(
        self,
        llm_called: bool = False,
        repair_reason: Optional[str] = None,
        status_code: str = "OK",
    ) -> StageTimingDiagnostic:
        t1 = time.perf_counter()
        ended_at = datetime.now(timezone.utc).isoformat()
        elapsed_ms = round((t1 - self.t0) * 1000.0, 2)
        meta = getattr(self.client, "last_call_metadata", {}) if (llm_called and self.client) else {}
        prompt_tokens = meta.get("prompt_tokens") if isinstance(meta, dict) else None
        generated_tokens = meta.get("generated_tokens") if isinstance(meta, dict) else None
        if not isinstance(prompt_tokens, int):
            prompt_tokens = None
        if not isinstance(generated_tokens, int):
            generated_tokens = None

        return StageTimingDiagnostic(
            stage=self.stage_name,
            started_at=self.started_at,
            ended_at=ended_at,
            elapsed_ms=elapsed_ms,
            llm_called=llm_called,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            repair_reason=repair_reason,
            status_code=status_code,
        )
