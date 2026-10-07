from fastapi import APIRouter, HTTPException, status
from app.core.analyzer import prompt_analyzer
from app.core.config import settings
from app.core.enhancer import prompt_enhancer
from app.llm.ollama_client import (
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    ollama_client,
)
from app.schemas.analysis import AnalyzeRequest, PromptAnalysis
from app.schemas.enhance import (
    AnalysisFailedResponse,
    ClarificationRequiredResponse,
    EnhancedResponse,
    EnhanceRequest,
    EnhanceResponse,
    HealthResponse,
    ModelsResponse,
    OllamaStatus,
    ValidationFailedResponse,
)
from app.templates.selector import get_template

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return backend status and Ollama connectivity status."""
    is_reachable = await ollama_client.check_health()
    return HealthResponse(
        status="ok",
        ollama=OllamaStatus(
            reachable=is_reachable,
            model=settings.ollama_model,
        ),
    )


@router.get("/models", response_model=ModelsResponse)
async def list_models() -> ModelsResponse:
    """List locally installed models from Ollama."""
    try:
        models = await ollama_client.list_models()
        return ModelsResponse(
            models=models,
            current_model=settings.ollama_model,
        )
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc


def _clean_ollama_error_detail(exc: OllamaResponseError) -> str:
    """Ensure user-facing error messages are clean and never leak raw Pydantic validation tracebacks."""
    err_str = str(exc)
    if any(keyword in err_str.lower() for keyword in ["validation error", "required schema", "invalid json", "structured response"]):
        return "HumanIntent couldn't produce a valid structured analysis for this request. Please try again or rephrase the request."
    return err_str


@router.post("/analyze", response_model=PromptAnalysis)
async def analyze_prompt(payload: AnalyzeRequest) -> PromptAnalysis:
    """Standalone endpoint to extract structured objective, requirements, and missing context."""
    try:
        return await prompt_analyzer.analyze(payload.prompt)
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OllamaModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_clean_ollama_error_detail(exc),
        ) from exc


@router.post("/enhance", response_model=EnhanceResponse)
async def enhance_prompt(payload: EnhanceRequest) -> EnhanceResponse:
    """Enhance the user prompt by analyzing intent, clarifying if critically ambiguous, or producing an enhanced prompt."""
    try:
        result = await prompt_enhancer.enhance(
            user_prompt=payload.prompt,
            clarification_answers=payload.clarification_answers,
            allow_reasonable_assumptions=payload.allow_reasonable_assumptions,
        )
        analysis, enhanced_text = result[0], result[1]

        # Analysis Grounding Decision
        if hasattr(analysis, "grounding") and analysis.grounding and not analysis.grounding.is_grounded:
            return AnalysisFailedResponse(
                status="analysis_failed",
                original_prompt=payload.prompt,
                analysis=analysis,
                message="HumanIntent could not verify that the analysis was grounded in your request.",
                model=settings.ollama_model,
            )

        # Ambiguity / Clarification Decision
        if analysis.ambiguity.needs_clarification and not payload.allow_reasonable_assumptions:
            return ClarificationRequiredResponse(
                status="needs_clarification",
                original_prompt=payload.prompt,
                analysis=analysis,
                questions=analysis.ambiguity.questions,
                model=settings.ollama_model,
            )

        # Check if validation failed after repair attempt
        validation = getattr(result, "validation", None)
        if validation is not None and not validation.is_valid:
            return ValidationFailedResponse(
                status="validation_failed",
                original_prompt=payload.prompt,
                analysis=analysis,
                validation=validation,
                message="HumanIntent could not produce a sufficiently faithful enhanced prompt after one repair attempt.",
                model=settings.ollama_model,
            )

        template_obj = get_template(analysis.classification.category)
        return EnhancedResponse(
            status="enhanced",
            original_prompt=payload.prompt,
            analysis=analysis,
            enhanced_prompt=enhanced_text,
            model=settings.ollama_model,
            template=template_obj.name,
            validation=validation,
            quality=getattr(result, "quality", None),
        )
    except OllamaConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OllamaModelNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OllamaTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except OllamaResponseError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=_clean_ollama_error_detail(exc),
        ) from exc
