from typing import Dict, Union
from app.schemas.analysis import PromptCategory
from app.templates.base import BaseTemplate
from app.templates.coding import CodingTemplate
from app.templates.debugging import DebuggingTemplate
from app.templates.research import ResearchTemplate
from app.templates.education import EducationTemplate
from app.templates.writing import WritingTemplate
from app.templates.data_analysis import DataAnalysisTemplate
from app.templates.planning import PlanningTemplate
from app.templates.creative import CreativeTemplate
from app.templates.image_generation import ImageGenerationTemplate
from app.templates.summarization import SummarizationTemplate
from app.templates.general import GeneralTemplate

# Static, deterministic registry mapping each locked category to its singleton template instance
_TEMPLATE_REGISTRY: Dict[PromptCategory, BaseTemplate] = {
    PromptCategory.CODING: CodingTemplate(),
    PromptCategory.DEBUGGING: DebuggingTemplate(),
    PromptCategory.RESEARCH: ResearchTemplate(),
    PromptCategory.EDUCATION: EducationTemplate(),
    PromptCategory.WRITING: WritingTemplate(),
    PromptCategory.DATA_ANALYSIS: DataAnalysisTemplate(),
    PromptCategory.PLANNING: PlanningTemplate(),
    PromptCategory.CREATIVE: CreativeTemplate(),
    PromptCategory.IMAGE_GENERATION: ImageGenerationTemplate(),
    PromptCategory.SUMMARIZATION: SummarizationTemplate(),
    PromptCategory.GENERAL: GeneralTemplate(),
}


def get_template(category: Union[PromptCategory, str]) -> BaseTemplate:
    """Deterministic, O(1) Python template selection.

    Maps every locked PromptCategory to its dedicated enhancement template.
    Uses zero LLM calls and falls back safely to GeneralTemplate for unknown inputs.
    """
    if isinstance(category, str):
        cleaned = category.strip().upper()
        try:
            cat_enum = PromptCategory(cleaned)
        except ValueError:
            return _TEMPLATE_REGISTRY[PromptCategory.GENERAL]
    elif isinstance(category, PromptCategory):
        cat_enum = category
    else:
        return _TEMPLATE_REGISTRY[PromptCategory.GENERAL]

    return _TEMPLATE_REGISTRY.get(cat_enum, _TEMPLATE_REGISTRY[PromptCategory.GENERAL])
