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
from app.templates.selector import get_template

__all__ = [
    "BaseTemplate",
    "CodingTemplate",
    "DebuggingTemplate",
    "ResearchTemplate",
    "EducationTemplate",
    "WritingTemplate",
    "DataAnalysisTemplate",
    "PlanningTemplate",
    "CreativeTemplate",
    "ImageGenerationTemplate",
    "SummarizationTemplate",
    "GeneralTemplate",
    "get_template",
]
