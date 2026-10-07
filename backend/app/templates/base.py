from abc import ABC, abstractmethod
from typing import List, Optional
from app.schemas.analysis import PromptAnalysis, PromptCategory


class BaseTemplate(ABC):
    """Abstract base class for all HumanIntent category-specific enhancement templates."""

    category: PromptCategory
    name: str
    description: str
    recommended_sections: List[str]
    anti_patterns: List[str]

    @abstractmethod
    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        """Return category-specific enhancement guidance for the downstream prompt optimizer."""
        pass

    def get_complexity_guideline(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        """Determine whether the request is simple or complex and provide length discipline guidance."""
        word_count = len(original_prompt.split())
        num_explicit = len(analysis.explicit_requirements)
        num_missing = len(analysis.missing_context)

        # Simple prompt heuristic: short word count and few requirements
        if word_count <= 8 and num_explicit <= 1 and num_missing <= 2:
            return (
                "OUTPUT LENGTH DISCIPLINE (SIMPLE PROMPT):\n"
                "- This is a concise request. Keep the enhanced prompt compact and focused.\n"
                "- Do NOT inflate this simple prompt into a giant multi-section specification.\n"
                "- Use only 2 to 4 essential sections that directly improve execution quality."
            )
        else:
            return (
                "OUTPUT LENGTH DISCIPLINE (STRUCTURED PROMPT):\n"
                "- Provide clear, professional structure tailored to this task.\n"
                "- Include relevant sections without unnecessary fluff or invented details."
            )
