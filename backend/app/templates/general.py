from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class GeneralTemplate(BaseTemplate):
    """Minimal, adaptive fallback enhancement template for broad or unspecialized tasks."""

    category = PromptCategory.GENERAL
    name = "General"
    description = "Adaptive, minimal structure for general requests not fitting specialized categories."
    recommended_sections = [
        "OBJECTIVE",
        "KNOWN REQUIREMENTS",
        "CONSTRAINTS",
        "PREFERENCES",
        "UNSPECIFIED DETAILS",
        "OUTPUT",
    ]
    anti_patterns = [
        "GENERAL must NOT become a giant catch-all prompt template with dozens of sections.",
        "Do NOT invent technologies, tools, roles, or details not supplied by the user.",
        "Keep structure minimal, clear, and directly proportional to user prompt complexity.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (GENERAL):\n"
            f"- Primary task type: General Intent Optimization.\n"
            f"- Objective focus: Provide a clear, natural instruction prompt directly stating the user's objective.\n"
            f"- Recommended sections (keep minimal):\n"
            f"  * OBJECTIVE (clear statement of core goal)\n"
            f"  * EXPLICIT REQUIREMENTS (only if supplied)\n"
            f"  * PREFERENCES (only if supplied)\n"
            f"  * UNSPECIFIED DETAILS (only if useful and material)\n"
            f"  * OUTPUT EXPECTATIONS (clear statement of desired deliverable)\n"
            f"- Strict Safeguards:\n"
            f"  * Keep prompt concise and direct; avoid artificial complexity.\n"
            f"{length_guidance}"
        )
