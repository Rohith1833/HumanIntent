from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class ResearchTemplate(BaseTemplate):
    """Enhancement template for literature, market, scientific, or conceptual investigation tasks."""

    category = PromptCategory.RESEARCH
    name = "Research"
    description = "Topic investigation, comparative analysis, literature review, and evidence synthesis."
    recommended_sections = [
        "RESEARCH QUESTION",
        "OBJECTIVE",
        "SCOPE",
        "KNOWN CONTEXT",
        "EVIDENCE REQUIREMENTS",
        "ANALYSIS REQUIREMENTS",
        "LIMITATIONS",
        "OUTPUT STRUCTURE",
    ]
    anti_patterns = [
        "Do NOT invent code snippets, software architectures, or programming sections.",
        "Do NOT invent artificial date ranges or time periods (e.g. '2020-2024') unless requested.",
        "Do NOT invent specific academic standards (e.g. APA format, peer-reviewed only) unless requested.",
        "Do NOT invent methodologies or conclusions beforehand.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (RESEARCH):\n"
            f"- Primary task type: Comprehensive Investigation & Synthesis.\n"
            f"- Role: Act as an objective, analytical researcher and subject matter analyst.\n"
            f"- Objective focus: Investigate the core research question rigorously without preconceived bias.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * RESEARCH QUESTION (clear, focused central question)\n"
            f"  * OBJECTIVE & SCOPE (boundaries of investigation based strictly on user prompt)\n"
            f"  * EVIDENCE REQUIREMENTS (request well-supported facts, data points, or recognized findings)\n"
            f"  * ANALYSIS REQUIREMENTS (synthesize perspectives, key benefits, risks, or mechanisms)\n"
            f"  * UNSPECIFIED DETAILS (if target depth, geography, or time period is missing, keep open)\n"
            f"  * OUTPUT STRUCTURE (e.g. executive summary, detailed analysis, key takeaways, limitations)\n"
            f"- Strict Safeguards:\n"
            f"  * Never output software development or coding headings.\n"
            f"  * Keep scope neutral and evidence-grounded.\n"
            f"{length_guidance}"
        )
