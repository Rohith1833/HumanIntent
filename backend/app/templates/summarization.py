from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class SummarizationTemplate(BaseTemplate):
    """Enhancement template for condensing, synthesizing, and extracting takeaways from content."""

    category = PromptCategory.SUMMARIZATION
    name = "Summarization"
    description = "Condensing text, extracting key insights, creating executive summaries, and bulleted takeaways."
    recommended_sections = [
        "SOURCE / CONTENT TO SUMMARIZE",
        "SUMMARY GOAL",
        "FOCUS",
        "LENGTH",
        "FORMAT",
        "CONSTRAINTS",
    ]
    anti_patterns = [
        "Do NOT invent outside facts, statistics, or claims absent from the source material.",
        "Do NOT transform summarization into an open-ended research or literature review task.",
        "Do NOT add software engineering architecture or code sections.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)
        prefs = ", ".join(analysis.preferences) if analysis.preferences else "concise, faithful summary"

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (SUMMARIZATION):\n"
            f"- Primary task type: Content Distillation & Executive Synthesis.\n"
            f"- Role: Act as an expert summarizer and editorial synthesizer.\n"
            f"- Objective focus: Condense the supplied content faithfully while capturing core concepts and nuance.\n"
            f"- Format / Style preferences to preserve: {prefs}.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * SOURCE / CONTEXT (what material is being summarized)\n"
            f"  * SUMMARY GOAL & FOCUS (key themes, decision points, or specific questions)\n"
            f"  * FORMAT & LENGTH (e.g. 5 bullet points, executive brief, TL;DR)\n"
            f"  * CONSTRAINTS (e.g. strict word limit, no outside speculation)\n"
            f"  * OUTPUT STRUCTURE (e.g. core thesis, key arguments, action items)\n"
            f"- Strict Safeguards:\n"
            f"  * Do not invent facts or extrapolate beyond provided source material.\n"
            f"{length_guidance}"
        )
