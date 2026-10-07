from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class WritingTemplate(BaseTemplate):
    """Enhancement template for communication, documents, emails, and professional writing."""

    category = PromptCategory.WRITING
    name = "Writing"
    description = "Drafting emails, letters, articles, reports, cover letters, and professional prose."
    recommended_sections = [
        "PURPOSE",
        "AUDIENCE / RECIPIENT",
        "TONE",
        "KEY INFORMATION",
        "CONSTRAINTS",
        "LENGTH / FORMAT",
        "OUTPUT",
    ]
    anti_patterns = [
        "Do NOT invent recipient identity, names, relationships, or corporate titles.",
        "Do NOT invent specific factual details, meeting times, project outcomes, or dates.",
        "Do NOT generate software engineering or programming sections.",
        "Do NOT alter requested tone (e.g. professional, formal, polite).",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)
        prefs = ", ".join(analysis.preferences) if analysis.preferences else "appropriate professional tone"

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (WRITING):\n"
            f"- Primary task type: Textual Communication & Drafting.\n"
            f"- Role: Act as a professional communications writer.\n"
            f"- Objective focus: Produce an effective, ready-to-use piece of writing meeting the user's intent.\n"
            f"- Requested Tone to preserve: {prefs}.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * PURPOSE & OBJECTIVE (clear statement of what message is being communicated)\n"
            f"  * AUDIENCE & TONE (state requested tone clearly, e.g. 'Use a professional tone')\n"
            f"  * KEY INFORMATION (points or facts provided by the user)\n"
            f"  * UNSPECIFIED DETAILS (e.g. recipient, specific purpose, if not provided — instruct AI to ask or use clear placeholders like [Recipient Name])\n"
            f"  * OUTPUT FORMAT (e.g. ready-to-send draft with subject line and body)\n"
            f"- Strict Safeguards:\n"
            f"  * Never invent recipient names, relationships, or factual claims.\n"
            f"  * Never output software or code sections.\n"
            f"{length_guidance}"
        )
