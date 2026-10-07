import re
from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class CreativeTemplate(BaseTemplate):
    """Enhancement template for storytelling, worldbuilding, creative writing, and conceptual ideation."""

    category = PromptCategory.CREATIVE
    name = "Creative"
    description = "Storytelling, fictional narratives, poetry, concept brainstorming, and artistic ideation."
    recommended_sections = [
        "OBJECTIVE",
        "PREFERENCES",
        "CONSTRAINTS",
        "UNSPECIFIED DETAILS",
        "OUTPUT EXPECTATIONS",
    ]
    anti_patterns = [
        "Do NOT force rigid technical prompt-engineering jargon into creative tasks.",
        "Do NOT invent restrictive plot twists, resolutions, or character names unless requested.",
        "Do NOT force storytelling, narrative voice, vivid imagery, or character design into brainstorming/ideation tasks.",
        "Do NOT output software engineering architecture or technical specifications.",
        "Do NOT restrict the generative model's artistic freedom beyond the user's premise.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)
        prompt_lower = original_prompt.lower()
        is_brainstorm = bool(
            re.search(r"\b(?:brainstorm|ideas?|concepts?|suggest|angles?)\b", prompt_lower)
            and not re.search(r"\b(?:story|narrative|novel|poem|poetry|fiction|script)\b", prompt_lower)
        )

        prefs = ", ".join(analysis.preferences) if analysis.preferences else ""

        if is_brainstorm:
            pref_line = f"  * PREFERENCES: {prefs}\n" if prefs else ""
            return (
                f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (CREATIVE BRAINSTORMING):\n"
                f"- Primary task type: Creative Ideation & Concept Generation.\n"
                f"- Role: Act as a creative conceptual thinker.\n"
                f"- Objective: {analysis.objective}\n"
                f"- Recommended structure:\n"
                f"  * OBJECTIVE: State the brainstorming goal directly.\n"
                f"{pref_line}"
                f"  * UNSPECIFIED DETAILS: List genuine unspecified dimensions only if helpful; omit if none.\n"
                f"  * OUTPUT EXPECTATIONS: Provide several distinct, creative ideas/concepts.\n"
                f"- Strict Safeguards:\n"
                f"  * Do NOT force storytelling, narrative voice, worldbuilding, vivid imagery, or character design.\n"
                f"  * Creative brainstorming is conceptual ideation, NOT fiction writing.\n"
                f"  * Do NOT place OBJECTIVE inside UNSPECIFIED DETAILS.\n"
                f"{length_guidance}"
            )

        pref_line = f"- Requested Style / Tone to preserve: {prefs}.\n" if prefs else ""
        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (CREATIVE WRITING):\n"
            f"- Primary task type: Creative Narrative & Artistic Writing.\n"
            f"- Role: Act as an imaginative, nuanced creative writer.\n"
            f"- Objective focus: {analysis.objective}\n"
            f"{pref_line}"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * OBJECTIVE & PREMISE (core story hook, theme, or concept)\n"
            f"  * TONE & ATMOSPHERE (sensory mood and style requested by user)\n"
            f"  * CONSTRAINTS (e.g. word count limit, specific themes)\n"
            f"  * UNSPECIFIED DETAILS (if genuine details are open)\n"
            f"  * OUTPUT EXPECTATIONS (e.g. short story, scene, outline)\n"
            f"- Strict Safeguards:\n"
            f"  * Avoid robotic or technical meta-prompting; preserve organic narrative freedom.\n"
            f"  * Do not invent unrequested plot resolutions, character names, or forced tropes.\n"
            f"{length_guidance}"
        )

