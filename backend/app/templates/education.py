from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class EducationTemplate(BaseTemplate):
    """Enhancement template for explaining concepts, tutoring, and pedagogical tasks."""

    category = PromptCategory.EDUCATION
    name = "Education"
    description = "Concept explanations, tutoring, step-by-step learning, and knowledge transfer."
    recommended_sections = [
        "TOPIC",
        "LEARNING OBJECTIVE",
        "STUDENT / AUDIENCE CONTEXT",
        "EXPLANATION STYLE",
        "EXAMPLES",
        "OUTPUT FORMAT",
    ]
    anti_patterns = [
        "Do NOT transform simple explanations into massive multi-week syllabi or exhaustive textbooks.",
        "Do NOT invent software architecture, backend designs, or technical frameworks.",
        "Do NOT invent prerequisites or assume high prior knowledge unless specified.",
        "Do NOT invent exam boards or educational institution guidelines.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)
        prefs = ", ".join(analysis.preferences) if analysis.preferences else "simple, clear explanation"

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (EDUCATION):\n"
            f"- Primary task type: Pedagogical Instruction & Concept Explanation.\n"
            f"- Role: Act as an effective teacher and educational communicator.\n"
            f"- Objective focus: Help the learner understand the topic clearly and build genuine intuition.\n"
            f"- Style preferences to preserve: {prefs}.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * TOPIC & LEARNING OBJECTIVE (what core concept is being learned)\n"
            f"  * EXPLANATION STYLE (e.g. intuitive explanation, clear step-by-step breakdown, simple analogy)\n"
            f"  * CONCRETE EXAMPLES (request a relatable walkthrough demonstrating the concept)\n"
            f"  * UNSPECIFIED DETAILS (e.g. audience background level, if not specified)\n"
            f"  * OUTPUT FORMAT (e.g. conceptual intuition, step-by-step example, common pitfalls)\n"
            f"- Strict Safeguards:\n"
            f"  * For simple requests (e.g. 'explain binary search simply'), KEEP IT COMPACT and beginner-friendly.\n"
            f"  * Never output software engineering architecture or deployment checklists.\n"
            f"{length_guidance}"
        )
