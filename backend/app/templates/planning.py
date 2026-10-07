from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class PlanningTemplate(BaseTemplate):
    """Enhancement template for roadmaps, execution strategies, schedules, and workflow design."""

    category = PromptCategory.PLANNING
    name = "Planning"
    description = "Roadmaps, milestone tracking, action strategies, and phased schedules."
    recommended_sections = [
        "GOAL",
        "CURRENT CONTEXT",
        "CONSTRAINTS",
        "PREFERENCES",
        "PLAN REQUIREMENTS",
        "MILESTONES / STEPS",
        "OUTPUT FORMAT",
    ]
    anti_patterns = [
        "Do NOT invent artificial budgets (e.g. '$10,000') or resource numbers.",
        "Do NOT invent rigid calendar dates or deadlines unless explicitly supplied.",
        "Do NOT invent team structures or geographic locations.",
        "Do NOT invent software technology choices unless requested.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (PLANNING):\n"
            f"- Primary task type: Strategic Roadmapping & Phased Action Planning.\n"
            f"- Role: Act as an experienced strategic planner and workflow advisor.\n"
            f"- Objective focus: Produce an actionable, realistic plan with clear milestones and logical sequence.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * GOAL & SCOPE (intended outcome and boundary of the plan)\n"
            f"  * CURRENT CONTEXT & PREREQUISITES (starting conditions provided by user)\n"
            f"  * PHASES / MILESTONES (logical stages of progression)\n"
            f"  * UNSPECIFIED DETAILS (if timeline, hours per week, or budget are missing, keep flexible)\n"
            f"  * OUTPUT FORMAT (e.g. phased breakdown with objectives, action items, checkpoint criteria)\n"
            f"- Strict Safeguards:\n"
            f"  * Never assume arbitrary dates, budgets, or team sizes.\n"
            f"{length_guidance}"
        )
