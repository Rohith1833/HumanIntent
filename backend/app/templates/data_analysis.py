from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class DataAnalysisTemplate(BaseTemplate):
    """Enhancement template for dataset exploration, metrics, statistical modeling, and insights."""

    category = PromptCategory.DATA_ANALYSIS
    name = "Data Analysis"
    description = "Analyzing tabular datasets, trend detection, metrics calculation, and reporting."
    recommended_sections = [
        "ANALYSIS OBJECTIVE",
        "DATA CONTEXT",
        "QUESTIONS TO ANSWER",
        "REQUIRED METRICS",
        "VISUALIZATION REQUIREMENTS",
        "CONSTRAINTS",
        "OUTPUT STRUCTURE",
    ]
    anti_patterns = [
        "Do NOT invent dataset columns, features, or schemas not described in the prompt.",
        "Do NOT assume specific statistical models (e.g. ARIMA, Random Forest) unless requested.",
        "Do NOT invent business metrics, KPIs, or charts not requested.",
        "Do NOT fabricate quantitative conclusions before analyzing the data.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (DATA_ANALYSIS):\n"
            f"- Primary task type: Quantitative Dataset Analysis & Insight Generation.\n"
            f"- Role: Act as an expert data analyst and analytical consultant.\n"
            f"- Objective focus: Uncover accurate, verifiable insights from data to answer core analytical questions.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * ANALYSIS OBJECTIVE (specific analytical goal)\n"
            f"  * DATA CONTEXT (data format or structure described by user)\n"
            f"  * QUESTIONS TO ANSWER (key hypotheses or patterns to evaluate)\n"
            f"  * REQUIRED METRICS / METHODOLOGY (statistical techniques or KPIs)\n"
            f"  * UNSPECIFIED DETAILS (if dataset schema, size, or format is missing, note neutrally)\n"
            f"  * OUTPUT STRUCTURE (e.g. data summary, findings breakdown, key trends, caveats)\n"
            f"- Strict Safeguards:\n"
            f"  * Do not invent column names, synthetic rows, or unrequested charts.\n"
            f"{length_guidance}"
        )
