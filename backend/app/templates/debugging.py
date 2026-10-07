from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class DebuggingTemplate(BaseTemplate):
    """Enhancement template for error diagnosis, troubleshooting, and bug fixing tasks."""

    category = PromptCategory.DEBUGGING
    name = "Debugging"
    description = "Diagnosing broken software, exceptions, 500 errors, stack traces, and unexpected behavior."
    recommended_sections = [
        "PROBLEM",
        "EXPECTED BEHAVIOR",
        "OBSERVED BEHAVIOR",
        "KNOWN CONTEXT",
        "ERROR INFORMATION",
        "CONSTRAINTS",
        "DEBUGGING TASK",
        "OUTPUT EXPECTATIONS",
    ]
    anti_patterns = [
        "Do NOT treat this as a greenfield software development or new feature creation task.",
        "Do NOT fabricate an error message, stack trace, or root cause that was not provided.",
        "Do NOT assume an unrequested framework, database, or architecture.",
        "Do NOT rewrite entire applications when a targeted bug fix is requested.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (DEBUGGING):\n"
            f"- Primary task type: Defect Diagnosis & Targeted Remediation.\n"
            f"- Role: Act as an expert software debugger and troubleshooter.\n"
            f"- Objective focus: Diagnose, isolate, and fix the specific failure described by the user.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * PROBLEM (concise description of the failure or symptom)\n"
            f"  * OBSERVED BEHAVIOR (what currently happens, e.g. HTTP 500 error, exception, freeze)\n"
            f"  * EXPECTED BEHAVIOR (what should happen under normal operation)\n"
            f"  * ERROR INFORMATION / CONTEXT (any provided error messages, logs, or code snippets)\n"
            f"  * UNSPECIFIED DETAILS (if stack traces, reproduction steps, or logs are missing, note neutrally)\n"
            f"  * DEBUGGING TASK (identify root cause hypothesis, provide minimal targeted fix)\n"
            f"  * OUTPUT EXPECTATIONS (e.g. root-cause explanation, code patch, verification steps)\n"
            f"- Strict Safeguards:\n"
            f"  * Never hallucinate an error cause or stack trace.\n"
            f"  * Never treat the request as building a brand new application.\n"
            f"{length_guidance}"
        )
