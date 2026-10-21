from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class CodingTemplate(BaseTemplate):
    """Enhancement template for software implementation and code creation tasks."""

    category = PromptCategory.CODING
    name = "Coding"
    description = "Software creation, feature implementation, and script development."
    recommended_sections = [
        "ROLE & OBJECTIVE",
        "CORE REQUIREMENTS",
        "EXPLICIT REQUIREMENTS",
        "PREFERENCES",
        "CONSTRAINTS",
        "UNSPECIFIED DETAILS",
        "OUTPUT EXPECTATIONS",
    ]
    anti_patterns = [
        "Do NOT invent frameworks or styling libraries (React, Vue, Angular, Next.js, Django, FastAPI, Tailwind) unless explicitly supplied.",
        "Do NOT invent databases or storage (SQLite, PostgreSQL, MongoDB, CSV) unless explicitly supplied.",
        "Do NOT invent cloud providers (AWS, GCP, Firebase, Azure) unless explicitly supplied.",
        "Do NOT invent testing frameworks or deployment architectures unless requested.",
        "Do NOT invent non-functional requirements (responsive design, accessibility, security, privacy) unless explicitly requested.",
        "Do NOT invent extra product/business features (pricing, price data, brands, sizes, user profiles, login/auth, user demographics like age/gender/location) unless explicitly requested.",
        "Do NOT assume UI type (CLI, GUI, Web) if unspecified — list as unspecified detail.",
        "Do NOT invent domain personas (faculty member, teacher, doctor, admin); use safe generic developer role.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        explicit_tools = ", ".join(analysis.explicit_requirements) if analysis.explicit_requirements else "None explicitly stated"
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (CODING):\n"
            f"- Primary task type: Software Implementation / Development.\n"
            f"- Role: Act as an experienced software developer (e.g., 'Act as an experienced Python developer' if Python is specified).\n"
            f"- Explicit tools and requirements to preserve: {explicit_tools}.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * ROLE & OBJECTIVE (state developer role and the user's primary goal)\n"
            f"  * CORE REQUIREMENTS (list strictly what the user asked for; DO NOT invent extra features, pricing, brands, demographics, or sub-questions)\n"
            f"  * PREFERENCES (if supplied by user, e.g. style/tone/presentation)\n"
            f"  * CONSTRAINTS (if supplied by user)\n"
            f"  * UNSPECIFIED DETAILS (state missing architecture/storage/interface neutrally without candidate options or answers)\n"
            f"  * OUTPUT EXPECTATIONS (e.g. implementation approach followed by code)\n"
            f"- STRICT SAFEGUARDS:\n"
            f"  * CATEGORY TEMPLATES MAY STRUCTURE KNOWN INFORMATION, BUT MUST NOT EXPAND THE PRODUCT IDEA.\n"
            f"  * Never invent extra product features (no pricing, no unrequested product brands/sizes, no demographic questions, no user accounts/login).\n"
            f"  * Never automatically inject non-functional requirements like responsive design, mobile layout, security practices, or privacy policies unless explicitly requested.\n"
            f"  * Never assume or suggest candidate frameworks, databases, or UI paradigms that were not supplied.\n"
            f"  * Do not invent user/domain personas (e.g. faculty, teachers, learners, administrators).\n"
            f"{length_guidance}"
        )
