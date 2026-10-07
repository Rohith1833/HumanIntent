from app.schemas.analysis import PromptAnalysis, PromptCategory
from app.templates.base import BaseTemplate


class ImageGenerationTemplate(BaseTemplate):
    """Enhancement template for visual generation prompts (Midjourney, DALL-E, Stable Diffusion)."""

    category = PromptCategory.IMAGE_GENERATION
    name = "Image Generation"
    description = "Visual descriptions, scene composition, lighting, camera perspectives, and aesthetic styles."
    recommended_sections = [
        "SUBJECT",
        "ENVIRONMENT",
        "COMPOSITION",
        "LIGHTING",
        "CAMERA / VIEWPOINT",
        "VISUAL STYLE",
        "COLOR / MOOD",
        "CONSTRAINTS",
    ]
    anti_patterns = [
        "Do NOT invent camera models (e.g. Sony A7R V) or lenses (e.g. 85mm f/1.4) unless requested.",
        "Do NOT invent artist references or aspect ratios unless requested.",
        "Do NOT add generic buzzword spam such as '8K', 'masterpiece', 'ultra detailed', 'trending on ArtStation', 'award-winning'.",
        "Do NOT produce programming code, software design, or research outlines.",
    ]

    def get_instructions(self, original_prompt: str, analysis: PromptAnalysis) -> str:
        length_guidance = self.get_complexity_guideline(original_prompt, analysis)
        prefs = ", ".join(analysis.preferences) if analysis.preferences else "visually descriptive"

        return (
            f"CATEGORY-SPECIFIC ENHANCEMENT GUIDANCE (IMAGE_GENERATION):\n"
            f"- Primary task type: Visual Prompt Optimization for Image Synthesis.\n"
            f"- Role: Act as a master prompt engineer and visual art director.\n"
            f"- Objective focus: Craft a vivid, precise description of visual elements without buzzword spam.\n"
            f"- Style / Aesthetic preferences to preserve: {prefs}.\n"
            f"- Recommended sections (use ONLY those relevant to this task):\n"
            f"  * SUBJECT (central figures, objects, or focal point with rich descriptive detail)\n"
            f"  * ENVIRONMENT & SETTING (spatial background, atmosphere, architecture, nature)\n"
            f"  * LIGHTING & COLOR (natural light, volumetric rays, color palette, shadows)\n"
            f"  * COMPOSITION & VIEWPOINT (camera angle, framing, depth of field)\n"
            f"  * VISUAL STYLE (e.g. cinematic photography, watercolor, cyberpunk digital art, oil painting)\n"
            f"- Strict Safeguards:\n"
            f"  * Never output software development or research sections.\n"
            f"  * Absolutely NO generic quality modifier spam (no '8K', 'masterpiece', 'photorealistic award winning').\n"
            f"{length_guidance}"
        )
