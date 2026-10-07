import React from "react";
import { PromptAnalysis, PromptCategory } from "@/lib/api";

interface UnderstoodIntentProps {
  analysis: PromptAnalysis;
}

const CATEGORY_LABELS: Record<PromptCategory, string> = {
  CODING: "Coding",
  DEBUGGING: "Debugging",
  RESEARCH: "Research",
  EDUCATION: "Education",
  WRITING: "Writing",
  DATA_ANALYSIS: "Data Analysis",
  PLANNING: "Planning",
  CREATIVE: "Creative",
  IMAGE_GENERATION: "Image Generation",
  SUMMARIZATION: "Summarization",
  GENERAL: "General",
};

const EMPTY_SENTINELS = new Set([
  "none",
  "none specified",
  "n/a",
  "na",
  "not applicable",
  "unknown",
  "no requirements",
  "no constraints",
  "no preferences",
  "no missing context",
]);

const cleanList = (list?: string[]): string[] => {
  if (!list) return [];
  return list.filter((item) => item && !EMPTY_SENTINELS.has(item.trim().toLowerCase()));
};

export const UnderstoodIntent: React.FC<UnderstoodIntentProps> = ({ analysis }) => {
  const explicitRequirements = cleanList(analysis.explicit_requirements);
  const constraints = cleanList(analysis.constraints);
  const preferences = cleanList(analysis.preferences);
  const missingContext = cleanList(analysis.missing_context);

  const hasConstraints = constraints.length > 0;
  const hasPreferences = preferences.length > 0;
  const hasMissingContext = missingContext.length > 0;
  const hasExplicit = explicitRequirements.length > 0;

  const categoryLabel =
    (analysis.classification?.category &&
      CATEGORY_LABELS[analysis.classification.category]) ||
    analysis.classification?.category ||
    "General";

  const confidencePercent = analysis.classification?.confidence != null
    ? Math.round(analysis.classification.confidence * 100)
    : null;

  return (
    <section className="analysis-card" aria-label="Understood Intent Analysis">
      <div className="analysis-header">
        <div>
          <h2 className="analysis-title">Understood Intent</h2>
          <p className="analysis-subtitle">
            Structured analysis extracted from your input to prevent invented requirements.
          </p>
        </div>

        {analysis.classification && (
          <div className="classification-pill-group">
            <span className="badge-category" title="Detected primary task category">
              <span className="category-dot" />
              Task Type: <strong>{categoryLabel}</strong>
            </span>

            {confidencePercent !== null && (
              <span
                className="badge-confidence"
                title="Model-estimated task classification confidence"
              >
                {confidencePercent}% Model confidence
              </span>
            )}

            {analysis.ambiguity && (
              <span
                className={`badge-ambiguity badge-ambiguity-${analysis.ambiguity.level.toLowerCase()}`}
                title="Estimated ambiguity heuristic (0 = fully specified, 100 = critical unknowns)"
              >
                Estimated ambiguity:{" "}
                <strong>
                  {analysis.ambiguity.level === "LOW"
                    ? "Low"
                    : analysis.ambiguity.level === "MEDIUM"
                    ? "Medium"
                    : "High"}
                </strong>{" "}
                ({analysis.ambiguity.score}/100)
              </span>
            )}
          </div>
        )}
      </div>

      <div className="objective-box">
        <span className="objective-label">Objective:</span>
        <span className="objective-text">{analysis.objective}</span>
      </div>

      <div className="analysis-grid">
        {/* Explicit Requirements Column */}
        <div className="analysis-column explicit-column">
          <div className="column-header">
            <span className="column-title">
              <span className="badge-dot dot-explicit" /> Explicit Requirements
            </span>
            <span className="column-caption">Actually stated by you</span>
          </div>

          <div className="tags-container">
            {hasExplicit ? (
              explicitRequirements.map((req, idx) => (
                <span key={idx} className="tag tag-explicit">
                  ✓ {req}
                </span>
              ))
            ) : (
              <span className="text-empty">None explicitly specified</span>
            )}
          </div>

          {/* Constraints & Preferences under explicit if present */}
          {(hasConstraints || hasPreferences) && (
            <div className="sub-section">
              {hasConstraints && (
                <div className="sub-group">
                  <span className="sub-label">Constraints:</span>
                  <div className="tags-container">
                    {constraints.map((c, idx) => (
                      <span key={idx} className="tag tag-constraint">
                        ⊘ {c}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {hasPreferences && (
                <div className="sub-group">
                  <span className="sub-label">Preferences:</span>
                  <div className="tags-container">
                    {preferences.map((p, idx) => (
                      <span key={idx} className="tag tag-preference">
                        ★ {p}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Missing Context Column */}
        <div className="analysis-column missing-column">
          <div className="column-header">
            <span className="column-title">
              <span className="badge-dot dot-missing" /> Missing Context
            </span>
            <span className="column-caption">NOT supplied &mdash; left open / neutral</span>
          </div>

          <div className="tags-container">
            {hasMissingContext ? (
              missingContext.map((item, idx) => (
                <span key={idx} className="tag tag-missing">
                  ? {item}
                </span>
              ))
            ) : (
              <span className="text-empty">No critical context missing</span>
            )}
          </div>
          <p className="missing-notice">
            These dimensions were not specified. The enhancer identifies them as unspecified rather than assuming or suggesting unrequested choices.
          </p>
        </div>
      </div>
    </section>
  );
};
