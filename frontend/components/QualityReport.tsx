import React from "react";
import { PromptQualityReport } from "@/lib/api";

interface QualityReportProps {
  report: PromptQualityReport;
}

export const QualityReport: React.FC<QualityReportProps> = ({ report }) => {
  const dimensions = [
    { name: "Intent Preservation", dim: report.intent_preservation },
    { name: "Requirement Coverage", dim: report.requirement_coverage },
    { name: "Constraint Preservation", dim: report.constraint_preservation },
    { name: "Clarity", dim: report.clarity },
    { name: "Specificity", dim: report.specificity },
    { name: "Output Definition", dim: report.output_definition },
    { name: "Structure", dim: report.structure },
    { name: "Concision", dim: report.concision },
  ];

  const getBadgeColor = (score: number) => {
    if (score >= 90) return "var(--status-green)";
    if (score >= 80) return "var(--accent-pine)";
    if (score >= 70) return "var(--status-amber)";
    return "var(--status-red)";
  };

  const getRiskColor = (risk: string) => {
    if (risk === "LOW") return "var(--status-green)";
    if (risk === "MEDIUM") return "var(--status-amber)";
    return "var(--status-red)";
  };

  return (
    <div
      style={{
        marginTop: "1.25rem",
        paddingTop: "1rem",
        borderTop: "1px solid var(--border-hairline)",
      }}
      aria-label="Prompt Quality Estimate"
    >
      {/* Header Row */}
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "0.5rem",
          marginBottom: "0.75rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "baseline", gap: "0.6rem" }}>
          <span style={{ fontSize: "0.92rem", fontWeight: 600, color: "var(--text-primary)" }}>
            Prompt Quality Estimate
          </span>
          <span
            style={{
              fontSize: "0.85rem",
              fontWeight: 700,
              fontFamily: "var(--font-mono)",
              color: getBadgeColor(report.overall_score),
            }}
          >
            {report.overall_score} / 100
          </span>
          <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>
            ({report.overall_label})
          </span>
        </div>

        {/* Inflation Diagnostic */}
        <div
          style={{
            fontSize: "0.76rem",
            color: "var(--text-muted)",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
          }}
        >
          <span>Prompt expansion: {report.inflation_ratio}×</span>
          <span>&bull;</span>
          <span>
            Inflation risk:{" "}
            <span style={{ color: getRiskColor(report.inflation_risk), fontWeight: 600 }}>
              {report.inflation_risk.charAt(0) + report.inflation_risk.slice(1).toLowerCase()}
            </span>
          </span>
        </div>
      </div>

      {/* Compact Dimension Grid */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
          gap: "0.45rem 0.85rem",
        }}
      >
        {dimensions.map(({ name, dim }) => (
          <div
            key={name}
            title={dim.reason}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: "0.8rem",
              padding: "0.3rem 0.6rem",
              borderRadius: "var(--radius-xs)",
              background: "var(--bg-canvas)",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <span style={{ color: "var(--text-secondary)" }}>{name}</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ fontWeight: 600, fontFamily: "var(--font-mono)", color: getBadgeColor(dim.score) }}>
                {dim.score}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
