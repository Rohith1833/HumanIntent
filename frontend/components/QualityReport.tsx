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
    if (score >= 90) return "#10b981"; // green
    if (score >= 80) return "#3b82f6"; // blue
    if (score >= 70) return "#f59e0b"; // amber
    return "#ef4444"; // red
  };

  const getRiskColor = (risk: string) => {
    if (risk === "LOW") return "#10b981";
    if (risk === "MEDIUM") return "#f59e0b";
    return "#ef4444";
  };

  return (
    <div
      style={{
        marginTop: "1.25rem",
        paddingTop: "1rem",
        borderTop: "1px solid var(--border, rgba(255, 255, 255, 0.1))",
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
          <span style={{ fontSize: "0.95rem", fontWeight: 600, color: "var(--foreground, #ededed)" }}>
            Prompt Quality Estimate
          </span>
          <span
            style={{
              fontSize: "0.85rem",
              fontWeight: 700,
              color: getBadgeColor(report.overall_score),
            }}
          >
            {report.overall_score} / 100
          </span>
          <span style={{ fontSize: "0.75rem", color: "#a1a1aa" }}>
            ({report.overall_label})
          </span>
        </div>

        {/* Inflation Diagnostic */}
        <div
          style={{
            fontSize: "0.75rem",
            color: "#a1a1aa",
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
          }}
        >
          <span>Prompt expansion: {report.inflation_ratio}×</span>
          <span>•</span>
          <span>
            Inflation risk:{" "}
            <span style={{ color: getRiskColor(report.inflation_risk), fontWeight: 500 }}>
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
          gap: "0.5rem 1rem",
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
              padding: "0.25rem 0.4rem",
              borderRadius: "4px",
              background: "rgba(255, 255, 255, 0.02)",
            }}
          >
            <span style={{ color: "#a1a1aa" }}>{name}</span>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem" }}>
              <span style={{ fontWeight: 600, color: getBadgeColor(dim.score) }}>
                {dim.score}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
