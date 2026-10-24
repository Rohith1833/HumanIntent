import React, { useState } from "react";
import { EnhancedResponse } from "@/lib/api";
import { QualityReport } from "@/components/QualityReport";
import { FeedbackWidget } from "@/components/FeedbackWidget";

interface EnhancedOutputProps {
  result: EnhancedResponse;
}

export const EnhancedOutput: React.FC<EnhancedOutputProps> = ({ result }) => {
  const [copied, setCopied] = useState(false);
  const [viewMode, setViewMode] = useState<"prompt" | "comparison">("prompt");

  const handleCopy = async (textToCopy: string = result.enhanced_prompt) => {
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback if clipboard API fails
      const textarea = document.createElement("textarea");
      textarea.value = textToCopy;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const originalWordCount = result.original_prompt.trim().split(/\s+/).filter(Boolean).length;
  const enhancedWordCount = result.enhanced_prompt.trim().split(/\s+/).filter(Boolean).length;

  return (
    <section className="output-card" aria-label="Enhanced Prompt Result">
      <div className="output-header">
        <div className="output-title-group">
          <span className="output-title">Enhanced Prompt</span>
          <span className="output-meta">via {result.model}</span>
          {result.mode && (
            <span className="output-meta">• Mode: {result.mode.charAt(0).toUpperCase() + result.mode.slice(1)}</span>
          )}
          {result.template && (
            <span className="output-meta">• Template: {result.template}</span>
          )}
          {result.validation && result.validation.is_valid && (!result.analysis.grounding || result.analysis.grounding.is_grounded) && (
            <span className="output-meta" style={{ color: "var(--status-green)", fontWeight: 600 }}>
              • Intent verified ✓
            </span>
          )}
        </div>

        <div className="output-actions">
          {/* View mode toggle */}
          <div className="view-toggle-group" role="tablist" aria-label="View Mode">
            <button
              type="button"
              role="tab"
              aria-selected={viewMode === "prompt"}
              className={`view-toggle-btn ${viewMode === "prompt" ? "active" : ""}`}
              onClick={() => setViewMode("prompt")}
            >
              Prompt
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={viewMode === "comparison"}
              className={`view-toggle-btn ${viewMode === "comparison" ? "active" : ""}`}
              onClick={() => setViewMode("comparison")}
            >
              Comparison
            </button>
          </div>

          <button
            type="button"
            onClick={() => handleCopy()}
            className={`btn-copy ${copied ? "copied" : ""}`}
            aria-label="Copy enhanced prompt to clipboard"
          >
            {copied ? (
              <>
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                Copied!
              </>
            ) : (
              <>
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                </svg>
                Copy Prompt
              </>
            )}
          </button>
        </div>
      </div>

      {/* Latency and telemetry metrics */}
      {(result.total_ms !== undefined || result.inference_ms !== undefined) && (
        <div className="telemetry-bar">
          {result.total_ms !== undefined && (
            <span className="telemetry-item" title="Total round-trip time">
              Total: {(result.total_ms / 1000).toFixed(2)}s
            </span>
          )}
          {result.inference_ms !== undefined && (
            <span className="telemetry-item" title="LLM inference time">
              Inference: {(result.inference_ms / 1000).toFixed(2)}s
            </span>
          )}
          {result.stage_timings && result.stage_timings.length > 0 && (
            <div className="stage-timings-pills">
              {result.stage_timings.map((st, idx) => (
                <span key={idx} className="stage-pill">
                  {st.stage}: {st.elapsed_ms < 1000 ? `${st.elapsed_ms}ms` : `${(st.elapsed_ms / 1000).toFixed(2)}s`}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {viewMode === "prompt" ? (
        <div className="output-content" tabIndex={0}>
          {result.enhanced_prompt}
        </div>
      ) : (
        <div className="comparison-container">
          <div className="comparison-column original-column">
            <div className="comparison-col-header">
              <span className="col-badge original-badge">Original Rough Prompt</span>
              <span className="col-stats">{originalWordCount} words &bull; {result.original_prompt.length} chars</span>
            </div>
            <div className="comparison-text-box original-text">
              {result.original_prompt}
            </div>
          </div>

          <div className="comparison-divider" aria-hidden="true">
            <span className="divider-arrow">➔</span>
          </div>

          <div className="comparison-column enhanced-column">
            <div className="comparison-col-header">
              <span className="col-badge enhanced-badge">Enhanced AI Instruction</span>
              <span className="col-stats">{enhancedWordCount} words &bull; {result.enhanced_prompt.length} chars</span>
            </div>
            <div className="comparison-text-box enhanced-text">
              {result.enhanced_prompt}
            </div>
          </div>
        </div>
      )}

      {result.quality && <QualityReport report={result.quality} />}

      {result.session_id && (
        <FeedbackWidget
          sessionId={result.session_id}
          enhancedPrompt={result.enhanced_prompt}
        />
      )}
    </section>
  );
};
