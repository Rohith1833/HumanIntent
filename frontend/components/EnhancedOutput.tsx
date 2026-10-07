import React, { useState } from "react";
import { EnhancedResponse } from "@/lib/api";
import { QualityReport } from "@/components/QualityReport";

interface EnhancedOutputProps {
  result: EnhancedResponse;
}

export const EnhancedOutput: React.FC<EnhancedOutputProps> = ({ result }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(result.enhanced_prompt);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback if clipboard API fails
      const textarea = document.createElement("textarea");
      textarea.value = result.enhanced_prompt;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <section className="output-card" aria-label="Enhanced Prompt Result">
      <div className="output-header">
        <div className="output-title-group">
          <span className="output-title">Enhanced Prompt</span>
          <span className="output-meta">via {result.model}</span>
          {result.template && (
            <span className="output-meta">• Template: {result.template}</span>
          )}
          {result.validation && result.validation.is_valid && (!result.analysis.grounding || result.analysis.grounding.is_grounded) && (
            <span className="output-meta" style={{ color: "#10b981", fontWeight: 500 }}>
              • Intent verified ✓
            </span>
          )}
        </div>

        <button
          type="button"
          onClick={handleCopy}
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

      <div className="output-content" tabIndex={0}>
        {result.enhanced_prompt}
      </div>

      {result.quality && <QualityReport report={result.quality} />}
    </section>
  );
};
