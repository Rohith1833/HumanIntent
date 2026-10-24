import React from "react";
import { HealthResponse } from "@/lib/api";

interface HeaderProps {
  health: HealthResponse | null;
  isLoadingHealth: boolean;
  onOpenHistory?: () => void;
  mode?: "quick" | "professional" | "deep";
  onScrollToWorkspace?: () => void;
  onScrollToHowItWorks?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  isLoadingHealth,
  onOpenHistory,
  mode = "professional",
  onScrollToWorkspace,
  onScrollToHowItWorks,
}) => {
  const isOllamaReachable = health?.ollama.reachable ?? false;
  const activeModel = health?.ollama.model || "llama3";

  return (
    <header className="site-header">
      <div className="header-inner">
        {/* Left: Refined Wordmark */}
        <div className="brand-group">
          <a href="#" className="brand-anchor" aria-label="HumanIntent Home">
            <span className="brand-mark" aria-hidden="true">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M12 2v20" />
                <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
              </svg>
            </span>
            <span className="brand-title">HumanIntent</span>
          </a>
          <span className="brand-tag">v1.0</span>
        </div>

        {/* Center / Right: Clean Navigation */}
        <nav className="header-nav" aria-label="Primary Navigation">
          <button
            type="button"
            onClick={onScrollToWorkspace}
            className="nav-link-btn"
          >
            Workspace
          </button>
          <button
            type="button"
            onClick={onScrollToHowItWorks}
            className="nav-link-btn"
          >
            How It Works
          </button>

          {onOpenHistory && (
            <button
              type="button"
              className="btn-header-history"
              onClick={onOpenHistory}
              title="Open locally stored prompt history"
              aria-label="Open Session History"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              <span>History</span>
            </button>
          )}

          {/* Status Badge */}
          <div
            className="status-pill"
            title={
              isLoadingHealth
                ? "Connecting to local Ollama..."
                : isOllamaReachable
                ? `Ollama online (${activeModel})`
                : "Ollama is offline (start local daemon)"
            }
          >
            <span
              className={`status-dot ${
                isLoadingHealth
                  ? "dot-amber"
                  : isOllamaReachable
                  ? "dot-green"
                  : "dot-red"
              }`}
            />
            <span className="status-label">
              {isLoadingHealth
                ? "Checking..."
                : isOllamaReachable
                ? "Local Llama 3"
                : "Ollama Offline"}
            </span>
          </div>
        </nav>
      </div>
    </header>
  );
};
