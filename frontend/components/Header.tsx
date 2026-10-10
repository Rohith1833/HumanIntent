import React from "react";
import { HealthResponse } from "@/lib/api";

interface HeaderProps {
  health: HealthResponse | null;
  isLoadingHealth: boolean;
  onOpenHistory?: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  health,
  isLoadingHealth,
  onOpenHistory,
}) => {
  const isOllamaReachable = health?.ollama.reachable ?? false;
  const activeModel = health?.ollama.model || "llama3.2";

  return (
    <header className="header">
      <div className="title-row">
        <div>
          <h1 className="app-title">HumanIntent</h1>
          <p className="app-subtitle">Intent-Preserving AI Prompt Enhancement</p>
        </div>

        <div className="status-bar">
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
              >
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              <span>History</span>
            </button>
          )}

          <span className="badge badge-mode" title="Enhancement Mode (Fixed in Phase 1)">
            Mode: Professional
          </span>

          <span className="badge badge-model" title="Configured Model">
            Model: {activeModel}
          </span>

          <span
            className="badge"
            title={
              isLoadingHealth
                ? "Checking Ollama status..."
                : isOllamaReachable
                ? "Ollama is running locally"
                : "Ollama is not reachable"
            }
          >
            <span
              className={`dot ${
                isLoadingHealth
                  ? "dot-amber"
                  : isOllamaReachable
                  ? "dot-green"
                  : "dot-red"
              }`}
            />
            {isLoadingHealth
              ? "Connecting..."
              : isOllamaReachable
              ? "Ollama Online"
              : "Ollama Offline"}
          </span>
        </div>
      </div>
    </header>
  );
};
