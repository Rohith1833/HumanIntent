import React from "react";
import { HealthResponse } from "@/lib/api";

interface HeaderProps {
  health: HealthResponse | null;
  isLoadingHealth: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, isLoadingHealth }) => {
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
