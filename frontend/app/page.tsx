"use client";

import React, { useEffect, useState } from "react";
import { Header } from "@/components/Header";
import { UnderstoodIntent } from "@/components/UnderstoodIntent";
import { EnhancedOutput } from "@/components/EnhancedOutput";
import { ClarificationCard } from "@/components/ClarificationCard";
import {
  enhancePrompt,
  getHealth,
  ClarificationAnswer,
  EnhanceResponse,
  HealthResponse,
} from "@/lib/api";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<EnhanceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState(true);

  // Check backend & Ollama health on initial load
  useEffect(() => {
    let isMounted = true;
    async function checkStatus() {
      try {
        const data = await getHealth();
        if (isMounted) {
          setHealth(data);
          setIsLoadingHealth(false);
        }
      } catch {
        if (isMounted) {
          setHealth({
            status: "offline",
            ollama: { reachable: false, model: "llama3.2" },
          });
          setIsLoadingHealth(false);
        }
      }
    }

    checkStatus();
    return () => {
      isMounted = false;
    };
  }, []);

  const [cumulativeAnswers, setCumulativeAnswers] = useState<ClarificationAnswer[]>([]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    setError(null);
    setCumulativeAnswers([]);

    try {
      const response = await enhancePrompt(trimmed);
      setResult(response);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred while enhancing the prompt.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleContinueWithAnswers = async (newAnswers: ClarificationAnswer[]) => {
    const trimmed = prompt.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    setError(null);

    // Merge newly provided answers into cumulative list, preserving earlier answers
    const merged = [...cumulativeAnswers];
    for (const na of newAnswers) {
      const idx = merged.findIndex(
        (existing) =>
          existing.question_id === na.question_id ||
          (na.target_dimension && existing.target_dimension === na.target_dimension) ||
          existing.question === na.question
      );
      if (idx >= 0) {
        merged[idx] = na;
      } else {
        merged.push(na);
      }
    }
    setCumulativeAnswers(merged);

    try {
      const response = await enhancePrompt(trimmed, {
        clarification_answers: merged,
      });
      setResult(response);
      if (response.status === "enhanced") {
        setCumulativeAnswers([]);
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred while applying clarification answers.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleUseReasonableAssumptions = async () => {
    const trimmed = prompt.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await enhancePrompt(trimmed, {
        allow_reasonable_assumptions: true,
        clarification_answers: cumulativeAnswers.length > 0 ? cumulativeAnswers : undefined,
      });
      setResult(response);
      if (response.status === "enhanced") {
        setCumulativeAnswers([]);
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("An unexpected error occurred while enhancing with reasonable assumptions.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const isInputEmpty = prompt.trim().length === 0;

  return (
    <main className="container">
      <Header health={health} isLoadingHealth={isLoadingHealth} />

      <form onSubmit={handleSubmit} className="main-card">
        <div className="input-label">
          <span>Input Prompt</span>
          <span className="char-count">{prompt.length} / 10,000</span>
        </div>

        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe what you want the AI to do... (e.g., 'make attendance app python' or 'explain binary search easy')"
          className="prompt-textarea"
          rows={5}
          disabled={isLoading}
          maxLength={10000}
        />

        {error && (
          <div className="error-banner" role="alert">
            <span className="error-icon" aria-hidden="true">
              ⚠️
            </span>
            <div>{error}</div>
          </div>
        )}

        <div className="actions-row">
          <button
            type="submit"
            disabled={isInputEmpty || isLoading}
            className="btn-primary"
          >
            {isLoading ? (
              <>
                <span className="spinner" />
                <span>Understanding your intent and analyzing ambiguity...</span>
              </>
            ) : (
              <span>Enhance Prompt</span>
            )}
          </button>
        </div>
      </form>

      {/* Understood Intent Section (rendered whenever analysis exists) */}
      {result && result.analysis && <UnderstoodIntent analysis={result.analysis} />}

      {/* Phase 5: High-Impact Clarification Card (when status === "needs_clarification") */}
      {result && result.status === "needs_clarification" && (
        <ClarificationCard
          questions={result.questions}
          onSubmitAnswers={handleContinueWithAnswers}
          onUseReasonableAssumptions={handleUseReasonableAssumptions}
          isLoading={isLoading}
        />
      )}

      {/* Enhanced Output Section (rendered only when status === "enhanced") */}
      {result && result.status === "enhanced" && <EnhancedOutput result={result} />}

      {/* Phase 6: Clean Validation Failure Notice (when status === "validation_failed") */}
      {result && result.status === "validation_failed" && (
        <section
          className="main-card"
          aria-label="Validation Notice"
          style={{ borderColor: "rgba(239, 68, 68, 0.4)", background: "rgba(239, 68, 68, 0.05)" }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem" }} aria-hidden="true">
              ⚠️
            </span>
            <h3 style={{ fontSize: "1rem", fontWeight: 600, margin: 0, color: "var(--foreground, #ededed)" }}>
              Enhancement Validation Notice
            </h3>
          </div>
          <p style={{ margin: "0 0 1rem 0", color: "#a1a1aa", fontSize: "0.875rem", lineHeight: 1.5 }}>
            HumanIntent couldn't produce a reliable enhancement for this request. Try again or rephrase the prompt.
          </p>
          <button
            type="button"
            onClick={handleSubmit}
            className="btn-primary"
            style={{ width: "auto", padding: "0.45rem 1.1rem", fontSize: "0.875rem" }}
            disabled={isLoading}
          >
            Try Again
          </button>
        </section>
      )}

      {/* Source Grounding Notice (when status === "analysis_failed") */}
      {result && result.status === "analysis_failed" && (
        <section
          className="main-card"
          aria-label="Analysis Reliability Notice"
          style={{ borderColor: "rgba(245, 158, 11, 0.4)", background: "rgba(245, 158, 11, 0.05)" }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "1.25rem" }} aria-hidden="true">
              ⚠️
            </span>
            <h3 style={{ fontSize: "1rem", fontWeight: 600, margin: 0, color: "var(--foreground, #ededed)" }}>
              Intent Verification Notice
            </h3>
          </div>
          <p style={{ margin: "0 0 1rem 0", color: "#a1a1aa", fontSize: "0.875rem", lineHeight: 1.5 }}>
            HumanIntent could not verify that the analysis was grounded in your request. Please rephrase or provide a bit more detail.
          </p>
          <button
            type="button"
            onClick={handleSubmit}
            className="btn-primary"
            style={{ width: "auto", padding: "0.45rem 1.1rem", fontSize: "0.875rem" }}
            disabled={isLoading}
          >
            Try Again
          </button>
        </section>
      )}

      <footer className="footer-note">
        HumanIntent Phase 6 &bull; Intent-Preserving &bull; Intent-Preservation Validator &bull; Zero Hallucinated Requirements
      </footer>
    </main>
  );
}
