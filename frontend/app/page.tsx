"use client";

import React, { useEffect, useRef, useState } from "react";
import { Header } from "@/components/Header";
import { UnderstoodIntent } from "@/components/UnderstoodIntent";
import { EnhancedOutput } from "@/components/EnhancedOutput";
import { ClarificationCard } from "@/components/ClarificationCard";
import { HistoryDrawer } from "@/components/HistoryDrawer";
import {
  enhancePrompt,
  getHealth,
  ClarificationAnswer,
  EnhanceResponse,
  HealthResponse,
  HistoryDetailResponse,
} from "@/lib/api";

export default function Home() {
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState<"quick" | "professional" | "deep">("professional");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<EnhanceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [isLoadingHealth, setIsLoadingHealth] = useState(true);
  const [isHistoryOpen, setIsHistoryOpen] = useState(false);
  const [cumulativeAnswers, setCumulativeAnswers] = useState<ClarificationAnswer[]>([]);

  const textareaRef = useRef<HTMLTextAreaElement>(null);

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
            ollama: { reachable: false, model: "llama3" },
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

  const scrollToWorkspace = () => {
    const el = document.getElementById("workspace");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
    setTimeout(() => {
      textareaRef.current?.focus();
    }, 350);
  };

  const scrollToHowItWorks = () => {
    const el = document.getElementById("how-it-works");
    if (el) {
      el.scrollIntoView({ behavior: "smooth" });
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = prompt.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    setError(null);
    setCumulativeAnswers([]);
    setSessionId(null);

    try {
      const response = await enhancePrompt(trimmed, { mode });
      setResult(response);
      setSessionId(response.session_id);
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

  const handleReset = () => {
    setPrompt("");
    setError(null);
    setResult(null);
    setSessionId(null);
    setCumulativeAnswers([]);
    if (textareaRef.current) {
      textareaRef.current.focus();
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      const trimmed = prompt.trim();
      if (trimmed && !isLoading) {
        handleSubmit(e);
      }
    }
  };

  const handleContinueWithAnswers = async (newAnswers: ClarificationAnswer[]) => {
    const trimmed = prompt.trim();
    if (!trimmed || isLoading) return;

    setIsLoading(true);
    setError(null);

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
        session_id: sessionId || undefined,
        clarification_answers: merged,
        mode,
      });
      setResult(response);
      setSessionId(response.session_id);
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
        session_id: sessionId || undefined,
        allow_reasonable_assumptions: true,
        clarification_answers: cumulativeAnswers.length > 0 ? cumulativeAnswers : undefined,
        mode,
      });
      setResult(response);
      setSessionId(response.session_id);
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

  const handleSelectHistorySession = (detail: HistoryDetailResponse) => {
    setPrompt(detail.original_prompt);
    setSessionId(detail.session_id);
    setError(null);
    setCumulativeAnswers([]);
    if (detail.mode && (detail.mode === "quick" || detail.mode === "professional" || detail.mode === "deep")) {
      setMode(detail.mode as "quick" | "professional" | "deep");
    }

    if (detail.status === "enhanced" && detail.latest_analysis) {
      setResult({
        status: "enhanced",
        session_id: detail.session_id,
        original_prompt: detail.original_prompt,
        analysis: detail.latest_analysis,
        enhanced_prompt: detail.final_enhanced_prompt || "",
        model: detail.model || "llama3",
        template: detail.template || undefined,
        validation: detail.validation || undefined,
        quality: detail.quality || undefined,
        history_saved: detail.history_saved,
      });
    } else if (detail.status === "needs_clarification" && detail.latest_analysis) {
      const lastRound = detail.clarification_rounds[detail.clarification_rounds.length - 1];
      const questions = lastRound?.questions || detail.latest_analysis.ambiguity?.questions || [];
      setResult({
        status: "needs_clarification",
        session_id: detail.session_id,
        original_prompt: detail.original_prompt,
        analysis: detail.latest_analysis,
        questions: questions,
        model: detail.model || "llama3",
        history_saved: detail.history_saved,
      });
    } else if (detail.status === "validation_failed" && detail.latest_analysis) {
      setResult({
        status: "validation_failed",
        session_id: detail.session_id,
        original_prompt: detail.original_prompt,
        analysis: detail.latest_analysis,
        validation: detail.validation!,
        message: "HumanIntent could not produce a sufficiently faithful enhanced prompt after one repair attempt.",
        model: detail.model || "llama3",
        history_saved: detail.history_saved,
      });
    } else if (detail.status === "analysis_failed" && detail.latest_analysis) {
      setResult({
        status: "analysis_failed",
        session_id: detail.session_id,
        original_prompt: detail.original_prompt,
        analysis: detail.latest_analysis,
        message: "HumanIntent could not verify that the analysis was grounded in your request.",
        model: detail.model || "llama3",
        history_saved: detail.history_saved,
      });
    }

    scrollToWorkspace();
  };

  const isInputEmpty = prompt.trim().length === 0;

  return (
    <div className="page-wrapper">
      <Header
        health={health}
        isLoadingHealth={isLoadingHealth}
        onOpenHistory={() => setIsHistoryOpen(true)}
        mode={mode}
        onScrollToWorkspace={scrollToWorkspace}
        onScrollToHowItWorks={scrollToHowItWorks}
      />

      <main className="content-container">
        {/* ==================== 1. PREMIUM HERO SECTION ==================== */}
        <section className="hero-section" aria-label="Product Overview">
          <div className="hero-grid">
            <div className="hero-copy-col">
              <div className="hero-pill-tag">
                <span className="hero-pill-dot" aria-hidden="true" />
                <span>Intent-Preserving Prompt Optimization</span>
              </div>

              <h1 className="hero-headline">
                Refine the prompt.<br />
                <span className="hero-headline-italic">Keep the intent.</span>
              </h1>

              <p className="hero-lead">
                Turn rough ideas into clearer instructions without losing what you actually meant.
                Preserves explicit technologies, quantities, and negative constraints while keeping instructional scope bounded.
              </p>

              <div className="hero-cta-group">
                <button
                  type="button"
                  onClick={scrollToWorkspace}
                  className="btn-hero-primary"
                >
                  <span>Enhance a prompt</span>
                  <span className="btn-arrow" aria-hidden="true">↓</span>
                </button>

                <button
                  type="button"
                  onClick={scrollToHowItWorks}
                  className="btn-hero-secondary"
                >
                  <span>See how it works</span>
                  <span className="btn-arrow" aria-hidden="true">→</span>
                </button>
              </div>

              <div className="hero-metrics-strip">
                <div className="metric-item">
                  <span className="metric-val">Sub-30s</span>
                  <span className="metric-lbl">Normal Latency</span>
                </div>
                <div className="metric-divider" aria-hidden="true" />
                <div className="metric-item">
                  <span className="metric-val">100%</span>
                  <span className="metric-lbl">Local-First</span>
                </div>
                <div className="metric-divider" aria-hidden="true" />
                <div className="metric-item">
                  <span className="metric-val">Single-Pass</span>
                  <span className="metric-lbl">Fast Path</span>
                </div>
              </div>
            </div>

            {/* Right: Authentic Transformation Preview */}
            <div className="hero-preview-col" aria-hidden="true">
              <div className="preview-card">
                <div className="preview-header">
                  <span className="preview-label">Prompt Transformation Sample</span>
                  <span className="preview-badge">Deterministic</span>
                </div>

                <div className="preview-body">
                  <div className="preview-block block-before">
                    <span className="block-tag tag-input">Original Input</span>
                    <p className="block-text">
                      &ldquo;create a rest api in fastapi without any database and do not use docker&rdquo;
                    </p>
                  </div>

                  <div className="preview-arrow-row">
                    <div className="preview-arrow-line" />
                    <span className="preview-arrow-text">Preserves constraints &bull; Structured instructions</span>
                    <div className="preview-arrow-line" />
                  </div>

                  <div className="preview-block block-after">
                    <span className="block-tag tag-output">Enhanced Output</span>
                    <div className="block-output-preview">
                      <div className="preview-code-line">
                        <strong className="preview-kw">ROLE &amp; OBJECTIVE:</strong> Implement a REST API using FastAPI.
                      </div>
                      <div className="preview-code-line">
                        <strong className="preview-kw">CONSTRAINTS:</strong>
                      </div>
                      <div className="preview-code-sub">
                        &bull; Do not use any database (in-memory storage only)
                      </div>
                      <div className="preview-code-sub">
                        &bull; Do not use Docker or containerization
                      </div>
                      <div className="preview-code-line">
                        <strong className="preview-kw">OUTPUT FORMAT:</strong> Complete endpoint definitions with type hints.
                      </div>
                    </div>
                  </div>
                </div>

                <div className="preview-footer">
                  <span className="preview-check-item">✓ Explicit constraints retained</span>
                  <span className="preview-check-item">✓ Bounded to stated intent</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ==================== 2. MAIN WORKSPACE ==================== */}
        <section id="workspace" className="workspace-section" aria-label="Prompt Enhancement Workspace">
          <div className="section-header-row">
            <div>
              <h2 className="section-title">Workspace</h2>
              <p className="section-subtitle">
                Compose or paste your prompt. HumanIntent will structure and optimize it for downstream AI models.
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="workspace-card">
            <div className="editor-top-bar">
              <div className="editor-label-group">
                <label htmlFor="prompt-input" className="editor-label">
                  Your Prompt
                </label>
                <span className="char-counter" aria-live="polite">
                  {prompt.length.toLocaleString()} / 10,000 characters
                </span>
              </div>

              {/* Mode Segmented Control */}
              <div className="mode-segmented-control" role="radiogroup" aria-label="Enhancement Mode">
                <button
                  type="button"
                  role="radio"
                  aria-checked={mode === "quick"}
                  className={`mode-seg-btn ${mode === "quick" ? "active" : ""}`}
                  onClick={() => setMode("quick")}
                  title="Quick: Fast, concise prompt enhancement"
                >
                  Quick
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={mode === "professional"}
                  className={`mode-seg-btn ${mode === "professional" ? "active" : ""}`}
                  onClick={() => setMode("professional")}
                  title="Professional: Thorough constraints, structure, and edge cases (Recommended)"
                >
                  Professional
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={mode === "deep"}
                  className={`mode-seg-btn ${mode === "deep" ? "active" : ""}`}
                  onClick={() => setMode("deep")}
                  title="Deep: Exhaustive instructions, verification criteria, and detailed guidance"
                >
                  Deep
                </button>
              </div>
            </div>

            <div className="textarea-container">
              <textarea
                ref={textareaRef}
                id="prompt-input"
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Describe what you want the AI to do... (e.g. 'Build a REST API endpoint in FastAPI to register users, but do not use SQLAlchemy and without any ORM, no Docker.')"
                className="workspace-textarea"
                rows={6}
                disabled={isLoading}
                maxLength={10000}
                aria-label="Prompt text input"
              />
            </div>

            {error && (
              <div className="error-alert" role="alert">
                <span className="alert-symbol" aria-hidden="true">⚠️</span>
                <div className="alert-message">{error}</div>
              </div>
            )}

            <div className="editor-bottom-bar">
              <div className="action-button-row">
                <button
                  type="submit"
                  disabled={isInputEmpty || isLoading}
                  className="btn-enhance-primary"
                >
                  {isLoading ? (
                    <>
                      <span className="btn-spinner" aria-hidden="true" />
                      <span>Refining your prompt…</span>
                    </>
                  ) : (
                    <>
                      <span>Enhance Prompt</span>
                      <span className="btn-arrow" aria-hidden="true">↵</span>
                    </>
                  )}
                </button>

                <button
                  type="button"
                  onClick={handleReset}
                  disabled={isLoading || (isInputEmpty && !result && !error)}
                  className="btn-workspace-reset"
                  title="Clear input and reset results"
                >
                  Reset
                </button>
              </div>

              <div className="shortcut-tag" aria-hidden="true">
                <kbd>Ctrl</kbd> + <kbd>Enter</kbd>
              </div>
            </div>
          </form>

          {/* ==================== 3. UNDERSTOOD INTENT ==================== */}
          {result && result.analysis && <UnderstoodIntent analysis={result.analysis} />}

          {/* ==================== 4. CLARIFICATION CARD ==================== */}
          {result && result.status === "needs_clarification" && (
            <ClarificationCard
              questions={result.questions}
              onSubmitAnswers={handleContinueWithAnswers}
              onUseReasonableAssumptions={handleUseReasonableAssumptions}
              isLoading={isLoading}
            />
          )}

          {/* ==================== 5. ENHANCED OUTPUT ==================== */}
          {result && result.status === "enhanced" && <EnhancedOutput result={result} />}

          {/* ==================== 6. VALIDATION FAILURE NOTICE ==================== */}
          {result && result.status === "validation_failed" && (
            <section className="notice-card notice-error" aria-label="Validation Notice">
              <div className="notice-header">
                <span className="notice-icon" aria-hidden="true">⚠️</span>
                <h3 className="notice-title">Enhancement Validation Notice</h3>
              </div>
              <p className="notice-body">
                HumanIntent could not produce a sufficiently faithful enhanced prompt for this request after automated verification. Please rephrase or adjust your constraints.
              </p>
              <button
                type="button"
                onClick={handleSubmit}
                className="btn-notice-action"
                disabled={isLoading}
              >
                Try Again
              </button>
            </section>
          )}

          {/* ==================== 7. ANALYSIS FAILURE NOTICE ==================== */}
          {result && result.status === "analysis_failed" && (
            <section className="notice-card notice-warning" aria-label="Intent Reliability Notice">
              <div className="notice-header">
                <span className="notice-icon" aria-hidden="true">⚠️</span>
                <h3 className="notice-title">Intent Verification Notice</h3>
              </div>
              <p className="notice-body">
                HumanIntent could not verify that the analysis was grounded in your original request. Please rephrase or provide a bit more context.
              </p>
              <button
                type="button"
                onClick={handleSubmit}
                className="btn-notice-action"
                disabled={isLoading}
              >
                Try Again
              </button>
            </section>
          )}
        </section>

        {/* ==================== 8. HOW IT WORKS SECTION ==================== */}
        <section id="how-it-works" className="how-it-works-section" aria-label="How HumanIntent Works">
          <div className="section-header-center">
            <span className="section-eyebrow">Design Principles</span>
            <h2 className="section-title-large">Engineered to Preserve Intent</h2>
            <p className="section-subtitle-wide">
              Most prompt optimizers hallucinate unrequested features, ignore negative constraints, and drag down latency. HumanIntent is architected differently.
            </p>
          </div>

          <div className="pillars-grid">
            <div className="pillar-card">
              <div className="pillar-num">01</div>
              <h3 className="pillar-heading">Bounded Feature Scope</h3>
              <p className="pillar-text">
                Rather than reflexively inserting unrequested frameworks, authentication stacks, or database layers, HumanIntent bounds prompt enhancement to what was explicitly specified or required.
              </p>
              <div className="pillar-tag-row">
                <span className="pillar-tag">Explicit scope only</span>
                <span className="pillar-tag">Bounded architecture</span>
              </div>
            </div>

            <div className="pillar-card">
              <div className="pillar-num">02</div>
              <h3 className="pillar-heading">Negative Constraint Retention</h3>
              <p className="pillar-text">
                Restrictions like &ldquo;without Docker&rdquo;, &ldquo;no ORM&rdquo;, and &ldquo;zero external dependencies&rdquo; are parsed as hard negative constraints and deterministically verified in the final prompt.
              </p>
              <div className="pillar-tag-row">
                <span className="pillar-tag">Active negation checks</span>
                <span className="pillar-tag">Rule preservation</span>
              </div>
            </div>

            <div className="pillar-card">
              <div className="pillar-num">03</div>
              <h3 className="pillar-heading">Single-Pass Local Architecture</h3>
              <p className="pillar-text">
                Rather than executing multiple slow sequential LLM analyzer and repair passes, HumanIntent couples deterministic NLP preprocessing with a single calibrated local generation pass to maintain responsive execution.
              </p>
              <div className="pillar-tag-row">
                <span className="pillar-tag">Sub-30s target</span>
                <span className="pillar-tag">100% private</span>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* ==================== 9. REFINED EDITORIAL FOOTER ==================== */}
      <footer className="site-footer">
        <div className="footer-inner">
          <div className="footer-brand-col">
            <div className="footer-brand-title">HumanIntent v1.0</div>
            <p className="footer-brand-desc">
              Intent-Preserving Adaptive Prompt Optimization System.
            </p>
          </div>

          <div className="footer-meta-col">
            <span className="footer-meta-item">Local Ollama Backend</span>
            <span className="footer-dot">&bull;</span>
            <span className="footer-meta-item">Private SQLite Persistence</span>
            <span className="footer-dot">&bull;</span>
            <span className="footer-meta-item">Zero Cloud Telemetry</span>
          </div>
        </div>
      </footer>

      {/* Session History Drawer */}
      <HistoryDrawer
        isOpen={isHistoryOpen}
        onClose={() => setIsHistoryOpen(false)}
        onSelectSession={handleSelectHistorySession}
      />
    </div>
  );
}
