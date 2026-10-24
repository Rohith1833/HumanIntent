import React, { useState } from "react";
import { ClarificationQuestion, ClarificationAnswer } from "@/lib/api";

interface ClarificationCardProps {
  questions: ClarificationQuestion[];
  onSubmitAnswers: (answers: ClarificationAnswer[]) => void;
  onUseReasonableAssumptions: () => void;
  isLoading: boolean;
}

export const ClarificationCard: React.FC<ClarificationCardProps> = ({
  questions,
  onSubmitAnswers,
  onUseReasonableAssumptions,
  isLoading,
}) => {
  const [answers, setAnswers] = useState<Record<string, string>>({});

  const handleInputChange = (questionId: string, value: string) => {
    setAnswers((prev) => ({
      ...prev,
      [questionId]: value,
    }));
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const formattedAnswers: ClarificationAnswer[] = questions.map((q) => ({
      question_id: q.id,
      question: q.question,
      answer: (answers[q.id] || "").trim(),
      target_dimension: q.target_dimension,
    })).filter((a) => a.answer.length > 0);

    onSubmitAnswers(formattedAnswers);
  };

  const hasAtLeastOneAnswer = Object.values(answers).some(
    (val) => val.trim().length > 0
  );

  return (
    <section className="clarification-card" aria-label="Prompt Clarification">
      <div className="clarification-header">
        <div className="clarification-icon-pill" aria-hidden="true">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <circle cx="12" cy="12" r="10" />
            <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        </div>
        <div>
          <h2 className="clarification-title">
            One or two details would improve this prompt
          </h2>
          <p className="clarification-subtitle">
            Providing these details helps tailor the enhanced prompt without making unsupported assumptions.
          </p>
        </div>
      </div>

      <form onSubmit={handleFormSubmit} className="clarification-form">
        <div className="clarification-questions-list">
          {questions.map((q, index) => (
            <div key={q.id} className="clarification-item">
              <label htmlFor={`clarification-${q.id}`} className="clarification-label">
                <span className="question-number">{index + 1}.</span> {q.question}
              </label>
              <input
                id={`clarification-${q.id}`}
                type="text"
                className="clarification-input"
                placeholder="Your answer (optional)..."
                value={answers[q.id] || ""}
                onChange={(e) => handleInputChange(q.id, e.target.value)}
                disabled={isLoading}
              />
            </div>
          ))}
        </div>

        <div className="clarification-actions">
          <button
            type="submit"
            className="btn-primary"
            disabled={!hasAtLeastOneAnswer || isLoading}
          >
            {isLoading ? (
              <>
                <span className="spinner" />
                <span>Applying answers...</span>
              </>
            ) : (
              <span>Continue with Answers</span>
            )}
          </button>

          <button
            type="button"
            className="btn-secondary"
            onClick={onUseReasonableAssumptions}
            disabled={isLoading}
          >
            <span>Enhance with Reasonable Assumptions</span>
          </button>
        </div>
      </form>
    </section>
  );
};
