import React, { useState } from "react";
import { submitFeedback, FeedbackRating } from "@/lib/api";

interface FeedbackWidgetProps {
  sessionId: string;
  enhancedPrompt: string;
}

export const FeedbackWidget: React.FC<FeedbackWidgetProps> = ({
  sessionId,
  enhancedPrompt,
}) => {
  const [rating, setRating] = useState<FeedbackRating | null>(null);
  const [reason, setReason] = useState("");
  const [editedPrompt, setEditedPrompt] = useState("");
  const [showEditPrompt, setShowEditPrompt] = useState(false);
  const [datasetConsent, setDatasetConsent] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleSelectRating = (selected: FeedbackRating) => {
    setRating(selected);
    setSubmitted(false);
    setErrorMessage(null);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!rating) return;

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      await submitFeedback({
        session_id: sessionId,
        rating,
        reason: reason.trim() ? reason.trim() : undefined,
        edited_prompt: showEditPrompt && editedPrompt.trim() ? editedPrompt.trim() : undefined,
        dataset_consent: datasetConsent,
      });
      setSubmitted(true);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setErrorMessage(err.message);
      } else {
        setErrorMessage("Failed to save feedback.");
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="feedback-container">
      <div className="feedback-prompt-row">
        <span className="feedback-label">Was this enhancement helpful?</span>
        <div className="feedback-buttons">
          <button
            type="button"
            className={`btn-feedback ${rating === "HELPFUL" ? "active helpful" : ""}`}
            onClick={() => handleSelectRating("HELPFUL")}
            aria-label="Mark helpful"
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
              <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
            </svg>
            Helpful
          </button>
          <button
            type="button"
            className={`btn-feedback ${rating === "NOT_HELPFUL" ? "active not-helpful" : ""}`}
            onClick={() => handleSelectRating("NOT_HELPFUL")}
            aria-label="Mark not helpful"
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
              <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3" />
            </svg>
            Not helpful
          </button>
        </div>
      </div>

      {rating !== null && (
        <form onSubmit={handleSubmit} className="feedback-form">
          <div className="feedback-field">
            <label htmlFor="feedback-reason" className="feedback-field-label">
              What could be improved? (optional)
            </label>
            <textarea
              id="feedback-reason"
              rows={2}
              className="feedback-textarea"
              placeholder="e.g., Too verbose, missed format constraint, or style could be more direct..."
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              disabled={isSubmitting}
            />
          </div>

          {!showEditPrompt ? (
            <button
              type="button"
              className="btn-link"
              onClick={() => {
                setShowEditPrompt(true);
                if (!editedPrompt) setEditedPrompt(enhancedPrompt);
              }}
            >
              + Edit the enhanced prompt with your preferred wording
            </button>
          ) : (
            <div className="feedback-field">
              <label htmlFor="feedback-edited-prompt" className="feedback-field-label">
                Your improved version of this prompt (optional)
              </label>
              <textarea
                id="feedback-edited-prompt"
                rows={4}
                className="feedback-textarea"
                placeholder="Paste or write your preferred version of the enhanced prompt..."
                value={editedPrompt}
                onChange={(e) => setEditedPrompt(e.target.value)}
                disabled={isSubmitting}
              />
            </div>
          )}

          <div className="feedback-consent-row">
            <label className="feedback-consent-label">
              <input
                type="checkbox"
                checked={datasetConsent}
                onChange={(e) => setDatasetConsent(e.target.checked)}
                disabled={isSubmitting}
                className="feedback-checkbox"
              />
              <span>Allow this feedback to be used for improving HumanIntent later</span>
            </label>
            <span className="feedback-consent-note">Default is off. Local history is never used for training without consent.</span>
          </div>

          {errorMessage && (
            <div className="feedback-error-msg" role="alert">
              ⚠️ {errorMessage}
            </div>
          )}

          <div className="feedback-actions">
            <button
              type="submit"
              disabled={isSubmitting}
              className="btn-feedback-submit"
            >
              {isSubmitting ? "Saving..." : submitted ? "Update Feedback" : "Submit Feedback"}
            </button>
            {submitted && (
              <span className="feedback-success-msg">
                ✓ Feedback saved locally!
              </span>
            )}
          </div>
        </form>
      )}
    </div>
  );
};
