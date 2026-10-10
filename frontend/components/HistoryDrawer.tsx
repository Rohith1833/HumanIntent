import React, { useEffect, useState } from "react";
import {
  getHistory,
  getHistoryDetail,
  deleteHistoryItem,
  clearHistory,
  HistorySummaryItem,
  HistoryDetailResponse,
} from "@/lib/api";

interface HistoryDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectSession: (detail: HistoryDetailResponse) => void;
}

export const HistoryDrawer: React.FC<HistoryDrawerProps> = ({
  isOpen,
  onClose,
  onSelectSession,
}) => {
  const [items, setItems] = useState<HistorySummaryItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [loadingSessionId, setLoadingSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getHistory(50, 0);
      setItems(data.items);
    } catch (err: unknown) {
      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Failed to load history.");
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchHistory();
    }
  }, [isOpen]);

  const handleSelectItem = async (sessionId: string) => {
    setLoadingSessionId(sessionId);
    try {
      const detail = await getHistoryDetail(sessionId);
      onSelectSession(detail);
      onClose();
    } catch (err: unknown) {
      if (err instanceof Error) {
        alert(`Failed to load historical session: ${err.message}`);
      }
    } finally {
      setLoadingSessionId(null);
    }
  };

  const handleDeleteItem = async (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    try {
      await deleteHistoryItem(sessionId);
      setItems((prev) => prev.filter((item) => item.session_id !== sessionId));
    } catch (err: unknown) {
      if (err instanceof Error) {
        alert(`Failed to delete item: ${err.message}`);
      }
    }
  };

  const handleClearAll = async () => {
    const confirmed = window.confirm(
      "Are you sure you want to clear all locally stored history? This cannot be undone."
    );
    if (!confirmed) return;

    try {
      await clearHistory();
      setItems([]);
    } catch (err: unknown) {
      if (err instanceof Error) {
        alert(`Failed to clear history: ${err.message}`);
      }
    }
  };

  const formatTimestamp = (isoString: string) => {
    try {
      const d = new Date(isoString);
      return d.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return isoString;
    }
  };

  const getStatusBadgeClass = (status: string) => {
    switch (status) {
      case "enhanced":
        return "badge-status-enhanced";
      case "needs_clarification":
        return "badge-status-clarification";
      case "validation_failed":
        return "badge-status-failed";
      case "analysis_failed":
        return "badge-status-warning";
      default:
        return "badge-status-default";
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "enhanced":
        return "Enhanced";
      case "needs_clarification":
        return "Clarification";
      case "validation_failed":
        return "Validation Failed";
      case "analysis_failed":
        return "Analysis Failed";
      default:
        return status;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="drawer-overlay" onClick={onClose} role="dialog" aria-modal="true" aria-label="History Panel">
      <div className="drawer-content" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <div className="drawer-title-row">
              <h2 className="drawer-title">History</h2>
              <span className="badge badge-local">Stored locally</span>
            </div>
            <p className="drawer-subtitle">
              Your prompt enhancement sessions on this machine.
            </p>
          </div>
          <button
            type="button"
            className="btn-close"
            onClick={onClose}
            aria-label="Close History"
          >
            ✕
          </button>
        </div>

        <div className="drawer-toolbar">
          <span className="drawer-count">
            {items.length} {items.length === 1 ? "session" : "sessions"}
          </span>
          {items.length > 0 && (
            <button
              type="button"
              className="btn-clear-history"
              onClick={handleClearAll}
              title="Delete all historical sessions"
            >
              Clear History
            </button>
          )}
        </div>

        <div className="drawer-body">
          {isLoading ? (
            <div className="drawer-empty">
              <span className="spinner" />
              <span>Loading local history...</span>
            </div>
          ) : error ? (
            <div className="drawer-empty" style={{ color: "#ef4444" }}>
              ⚠️ {error}
            </div>
          ) : items.length === 0 ? (
            <div className="drawer-empty">
              <p>No historical sessions found.</p>
              <p className="drawer-empty-sub">
                Prompts you enhance will be saved locally here for easy recall.
              </p>
            </div>
          ) : (
            <ul className="history-list">
              {items.map((item) => (
                <li
                  key={item.session_id}
                  className={`history-item ${
                    loadingSessionId === item.session_id ? "loading" : ""
                  }`}
                  onClick={() => handleSelectItem(item.session_id)}
                >
                  <div className="history-item-top">
                    <span className="history-item-time">
                      {formatTimestamp(item.created_at)}
                    </span>
                    <div className="history-item-badges">
                      <span className={`badge-status ${getStatusBadgeClass(item.status)}`}>
                        {getStatusLabel(item.status)}
                      </span>
                      {item.category && (
                        <span className="badge-category">{item.category}</span>
                      )}
                      {item.quality_score !== null && item.quality_score !== undefined && (
                        <span className="badge-quality">
                          {Math.round(item.quality_score)}/100
                        </span>
                      )}
                    </div>
                  </div>

                  <p className="history-item-preview">
                    {item.original_prompt_preview}
                  </p>

                  <div className="history-item-actions">
                    <span className="history-load-hint">
                      {loadingSessionId === item.session_id ? "Loading..." : "Click to view result"}
                    </span>
                    <button
                      type="button"
                      className="btn-delete-item"
                      onClick={(e) => handleDeleteItem(e, item.session_id)}
                      title="Delete this session"
                      aria-label="Delete this session"
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
                        <polyline points="3 6 5 6 21 6" />
                        <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                      </svg>
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};
