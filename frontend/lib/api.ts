export interface OllamaStatus {
  reachable: boolean;
  model: string;
}

export interface HealthResponse {
  status: string;
  ollama: OllamaStatus;
}

export type PromptCategory =
  | "CODING"
  | "DEBUGGING"
  | "RESEARCH"
  | "EDUCATION"
  | "WRITING"
  | "DATA_ANALYSIS"
  | "PLANNING"
  | "CREATIVE"
  | "IMAGE_GENERATION"
  | "SUMMARIZATION"
  | "GENERAL";

export interface PromptClassification {
  category: PromptCategory;
  confidence: number;
}

export type AmbiguityLevel = "LOW" | "MEDIUM" | "HIGH";

export interface ClarificationQuestion {
  id: string;
  question: string;
  reason: string;
  target_dimension: string;
}

export interface AmbiguityAssessment {
  score: number;
  level: AmbiguityLevel;
  needs_clarification: boolean;
  reasons: string[];
  questions: ClarificationQuestion[];
}

export type SourceType = "ORIGINAL_PROMPT" | "CLARIFICATION";

export type GroundingRelation =
  | "DIRECT"
  | "MODIFIES_DELIVERABLE"
  | "MODIFIES_OUTPUT_STYLE"
  | "CONSTRAINT"
  | "CLARIFICATION"
  | "PARAPHRASE";

export interface GroundingEvidence {
  field: string;
  value: string;
  source_type: SourceType;
  evidence: string;
  relation?: GroundingRelation;
}

export interface AnalysisGrounding {
  is_grounded: boolean;
  evidence: GroundingEvidence[];
  unsupported_items: string[];
}

export interface DeliverableSemantics {
  deliverable_type?: string | null;
  intended_outcome?: string | null;
  is_functionally_complete: boolean;
}

export interface PromptAnalysis {
  classification: PromptClassification;
  objective: string;
  explicit_requirements: string[];
  constraints: string[];
  preferences: string[];
  missing_context: string[];
  ambiguity?: AmbiguityAssessment;
  grounding?: AnalysisGrounding;
  deliverable_semantics?: DeliverableSemantics;
}

export interface ClarificationAnswer {
  question_id: string;
  question: string;
  answer: string;
  target_dimension?: string;
}

export type ValidationSeverity = "INFO" | "WARNING" | "ERROR";

export type ValidationIssueType =
  | "OBJECTIVE_DRIFT"
  | "ANALYSIS_OBJECTIVE_DRIFT"
  | "UNSUPPORTED_ANALYSIS_REQUIREMENT"
  | "UNSUPPORTED_ANALYSIS_CONSTRAINT"
  | "UNSUPPORTED_ANALYSIS_PREFERENCE"
  | "ANALYSIS_GROUNDING_FAILURE"
  | "SEMANTIC_GROUNDING_MISMATCH"
  | "MISSING_REQUIREMENT"
  | "MISSING_CONSTRAINT"
  | "MISSING_PREFERENCE"
  | "QUANTITY_MISMATCH"
  | "INVENTED_REQUIREMENT"
  | "INVENTED_TECHNOLOGY"
  | "INVENTED_FEATURE"
  | "CONSTRAINT_CONTRADICTION"
  | "REJECTED_OPTION_REINTRODUCED"
  | "MISSING_CONTEXT_PROMOTED"
  | "COMPOUND_TASK_LOSS"
  | "CLARIFICATION_MISMATCH"
  | "META_COMMENTARY"
  | "OTHER";

export interface ValidationIssue {
  type: ValidationIssueType;
  severity: ValidationSeverity;
  message: string;
  evidence?: string | null;
}

export interface PromptValidation {
  is_valid: boolean;
  issues: ValidationIssue[];
}

export interface QualityDimension {
  score: number;
  label: string;
  reason: string;
}

export type InflationRisk = "LOW" | "MEDIUM" | "HIGH";

export interface PromptQualityReport {
  overall_score: number;
  overall_label: string;
  intent_preservation: QualityDimension;
  requirement_coverage: QualityDimension;
  constraint_preservation: QualityDimension;
  clarity: QualityDimension;
  specificity: QualityDimension;
  output_definition: QualityDimension;
  structure: QualityDimension;
  concision: QualityDimension;
  inflation_ratio: number;
  inflation_risk: InflationRisk;
}

export interface EnhancedResponse {
  status: "enhanced";
  session_id: string;
  original_prompt: string;
  analysis: PromptAnalysis;
  enhanced_prompt: string;
  model: string;
  template?: string;
  validation?: PromptValidation;
  quality?: PromptQualityReport;
  mode?: string;
  total_ms?: number;
  inference_ms?: number;
  stage_timings?: Array<{ stage: string; elapsed_ms: number; llm_called: boolean }>;
  history_saved?: boolean;
}

export interface ClarificationRequiredResponse {
  status: "needs_clarification";
  session_id: string;
  original_prompt: string;
  analysis: PromptAnalysis;
  questions: ClarificationQuestion[];
  model: string;
  mode?: string;
  total_ms?: number;
  history_saved?: boolean;
}

export interface ValidationFailedResponse {
  status: "validation_failed";
  session_id: string;
  original_prompt: string;
  analysis: PromptAnalysis;
  validation: PromptValidation;
  message: string;
  model: string;
  mode?: string;
  total_ms?: number;
  history_saved?: boolean;
}

export interface AnalysisFailedResponse {
  status: "analysis_failed";
  session_id: string;
  original_prompt: string;
  analysis: PromptAnalysis;
  message: string;
  model: string;
  mode?: string;
  total_ms?: number;
  history_saved?: boolean;
}

export type EnhanceResponse =
  | EnhancedResponse
  | ClarificationRequiredResponse
  | ValidationFailedResponse
  | AnalysisFailedResponse;

export interface EnhanceOptions {
  session_id?: string;
  clarification_answers?: ClarificationAnswer[];
  allow_reasonable_assumptions?: boolean;
  mode?: "quick" | "professional" | "deep";
}

export interface HistorySummaryItem {
  session_id: string;
  created_at: string;
  updated_at: string;
  original_prompt_preview: string;
  status: string;
  category?: string | null;
  quality_score?: number | null;
}

export interface HistoryListResponse {
  items: HistorySummaryItem[];
  total: number;
  limit: number;
  offset: number;
}

export interface ClarificationRoundDetail {
  round_number: number;
  created_at: string;
  questions?: ClarificationQuestion[] | null;
  answers?: ClarificationAnswer[] | null;
  allow_reasonable_assumptions: boolean;
}

export interface HistoryDetailResponse {
  session_id: string;
  created_at: string;
  updated_at: string;
  original_prompt: string;
  status: string;
  mode?: string | null;
  model?: string | null;
  allow_reasonable_assumptions: boolean;
  category?: string | null;
  template?: string | null;
  clarification_rounds: ClarificationRoundDetail[];
  latest_analysis?: PromptAnalysis | null;
  final_enhanced_prompt?: string | null;
  validation?: PromptValidation | null;
  quality?: PromptQualityReport | null;
  feedback?: FeedbackResponse | null;
  history_saved: boolean;
  dataset_consent: boolean;
}

export type FeedbackRating = "HELPFUL" | "NOT_HELPFUL";

export interface FeedbackRequest {
  session_id: string;
  rating: FeedbackRating;
  reason?: string | null;
  edited_prompt?: string | null;
  dataset_consent?: boolean;
}

export interface FeedbackResponse {
  id: string;
  session_id: string;
  rating: FeedbackRating;
  reason?: string | null;
  edited_prompt?: string | null;
  dataset_consent: boolean;
  created_at: string;
  updated_at: string;
}

export interface DeleteResponse {
  status: string;
  session_id?: string | null;
  deleted_count?: number | null;
}

export interface ModelsResponse {
  models: string[];
  current_model: string;
}

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/+$/, "") || "http://localhost:8000";

export class ApiError extends Error {
  statusCode?: number;
  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
  }
}

function sanitizeErrorMessage(detail: string): string {
  if (
    /validation error|required schema|field required|pydantic|traceback/i.test(detail)
  ) {
    return "HumanIntent couldn't produce a valid structured analysis for this request. Please try again or rephrase the request.";
  }
  return detail;
}

/**
 * Fetch backend and Ollama connectivity status.
 */
export async function getHealth(): Promise<HealthResponse> {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" },
      cache: "no-store",
    });

    if (!res.ok) {
      throw new ApiError(`Health check failed with status ${res.status}`, res.status);
    }

    return await res.json();
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      `Cannot connect to HumanIntent backend at ${API_BASE_URL}. Ensure the backend is running.`
    );
  }
}

/**
 * Standalone analysis of a rough prompt.
 */
export async function analyzePrompt(prompt: string): Promise<PromptAnalysis> {
  const trimmed = prompt.trim();
  if (!trimmed) {
    throw new ApiError("Prompt cannot be empty or contain only whitespace.", 422);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: trimmed }),
    });
  } catch {
    throw new ApiError(
      `Cannot connect to HumanIntent backend at ${API_BASE_URL}. Ensure the backend server is running.`
    );
  }

  if (!res.ok) {
    let errorDetail = "";
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail || "";
    } catch {
      errorDetail = await res.text();
    }

    const sanitized = sanitizeErrorMessage(errorDetail);

    if (res.status === 503) {
      throw new ApiError(
        sanitized ||
          "HumanIntent backend is running, but Ollama could not be reached. Make sure Ollama is running locally.",
        503
      );
    }

    if (res.status === 502) {
      throw new ApiError(
        sanitized || "HumanIntent couldn't produce a valid structured analysis for this request. Please try again or rephrase the request.",
        502
      );
    }

    throw new ApiError(sanitized || `Analysis failed with status ${res.status}.`, res.status);
  }

  return await res.json();
}

/**
 * Submit rough human prompt to be analyzed and enhanced by the backend.
 */
export async function enhancePrompt(
  prompt: string,
  options?: EnhanceOptions
): Promise<EnhanceResponse> {
  const trimmed = prompt.trim();
  if (!trimmed) {
    throw new ApiError("Prompt cannot be empty or contain only whitespace.", 422);
  }

  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/enhance`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: trimmed,
        session_id: options?.session_id,
        clarification_answers: options?.clarification_answers,
        allow_reasonable_assumptions: options?.allow_reasonable_assumptions ?? false,
        mode: options?.mode ?? "professional",
      }),
    });
  } catch {
    throw new ApiError(
      `Cannot connect to HumanIntent backend at ${API_BASE_URL}. Ensure the backend server is running.`
    );
  }

  if (!res.ok) {
    let errorDetail = "";
    try {
      const errorJson = await res.json();
      errorDetail = errorJson.detail || "";
    } catch {
      errorDetail = await res.text();
    }

    const sanitized = sanitizeErrorMessage(errorDetail);

    if (res.status === 503) {
      throw new ApiError(
        sanitized ||
          "HumanIntent backend is running, but Ollama could not be reached. Make sure Ollama is running locally.",
        503
      );
    }

    if (res.status === 404) {
      throw new ApiError(
        sanitized ||
          "The configured Ollama model is not installed. Please check your Ollama installation.",
        404
      );
    }

    if (res.status === 502) {
      throw new ApiError(
        sanitized || "HumanIntent couldn't produce a valid structured analysis for this request. Please try again or rephrase the request.",
        502
      );
    }

    if (res.status === 504) {
      throw new ApiError(
        sanitized ||
          "Ollama request timed out. The local model took too long to generate a response.",
        504
      );
    }

    if (res.status === 422) {
      throw new ApiError("Prompt cannot be empty or contain only whitespace.", 422);
    }

    throw new ApiError(sanitized || `Enhancement failed with status ${res.status}.`, res.status);
  }

  return await res.json();
}

/**
 * Retrieve paginated lightweight history summaries. Zero LLM calls.
 */
export async function getHistory(limit: number = 20, offset: number = 0): Promise<HistoryListResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/history?limit=${limit}&offset=${offset}`);
  } catch {
    throw new ApiError(`Cannot connect to HumanIntent backend at ${API_BASE_URL}.`);
  }

  if (!res.ok) {
    throw new ApiError(`Failed to fetch history (status ${res.status}).`, res.status);
  }

  return await res.json();
}

/**
 * Reconstruct complete historical enhancement result. Zero LLM calls.
 */
export async function getHistoryDetail(sessionId: string): Promise<HistoryDetailResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/history/${sessionId}`);
  } catch {
    throw new ApiError(`Cannot connect to HumanIntent backend at ${API_BASE_URL}.`);
  }

  if (!res.ok) {
    throw new ApiError(`Failed to fetch history detail for session ${sessionId}.`, res.status);
  }

  return await res.json();
}

/**
 * Delete a single historical session and its associated snapshots.
 */
export async function deleteHistoryItem(sessionId: string): Promise<DeleteResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/history/${sessionId}`, {
      method: "DELETE",
    });
  } catch {
    throw new ApiError(`Cannot connect to HumanIntent backend at ${API_BASE_URL}.`);
  }

  if (!res.ok) {
    throw new ApiError(`Failed to delete session ${sessionId}.`, res.status);
  }

  return await res.json();
}

/**
 * Clear all local history sessions.
 */
export async function clearHistory(): Promise<DeleteResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/history`, {
      method: "DELETE",
    });
  } catch {
    throw new ApiError(`Cannot connect to HumanIntent backend at ${API_BASE_URL}.`);
  }

  if (!res.ok) {
    throw new ApiError(`Failed to clear history.`, res.status);
  }

  return await res.json();
}

/**
 * Submit or update user feedback for an enhancement session.
 */
export async function submitFeedback(payload: FeedbackRequest): Promise<FeedbackResponse> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE_URL}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    throw new ApiError(`Cannot connect to HumanIntent backend at ${API_BASE_URL}.`);
  }

  if (!res.ok) {
    throw new ApiError(`Failed to submit feedback.`, res.status);
  }

  return await res.json();
}
