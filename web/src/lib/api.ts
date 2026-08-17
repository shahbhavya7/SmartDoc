/**
 * The single place this app talks to FastAPI.
 *
 * Every call goes through `request()`, which attaches the bearer token and
 * normalises the API's `{"error": {type, message}}` body into an `ApiError`.
 * Centralising it is what makes two guarantees checkable by reading one file:
 *
 * 1. **No endpoint is ever called without the token**, because attaching it is
 *    not the caller's job.
 * 2. **No request names a user.** There is no `user_id` parameter anywhere in
 *    this module the server derives identity from the token's `sub` claim, so
 *    a client-supplied id would be ignored even if one were sent. Data isolation
 *    is a server property; the frontend's part is simply not to invent a second
 *    notion of "whose data this is".
 */

import type {
  AskResponse,
  ChatMessage,
  ChatSession,
  DeleteDocumentResponse,
  DocumentList,
  EvalCalibration,
  EvalGoldSetOverview,
  EvalJob,
  EvalMethod,
  EvalRun,
  EvalRunSummary,
  EvalTestSetUpload,
  HealthResponse,
  TokenResponse,
  UploadResponse,
  User,
} from "./types";

export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000"
).replace(/\/$/, "");

/** An error the API reported, carrying the status so callers can branch on 401. */
export class ApiError extends Error {
  readonly status: number;
  readonly type: string;

  constructor(status: number, type: string, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
  }

  /** True when the token is missing, invalid, or expired. */
  get isUnauthenticated(): boolean {
    return this.status === 401;
  }
}

/** Raised when the browser could not reach the API at all. */
export class NetworkError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "NetworkError";
  }
}

interface RequestOptions {
  method?: string;
  /** JSON body. Mutually exclusive with `formData`. */
  body?: unknown;
  formData?: FormData;
  token?: string | null;
  signal?: AbortSignal;
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { method = "GET", body, formData, token, signal } = options;

  const headers: Record<string, string> = {};
  if (token) headers.Authorization = `Bearer ${token}`;
  // Content-Type is left unset for FormData so the browser can add the
  // multipart boundary; setting it by hand produces an unparseable body.
  if (body !== undefined) headers["Content-Type"] = "application/json";

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method,
      headers,
      body: formData ?? (body !== undefined ? JSON.stringify(body) : undefined),
      signal,
      cache: "no-store",
    });
  } catch (error) {
    // An aborted request is a caller decision, not a failure to surface.
    if (error instanceof DOMException && error.name === "AbortError") throw error;
    throw new NetworkError(
      `Could not reach the SmartDoc API at ${API_BASE}. Check that the backend is running.`,
    );
  }

  if (response.status === 204) return undefined as T;

  const raw = await response.text();
  let parsed: unknown = null;
  if (raw) {
    try {
      parsed = JSON.parse(raw);
    } catch {
      parsed = null;
    }
  }

  if (!response.ok) {
    // The API returns one error shape everywhere; fall back for anything that
    // slipped past it (a proxy error page, say) rather than showing "[object].
    const detail = (parsed as { error?: { type?: string; message?: string } } | null)?.error;
    throw new ApiError(
      response.status,
      detail?.type ?? "http_error",
      detail?.message ?? `Request failed with status ${response.status}.`,
    );
  }

  return parsed as T;
}

/* -------------------------------------------------------------------------- */
/* Auth                                                                       */
/* -------------------------------------------------------------------------- */

export const api = {
  health: () => request<HealthResponse>("/health"),

  signup: (email: string, password: string) =>
    request<TokenResponse>("/auth/signup", {
      method: "POST",
      body: { email, password },
    }),

  login: (email: string, password: string) =>
    request<TokenResponse>("/auth/login", {
      method: "POST",
      body: { email, password },
    }),

  /** Validates a stored token AND returns the live user row behind it. */
  me: (token: string) => request<User>("/auth/me", { token }),

  /* ------------------------------------------------------------------------ */
  /* Documents                                                               */
  /* ------------------------------------------------------------------------ */

  documents: (token: string, signal?: AbortSignal) =>
    request<DocumentList>("/documents", { token, signal }),

  deleteDocument: (token: string, documentId: string) =>
    request<DeleteDocumentResponse>(`/documents/${encodeURIComponent(documentId)}`, {
      method: "DELETE",
      token,
    }),

  upload: (token: string, files: File[], signal?: AbortSignal) => {
    const formData = new FormData();
    // The field name is repeated per file: FastAPI binds `list[UploadFile]`
    // from repeated `files` parts, not from a single array-valued one.
    for (const file of files) formData.append("files", file);
    return request<UploadResponse>("/upload", {
      method: "POST",
      formData,
      token,
      signal,
    });
  },

  /* ------------------------------------------------------------------------ */
  /* Sessions and messages                                                   */
  /* ------------------------------------------------------------------------ */

  sessions: (token: string, limit = 10, signal?: AbortSignal) =>
    request<ChatSession[]>(`/sessions?limit=${limit}`, { token, signal }),

  createSession: (token: string, title = "") =>
    request<ChatSession>("/sessions", { method: "POST", body: { title }, token }),

  deleteSession: (token: string, sessionId: string) =>
    request<{ session_id: string; deleted: boolean }>(
      `/sessions/${encodeURIComponent(sessionId)}`,
      { method: "DELETE", token },
    ),

  messages: (token: string, sessionId: string, signal?: AbortSignal) =>
    request<ChatMessage[]>(
      `/sessions/${encodeURIComponent(sessionId)}/messages`,
      { token, signal },
    ),

  /* ------------------------------------------------------------------------ */
  /* Ask                                                                      */
  /* ------------------------------------------------------------------------ */

  /**
   * Ask a question. Passing `sessionId` makes it a remembered turn: the server
   * stores both messages, resolves references from the session's summary, and
   * updates that summary in a background task after responding.
   */
  ask: (
    token: string,
    question: string,
    sessionId?: string | null,
    signal?: AbortSignal,
  ) =>
    request<AskResponse>("/ask", {
      method: "POST",
      body: { question, session_id: sessionId ?? null },
      token,
      signal,
    }),

  /* ------------------------------------------------------------------------ */
  /* Evaluation                                                              */
  /*                                                                          */
  /* Reading is cheap; starting a run is not. `startEvalRun` returns as soon   */
  /* as the job is queued and progress is polled from `evalJob`, because a     */
  /* full run is minutes long and holding a request open for it would time     */
  /* out in every layer between here and the server.                          */
  /* ------------------------------------------------------------------------ */

  evalMethod: (token: string, signal?: AbortSignal) =>
    request<EvalMethod>("/eval/method", { token, signal }),

  evalGoldSet: (token: string, signal?: AbortSignal) =>
    request<EvalGoldSetOverview>("/eval/gold-set", { token, signal }),

  evalCalibration: (token: string, signal?: AbortSignal) =>
    request<EvalCalibration>("/eval/calibration", { token, signal }),

  evalRuns: (token: string, limit = 25, signal?: AbortSignal) =>
    request<{ runs: EvalRunSummary[] }>(`/eval/runs?limit=${limit}`, {
      token,
      signal,
    }),

  evalRun: (token: string, runId: string, signal?: AbortSignal) =>
    request<EvalRun>(`/eval/runs/${encodeURIComponent(runId)}`, { token, signal }),

  evalLatestRun: (token: string, signal?: AbortSignal) =>
    request<EvalRun>("/eval/runs/latest", { token, signal }),

  uploadTestSet: (token: string, file: File, signal?: AbortSignal) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<EvalTestSetUpload>("/eval/test-sets", {
      method: "POST",
      formData,
      token,
      signal,
    });
  },

  startEvalRun: (
    token: string,
    options: {
      testSetId?: string | null;
      categories?: string[] | null;
      label?: string;
    } = {},
  ) =>
    request<EvalJob>("/eval/runs", {
      method: "POST",
      body: {
        test_set_id: options.testSetId ?? null,
        categories: options.categories ?? null,
        label: options.label ?? "",
        skip_consistency_wait: true,
      },
      token,
    }),

  evalJob: (token: string, jobId: string, signal?: AbortSignal) =>
    request<EvalJob>(`/eval/jobs/${encodeURIComponent(jobId)}`, { token, signal }),

  /** The caller's in-flight run, if any — lets the page resume after a reload. */
  evalActiveJob: (token: string, signal?: AbortSignal) =>
    request<{ job: EvalJob | null }>("/eval/jobs", { token, signal }),
};

/** Where to send the browser to begin Google sign-in (a server-side redirect). */
export const googleLoginUrl = () => `${API_BASE}/auth/google/login`;
