/**
 * Wire types mirroring the FastAPI response models in `backend/main.py`.
 *
 * These are a transcription of the server's contract, not a second source of
 * truth: nothing here is computed, defaulted, or reinterpreted. The frontend
 * makes no identity or retrieval decisions, so there is no client-side model of
 * a user's permissions or of how an answer was produced — only of what the API
 * returned.
 */

export interface User {
  id: string;
  email: string;
  created_at: string;
  /** "password" and/or "google". */
  auth_methods: string[];
}

export interface TokenResponse {
  access_token: string;
  token_type: "bearer";
  /** Lifetime in seconds, used to expire the stored token client-side too. */
  expires_in: number;
  user: User;
}

export interface Source {
  source: string;
  page: number;
  snippet: string;
  section: string;
  page_end: number | null;
}

export interface Grounding {
  checked: boolean;
  faithful: boolean | null;
  unsupported_claims: string[];
  /** Informational: a legitimately derived figure lands here too. */
  unverified_numbers: string[];
  note: string;
  /** "regenerated" | "pruned" | "declined" | "none" | "". */
  repaired: string;
  removed_claims: string[];
}

export interface AskResponse {
  answer: string;
  sources: Source[];
  query_type: string;
  grounding: Grounding | null;
  diagnostics: Record<string, unknown> | null;
  session_id: string | null;
}

export interface DocumentRecord {
  id: string;
  filename: string;
  created_at: string;
  chunks: number | null;
  /** null means genuinely unknown, not empty. See backend/documents.py. */
  size_bytes: number | null;
}

export interface DocumentList {
  documents: DocumentRecord[];
  total_chunks: number;
  /** Lower bound: documents of unknown size contribute nothing. */
  total_bytes: number;
  documents_with_unknown_size: number;
}

export interface UploadFileResult {
  filename: string;
  status: "success" | "error";
  document_id: string | null;
  pages_parsed: number | null;
  chunks_created: number | null;
  chunks_indexed: number | null;
  error: string | null;
}

export interface UploadResponse {
  files: UploadFileResult[];
  total_chunks_indexed: number;
  collection_name: string;
  collection_count: number;
}

export interface DeleteDocumentResponse {
  document_id: string;
  filename: string;
  chunks_deleted: number;
  row_deleted: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  /** Running conversation summary, maintained server-side. */
  summary: string;
  last_document: string | null;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
}

export interface HealthResponse {
  status: "ok";
  collection: string | null;
  embedding_model: string | null;
  /** Whether the server has Google OAuth credentials configured. */
  google_oauth_enabled: boolean;
}
