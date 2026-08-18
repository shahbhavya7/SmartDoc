/**
 * Wire types mirroring the FastAPI response models in `backend/main.py`.
 *
 * These are a transcription of the server's contract, not a second source of
 * truth: nothing here is computed, defaulted, or reinterpreted. The frontend
 * makes no identity or retrieval decisions, so there is no client-side model of
 * a user's permissions or of how an answer was produced only of what the API
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

/** colpali branch experiment: which pipeline answers /ask. See backend/main.py. */
export type RetrievalBackend = "hybrid" | "colpali";

export interface AskResponse {
  answer: string;
  sources: Source[];
  query_type: string;
  grounding: Grounding | null;
  diagnostics: Record<string, unknown> | null;
  session_id: string | null;
  /** Which pipeline actually answered -- 'hybrid' or 'colpali' (visual). */
  backend: RetrievalBackend;
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

/* -------------------------------------------------------------------------- */
/* Evaluation                                                                 */
/*                                                                            */
/* Mirrors backend/evaluation.py and the JSON written by                      */
/* eval/eval_tool/report.py. As above, this is a transcription of the         */
/* server's contract: no score is computed on this side, only displayed.      */
/* -------------------------------------------------------------------------- */

/** One question's full result. Every field the harness measured is kept. */
export interface EvalQuestionResult {
  id: string;
  question: string;
  category: string;
  expected_answer: string;
  expected_source: string | null;
  generated_answer: string;
  similarity: number;
  passed: boolean;
  fail_reason: string;

  http_status: number;
  latency_ms: number;
  retrieved_sources: string[];
  cited_expected_source: boolean | null;
  query_type: string;

  /** The exact-value guard: numbers, codes, and IDs that had to match. */
  exact_match_applicable: boolean;
  exact_match_passed: boolean;
  expected_values: string[];
  found_values: string[];
  missing_values: string[];

  /** List questions: how many expected items actually appeared. */
  completeness_applicable: boolean;
  items_expected: number;
  items_found: number;
  missing_items: string[];

  /** comparison: did the answer render an actual table? */
  rendered_table: boolean | null;
  /** out_of_scope_*: did it correctly decline? */
  correctly_declined: boolean | null;

  /** consistency_pair: the same question asked twice. */
  run1_answer: string;
  run2_answer: string;
  self_similarity: number | null;
  stable: boolean | null;

  /** input_edge_*: judged on behaviour, not similarity. */
  expected_behavior: string;
  actual_behavior: string;
}

export interface EvalCategoryStat {
  total: number;
  passed: number;
  pass_rate: number;
  mean_similarity: number | null;
}

export interface EvalRun {
  meta: {
    timestamp: string;
    threshold: number;
    consistency_threshold: number;
    embedding_model: string;
    api_base_url: string;
    gold_set: string;
    question_count: number;
    consistency_wait_seconds: number;
    label?: string;
    user_id?: string;
    source?: string;
  };
  summary: {
    total: number;
    passed: number;
    pass_rate: number;
    mean_similarity: number | null;
  };
  by_category: Record<string, EvalCategoryStat>;
  results: EvalQuestionResult[];
}

/** A row in the run history list. */
export interface EvalRunSummary {
  run_id: string;
  timestamp: string;
  /** True for the project's baseline runs, which predate per-user tagging. */
  shared: boolean;
  label: string;
  gold_set: string;
  question_count: number;
  passed: number;
  pass_rate: number;
  mean_similarity: number | null;
  threshold: number | null;
}

export interface EvalJob {
  job_id: string;
  status: "queued" | "running" | "done" | "error";
  phase: string;
  total: number;
  completed: number;
  run_id: string | null;
  error: string;
  started_at: string;
  finished_at: string;
}

export interface EvalTestSetUpload {
  test_set_id: string;
  filename: string;
  question_count: number;
  categories: { name: string; count: number }[];
}

export interface EvalGoldSetOverview {
  total: number;
  minimum_per_category: number;
  categories: {
    name: string;
    count: number;
    meets_minimum: boolean;
    scored_by: string;
  }[];
  error?: string;
}

/** The plain-English explanation of scoring, served by the backend. */
export interface EvalMethod {
  summary: string;
  steps: { title: string; body: string }[];
  why_exact_match: string;
  metrics: { name: string; plain: string; detail: string }[];
  categories_note: string;
}

export interface EvalCalibration {
  proposed_threshold: number;
  current_threshold: number;
  threshold_basis?: string;
  gap: number;
  wrong_section_p95?: number;
  correct_median?: number;
  embedding_model: string;
  distributions: {
    correct: DistributionStats;
    wrong: DistributionStats;
    wrong_by_kind: Record<string, { n: number; mean: number; max: number }>;
  };
}

export interface DistributionStats {
  n: number;
  min: number;
  max: number;
  mean: number;
  median: number;
  p05?: number;
  p95?: number;
}

/* -------------------------------------------------------------------------- */
/* ColPali vs Hybrid comparison (colpali branch experiment)                   */
/* -------------------------------------------------------------------------- */

export interface ComparisonBackendSummary {
  label: string;
  run_path: string;
  timestamp: string;
  total: number;
  passed: number;
  pass_rate: number;
  mean_similarity: number | null;
  mean_latency_ms: number | null;
  mean_cost_usd: number | null;
  cost_coverage: string;
  by_category: Record<string, EvalCategoryStat>;
}

export interface ComparisonCategoryRow {
  category: string;
  hybrid_pass_rate: number | null;
  hybrid_total: number;
  colpali_pass_rate: number | null;
  colpali_total: number;
  delta: number | null;
  /** True for the two hypothesis-relevant category groups. */
  watch: boolean;
}

export interface ComparisonHypothesis {
  categories: string[];
  held: boolean | null;
  detail: string;
  /** Root-cause note for the 'comparison' category's ColPali failures, when applicable. */
  comparison_formatting_note?: string;
}

export interface ComparisonIntentRow {
  id: string;
  question: string;
  hybrid_detector: string;
  hybrid_fired: boolean | null;
  hybrid_passed: boolean;
  colpali_detector: string;
  colpali_fired: boolean | null;
  colpali_passed: boolean;
}

export interface ComparisonReport {
  generated_at: string | null;
  hybrid: ComparisonBackendSummary;
  colpali: ComparisonBackendSummary;
  by_category: ComparisonCategoryRow[];
  hypotheses: {
    sql_exact_match_outperforms: ComparisonHypothesis;
    colpali_layout_outperforms: ComparisonHypothesis;
  };
  surprising_categories: ComparisonCategoryRow[];
  intent_classifier_asymmetry: {
    per_question: ComparisonIntentRow[];
    hybrid_blind_spot_questions: { id: string; question: string }[];
  };
}
