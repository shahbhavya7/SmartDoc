/**
 * Keeps citations attached to an answer across a page reload.
 *
 * `GET /sessions/{id}/messages` returns `{role, content}` — the durable record of
 * *what was said*. It does not carry the sources or the grounding verdict, which
 * `POST /ask` returns alongside the answer and which the server does not persist.
 * So on a reload, message history alone would render every past answer with its
 * citations missing, and structural citations are the feature that makes an
 * answer checkable.
 *
 * This is a presentation cache, and it is deliberately *not* treated as data:
 *
 * - It is keyed by the assistant message's server-assigned id, so an entry can
 *   only ever be shown against the exact message it was produced for.
 * - A miss renders the answer text with no sources panel — never a guess, and
 *   never another answer's citations.
 * - It is per-origin browser storage, so it does not follow the user to another
 *   device. That is a known limitation of not persisting sources server-side,
 *   and the alternative (writing them into the `messages` table) is a schema
 *   change outside this phase.
 */

import type { AskResponse } from "./types";

const KEY = "smartdoc.answers.v1";
/** Bounded so a heavy user does not eventually fill the origin's quota. */
const MAX_ENTRIES = 400;

export interface CachedAnswer {
  sources: AskResponse["sources"];
  grounding: AskResponse["grounding"];
  query_type: string;
}

type CacheShape = Record<string, CachedAnswer>;

function read(): CacheShape {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? (parsed as CacheShape) : {};
  } catch {
    return {};
  }
}

function write(cache: CacheShape): void {
  try {
    const keys = Object.keys(cache);
    let next = cache;
    if (keys.length > MAX_ENTRIES) {
      // Insertion order is preserved by JSON round-trips, so dropping from the
      // front evicts the oldest answers.
      next = Object.fromEntries(
        keys.slice(keys.length - MAX_ENTRIES).map((key) => [key, cache[key]]),
      );
    }
    window.localStorage.setItem(KEY, JSON.stringify(next));
  } catch {
    // Quota exceeded or storage disabled: citations then simply do not survive
    // a reload, which degrades cleanly.
  }
}

export function rememberAnswer(messageId: string, response: AskResponse): void {
  if (typeof window === "undefined" || !messageId) return;
  const cache = read();
  cache[messageId] = {
    sources: response.sources,
    grounding: response.grounding,
    query_type: response.query_type,
  };
  write(cache);
}

export function recallAnswers(messageIds: string[]): Record<string, CachedAnswer> {
  if (typeof window === "undefined" || messageIds.length === 0) return {};
  const cache = read();
  const found: Record<string, CachedAnswer> = {};
  for (const id of messageIds) {
    if (cache[id]) found[id] = cache[id];
  }
  return found;
}

/** Called when a session is deleted, so its entries do not linger forever. */
export function forgetAnswers(messageIds: string[]): void {
  if (typeof window === "undefined" || messageIds.length === 0) return;
  const cache = read();
  let changed = false;
  for (const id of messageIds) {
    if (cache[id]) {
      delete cache[id];
      changed = true;
    }
  }
  if (changed) write(cache);
}
