/**
 * Where the JWT lives in the browser.
 *
 * `localStorage`, not a cookie. The API authorizes on an `Authorization: Bearer`
 * header and runs CORS with `allow_credentials=False`, so a cookie would never
 * be sent on an API call anyway — and a non-HttpOnly cookie is readable by the
 * same script that can read localStorage, so it would buy nothing while adding a
 * CSRF surface that bearer headers do not have.
 *
 * The trade-off is honest: localStorage is readable by any script that achieves
 * XSS on this origin. Mitigations are that the app renders no third-party
 * script, and that the stored expiry is enforced client-side too, so a token
 * left in a shared browser stops being presented once it is stale.
 */

import type { User } from "./types";

const TOKEN_KEY = "smartdoc.token";
const EXPIRY_KEY = "smartdoc.token.expires_at";
const USER_KEY = "smartdoc.user";

/**
 * Re-read on the next tick after a cross-tab change, so signing out in one tab
 * takes effect in the others rather than leaving a stale authenticated view.
 */
export const AUTH_STORAGE_KEYS = [TOKEN_KEY, EXPIRY_KEY, USER_KEY];

function available(): boolean {
  // Guarded for SSR, and for a browser where storage is disabled outright
  // (Safari private mode historically threw on access rather than returning
  // null), which must degrade to "not signed in" instead of crashing render.
  try {
    return typeof window !== "undefined" && !!window.localStorage;
  } catch {
    return false;
  }
}

export interface StoredSession {
  token: string;
  user: User | null;
}

export function saveSession(token: string, expiresInSeconds: number, user: User): void {
  if (!available()) return;
  try {
    const expiresAt = Date.now() + expiresInSeconds * 1000;
    window.localStorage.setItem(TOKEN_KEY, token);
    window.localStorage.setItem(EXPIRY_KEY, String(expiresAt));
    window.localStorage.setItem(USER_KEY, JSON.stringify(user));
  } catch {
    // A full or blocked store must not break sign-in; the session simply does
    // not survive a reload.
  }
}

/**
 * Stores a token that arrived without a user body — the Google callback
 * redirect carries only `?token=`. The caller then resolves the user via
 * `GET /auth/me`, which is the authoritative source for it regardless.
 */
export function saveBareToken(token: string): void {
  if (!available()) return;
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
    window.localStorage.removeItem(EXPIRY_KEY);
    window.localStorage.removeItem(USER_KEY);
  } catch {
    /* see saveSession */
  }
}

export function saveUser(user: User): void {
  if (!available()) return;
  try {
    window.localStorage.setItem(USER_KEY, JSON.stringify(user));
  } catch {
    /* see saveSession */
  }
}

/**
 * The stored session, or null. A token past its recorded expiry is cleared and
 * reported as absent: presenting it would only earn a 401, and treating it as
 * present would flash an authenticated shell before the redirect.
 */
export function loadSession(): StoredSession | null {
  if (!available()) return null;
  try {
    const token = window.localStorage.getItem(TOKEN_KEY);
    if (!token) return null;

    const expiresAt = Number(window.localStorage.getItem(EXPIRY_KEY) ?? 0);
    if (expiresAt && Date.now() >= expiresAt) {
      clearSession();
      return null;
    }

    let user: User | null = null;
    const rawUser = window.localStorage.getItem(USER_KEY);
    if (rawUser) {
      try {
        user = JSON.parse(rawUser) as User;
      } catch {
        // Corrupt cached user: harmless, since /auth/me is what we trust.
        user = null;
      }
    }
    return { token, user };
  } catch {
    return null;
  }
}

export function clearSession(): void {
  if (!available()) return;
  try {
    for (const key of AUTH_STORAGE_KEYS) window.localStorage.removeItem(key);
  } catch {
    /* nothing useful to do */
  }
}
