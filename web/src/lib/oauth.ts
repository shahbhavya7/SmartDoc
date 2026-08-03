/**
 * Shared between the Google button and the callback page, so the two cannot
 * disagree about where the post-OAuth destination was parked.
 */
export const OAUTH_NEXT_KEY = "smartdoc.oauth.next";

/** Reads and clears the parked destination, validated as a same-origin path. */
export function takeOAuthNext(fallback = "/dashboard"): string {
  try {
    const stored = window.sessionStorage.getItem(OAUTH_NEXT_KEY);
    window.sessionStorage.removeItem(OAUTH_NEXT_KEY);
    // Only a plain path is honoured, so a tampered value cannot turn the
    // callback into an open redirect to another site.
    if (stored && stored.startsWith("/") && !stored.startsWith("//")) return stored;
  } catch {
    /* storage unavailable fall through */
  }
  return fallback;
}
