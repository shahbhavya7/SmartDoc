/** Presentation helpers. Nothing here makes a decision only formats one. */

/**
 * Human-readable byte count. Binary units, because that is what a file manager
 * shows for the same PDF, and a mismatch reads as a bug.
 */
export function formatBytes(bytes: number | null | undefined): string {
  if (bytes === null || bytes === undefined) return "—";
  if (bytes <= 0) return "0 KB";

  const units = ["B", "KB", "MB", "GB"];
  const exponent = Math.min(
    Math.floor(Math.log(bytes) / Math.log(1024)),
    units.length - 1,
  );
  const value = bytes / 1024 ** exponent;
  // One decimal below 10 keeps "1.4 MB" informative without "1.437 MB" noise.
  const digits = exponent === 0 ? 0 : value < 10 ? 1 : 0;
  return `${value.toFixed(digits)} ${units[exponent]}`;
}

/**
 * "4m ago" / "Yesterday" / "12 Mar". Falls back to the raw string if the API
 * ever hands over something unparseable, rather than rendering "Invalid Date".
 */
export function formatRelativeTime(iso: string | null | undefined): string {
  if (!iso) return "";
  const then = parseUtc(iso);
  if (!then) return iso;

  const seconds = Math.round((Date.now() - then.getTime()) / 1000);
  if (seconds < 45) return "just now";
  if (seconds < 90) return "1m ago";
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 7200) return "1h ago";
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  if (seconds < 172800) return "yesterday";
  if (seconds < 604800) return `${Math.round(seconds / 86400)}d ago`;

  return then.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const parsed = parseUtc(iso);
  if (!parsed) return iso;
  return parsed.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/**
 * The API writes timestamps with `datetime.utcnow().isoformat()`-style output,
 * which carries no timezone designator. `new Date()` would read that as LOCAL
 * time, making every "2h ago" wrong by the UTC offset so a bare timestamp is
 * explicitly marked as UTC before parsing.
 */
function parseUtc(iso: string): Date | null {
  const hasZone = /(?:Z|[+-]\d{2}:?\d{2})$/.test(iso);
  const parsed = new Date(hasZone ? iso : `${iso}Z`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

/** A session's sidebar label: its title, or its first question, or a fallback. */
export function sessionLabel(title: string, fallback = "New chat"): string {
  const trimmed = (title ?? "").trim();
  if (!trimmed) return fallback;
  return trimmed.length > 60 ? `${trimmed.slice(0, 57)}…` : trimmed;
}

/** Drops the `.pdf` suffix for display; the badge alongside already says PDF. */
export function displayFilename(filename: string): string {
  return filename.replace(/\.pdf$/i, "");
}

export function initialsFor(email: string): string {
  const local = (email ?? "").split("@")[0] ?? "";
  const parts = local.split(/[._-]+/).filter(Boolean);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return (local.slice(0, 2) || "?").toUpperCase();
}
