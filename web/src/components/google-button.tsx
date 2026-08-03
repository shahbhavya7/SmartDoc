"use client";

/**
 * Google sign-in.
 *
 * This is a full-page navigation to `GET /auth/google/login`, not a fetch: the
 * endpoint answers with a 302 to Google's consent screen, and the OAuth `state`
 * value rides in a cookie the API sets on that same response. An XHR would
 * follow the redirect inside the tab's fetch context and hand back Google's HTML
 * instead of taking the user there.
 *
 * The button reflects `google_oauth_enabled` from `/health` rather than
 * optimistically linking: with credentials unset the endpoint returns 503, and a
 * plain link would drop the user on a raw JSON error page.
 */

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { api, googleLoginUrl } from "@/lib/api";
import { OAUTH_NEXT_KEY } from "@/lib/oauth";

function GoogleMark() {
  return (
    <svg viewBox="0 0 18 18" className="size-4" aria-hidden focusable="false">
      <path
        fill="#EA4335"
        d="M9 3.48c1.69 0 2.83.73 3.48 1.34l2.54-2.48C13.46.89 11.43 0 9 0 5.48 0 2.44 2.02.96 4.96l2.91 2.26C4.6 5.05 6.62 3.48 9 3.48Z"
      />
      <path
        fill="#4285F4"
        d="M17.64 9.2c0-.64-.06-1.11-.13-1.6H9v3.03h4.84c-.1.8-.62 2.01-1.79 2.82l2.84 2.2c1.7-1.57 2.75-3.88 2.75-6.45Z"
      />
      <path
        fill="#FBBC05"
        d="M3.88 10.78A5.54 5.54 0 0 1 3.58 9c0-.62.11-1.22.29-1.78L.96 4.96A9 9 0 0 0 0 9c0 1.45.35 2.82.96 4.04l2.92-2.26Z"
      />
      <path
        fill="#34A853"
        d="M9 18c2.43 0 4.47-.8 5.95-2.18l-2.84-2.2c-.76.53-1.78.9-3.11.9-2.38 0-4.4-1.57-5.13-3.74L.96 13.04C2.44 15.98 5.48 18 9 18Z"
      />
    </svg>
  );
}

export function GoogleButton({
  disabled = false,
  next = "/dashboard",
}: {
  disabled?: boolean;
  next?: string;
}) {
  // `null` = still checking. The button renders in a neutral disabled state
  // until then, so it never briefly offers a route that would 503.
  const [enabled, setEnabled] = useState<boolean | null>(null);
  const [redirecting, setRedirecting] = useState(false);

  useEffect(() => {
    let cancelled = false;
    api
      .health()
      .then((health) => {
        if (!cancelled) setEnabled(Boolean(health.google_oauth_enabled));
      })
      .catch(() => {
        // Backend unreachable: the email/password path will report that clearly
        // enough on submit, so this just stays unavailable.
        if (!cancelled) setEnabled(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function start() {
    // The OAuth round trip leaves this origin entirely, so the post-login
    // destination is parked in sessionStorage. It cannot ride in the redirect
    // URI: that value is registered with Google and must match exactly.
    try {
      window.sessionStorage.setItem(OAUTH_NEXT_KEY, next);
    } catch {
      // Losing it only costs a redirect to the default landing page.
    }
    setRedirecting(true);
    window.location.assign(googleLoginUrl());
  }

  const unavailable = enabled === false;

  return (
    <div className="space-y-2">
      <Button
        type="button"
        variant="outline"
        onClick={start}
        disabled={disabled || redirecting || enabled !== true}
        className="h-11 w-full gap-2.5 border-white/15 bg-white/[0.06] font-medium hover:bg-white/[0.11]"
      >
        <GoogleMark />
        {redirecting ? "Redirecting to Google…" : "Continue with Google"}
      </Button>
      {unavailable ? (
        <p className="text-center text-xs text-muted-foreground">
          Google sign-in is not configured on this server. Use email and password.
        </p>
      ) : null}
    </div>
  );
}
