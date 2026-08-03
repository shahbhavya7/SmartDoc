"use client";

/**
 * Landing point for the Google flow.
 *
 * The backend's `/auth/google/callback` finishes the code exchange server-side,
 * mints the same kind of JWT that password login issues, and redirects here with
 * `?token=`. So this page never sees an authorization code, a client secret, or
 * Google's tokens — only the finished bearer token. That is the whole reason auth
 * is owned by FastAPI: the browser is handed a result, not a credential exchange
 * to perform.
 *
 * The token is validated by `adoptToken`, which calls `/auth/me` before the app
 * renders as signed in, so a token that has been tampered with in the URL bar
 * fails here rather than producing a shell full of 401s.
 *
 * It is also stripped from the address bar on success: a bearer token in a URL
 * lands in history and in any copied link.
 */

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { TriangleAlert } from "lucide-react";

import { useAuth } from "@/components/auth-provider";
import { AppLoading } from "@/components/app-loading";
import { Brand } from "@/components/brand";
import { Button } from "@/components/ui/button";
import { ApiError, NetworkError } from "@/lib/api";
import { takeOAuthNext } from "@/lib/oauth";

export function OAuthCallback() {
  const { adoptToken } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();
  const [failure, setFailure] = useState<string | null>(null);

  // React runs effects twice in development's strict mode; adopting the same
  // token twice would fire a second, pointless /auth/me.
  const handled = useRef(false);

  const token = searchParams.get("token");

  // Arriving with no token at all — the user cancelled at Google's consent
  // screen, or OAUTH_SUCCESS_REDIRECT is misconfigured — is derived straight
  // from the URL rather than stored, so no state write is needed to render it.
  const error =
    failure ?? (token ? null : "Google sign-in did not complete. No token was returned.");

  useEffect(() => {
    if (handled.current || !token) return;
    handled.current = true;

    const next = takeOAuthNext();

    adoptToken(token)
      .then(() => {
        // replace(), not push(): the token-bearing URL must not be reachable by
        // pressing Back.
        window.history.replaceState(null, "", "/auth/callback");
        router.replace(next);
      })
      .catch((caught) => {
        if (caught instanceof ApiError || caught instanceof NetworkError) {
          setFailure(caught.message);
        } else {
          setFailure("Could not complete Google sign-in.");
        }
      });
  }, [adoptToken, router, token]);

  if (!error) return <AppLoading label="Completing Google sign-in…" />;

  return (
    <div className="grid min-h-dvh place-items-center px-4">
      <div className="w-full max-w-md animate-rise">
        <div className="mb-7 flex justify-center">
          <Brand size="lg" />
        </div>
        <div className="glass rounded-2xl p-6 text-center">
          <TriangleAlert className="mx-auto mb-3 size-8 text-destructive" aria-hidden />
          <h1 className="text-lg font-semibold">Sign-in failed</h1>
          <p className="mt-2 text-sm text-muted-foreground">{error}</p>
          <Button
            className="mt-6 h-11 w-full font-semibold"
            render={<Link href="/login">Back to sign in</Link>}
          />
        </div>
      </div>
    </div>
  );
}
