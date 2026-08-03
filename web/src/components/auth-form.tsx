"use client";

/**
 * The shared sign-in / sign-up form.
 *
 * One component for both modes because they differ only in which API call runs
 * and what the copy says — duplicating it would let the two screens drift apart
 * visually, which on an auth screen reads as a phishing page.
 *
 * Error handling deliberately shows the server's message verbatim. The API
 * returns the *same* text for an unknown email and a wrong password so the form
 * cannot be used to discover which addresses are registered; paraphrasing it
 * client-side ("no account with that email") would undo that on the frontend.
 */

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { ArrowRight, Loader2, LockKeyhole, Mail } from "lucide-react";

import { useAuth } from "@/components/auth-provider";
import { Brand } from "@/components/brand";
import { GoogleButton } from "@/components/google-button";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ApiError, NetworkError } from "@/lib/api";

const MIN_PASSWORD_CHARS = 8;

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const isSignup = mode === "signup";
  const { user, login, signup, loading } = useAuth();
  const router = useRouter();
  const searchParams = useSearchParams();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Only paths are honoured, so `?next=https://evil.example` cannot turn the
  // login page into an open redirect.
  const rawNext = searchParams.get("next");
  const next = rawNext && rawNext.startsWith("/") && !rawNext.startsWith("//")
    ? rawNext
    : "/dashboard";

  // Covers both "already signed in, visited /login" and "just authenticated".
  useEffect(() => {
    if (!loading && user) router.replace(next);
  }, [loading, user, router, next]);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);

    const trimmedEmail = email.trim();
    if (!trimmedEmail || !password) {
      setError("Enter your email and password.");
      return;
    }
    // Checked here as well as server-side purely to save a round trip; the
    // server's own validation is what actually enforces it.
    if (isSignup && password.length < MIN_PASSWORD_CHARS) {
      setError(`Choose a password of at least ${MIN_PASSWORD_CHARS} characters.`);
      return;
    }

    setSubmitting(true);
    try {
      await (isSignup ? signup(trimmedEmail, password) : login(trimmedEmail, password));
      router.replace(next);
    } catch (caught) {
      if (caught instanceof ApiError || caught instanceof NetworkError) {
        setError(caught.message);
      } else {
        setError("Something went wrong. Please try again.");
      }
      setSubmitting(false);
    }
  }

  const busy = submitting || loading;

  return (
    <div className="grid min-h-dvh place-items-center px-4 py-10">
      <div className="w-full max-w-[26rem] animate-rise">
        <div className="mb-8 flex flex-col items-center gap-5 text-center">
          <Brand size="lg" />
          <div className="space-y-1.5">
            <h1 className="text-2xl font-semibold tracking-tight">
              {isSignup ? "Create your workspace" : "Welcome back"}
            </h1>
            <p className="text-sm text-muted-foreground">
              {isSignup
                ? "Upload your documents and ask them anything."
                : "Sign in to ask your documents a question."}
            </p>
          </div>
        </div>

        <div className="glass rounded-2xl p-6 sm:p-7">
          <form onSubmit={onSubmit} className="space-y-4" noValidate>
            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <div className="relative">
                <Mail
                  className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden
                />
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  placeholder="you@company.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={busy}
                  className="h-11 pl-9"
                  required
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <LockKeyhole
                  className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground"
                  aria-hidden
                />
                <Input
                  id="password"
                  type="password"
                  autoComplete={isSignup ? "new-password" : "current-password"}
                  placeholder={isSignup ? `At least ${MIN_PASSWORD_CHARS} characters` : "••••••••"}
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  disabled={busy}
                  className="h-11 pl-9"
                  required
                />
              </div>
            </div>

            {error ? (
              <p
                role="alert"
                className="rounded-lg border border-destructive/35 bg-destructive/12 px-3 py-2.5 text-sm text-red-200"
              >
                {error}
              </p>
            ) : null}

            <Button type="submit" disabled={busy} className="h-11 w-full gap-2 font-semibold">
              {submitting ? (
                <>
                  <Loader2 className="size-4 animate-spin" aria-hidden />
                  {isSignup ? "Creating account…" : "Signing in…"}
                </>
              ) : (
                <>
                  {isSignup ? "Create account" : "Sign in"}
                  <ArrowRight className="size-4" aria-hidden />
                </>
              )}
            </Button>
          </form>

          <div className="my-5 flex items-center gap-3">
            <span className="h-px flex-1 bg-border" />
            <span className="text-xs uppercase tracking-wider text-muted-foreground">or</span>
            <span className="h-px flex-1 bg-border" />
          </div>

          <GoogleButton disabled={busy} next={next} />
        </div>

        <p className="mt-6 text-center text-sm text-muted-foreground">
          {isSignup ? "Already have an account?" : "New to SmartDoc?"}{" "}
          <Link
            href={isSignup ? "/login" : "/signup"}
            className="font-medium text-primary underline-offset-4 hover:underline"
          >
            {isSignup ? "Sign in" : "Create an account"}
          </Link>
        </p>
      </div>
    </div>
  );
}
