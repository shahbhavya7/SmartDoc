"use client";

/**
 * Client-side route protection for the authenticated shell.
 *
 * This is a UX guard, not a security boundary and the distinction is worth
 * being explicit about. Nothing here decides what data anyone may see: every
 * protected endpoint verifies the JWT server-side and derives `user_id` from the
 * token, so bypassing this component yields an authenticated-looking chrome with
 * 401s inside it, not someone else's documents. Its job is only to send an
 * unauthenticated visitor to `/login` instead of showing them empty panels.
 *
 * The guard waits for `loading` to clear before redirecting: on a hard reload
 * the stored token is still being checked against `/auth/me`, and redirecting
 * during that window would bounce a signed-in user to the login page.
 */

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";

import { useAuth } from "@/components/auth-provider";
import { AppLoading } from "@/components/app-loading";

export function RequireAuth({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const pathname = usePathname();

  useEffect(() => {
    if (loading || user) return;
    // `next` lets the login page return the visitor to where they were headed,
    // so a bookmarked chat URL survives an expired session.
    const next = pathname && pathname !== "/" ? `?next=${encodeURIComponent(pathname)}` : "";
    router.replace(`/login${next}`);
  }, [loading, user, router, pathname]);

  if (loading) return <AppLoading label="Restoring your session…" />;
  if (!user) return <AppLoading label="Redirecting to sign in…" />;

  return <>{children}</>;
}
