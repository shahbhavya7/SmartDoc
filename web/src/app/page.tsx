"use client";

/**
 * `/` is a router, not a landing page: signed in goes to the dashboard, signed
 * out to sign-in. The decision waits for the stored token to be validated, so a
 * returning user is not bounced to /login while that check is in flight.
 */

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { AppLoading } from "@/components/app-loading";
import { useAuth } from "@/components/auth-provider";

export default function RootPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (loading) return;
    router.replace(user ? "/dashboard" : "/login");
  }, [loading, user, router]);

  return <AppLoading label="Starting SmartDoc…" />;
}
