import { Suspense } from "react";

import { AppLoading } from "@/components/app-loading";
import { AuthForm } from "@/components/auth-form";

export const metadata = { title: "Sign in SmartDoc" };

export default function LoginPage() {
  // `useSearchParams` (for `?next=`) suspends during prerender, so the boundary
  // is required rather than decorative.
  return (
    <Suspense fallback={<AppLoading label="Loading sign in…" />}>
      <AuthForm mode="login" />
    </Suspense>
  );
}
