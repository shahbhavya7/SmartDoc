import { Suspense } from "react";

import { AppLoading } from "@/components/app-loading";
import { OAuthCallback } from "@/components/oauth-callback";

export const metadata = { title: "Signing in — SmartDoc" };

export default function OAuthCallbackPage() {
  return (
    <Suspense fallback={<AppLoading label="Completing sign in…" />}>
      <OAuthCallback />
    </Suspense>
  );
}
