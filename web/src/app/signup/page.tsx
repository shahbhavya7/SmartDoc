import { Suspense } from "react";

import { AppLoading } from "@/components/app-loading";
import { AuthForm } from "@/components/auth-form";

export const metadata = { title: "Create an account SmartDoc" };

export default function SignupPage() {
  return (
    <Suspense fallback={<AppLoading label="Loading sign up…" />}>
      <AuthForm mode="signup" />
    </Suspense>
  );
}
