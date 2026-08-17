import { AppShell } from "@/components/app-shell";
import { Evaluation } from "@/components/evaluation/evaluation";
import { RequireAuth } from "@/components/require-auth";

export const metadata = { title: "Evaluation · SmartDoc" };

export default function EvaluationPage() {
  return (
    <RequireAuth>
      <AppShell>
        <Evaluation />
      </AppShell>
    </RequireAuth>
  );
}
