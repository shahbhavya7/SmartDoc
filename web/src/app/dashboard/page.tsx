import { AppShell } from "@/components/app-shell";
import { Dashboard } from "@/components/dashboard/dashboard";
import { RequireAuth } from "@/components/require-auth";

export const metadata = { title: "Dashboard SmartDoc" };

export default function DashboardPage() {
  return (
    <RequireAuth>
      <AppShell>
        <Dashboard />
      </AppShell>
    </RequireAuth>
  );
}
