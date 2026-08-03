import { Brand } from "@/components/brand";

/** Full-height placeholder for the moment before the shell can be rendered. */
export function AppLoading({ label = "Loading…" }: { label?: string }) {
  return (
    <div className="grid min-h-dvh place-items-center px-6">
      <div className="flex flex-col items-center gap-4">
        <Brand size="lg" />
        <p className="shimmer-text text-sm font-medium" role="status">
          {label}
        </p>
      </div>
    </div>
  );
}
