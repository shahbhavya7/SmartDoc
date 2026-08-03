import Link from "next/link";

import { cn } from "@/lib/utils";

/**
 * The wordmark. A glass tile holding the mark, then champagne gradient type —
 * the two design-system signatures in one element, so it reads as the same
 * product on the auth screens and inside the app.
 */
export function Brand({
  href,
  className,
  size = "md",
}: {
  /** Rendered as a link when set; as plain text otherwise (e.g. on /login). */
  href?: string;
  className?: string;
  size?: "sm" | "md" | "lg";
}) {
  const tile = {
    sm: "size-7 rounded-lg text-[13px]",
    md: "size-9 rounded-xl text-[15px]",
    lg: "size-12 rounded-2xl text-lg",
  }[size];

  const word = {
    sm: "text-base",
    md: "text-lg",
    lg: "text-2xl",
  }[size];

  const content = (
    <span className={cn("flex items-center gap-2.5", className)}>
      <span
        className={cn(
          "glass-accent grid shrink-0 place-items-center font-semibold text-primary",
          tile,
        )}
        aria-hidden
      >
        SD
      </span>
      <span className={cn("text-accent-gradient font-semibold tracking-tight", word)}>
        SmartDoc
      </span>
    </span>
  );

  if (!href) return content;

  return (
    <Link
      href={href}
      className="rounded-xl outline-none transition-opacity hover:opacity-85 focus-visible:ring-2 focus-visible:ring-ring"
    >
      {content}
    </Link>
  );
}
