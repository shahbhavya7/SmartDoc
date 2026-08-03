"use client";

/**
 * Renders an answer's prose.
 *
 * Answers are markdown-ish: the exhaustive and comparison prompts produce
 * bulleted lists and occasionally tables, and a synthesis answer is structured
 * into sections. Rendering that as preformatted text would make the system's
 * best answers its least readable ones.
 *
 * `react-markdown` is used with no raw-HTML plugin, so model output cannot inject
 * markup the answer is untrusted text as far as this component is concerned,
 * even though it originates from our own backend.
 */

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { cn } from "@/lib/utils";

export function AnswerText({
  text,
  className,
}: {
  text: string;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "space-y-3 text-[15px] leading-relaxed text-foreground/95",
        // Tables come from the fault-code / entitlement documents; they need to
        // scroll inside the bubble rather than widening the whole panel.
        "[&_table]:my-2 [&_table]:block [&_table]:w-full [&_table]:overflow-x-auto",
        className,
      )}
    >
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="whitespace-pre-wrap">{children}</p>,
          ul: ({ children }) => (
            <ul className="ml-1 list-outside list-disc space-y-1.5 pl-4 marker:text-primary/70">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="ml-1 list-outside list-decimal space-y-1.5 pl-4 marker:text-primary/70">
              {children}
            </ol>
          ),
          li: ({ children }) => <li className="pl-0.5">{children}</li>,
          strong: ({ children }) => (
            <strong className="font-semibold text-foreground">{children}</strong>
          ),
          h1: ({ children }) => (
            <h3 className="pt-1 text-[15px] font-semibold">{children}</h3>
          ),
          h2: ({ children }) => (
            <h3 className="pt-1 text-[15px] font-semibold">{children}</h3>
          ),
          h3: ({ children }) => (
            <h4 className="pt-1 text-sm font-semibold text-foreground/90">{children}</h4>
          ),
          code: ({ children }) => (
            <code className="rounded bg-white/[0.09] px-1.5 py-0.5 font-mono text-[13px] text-primary/95">
              {children}
            </code>
          ),
          pre: ({ children }) => (
            <pre className="overflow-x-auto rounded-lg bg-black/40 p-3 font-mono text-[13px]">
              {children}
            </pre>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-primary/40 pl-3 text-foreground/85">
              {children}
            </blockquote>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto rounded-lg border border-white/10">
              <table className="w-full border-collapse text-sm">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border-b border-white/10 bg-white/[0.05] px-3 py-2 text-left text-xs font-semibold">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border-b border-white/[0.06] px-3 py-2 align-top">
              {children}
            </td>
          ),
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary underline underline-offset-2"
            >
              {children}
            </a>
          ),
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}
