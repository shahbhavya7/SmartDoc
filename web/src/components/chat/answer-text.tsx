"use client";

/**
 * Renders an answer's prose.
 *
 * Phase 4 asks the model to shape each answer to its content a table for a
 * comparison, a list for steps, prose for an explanation so this component is
 * the other half of that instruction: every markdown block the prompt can
 * produce maps to the same shadcn primitives the rest of the app uses, rather
 * than to bare HTML that happens to inherit some styling.
 *
 * `react-markdown` is used with no raw-HTML plugin, so model output cannot inject
 * markup the answer is untrusted text as far as this component is concerned,
 * even though it originates from our own backend. That is also why there is no
 * `rehypeRaw` here and no `dangerouslySetInnerHTML` anywhere: a document
 * containing `<img onerror=...>` gets rendered as the literal characters.
 */

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
      className={cn("space-y-3 text-[15px] leading-relaxed text-foreground/95", className)}
    >
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          // `whitespace-pre-wrap` keeps intentional line breaks inside a
          // paragraph, which matters for an answer that lays out two values on
          // separate lines without reaching for a table.
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
          // A nested list needs its own top margin, or the child items sit flush
          // against the parent item's text.
          li: ({ children }) => (
            <li className="pl-0.5 [&>ol]:mt-1.5 [&>ul]:mt-1.5">{children}</li>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-foreground">{children}</strong>
          ),
          em: ({ children }) => <em className="italic text-foreground/90">{children}</em>,
          // Answer headings are section labels inside a bubble, not page
          // headings, so h1/h2 are levelled down to keep the document outline
          // sane for a screen reader.
          h1: ({ children }) => (
            <h3 className="pt-1 text-[15px] font-semibold">{children}</h3>
          ),
          h2: ({ children }) => (
            <h3 className="pt-1 text-[15px] font-semibold">{children}</h3>
          ),
          h3: ({ children }) => (
            <h4 className="pt-1 text-sm font-semibold text-foreground/90">{children}</h4>
          ),
          h4: ({ children }) => (
            <h5 className="pt-1 text-sm font-semibold text-foreground/85">{children}</h5>
          ),
          hr: () => <hr className="my-3 border-white/10" />,
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

          // --- GFM tables -> shadcn Table ---------------------------------
          // `Table` supplies its own overflow-x container, which is what stops a
          // wide comparison table from widening the message panel.
          table: ({ children }) => (
            <div className="my-2 overflow-hidden rounded-lg border border-white/10">
              <Table>{children}</Table>
            </div>
          ),
          thead: ({ children }) => <TableHeader>{children}</TableHeader>,
          tbody: ({ children }) => <TableBody>{children}</TableBody>,
          tr: ({ children }) => <TableRow>{children}</TableRow>,
          // `style` carries remark-gfm's column alignment (`|---:|`); dropping
          // it would silently left-align a column of figures the document
          // right-aligned. Cells wrap rather than scroll: an answer cell holds a
          // phrase, not an identifier.
          th: ({ children, style }) => (
            <TableHead style={style} className="whitespace-normal">
              {children}
            </TableHead>
          ),
          td: ({ children, style }) => (
            <TableCell style={style} className="whitespace-normal">
              {children}
            </TableCell>
          ),
          caption: ({ children }) => <TableCaption>{children}</TableCaption>,

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
