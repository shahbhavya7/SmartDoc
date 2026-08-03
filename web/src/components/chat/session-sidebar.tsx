"use client";

/**
 * The sessions sidebar: the user's last 10 chats.
 *
 * Selecting one is a plain state change in the parent — no navigation — which is
 * what makes switching instant. Hovering or focusing an entry prefetches its
 * history into the parent's cache, so by the time the click lands the messages
 * are usually already there.
 *
 * The list order comes from the server, which ranks by *last activity* rather
 * than creation time, so a chat replied to a minute ago sits above one opened an
 * hour ago and abandoned. Re-sorting here would fight that.
 */

import { Loader2, MessageSquare, Plus, Trash2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { formatRelativeTime, sessionLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { ChatSession } from "@/lib/types";

export function SessionSidebar({
  sessions,
  activeId,
  loading,
  error,
  creating,
  deletingId,
  onSelect,
  onPrefetch,
  onCreate,
  onDelete,
  onRetry,
}: {
  sessions: ChatSession[];
  activeId: string | null;
  loading: boolean;
  error: string | null;
  creating: boolean;
  deletingId: string | null;
  onSelect: (sessionId: string) => void;
  onPrefetch: (sessionId: string) => void;
  onCreate: () => void;
  onDelete: (session: ChatSession) => void;
  onRetry: () => void;
}) {
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="p-3">
        <Button
          onClick={onCreate}
          disabled={creating}
          size="lg"
          className="glow-accent h-10 w-full gap-2 font-semibold"
        >
          {creating ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <Plus className="size-4" aria-hidden />
          )}
          New chat
        </Button>
      </div>

      <p className="px-4 pb-2 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
        Recent chats
      </p>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {error ? (
          <div className="mx-1 rounded-lg border border-destructive/30 bg-destructive/10 p-3">
            <p className="text-xs text-red-200">{error}</p>
            <Button
              variant="ghost"
              size="sm"
              onClick={onRetry}
              className="mt-2 h-6 px-2 text-[11px] hover:bg-white/[0.08]"
            >
              Try again
            </Button>
          </div>
        ) : loading ? (
          <ul className="space-y-1.5 px-1">
            {[0, 1, 2, 3, 4].map((row) => (
              <li key={row} className="space-y-1.5 rounded-lg px-2 py-2.5">
                <Skeleton className="h-3.5 w-4/5 bg-white/10" />
                <Skeleton className="h-2.5 w-1/3 bg-white/[0.07]" />
              </li>
            ))}
          </ul>
        ) : sessions.length === 0 ? (
          <p className="px-3 py-6 text-center text-xs leading-relaxed text-muted-foreground">
            No chats yet.
            <br />
            Start one to ask your documents a question.
          </p>
        ) : (
          <ul className="space-y-0.5">
            {sessions.map((session) => {
              const active = session.id === activeId;
              return (
                <li key={session.id} className="group/item relative">
                  <button
                    type="button"
                    onClick={() => onSelect(session.id)}
                    onMouseEnter={() => onPrefetch(session.id)}
                    onFocus={() => onPrefetch(session.id)}
                    aria-current={active ? "true" : undefined}
                    className={cn(
                      "w-full rounded-lg border px-3 py-2.5 pr-9 text-left outline-none transition-colors",
                      "focus-visible:ring-2 focus-visible:ring-ring",
                      active
                        ? "border-primary/25 bg-primary/[0.1]"
                        : "border-transparent hover:border-white/[0.08] hover:bg-white/[0.05]",
                    )}
                  >
                    <span className="flex items-center gap-2">
                      <MessageSquare
                        className={cn(
                          "size-3.5 shrink-0",
                          active ? "text-primary" : "text-muted-foreground",
                        )}
                        aria-hidden
                      />
                      <span
                        className={cn(
                          "min-w-0 flex-1 truncate text-[13px] font-medium",
                          active ? "text-foreground" : "text-foreground/85",
                        )}
                      >
                        {sessionLabel(session.title, "Untitled chat")}
                      </span>
                    </span>
                    <span className="mt-1 flex items-center gap-1.5 pl-5 text-[11px] text-muted-foreground">
                      <span className="tabular-nums">
                        {formatRelativeTime(session.created_at)}
                      </span>
                      {session.last_document ? (
                        <>
                          <span aria-hidden>·</span>
                          <span className="min-w-0 truncate">
                            {session.last_document}
                          </span>
                        </>
                      ) : null}
                    </span>
                  </button>

                  <Button
                    variant="ghost"
                    size="icon-sm"
                    onClick={() => onDelete(session)}
                    disabled={deletingId === session.id}
                    aria-label={`Delete chat ${sessionLabel(session.title, "Untitled chat")}`}
                    className={cn(
                      "absolute right-1.5 top-2 size-7 text-muted-foreground transition-opacity",
                      "hover:bg-destructive/15 hover:text-red-300",
                      // Kept reachable by keyboard even while visually hidden.
                      "opacity-0 focus-visible:opacity-100 group-hover/item:opacity-100",
                    )}
                  >
                    {deletingId === session.id ? (
                      <Loader2 className="size-3.5 animate-spin" aria-hidden />
                    ) : (
                      <Trash2 className="size-3.5" aria-hidden />
                    )}
                  </Button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    </div>
  );
}
