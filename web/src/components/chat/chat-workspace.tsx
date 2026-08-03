"use client";

/**
 * The chat window: sessions sidebar on the left, the active conversation on the
 * right.
 *
 * Switching sessions is instant, and that is a structural choice rather than an
 * optimisation. The active session id lives in component state, and the URL is
 * kept in sync with `history.replaceState` — NOT with `router.push`. An App Router
 * navigation would fetch an RSC payload for a route whose content is entirely
 * client-fetched anyway, adding a network hop to a switch that needs none. The URL
 * still updates, so a chat stays linkable and reload-safe.
 *
 * History is cached per session and prefetched on hover, so a switch usually
 * renders from memory with no request at all.
 *
 * The server owns memory. Each `POST /ask` with a `session_id` stores both
 * messages, resolves references from that session's running summary, and updates
 * the summary in a background task after responding. This component therefore
 * never sends conversation history — sending it would duplicate the memory the
 * server already maintains, and let the two disagree.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { FileWarning, MessageSquarePlus, PanelLeft } from "lucide-react";
import { toast } from "sonner";

import { useAuth } from "@/components/auth-provider";
import { Composer } from "@/components/chat/composer";
import {
  AnswerPanel,
  ErrorPanel,
  QuestionPanel,
  ThinkingPanel,
} from "@/components/chat/message-panels";
import { SessionSidebar } from "@/components/chat/session-sidebar";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, NetworkError, api } from "@/lib/api";
import {
  forgetAnswers,
  recallAnswers,
  rememberAnswer,
  type CachedAnswer,
} from "@/lib/answer-cache";
import { useDocuments, useSessions } from "@/lib/hooks";
import { sessionLabel } from "@/lib/format";
import type { AskResponse, ChatMessage } from "@/lib/types";
import { cn } from "@/lib/utils";

/** A turn still in flight, held outside the message list until it resolves. */
interface PendingTurn {
  question: string;
  startedAt: number;
}

function describe(error: unknown, fallback: string): string {
  if (error instanceof ApiError || error instanceof NetworkError) return error.message;
  return fallback;
}

export function ChatWorkspace() {
  const { authorizedFetch } = useAuth();
  const searchParams = useSearchParams();

  const sessions = useSessions(10);
  const documents = useDocuments();

  const [activeId, setActiveId] = useState<string | null>(null);
  const [messagesBySession, setMessagesBySession] = useState<
    Record<string, ChatMessage[]>
  >({});
  const [answerMeta, setAnswerMeta] = useState<Record<string, CachedAnswer>>({});
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [historyError, setHistoryError] = useState<string | null>(null);

  const [draft, setDraft] = useState("");
  const [pending, setPending] = useState<PendingTurn | null>(null);
  const [turnError, setTurnError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);

  const [creating, setCreating] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  /**
   * The one message allowed to play the reveal animation: the answer to the turn
   * just asked. Explicit state rather than a "seen" set, so nothing is mutated
   * during render and history can never re-animate — a reloaded conversation
   * whose answers all typed themselves back in would look broken.
   */
  const [animatingId, setAnimatingId] = useState<string | null>(null);

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inFlightRef = useRef<Set<string>>(new Set());

  const sessionList = useMemo(() => sessions.data ?? [], [sessions.data]);
  const hasDocuments = (documents.data?.documents.length ?? 0) > 0;

  /* ---------------------------------------------------------------------- */
  /* History loading                                                        */
  /* ---------------------------------------------------------------------- */

  const loadHistory = useCallback(
    async (sessionId: string, { quiet = false }: { quiet?: boolean } = {}) => {
      // De-duplicate: hover-prefetch and selection can both ask at once.
      if (inFlightRef.current.has(sessionId)) return;
      inFlightRef.current.add(sessionId);

      if (!quiet) {
        setLoadingHistory(true);
        setHistoryError(null);
      }
      try {
        const rows = await authorizedFetch((token) => api.messages(token, sessionId));
        setMessagesBySession((current) => ({ ...current, [sessionId]: rows }));

        const recalled = recallAnswers(
          rows.filter((row) => row.role === "assistant").map((row) => row.id),
        );
        if (Object.keys(recalled).length > 0) {
          setAnswerMeta((current) => ({ ...current, ...recalled }));
        }
      } catch (caught) {
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        if (!quiet) {
          setHistoryError(describe(caught, "Could not load this chat's history."));
        }
      } finally {
        inFlightRef.current.delete(sessionId);
        if (!quiet) setLoadingHistory(false);
      }
    },
    [authorizedFetch],
  );

  const prefetch = useCallback(
    (sessionId: string) => {
      if (messagesBySession[sessionId]) return;
      void loadHistory(sessionId, { quiet: true });
    },
    [messagesBySession, loadHistory],
  );

  /* ---------------------------------------------------------------------- */
  /* Choosing the active session                                            */
  /* ---------------------------------------------------------------------- */

  const selectSession = useCallback(
    (sessionId: string) => {
      if (sessionId === activeId) {
        setSidebarOpen(false);
        return;
      }
      setActiveId(sessionId);
      setTurnError(null);
      setPending(null);
      setAnimatingId(null);
      setDraft("");
      setSidebarOpen(false);

      // The URL follows the selection without a route change, so the switch
      // costs nothing and the address bar still names the open chat.
      window.history.replaceState(
        null,
        "",
        `/chat?session=${encodeURIComponent(sessionId)}`,
      );

      if (!messagesBySession[sessionId]) void loadHistory(sessionId);
    },
    [activeId, messagesBySession, loadHistory],
  );

  // Resolve the initial session once the list is known: the `?session=` param if
  // it is genuinely one of the user's own, otherwise the most recent chat. A
  // param naming someone else's session simply is not in this list, so it falls
  // through to the default rather than issuing a request that would 404.
  const resolvedInitial = useRef(false);
  useEffect(() => {
    if (resolvedInitial.current || sessions.loading) return;

    const requested = searchParams.get("session");
    const wanted = requested && sessionList.some((row) => row.id === requested)
      ? requested
      : (sessionList[0]?.id ?? null);

    resolvedInitial.current = true;
    if (!wanted) return;

    // The choice depends on the session list, which arrives from an
    // authenticated fetch after mount, so there is no render-time value to
    // derive it from.
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see above
    setActiveId(wanted);
    if (requested !== wanted) {
      window.history.replaceState(null, "", `/chat?session=${encodeURIComponent(wanted)}`);
    }
    void loadHistory(wanted);
  }, [sessions.loading, sessionList, searchParams, loadHistory]);

  /* ---------------------------------------------------------------------- */
  /* Creating and deleting sessions                                         */
  /* ---------------------------------------------------------------------- */

  const createSession = useCallback(async () => {
    setCreating(true);
    try {
      const session = await authorizedFetch((token) => api.createSession(token));
      // Seed the cache so the new chat renders its empty state immediately
      // instead of showing a history skeleton for a chat with no history.
      setMessagesBySession((current) => ({ ...current, [session.id]: [] }));
      await sessions.refresh(true);
      setActiveId(session.id);
      setPending(null);
      setTurnError(null);
      setDraft("");
      setSidebarOpen(false);
      window.history.replaceState(
        null,
        "",
        `/chat?session=${encodeURIComponent(session.id)}`,
      );
    } catch (caught) {
      if (caught instanceof ApiError && caught.isUnauthenticated) return;
      toast.error("Could not start a new chat", {
        description: describe(caught, "Something went wrong."),
      });
    } finally {
      setCreating(false);
    }
  }, [authorizedFetch, sessions]);

  const deleteSession = useCallback(
    async (sessionId: string) => {
      setDeletingId(sessionId);
      try {
        await authorizedFetch((token) => api.deleteSession(token, sessionId));

        const removedMessages = messagesBySession[sessionId] ?? [];
        forgetAnswers(removedMessages.map((row) => row.id));

        setMessagesBySession((current) => {
          const next = { ...current };
          delete next[sessionId];
          return next;
        });

        const remaining = sessionList.filter((row) => row.id !== sessionId);
        await sessions.refresh(true);

        if (sessionId === activeId) {
          const nextId = remaining[0]?.id ?? null;
          setActiveId(nextId);
          setPending(null);
          setTurnError(null);
          window.history.replaceState(
            null,
            "",
            nextId ? `/chat?session=${encodeURIComponent(nextId)}` : "/chat",
          );
          if (nextId && !messagesBySession[nextId]) void loadHistory(nextId);
        }
        toast.success("Chat deleted");
      } catch (caught) {
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        toast.error("Could not delete that chat", {
          description: describe(caught, "Something went wrong."),
        });
      } finally {
        setDeletingId(null);
      }
    },
    [authorizedFetch, messagesBySession, sessionList, sessions, activeId, loadHistory],
  );

  /* ---------------------------------------------------------------------- */
  /* Asking                                                                 */
  /* ---------------------------------------------------------------------- */

  /**
   * Files a completed turn's citations against the server's own message id.
   *
   * `POST /ask` returns the answer but not the ids of the two rows it stored, and
   * the citation cache is keyed by the assistant message's real id so it survives
   * a reload. So the message list is re-read once per turn — a cheap indexed
   * SQLite query — purely to learn that id.
   *
   * It deliberately does NOT swap the optimistic rows for the fetched ones. Their
   * text is identical, and replacing them would change the rendered answer's
   * React key mid-animation, remounting it and restarting the reveal from an
   * empty panel.
   */
  const recordTurnCitations = useCallback(
    async (sessionId: string, response: AskResponse) => {
      try {
        const rows = await authorizedFetch((token) => api.messages(token, sessionId));
        const lastAssistant = [...rows].reverse().find((row) => row.role === "assistant");
        if (!lastAssistant) return;

        setAnswerMeta((current) => ({
          ...current,
          [lastAssistant.id]: {
            sources: response.sources,
            grounding: response.grounding,
            query_type: response.query_type,
          },
        }));
        rememberAnswer(lastAssistant.id, response);
      } catch {
        // The answer is already on screen from the optimistic rows; failing to
        // re-read only costs durable citations after a reload.
      }
    },
    [authorizedFetch],
  );

  const ask = useCallback(
    async (question: string, sessionId: string) => {
      const trimmed = question.trim();
      if (!trimmed) return;

      setTurnError(null);
      setPending({ question: trimmed, startedAt: Date.now() });
      setElapsed(0);
      setDraft("");

      try {
        const response = await authorizedFetch((token) =>
          api.ask(token, trimmed, sessionId),
        );

        // Optimistic rows so the answer appears the moment it arrives. Temporary
        // ids are prefixed to guarantee they cannot collide with a server uuid.
        const optimisticAnswerId = `local-a-${Date.now()}`;
        const now = new Date().toISOString();
        setMessagesBySession((current) => ({
          ...current,
          [sessionId]: [
            ...(current[sessionId] ?? []),
            {
              id: `local-q-${Date.now()}`,
              session_id: sessionId,
              role: "user",
              content: trimmed,
              created_at: now,
            },
            {
              id: optimisticAnswerId,
              session_id: sessionId,
              role: "assistant",
              content: response.answer,
              created_at: now,
            },
          ],
        }));
        setAnswerMeta((current) => ({
          ...current,
          [optimisticAnswerId]: {
            sources: response.sources,
            grounding: response.grounding,
            query_type: response.query_type,
          },
        }));
        // This — and only this — answer gets the progressive reveal.
        setAnimatingId(optimisticAnswerId);
        setPending(null);

        // Refresh the sidebar: this session is now the most recently active, and
        // `last_document` has just been set from the answer's first citation.
        void sessions.refresh(true);
        void recordTurnCitations(sessionId, response);
      } catch (caught) {
        setPending(null);
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        setTurnError(describe(caught, "The question could not be answered."));
        // Put the text back so a failed question is not lost.
        setDraft(trimmed);
      }
    },
    [authorizedFetch, sessions, recordTurnCitations],
  );

  /**
   * Sending with no session open creates one first, so the composer works on a
   * cold start rather than being disabled until the user finds "New chat".
   */
  const submit = useCallback(async () => {
    const question = draft.trim();
    if (!question) return;

    if (activeId) {
      await ask(question, activeId);
      return;
    }

    setCreating(true);
    try {
      const session = await authorizedFetch((token) =>
        // Seed the title from the question so the sidebar entry is meaningful
        // straight away; the server stores whatever title it is given.
        api.createSession(token, question.slice(0, 120)),
      );
      setMessagesBySession((current) => ({ ...current, [session.id]: [] }));
      setActiveId(session.id);
      window.history.replaceState(
        null,
        "",
        `/chat?session=${encodeURIComponent(session.id)}`,
      );
      await sessions.refresh(true);
      setCreating(false);
      await ask(question, session.id);
    } catch (caught) {
      setCreating(false);
      if (caught instanceof ApiError && caught.isUnauthenticated) return;
      toast.error("Could not start a chat for that question", {
        description: describe(caught, "Something went wrong."),
      });
    }
  }, [draft, activeId, ask, authorizedFetch, sessions]);

  /* ---------------------------------------------------------------------- */
  /* Effects: elapsed timer, autoscroll                                     */
  /* ---------------------------------------------------------------------- */

  useEffect(() => {
    if (!pending) return;
    const timer = window.setInterval(() => {
      setElapsed(Math.floor((Date.now() - pending.startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [pending]);

  const activeMessages = activeId ? messagesBySession[activeId] : undefined;

  // Pin to the bottom as the conversation grows. Keyed on the message count and
  // the pending turn rather than on every render, so it does not fight a user
  // who has scrolled up to re-read an earlier answer.
  useEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    node.scrollTo({ top: node.scrollHeight, behavior: "smooth" });
  }, [activeMessages?.length, pending, activeId]);

  /* ---------------------------------------------------------------------- */
  /* Render                                                                 */
  /* ---------------------------------------------------------------------- */

  const activeSession = sessionList.find((row) => row.id === activeId) ?? null;
  const showEmptyChat =
    !!activeId && !loadingHistory && (activeMessages?.length ?? 0) === 0 && !pending;

  return (
    // Fills the viewport below the 4rem top bar; the two columns scroll
    // independently so neither the sidebar nor the composer moves with the
    // conversation.
    <div className="flex h-[calc(100dvh-4rem)] min-h-0">
      {/* Sidebar — a fixed rail from lg up, an overlay drawer below it. */}
      <aside
        className={cn(
          "glass-chrome z-30 w-[17.5rem] shrink-0 border-r border-white/[0.07]",
          "max-lg:fixed max-lg:inset-y-16 max-lg:left-0 max-lg:transition-transform",
          sidebarOpen ? "max-lg:translate-x-0" : "max-lg:-translate-x-full",
          "lg:block",
        )}
        aria-label="Chat sessions"
      >
        <SessionSidebar
          sessions={sessionList}
          activeId={activeId}
          loading={sessions.loading}
          error={sessions.error}
          creating={creating}
          deletingId={deletingId}
          onSelect={selectSession}
          onPrefetch={prefetch}
          onCreate={() => void createSession()}
          onDelete={(session) => void deleteSession(session.id)}
          onRetry={() => void sessions.refresh()}
        />
      </aside>

      {sidebarOpen ? (
        <button
          type="button"
          aria-label="Close chat list"
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 top-16 z-20 bg-black/50 lg:hidden"
        />
      ) : null}

      {/* Conversation column */}
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        <div className="flex items-center gap-3 border-b border-white/[0.06] px-4 py-3 lg:px-6">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(true)}
            aria-label="Open chat list"
            className="size-9 lg:hidden"
          >
            <PanelLeft className="size-4" aria-hidden />
          </Button>

          <div className="min-w-0 flex-1">
            <h1 className="truncate text-sm font-semibold">
              {activeSession
                ? sessionLabel(activeSession.title, "Untitled chat")
                : "New chat"}
            </h1>
            <p className="mt-0.5 truncate text-xs text-muted-foreground">
              {activeSession?.last_document
                ? `Following up on ${activeSession.last_document}`
                : "Answers are grounded in your documents only"}
            </p>
          </div>
        </div>

        <div ref={scrollRef} className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-4xl space-y-5 px-4 py-6 lg:px-6">
            {historyError ? (
              <ErrorPanel
                message={historyError}
                onRetry={() => activeId && void loadHistory(activeId)}
              />
            ) : loadingHistory ? (
              <div className="space-y-5">
                {[0, 1].map((row) => (
                  <div key={row} className="space-y-3">
                    <Skeleton className="ml-auto h-12 w-2/5 rounded-2xl bg-white/[0.07]" />
                    <Skeleton className="h-28 w-4/5 rounded-2xl bg-white/[0.05]" />
                  </div>
                ))}
              </div>
            ) : showEmptyChat ? (
              <EmptyChat
                hasDocuments={hasDocuments}
                documentsLoading={documents.loading}
                onPick={(question) => setDraft(question)}
              />
            ) : !activeId ? (
              <NoSessionYet onCreate={() => void createSession()} creating={creating} />
            ) : null}

            {(activeMessages ?? []).map((message) => {
              if (message.role === "user") {
                return <QuestionPanel key={message.id} content={message.content} />;
              }
              if (message.role !== "assistant") return null;

              const meta = answerMeta[message.id];
              return (
                <AnswerPanel
                  key={message.id}
                  content={message.content}
                  sources={meta?.sources ?? []}
                  grounding={meta?.grounding ?? null}
                  queryType={meta?.query_type}
                  animate={message.id === animatingId}
                />
              );
            })}

            {pending ? (
              <>
                <QuestionPanel content={pending.question} />
                <ThinkingPanel elapsedSeconds={elapsed} />
              </>
            ) : null}

            {turnError ? (
              <ErrorPanel
                message={turnError}
                onRetry={() => {
                  setTurnError(null);
                  void submit();
                }}
              />
            ) : null}
          </div>
        </div>

        <div className="border-t border-white/[0.06] px-4 py-3 lg:px-6">
          <div className="mx-auto w-full max-w-4xl">
            {!documents.loading && !hasDocuments ? (
              <p className="mb-2 flex items-center gap-2 text-xs text-amber-200/85">
                <FileWarning className="size-3.5 shrink-0" aria-hidden />
                You have no documents indexed yet — upload a PDF or every question
                will be refused.
              </p>
            ) : null}
            <Composer
              value={draft}
              onChange={setDraft}
              onSubmit={() => void submit()}
              onUploaded={() => void documents.refresh(true)}
              busy={!!pending || creating}
            />
          </div>
        </div>
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/* Empty states                                                              */
/* -------------------------------------------------------------------------- */

const STARTERS = [
  "What are the annual leave entitlements by band?",
  "List every requirement in the onboarding checklist.",
  "How do the employee and contractor policies differ on leave?",
];

function EmptyChat({
  hasDocuments,
  documentsLoading,
  onPick,
}: {
  hasDocuments: boolean;
  documentsLoading: boolean;
  onPick: (question: string) => void;
}) {
  return (
    <div className="animate-rise py-10 text-center">
      <span
        className="glass-accent mx-auto mb-5 grid size-14 place-items-center rounded-2xl text-primary"
        aria-hidden
      >
        <MessageSquarePlus className="size-6" />
      </span>
      <h2 className="text-lg font-semibold">Ask your documents anything</h2>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-muted-foreground">
        Every answer comes with the document, page, and passage it was drawn from.
        If your documents do not cover it, SmartDoc says so instead of guessing.
      </p>

      {!documentsLoading && hasDocuments ? (
        <ul className="mx-auto mt-6 flex max-w-lg flex-col gap-2">
          {STARTERS.map((question) => (
            <li key={question}>
              <button
                type="button"
                onClick={() => onPick(question)}
                className="glass w-full rounded-xl px-4 py-2.5 text-left text-[13px] text-foreground/85 outline-none transition-colors hover:bg-white/[0.09] focus-visible:ring-2 focus-visible:ring-ring"
              >
                {question}
              </button>
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function NoSessionYet({
  onCreate,
  creating,
}: {
  onCreate: () => void;
  creating: boolean;
}) {
  return (
    <div className="animate-rise py-14 text-center">
      <span
        className="glass-accent mx-auto mb-5 grid size-14 place-items-center rounded-2xl text-primary"
        aria-hidden
      >
        <MessageSquarePlus className="size-6" />
      </span>
      <h2 className="text-lg font-semibold">Start a chat</h2>
      <p className="mx-auto mt-2 max-w-md text-sm text-muted-foreground">
        Each chat keeps its own memory, so follow-up questions like
        &ldquo;what about the executive band?&rdquo; resolve against what you just
        asked.
      </p>
      <Button
        onClick={onCreate}
        disabled={creating}
        size="lg"
        className="glow-accent mt-6 h-10 gap-2 font-semibold"
      >
        <MessageSquarePlus className="size-4" aria-hidden />
        New chat
      </Button>
      <p className="mt-3 text-xs text-muted-foreground">
        Or just type your question below.
      </p>
    </div>
  );
}
