"use client";

/**
 * The dashboard: what the signed-in user has, and the fastest route to using it.
 *
 * Everything on this page comes from two authenticated calls `GET /documents`
 * and `GET /sessions`. Both are scoped server-side to the token's user, so the
 * counts are per-user by construction; there is no client-side filtering step
 * that could be got wrong.
 *
 * Deletion updates the list optimistically and then re-fetches. The optimistic
 * step is what makes the row disappear instantly; the re-fetch is what makes the
 * *counts* right, since the chunk total is a live Chroma read the client cannot
 * recompute. On failure the removed row is restored, so the UI never claims a
 * document is gone when the server still has it.
 */

import { useCallback, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  ArrowRight,
  Blocks,
  FileText,
  HardDrive,
  Loader2,
  MessageSquare,
  Plus,
  TriangleAlert,
} from "lucide-react";
import { toast } from "sonner";

import { useAuth } from "@/components/auth-provider";
import { DocumentRow } from "@/components/dashboard/document-row";
import { StatCard } from "@/components/dashboard/stat-card";
import { UploadControl } from "@/components/upload-button";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { ApiError, NetworkError, api } from "@/lib/api";
import { formatBytes, formatRelativeTime, sessionLabel } from "@/lib/format";
import { useDocuments, useSessions } from "@/lib/hooks";
import type { DocumentRecord } from "@/lib/types";

export function Dashboard() {
  const { user, authorizedFetch } = useAuth();
  const router = useRouter();
  const documents = useDocuments();
  const sessions = useSessions(10);

  const [deleting, setDeleting] = useState<string | null>(null);
  const [creatingChat, setCreatingChat] = useState(false);

  const docs = documents.data?.documents ?? [];
  const unknownSizes = documents.data?.documents_with_unknown_size ?? 0;

  const onDelete = useCallback(
    async (target: DocumentRecord) => {
      const snapshot = documents.data;
      setDeleting(target.id);
      // Remove locally first; the re-fetch below is what corrects the totals.
      if (snapshot) {
        documents.set({
          ...snapshot,
          documents: snapshot.documents.filter((doc) => doc.id !== target.id),
        });
      }
      try {
        const result = await authorizedFetch((token) =>
          api.deleteDocument(token, target.id),
        );
        toast.success(`Deleted ${result.filename}`, {
          description: `${result.chunks_deleted} indexed ${
            result.chunks_deleted === 1 ? "chunk" : "chunks"
          } removed.`,
        });
        await documents.refresh(true);
      } catch (caught) {
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        // Put it back: it is still on the server, and showing it as gone would
        // be a lie the user only discovers when an answer still cites it.
        if (snapshot) documents.set(snapshot);
        toast.error("Could not delete that document", {
          description:
            caught instanceof ApiError || caught instanceof NetworkError
              ? caught.message
              : "Something went wrong.",
          duration: 8000,
        });
      } finally {
        setDeleting(null);
      }
    },
    [authorizedFetch, documents],
  );

  const startChat = useCallback(async () => {
    setCreatingChat(true);
    try {
      const session = await authorizedFetch((token) => api.createSession(token));
      router.push(`/chat?session=${encodeURIComponent(session.id)}`);
    } catch (caught) {
      if (caught instanceof ApiError && caught.isUnauthenticated) return;
      toast.error("Could not start a new chat", {
        description:
          caught instanceof ApiError || caught instanceof NetworkError
            ? caught.message
            : "Something went wrong.",
      });
      setCreatingChat(false);
    }
  }, [authorizedFetch, router]);

  return (
    <div className="space-y-7">
      {/* ---------------------------------------------------------------- */}
      {/* Header + primary action                                          */}
      {/* ---------------------------------------------------------------- */}
      <section className="flex flex-col gap-5 sm:flex-row sm:items-end sm:justify-between">
        <div className="animate-rise space-y-1.5">
          <p className="text-sm text-muted-foreground">
            {user?.email ? `Signed in as ${user.email}` : "Your workspace"}
          </p>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
            Your document workspace
          </h1>
          <p className="max-w-xl text-sm text-muted-foreground">
            Ask a question and SmartDoc answers from your PDFs only with the
            document, page, and passage it used.
          </p>
        </div>

        <Button
          size="lg"
          onClick={startChat}
          disabled={creatingChat}
          className="glow-accent h-11 shrink-0 gap-2 px-5 text-[15px] font-semibold"
        >
          {creatingChat ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <Plus className="size-4" aria-hidden />
          )}
          New chat
        </Button>
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Stats                                                            */}
      {/* ---------------------------------------------------------------- */}
      <section
        aria-label="Workspace summary"
        className="grid grid-cols-2 gap-4 lg:grid-cols-4"
      >
        <StatCard
          label="Documents"
          value={docs.length}
          hint={docs.length === 0 ? "Upload a PDF to begin" : "Indexed and searchable"}
          icon={FileText}
          loading={documents.loading}
          accent
        />
        <StatCard
          label="Storage used"
          value={formatBytes(documents.data?.total_bytes ?? 0)}
          hint={
            unknownSizes > 0
              ? `At least ${unknownSizes} of unknown size`
              : `Across ${docs.length} ${docs.length === 1 ? "file" : "files"}`
          }
          icon={HardDrive}
          loading={documents.loading}
        />
        <StatCard
          label="Indexed chunks"
          value={documents.data?.total_chunks ?? 0}
          hint="Passages available to retrieval"
          icon={Blocks}
          loading={documents.loading}
        />
        <StatCard
          label="Chats"
          value={sessions.data?.length ?? 0}
          hint="Most recent 10"
          icon={MessageSquare}
          loading={sessions.loading}
        />
      </section>

      {/* ---------------------------------------------------------------- */}
      {/* Documents + recent sessions                                      */}
      {/* ---------------------------------------------------------------- */}
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <section className="glass rounded-2xl p-5 sm:p-6">
          <header className="mb-4 flex items-baseline justify-between gap-3">
            <div>
              <h2 className="text-base font-semibold">Documents</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Only yours. Deleting one removes its passages from every future
                answer.
              </p>
            </div>
            {docs.length > 0 ? (
              <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
                {docs.length} total
              </span>
            ) : null}
          </header>

          {documents.error ? (
            <ErrorPanel
              message={documents.error}
              onRetry={() => void documents.refresh()}
            />
          ) : documents.loading ? (
            <ul className="space-y-2">
              {[0, 1, 2].map((row) => (
                <li key={row} className="flex items-center gap-3 px-3 py-3">
                  <Skeleton className="size-9 shrink-0 rounded-lg bg-white/10" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-3.5 w-1/2 bg-white/10" />
                    <Skeleton className="h-3 w-2/3 bg-white/[0.07]" />
                  </div>
                </li>
              ))}
            </ul>
          ) : docs.length === 0 ? (
            <div className="space-y-4">
              <p className="text-sm text-muted-foreground">
                No documents yet. SmartDoc can only answer from what you upload —
                with nothing indexed, every question returns{" "}
                <em>&ldquo;I don&rsquo;t know based on the available
                documents.&rdquo;</em>
              </p>
              <UploadControl onUploaded={() => void documents.refresh(true)} />
            </div>
          ) : (
            <>
              <ul className="-mx-1 space-y-0.5">
                {docs.map((doc) => (
                  <DocumentRow
                    key={doc.id}
                    document={doc}
                    deleting={deleting === doc.id}
                    onDelete={onDelete}
                  />
                ))}
              </ul>
              <UploadControl
                onUploaded={() => void documents.refresh(true)}
                className="mt-5"
              />
            </>
          )}
        </section>

        <section className="glass rounded-2xl p-5 sm:p-6">
          <header className="mb-4 flex items-baseline justify-between gap-3">
            <div>
              <h2 className="text-base font-semibold">Recent chats</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">
                Most recently active first.
              </p>
            </div>
            <Link
              href="/chat"
              className="shrink-0 text-xs font-medium text-primary underline-offset-4 hover:underline"
            >
              Open chat
            </Link>
          </header>

          {sessions.error ? (
            <ErrorPanel message={sessions.error} onRetry={() => void sessions.refresh()} />
          ) : sessions.loading ? (
            <ul className="space-y-2">
              {[0, 1, 2, 3].map((row) => (
                <li key={row} className="space-y-2 px-1 py-2.5">
                  <Skeleton className="h-3.5 w-3/4 bg-white/10" />
                  <Skeleton className="h-3 w-1/3 bg-white/[0.07]" />
                </li>
              ))}
            </ul>
          ) : (sessions.data ?? []).length === 0 ? (
            <div className="hatch rounded-xl border border-dashed border-white/12 p-6 text-center">
              <MessageSquare
                className="mx-auto mb-3 size-6 text-muted-foreground"
                aria-hidden
              />
              <p className="text-sm font-medium">No chats yet</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Each chat keeps its own memory of what you discussed.
              </p>
            </div>
          ) : (
            <ul className="-mx-1 space-y-0.5">
              {(sessions.data ?? []).map((session) => (
                <li key={session.id}>
                  <Link
                    href={`/chat?session=${encodeURIComponent(session.id)}`}
                    className="group flex items-center gap-3 rounded-xl border border-transparent px-3 py-2.5 outline-none transition-colors hover:border-white/[0.08] hover:bg-white/[0.05] focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium">
                        {sessionLabel(session.title, "Untitled chat")}
                      </span>
                      <span className="mt-0.5 block truncate text-xs text-muted-foreground">
                        {session.last_document
                          ? `Last discussed ${session.last_document}`
                          : "No answer yet"}
                      </span>
                    </span>
                    <span className="shrink-0 text-[11px] tabular-nums text-muted-foreground">
                      {formatRelativeTime(session.created_at)}
                    </span>
                    <ArrowRight
                      className="size-4 shrink-0 text-muted-foreground opacity-0 transition-opacity group-hover:opacity-100"
                      aria-hidden
                    />
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  );
}

function ErrorPanel({
  message,
  onRetry,
}: {
  message: string;
  onRetry: () => void;
}) {
  return (
    <div
      role="alert"
      className="rounded-xl border border-destructive/30 bg-destructive/10 p-4"
    >
      <div className="flex items-start gap-3">
        <TriangleAlert className="mt-0.5 size-4 shrink-0 text-destructive" aria-hidden />
        <div className="min-w-0 flex-1">
          <p className="text-sm font-medium text-red-200">{message}</p>
          <Button
            variant="ghost"
            size="sm"
            onClick={onRetry}
            className="mt-2 h-7 px-2 text-xs hover:bg-white/[0.08]"
          >
            Try again
          </Button>
        </div>
      </div>
    </div>
  );
}
