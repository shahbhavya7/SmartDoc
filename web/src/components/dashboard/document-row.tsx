"use client";

/**
 * One document, with its delete control.
 *
 * Deletion is confirmed rather than one-click. It is not undoable and it is not
 * cheap: the API removes the document's vectors and parent records, then its row
 * and the stored PDF, so re-adding it means re-uploading and re-embedding. The
 * dialog names the file and says what goes with it, so "Delete" is never a
 * surprise.
 */

import { useState } from "react";
import { FileText, Loader2, Trash2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { displayFilename, formatBytes, formatDate } from "@/lib/format";
import type { DocumentRecord } from "@/lib/types";

export function DocumentRow({
  document,
  deleting,
  onDelete,
}: {
  document: DocumentRecord;
  deleting: boolean;
  onDelete: (document: DocumentRecord) => Promise<void>;
}) {
  const [open, setOpen] = useState(false);

  async function confirm() {
    // The dialog closes first so the row's own pending state is what the user
    // watches, rather than a modal hanging over a list that is already changing.
    setOpen(false);
    await onDelete(document);
  }

  return (
    <li className="group flex items-center gap-3 rounded-xl border border-transparent px-3 py-3 transition-colors hover:border-white/[0.08] hover:bg-white/[0.04]">
      <span
        className="grid size-9 shrink-0 place-items-center rounded-lg bg-primary/12 text-primary"
        aria-hidden
      >
        <FileText className="size-4" />
      </span>

      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium" title={document.filename}>
          {displayFilename(document.filename)}
        </p>
        <p className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted-foreground">
          <span>{formatBytes(document.size_bytes)}</span>
          <span aria-hidden>·</span>
          <span>
            {document.chunks ?? 0} {document.chunks === 1 ? "chunk" : "chunks"}
          </span>
          <span aria-hidden>·</span>
          <span>Added {formatDate(document.created_at)}</span>
        </p>
      </div>

      <Badge
        variant="outline"
        className="hidden shrink-0 border-white/12 bg-white/[0.05] text-[10px] font-semibold tracking-wider text-muted-foreground sm:inline-flex"
      >
        PDF
      </Badge>

      <Dialog open={open} onOpenChange={setOpen}>
        <Tooltip>
          <TooltipTrigger
            render={
              <DialogTrigger
                render={
                  <Button
                    variant="ghost"
                    size="icon"
                    disabled={deleting}
                    aria-label={`Delete ${document.filename}`}
                    className="size-9 shrink-0 text-muted-foreground hover:bg-destructive/15 hover:text-red-300"
                  >
                    {deleting ? (
                      <Loader2 className="size-4 animate-spin" aria-hidden />
                    ) : (
                      <Trash2 className="size-4" aria-hidden />
                    )}
                  </Button>
                }
              />
            }
          />
          <TooltipContent>Delete document</TooltipContent>
        </Tooltip>

        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete this document?</DialogTitle>
            <DialogDescription className="space-y-3 pt-1">
              <span className="block">
                <span className="font-medium text-foreground">{document.filename}</span>{" "}
                and its {document.chunks ?? 0} indexed{" "}
                {document.chunks === 1 ? "chunk" : "chunks"} will be removed. Answers
                will no longer cite it.
              </span>
              <span className="block text-xs">
                This cannot be undone restoring it means uploading the file again.
                Past chats keep their text, but their citations to this document will
                no longer resolve.
              </span>
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <DialogClose
              render={
                <Button variant="ghost" className="hover:bg-white/[0.07]">
                  Cancel
                </Button>
              }
            />
            <Button
              variant="destructive"
              onClick={confirm}
              className="gap-2 font-semibold"
            >
              <Trash2 className="size-4" aria-hidden />
              Delete document
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </li>
  );
}
