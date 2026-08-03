"use client";

/**
 * PDF upload. Used by both the dashboard panel and the chat composer.
 *
 * The API accepts a batch and reports **per file**, because one bad PDF does not
 * fail the rest. So this reports per file too: a batch where two of three
 * succeeded says exactly that, instead of a single "upload failed" that hides
 * the two that worked.
 *
 * The client-side checks (extension, size) exist to save a pointless round trip
 * on an obviously wrong file. They are not the enforcement the server
 * sanitizes the filename, caps the byte count as it reads, and rejects
 * non-PDF content types regardless of what this component let through.
 */

import { useCallback, useRef, useState } from "react";
import { Loader2, Upload } from "lucide-react";
import { toast } from "sonner";

import { useAuth } from "@/components/auth-provider";
import { Button } from "@/components/ui/button";
import { ApiError, NetworkError, api } from "@/lib/api";
import { cn } from "@/lib/utils";
import type { UploadFileResult } from "@/lib/types";

/** Mirrors MAX_UPLOAD_MB in backend/config.py. */
const MAX_UPLOAD_MB = 20;

export interface UploadOutcome {
  succeeded: UploadFileResult[];
  failed: UploadFileResult[];
}

interface UploadControlProps {
  /** Called after a batch in which at least one file indexed, to refresh lists. */
  onUploaded: (outcome: UploadOutcome) => void;
  className?: string;
  /** "panel" is the dashboard drop zone; "compact" is the chat composer icon. */
  variant?: "panel" | "compact";
  disabled?: boolean;
}

function reportOutcome(outcome: UploadOutcome) {
  const { succeeded, failed } = outcome;

  for (const file of failed) {
    toast.error(`Could not index ${file.filename}`, {
      description: file.error ?? "The server did not say why.",
      duration: 8000,
    });
  }

  if (succeeded.length === 1) {
    const [file] = succeeded;
    toast.success(`Indexed ${file.filename}`, {
      description: `${file.pages_parsed ?? 0} pages · ${file.chunks_indexed ?? 0} chunks searchable`,
    });
  } else if (succeeded.length > 1) {
    const chunks = succeeded.reduce((sum, file) => sum + (file.chunks_indexed ?? 0), 0);
    toast.success(`Indexed ${succeeded.length} documents`, {
      description: `${chunks} chunks searchable`,
    });
  }
}

export function UploadControl({
  onUploaded,
  className,
  variant = "panel",
  disabled = false,
}: UploadControlProps) {
  const { authorizedFetch } = useAuth();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  // Drag events fire for every child element; a depth counter is what stops the
  // highlight flickering as the pointer crosses the zone's inner nodes.
  const dragDepth = useRef(0);

  const send = useCallback(
    async (files: File[]) => {
      const pdfs: File[] = [];
      for (const file of files) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
          toast.error(`${file.name} is not a PDF`, {
            description: "SmartDoc indexes PDF documents only.",
          });
          continue;
        }
        if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
          toast.error(`${file.name} is too large`, {
            description: `The limit is ${MAX_UPLOAD_MB} MB per file.`,
          });
          continue;
        }
        pdfs.push(file);
      }
      if (pdfs.length === 0) return;

      setUploading(true);
      try {
        const response = await authorizedFetch((token) => api.upload(token, pdfs));
        const outcome: UploadOutcome = {
          succeeded: response.files.filter((file) => file.status === "success"),
          failed: response.files.filter((file) => file.status !== "success"),
        };
        reportOutcome(outcome);
        onUploaded(outcome);
      } catch (caught) {
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        toast.error("Upload failed", {
          description:
            caught instanceof ApiError || caught instanceof NetworkError
              ? caught.message
              : "Something went wrong while uploading.",
          duration: 8000,
        });
      } finally {
        setUploading(false);
      }
    },
    [authorizedFetch, onUploaded],
  );

  function onPick(event: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    // Reset first, so picking the same file twice in a row still fires onChange.
    event.target.value = "";
    void send(files);
  }

  const busy = uploading || disabled;

  const input = (
    <input
      ref={inputRef}
      type="file"
      accept="application/pdf,.pdf"
      multiple
      onChange={onPick}
      className="sr-only"
      tabIndex={-1}
      aria-hidden
    />
  );

  if (variant === "compact") {
    return (
      <>
        {input}
        <Button
          type="button"
          variant="ghost"
          size="icon"
          disabled={busy}
          onClick={() => inputRef.current?.click()}
          aria-label="Upload a PDF"
          className={cn("size-9 text-muted-foreground hover:text-foreground", className)}
        >
          {uploading ? (
            <Loader2 className="size-4 animate-spin" aria-hidden />
          ) : (
            <Upload className="size-4" aria-hidden />
          )}
        </Button>
      </>
    );
  }

  return (
    <div
      onDragEnter={(event) => {
        event.preventDefault();
        dragDepth.current += 1;
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        event.preventDefault();
        dragDepth.current = Math.max(0, dragDepth.current - 1);
        if (dragDepth.current === 0) setDragging(false);
      }}
      onDrop={(event) => {
        event.preventDefault();
        dragDepth.current = 0;
        setDragging(false);
        if (!busy) void send(Array.from(event.dataTransfer.files ?? []));
      }}
      className={cn(
        "hatch rounded-xl border border-dashed p-6 text-center transition-colors",
        dragging
          ? "border-primary/55 bg-primary/[0.07]"
          : "border-white/15 hover:border-white/25",
        className,
      )}
    >
      {input}
      <Upload
        className={cn(
          "mx-auto mb-3 size-6 transition-colors",
          dragging ? "text-primary" : "text-muted-foreground",
        )}
        aria-hidden
      />
      <p className="text-sm font-medium">
        {dragging ? "Drop to upload" : "Drop PDFs here"}
      </p>
      <p className="mt-1 text-xs text-muted-foreground">
        Up to {MAX_UPLOAD_MB} MB each. Re-uploading a filename replaces your copy.
      </p>
      <Button
        type="button"
        variant="outline"
        size="lg"
        disabled={busy}
        onClick={() => inputRef.current?.click()}
        className="mt-4 gap-2 border-white/15 bg-white/[0.06] hover:bg-white/[0.11]"
      >
        {uploading ? (
          <>
            <Loader2 className="size-4 animate-spin" aria-hidden />
            Indexing…
          </>
        ) : (
          "Choose files"
        )}
      </Button>
    </div>
  );
}
