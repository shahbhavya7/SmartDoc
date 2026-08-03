"use client";

/**
 * Data hooks over the API client.
 *
 * Every fetch here runs through `authorizedFetch`, so the token is attached and
 * a 401 signs the user out globally rather than surfacing as a broken panel.
 * None of these hooks takes a `user_id`: the server derives the owner from the
 * token, which is why "the UI must not leak across accounts" needs no filtering
 * on this side — there is nothing to filter, only what the server returned.
 *
 * In-flight requests are aborted on unmount and superseded by newer ones, so a
 * slow response cannot land after a faster one and overwrite it.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { useAuth } from "@/components/auth-provider";
import { ApiError, NetworkError, api } from "@/lib/api";
import type { ChatSession, DocumentList } from "@/lib/types";

function messageFor(error: unknown, fallback: string): string {
  if (error instanceof ApiError || error instanceof NetworkError) return error.message;
  return fallback;
}

interface Resource<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
  /** Re-fetch. `quiet` skips the loading state, for background refreshes. */
  refresh: (quiet?: boolean) => Promise<void>;
  /** Optimistic local write, for delete before the list is re-fetched. */
  set: (next: T | null) => void;
}

function useResource<T>(
  fetcher: (token: string, signal: AbortSignal) => Promise<T>,
  fallbackMessage: string,
): Resource<T> {
  const { authorizedFetch, user } = useAuth();
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const controllerRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      controllerRef.current?.abort();
    };
  }, []);

  const refresh = useCallback(
    async (quiet = false) => {
      // Supersede any request still in flight; without this a stale response
      // can resolve last and overwrite fresher data.
      controllerRef.current?.abort();
      const controller = new AbortController();
      controllerRef.current = controller;

      if (!quiet) setLoading(true);
      setError(null);
      try {
        const result = await authorizedFetch((token) =>
          fetcher(token, controller.signal),
        );
        if (!controller.signal.aborted && mountedRef.current) setData(result);
      } catch (caught) {
        if (controller.signal.aborted || !mountedRef.current) return;
        // A 401 has already triggered a global sign-out; showing an error for it
        // would flash under the redirect.
        if (caught instanceof ApiError && caught.isUnauthenticated) return;
        setError(messageFor(caught, fallbackMessage));
      } finally {
        if (!controller.signal.aborted && mountedRef.current) setLoading(false);
      }
    },
    // `fetcher` must be referentially stable or this callback is rebuilt every
    // render and the effect below re-fires forever. Callers guarantee that by
    // passing a module-level function or a useCallback-wrapped one.
    [authorizedFetch, fallbackMessage, fetcher],
  );

  // Keyed on the user id so switching accounts in the same tab refetches rather
  // than showing the previous account's list.
  //
  // `refresh` flips `loading`, which trips `set-state-in-effect`. Fetching on
  // mount is nonetheless the "subscribe to an external system" case: the data
  // lives behind an authenticated HTTP call that cannot run during render or on
  // the server, because the token is in browser storage.
  useEffect(() => {
    if (!user) return;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- see above
    void refresh();
  }, [user, refresh]);

  return { data, loading, error, refresh, set: setData };
}

// Declared at module scope so the reference never changes between renders.
const fetchDocuments = (token: string, signal: AbortSignal) =>
  api.documents(token, signal);

export function useDocuments(): Resource<DocumentList> {
  return useResource<DocumentList>(fetchDocuments, "Could not load your documents.");
}

export function useSessions(limit = 10): Resource<ChatSession[]> {
  const fetcher = useCallback(
    (token: string, signal: AbortSignal) => api.sessions(token, limit, signal),
    [limit],
  );
  return useResource<ChatSession[]>(fetcher, "Could not load your chats.");
}
