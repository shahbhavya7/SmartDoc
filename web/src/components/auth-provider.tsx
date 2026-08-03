"use client";

/**
 * Holds the signed-in identity for the whole client tree.
 *
 * Two things matter here.
 *
 * **The stored token is verified against the server before the app trusts it.**
 * On mount, a stored token is presented to `GET /auth/me`; only its reply
 * populates `user`. A token whose account was deleted, or which was hand-edited
 * in devtools, therefore never yields a signed-in UI the backend re-reads the
 * user row on every request, so `/auth/me` is the authoritative check.
 *
 * **A 401 from anywhere ends the session once.** `authorizedFetch` funnels every
 * call through here, so an expired token clears state and redirects rather than
 * leaving individual panels to each render their own error.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useRouter } from "next/navigation";

import { ApiError, api } from "@/lib/api";
import {
  AUTH_STORAGE_KEYS,
  clearSession,
  loadSession,
  saveBareToken,
  saveSession,
  saveUser,
} from "@/lib/auth-store";
import type { User } from "@/lib/types";

interface AuthContextValue {
  user: User | null;
  token: string | null;
  /** True until the stored token has been checked against the server. */
  loading: boolean;
  login: (email: string, password: string) => Promise<void>;
  signup: (email: string, password: string) => Promise<void>;
  /** Adopts a token handed back by the Google OAuth redirect. */
  adoptToken: (token: string) => Promise<void>;
  logout: () => void;
  /**
   * Runs an API call with the current token, converting a 401 into a single
   * global sign-out. Callers get the resolved value or the original error.
   */
  authorizedFetch: <T>(call: (token: string) => Promise<T>) => Promise<T>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Read inside callbacks that must not be re-created when the token changes,
  // so `authorizedFetch` stays referentially stable for effect dependencies.
  // Written from an effect rather than during render; every code path that sets
  // `token` also assigns the ref synchronously, so a call issued in the same tick
  // as a sign-in still sees the new token rather than waiting for this to commit.
  const tokenRef = useRef<string | null>(null);
  useEffect(() => {
    tokenRef.current = token;
  }, [token]);

  const endSession = useCallback(
    (redirect: boolean) => {
      clearSession();
      tokenRef.current = null;
      setToken(null);
      setUser(null);
      if (redirect) router.replace("/login");
    },
    [router],
  );

  // Validate whatever is in storage, exactly once per mount.
  //
  // The state writes below are flagged by `set-state-in-effect` and are
  // nonetheless correct: the stored token lives in localStorage, which is
  // unreadable during render and absent on the server. Deriving it with a lazy
  // `useState` initialiser instead would make the first client render disagree
  // with the prerendered HTML for any signed-in user a hydration mismatch
  // rather than a fix.
  useEffect(() => {
    let cancelled = false;
    const stored = loadSession();

    if (!stored) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- see above
      setLoading(false);
      return;
    }

    // Show the cached user immediately so the shell does not flash empty, but
    // treat it as provisional until /auth/me confirms it.
    if (stored.user) setUser(stored.user);
    setToken(stored.token);
    tokenRef.current = stored.token;

    api
      .me(stored.token)
      .then((fresh) => {
        if (cancelled) return;
        setUser(fresh);
        saveUser(fresh);
      })
      .catch((error) => {
        if (cancelled) return;
        // Only an explicit 401 means the token is bad. A network failure means
        // the backend is down, and signing the user out for that would lose
        // their session over a restart of the API.
        if (error instanceof ApiError && error.isUnauthenticated) {
          endSession(false);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [endSession]);

  // Keep tabs consistent: signing out in one tab must not leave another showing
  // an authenticated view built on a token that is no longer there.
  useEffect(() => {
    function onStorage(event: StorageEvent) {
      if (event.key && !AUTH_STORAGE_KEYS.includes(event.key)) return;
      const stored = loadSession();
      if (!stored) {
        tokenRef.current = null;
        setToken(null);
        setUser(null);
      } else if (stored.token !== tokenRef.current) {
        tokenRef.current = stored.token;
        setToken(stored.token);
        setUser(stored.user);
      }
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const adopt = useCallback(
    (accessToken: string, expiresIn: number, nextUser: User) => {
      saveSession(accessToken, expiresIn, nextUser);
      tokenRef.current = accessToken;
      setToken(accessToken);
      setUser(nextUser);
    },
    [],
  );

  const login = useCallback(
    async (email: string, password: string) => {
      const result = await api.login(email, password);
      adopt(result.access_token, result.expires_in, result.user);
    },
    [adopt],
  );

  const signup = useCallback(
    async (email: string, password: string) => {
      const result = await api.signup(email, password);
      adopt(result.access_token, result.expires_in, result.user);
    },
    [adopt],
  );

  const adoptToken = useCallback(async (accessToken: string) => {
    // The redirect carries no user body and no expiry, so both are resolved
    // from the server rather than guessed. A bad token fails here, before the
    // app ever renders as signed in.
    saveBareToken(accessToken);
    const fresh = await api.me(accessToken);
    saveUser(fresh);
    tokenRef.current = accessToken;
    setToken(accessToken);
    setUser(fresh);
  }, []);

  const logout = useCallback(() => endSession(true), [endSession]);

  const authorizedFetch = useCallback(
    async <T,>(call: (activeToken: string) => Promise<T>): Promise<T> => {
      const active = tokenRef.current;
      if (!active) throw new ApiError(401, "unauthenticated", "You are not signed in.");
      try {
        return await call(active);
      } catch (error) {
        if (error instanceof ApiError && error.isUnauthenticated) endSession(true);
        throw error;
      }
    },
    [endSession],
  );

  const value = useMemo<AuthContextValue>(
    () => ({ user, token, loading, login, signup, adoptToken, logout, authorizedFetch }),
    [user, token, loading, login, signup, adoptToken, logout, authorizedFetch],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside <AuthProvider>.");
  return context;
}
