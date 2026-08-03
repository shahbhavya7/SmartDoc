"use client";

/**
 * The authenticated chrome: a floating frosted nav bar with navigation and the
 * account menu, wrapped around whatever the route renders.
 *
 * The bar is sticky and translucent so content scrolls *under* it, which is what
 * sells the glass an opaque bar would just look like a header. It floats with
 * a fixed margin on every side (rather than sitting flush against the viewport
 * edge) so it reads as a card above the page instead of a bolted-on strip.
 *
 * A `fullBleed` page (currently just /chat) gets the SAME horizontal margin
 * scale on `main` as the bar has on itself, so a page that builds its own
 * floating card below the bar lines up with it edge to edge instead of that
 * card sitting flush against the viewport under a bar that visibly doesn't.
 * The vertical margin (`mt-3`/`mb-3`/`top-3`) is kept the SAME value at every
 * breakpoint on both: `main` then resolves to one exact, breakpoint-stable
 * pixel height via flexbox (this bar's flow height plus its own margins), and
 * `chat-workspace.tsx` sizes its card as `h-full` of that rather than
 * duplicating the arithmetic in a second hardcoded calc.
 */

import Link from "next/link";
import { usePathname } from "next/navigation";
import { LayoutDashboard, LogOut, MessageSquare } from "lucide-react";

import { useAuth } from "@/components/auth-provider";
import { Brand } from "@/components/brand";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { initialsFor } from "@/lib/format";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/chat", label: "Chat", icon: MessageSquare },
];

export function AppShell({
  children,
  /** Chat manages its own scrolling regions, so it opts out of page padding. */
  fullBleed = false,
}: {
  children: React.ReactNode;
  fullBleed?: boolean;
}) {
  const { user, logout } = useAuth();
  const pathname = usePathname();

  return (
    <div
      className={cn(
        "flex flex-col",
        // `min-h-dvh` is a FLOOR: a normal page (dashboard) can still grow
        // taller and let the document scroll, which is exactly the ordinary
        // page-scroll behaviour it wants. fullBleed (chat) needs the opposite
        // -- a hard CEILING -- because it owns its own internal scroll region
        // for messages and expects everything above that region to be
        // capped at exactly the viewport, never bigger. Without this, a long
        // conversation doesn't get clipped by anything: every flex-1
        // ancestor down to this div just grows to fit the content (verified
        // in Chromium -- with enough messages this div measured 1120px
        // against a 900px viewport), so the DOCUMENT scrolls and the
        // composer drifts down with it, instead of the message list alone
        // scrolling inside a fixed-height card.
        fullBleed ? "h-dvh overflow-hidden" : "min-h-dvh",
      )}
    >
      <header className="sticky top-3 z-40 mx-3 mt-3 sm:mx-4 lg:mx-6">
        <div className="glass-chrome mx-auto flex h-16 w-full max-w-[1400px] items-center gap-4 rounded-2xl border border-white/10 px-4 shadow-[0_10px_30px_-10px_rgba(0,0,0,0.65)] sm:px-6">
          <Brand href="/dashboard" size="sm" />

          <nav className="ml-2 flex items-center gap-1">
            {NAV.map(({ href, label, icon: Icon }) => {
              const active = pathname === href || pathname.startsWith(`${href}/`);
              return (
                <Link
                  key={href}
                  href={href}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "flex items-center gap-2 rounded-lg px-3 py-1.5 text-sm font-medium transition-colors",
                    "outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    active
                      ? "bg-white/10 text-foreground"
                      : "text-muted-foreground hover:bg-white/[0.06] hover:text-foreground",
                  )}
                >
                  <Icon className="size-4" aria-hidden />
                  <span className="hidden sm:inline">{label}</span>
                </Link>
              );
            })}
          </nav>

          <div className="ml-auto flex items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger
                render={
                  <Button
                    variant="ghost"
                    className="h-10 gap-2.5 rounded-xl px-2 hover:bg-white/[0.06]"
                  >
                    <Avatar className="size-7">
                      <AvatarFallback className="bg-primary/18 text-[11px] font-semibold text-primary">
                        {initialsFor(user?.email ?? "")}
                      </AvatarFallback>
                    </Avatar>
                    <span className="hidden max-w-[180px] truncate text-sm text-muted-foreground md:inline">
                      {user?.email}
                    </span>
                  </Button>
                }
              />
              <DropdownMenuContent align="end" className="w-64">
                {/* The Group wrapper is required, not decorative: this shadcn
                    build is Base UI, whose GroupLabel throws outside a Group —
                    which silently prevents the whole menu from opening. */}
                <DropdownMenuGroup>
                  <DropdownMenuLabel className="flex flex-col gap-1">
                    <span className="truncate text-sm font-medium">{user?.email}</span>
                    <span className="text-xs font-normal text-muted-foreground">
                      Signed in with{" "}
                      {(user?.auth_methods ?? []).join(" and ") || "password"}
                    </span>
                  </DropdownMenuLabel>
                </DropdownMenuGroup>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={logout} className="gap-2">
                  <LogOut className="size-4" aria-hidden />
                  Sign out
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </header>

      <main
        className={cn(
          "flex-1",
          fullBleed
            // Same horizontal scale as the header above, so a fullBleed page's
            // own floating card lines up edge-to-edge with the nav bar instead
            // of sitting flush against the viewport under a bar that floats —
            // that mismatch is exactly what looked "fixed to the edges and
            // odd" before this.
            //
            // `flex flex-col` turns `main` into a flex container so its child
            // can fill it with `flex-1` rather than `h-full`. That is NOT a
            // style preference: a percentage height does not reliably resolve
            // against a flex ITEM's flex-computed size in this stack (verified
            // in Chromium even an inline `height:100%` on the child measured
            // short of `main`'s actual height). Flex-grow has no percentage to
            // resolve, so it isn't exposed to that failure; it's also exactly
            // how `main` itself already gets its own height from ITS flex-col
            // parent above, so this chains the same proven mechanism one level
            // deeper instead of introducing a different one.
            ? "flex min-h-0 flex-col mx-3 mt-3 mb-3 sm:mx-4 lg:mx-6"
            : "mx-auto w-full max-w-[1400px] px-4 py-8 sm:px-6 lg:py-10",
        )}
      >
        {children}
      </main>
    </div>
  );
}
