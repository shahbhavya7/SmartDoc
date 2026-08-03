"use client";

/**
 * The authenticated chrome: a frosted top bar with navigation and the account
 * menu, wrapped around whatever the route renders.
 *
 * The bar is sticky and translucent so content scrolls *under* it, which is what
 * sells the glass — an opaque bar would just look like a header.
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
    <div className="flex min-h-dvh flex-col">
      <header className="glass-chrome sticky top-0 z-40 border-b border-white/[0.07]">
        <div className="mx-auto flex h-16 w-full max-w-[1400px] items-center gap-4 px-4 sm:px-6">
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
            ? "min-h-0"
            : "mx-auto w-full max-w-[1400px] px-4 py-8 sm:px-6 lg:py-10",
        )}
      >
        {children}
      </main>
    </div>
  );
}
