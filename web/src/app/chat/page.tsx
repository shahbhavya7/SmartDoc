import { Suspense } from "react";

import { AppLoading } from "@/components/app-loading";
import { AppShell } from "@/components/app-shell";
import { ChatWorkspace } from "@/components/chat/chat-workspace";
import { RequireAuth } from "@/components/require-auth";

export const metadata = { title: "Chat SmartDoc" };

export default function ChatPage() {
  return (
    <RequireAuth>
      {/* fullBleed: the chat manages its own scroll regions, so the page must
          not add padding or a second scroll container around them. */}
      <AppShell fullBleed>
        <Suspense fallback={<AppLoading label="Opening your chats…" />}>
          <ChatWorkspace />
        </Suspense>
      </AppShell>
    </RequireAuth>
  );
}
