import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";

import { AuthProvider } from "@/components/auth-provider";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";

import "./globals.css";

const geistSans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "SmartDoc — Ask your documents",
  description:
    "Ask plain-English questions and get cited answers from your company's PDFs.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    // `dark` is fixed rather than toggleable: the design system is glass on
    // black, and a light variant would need a second, differently-tuned set of
    // glass tints to stay legible.
    <html
      lang="en"
      className={`dark ${geistSans.variable} ${geistMono.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <body className="min-h-full">
        <AuthProvider>
          <TooltipProvider delay={250}>{children}</TooltipProvider>
          <Toaster position="bottom-right" />
        </AuthProvider>
      </body>
    </html>
  );
}
