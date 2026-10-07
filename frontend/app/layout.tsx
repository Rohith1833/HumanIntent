import type { Metadata } from "next";
import React from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "HumanIntent — Intent-Preserving AI Prompt Enhancement",
  description:
    "Convert informal, incomplete human requests into high-quality AI prompts while strictly preserving original intent.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
