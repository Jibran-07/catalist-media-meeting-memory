import type { Metadata } from "next";
import "./globals.css";
export const metadata: Metadata = { title: "Meeting Memory", description: "Find decisions and references in meeting transcripts" };
export default function RootLayout({ children }: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body>{children}</body></html>;
}
