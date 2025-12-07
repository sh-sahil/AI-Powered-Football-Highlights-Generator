import Link from "next/link"

export function SiteFooter() {
  return (
    <footer className="mt-16 border-t border-border/60 bg-card/50">
      <div className="mx-auto max-w-7xl px-4 py-12 md:px-6 md:py-16">
        <div className="grid gap-8 md:grid-cols-3 md:gap-12">
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <img src="/images/kheliq-logo-white.png" alt="KhelIQ" className="h-8 w-auto" draggable={false} />
              <span className="text-sm font-medium text-foreground">KhelIQ</span>
            </div>
            <p className="text-sm text-muted-foreground">
              Digitizing grassroots football with AI-powered highlights and live streaming.
            </p>
          </div>

          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-semibold text-foreground">Quick Links</h3>
            <nav className="flex flex-col gap-2 text-sm text-muted-foreground">
              <Link href="/" className="hover:text-foreground transition-colors">
                Football
              </Link>
              <Link href="/chat" className="hover:text-foreground transition-colors">
                Chat
              </Link>
              <Link href="/upload" className="hover:text-foreground transition-colors">
                Upload Match
              </Link>
            </nav>
          </div>

          <div className="flex flex-col gap-3">
            <h3 className="text-sm font-semibold text-foreground">Legal</h3>
            <nav className="flex flex-col gap-2 text-sm text-muted-foreground">
              <Link href="#" className="hover:text-foreground transition-colors">
                Privacy Policy
              </Link>
              <Link href="#" className="hover:text-foreground transition-colors">
                Terms of Use
              </Link>
              <Link href="#" className="hover:text-foreground transition-colors">
                Careers
              </Link>
            </nav>
          </div>
        </div>

        <div className="mt-8 border-t border-border/40 pt-8 text-xs text-muted-foreground">
          <p>©2025 KhelIQ. All rights reserved. Built for grassroots football.</p>
        </div>
      </div>
    </footer>
  )
}
