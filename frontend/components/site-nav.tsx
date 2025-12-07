import Link from "next/link"

const navItems = [
  { label: "Home", href: "/" },
  { label: "Chat", href: "/chat" },
  { label: "Matches", href: "/matches" },
  { label: "Upload Match", href: "/upload" },
]

export function SiteNav() {
  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/50 bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-20 max-w-7xl items-center gap-8 px-4 md:h-24 md:px-6">
        <Link href="/" className="flex items-center gap-3 flex-shrink-0">
          <img src="/images/kheliq-logo-white.png" alt="KhelIQ" className="h-10 w-auto" draggable={false} />
          <span className="sr-only">KhelIQ</span>
        </Link>

        <nav className="ml-2 flex items-center gap-8">
          {navItems.map((item) => (
            <Link
              key={item.label}
              href={item.href}
              className="text-sm font-semibold text-muted-foreground transition-colors hover:text-primary"
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  )
}
