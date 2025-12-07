import { Button } from "@/components/ui/button"

const filters = ["All", "Live", "Upcoming", "Highlights", "Replays", "Clips"]

export function FilterPills() {
  return (
    <div className="mx-auto w-full max-w-7xl px-3 md:px-4">
      <div className="flex flex-wrap items-center gap-2 py-3 md:py-4">
        {filters.map((f, i) => (
          <Button
            key={f}
            variant={i === 0 ? "default" : "secondary"}
            className={
              i === 0
                ? "h-8 rounded-full bg-[var(--color-brand)] px-4 text-[12px] leading-none text-[var(--color-brand-foreground)] hover:opacity-90"
                : "h-8 rounded-full border border-border/60 bg-muted/50 px-4 text-[12px] leading-none text-muted-foreground hover:bg-muted"
            }
          >
            {f}
          </Button>
        ))}
      </div>
    </div>
  )
}
