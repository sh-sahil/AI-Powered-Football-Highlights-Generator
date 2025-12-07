import { Card, CardContent } from "@/components/ui/card"

type Props = {
  title: string
  meta?: string
}

export function ContentCard({ title, meta }: Props) {
  return (
    <Card className="group overflow-hidden border-border/60 bg-card/40 transition-all hover:shadow-lg hover:border-primary/40">
      <div className="relative w-full overflow-hidden">
        <img
          src="/sports-match-thumbnail.jpg"
          alt="Sports video thumbnail"
          className="h-auto w-full select-none bg-muted object-cover transition-transform group-hover:scale-105"
          draggable={false}
        />
        {meta ? (
          <span className="absolute left-3 top-3 rounded-md bg-primary px-2.5 py-1 text-xs font-bold text-primary-foreground shadow-md">
            {meta.toUpperCase()}
          </span>
        ) : null}
      </div>
      <CardContent className="space-y-2 p-4">
        <p className="line-clamp-2 text-sm font-semibold leading-snug">{title}</p>
        {meta ? <p className="text-xs text-muted-foreground">Football</p> : null}
      </CardContent>
    </Card>
  )
}
