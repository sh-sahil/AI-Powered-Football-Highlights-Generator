import Link from "next/link"

export function HeroPlaceholder() {
  return (
    <section aria-label="AI Football Highlights" className="relative w-full bg-background">
      <div className="relative mx-auto h-[70vh] w-full max-w-7xl overflow-hidden rounded-2xl px-4 md:h-[80vh] md:px-6">
        <div className="absolute inset-0 -z-10">
          <img
            src="/images/hero-football.jpg"
            alt="AI Football Highlights Generation"
            className="h-full w-full object-cover opacity-50"
            draggable={false}
          />
          <div className="absolute inset-0 bg-gradient-to-t from-background via-background/60 to-background/20" />
        </div>

        <div className="flex h-full flex-col justify-end pb-12 md:pb-16">
          <span className="mb-6 inline-flex w-fit rounded-full bg-primary/20 px-4 py-2 text-xs font-bold tracking-widest text-primary ring-1 ring-primary/30 uppercase">
            AI-Powered Highlights
          </span>
          <h1 className="max-w-4xl text-pretty text-5xl font-extrabold leading-tight md:text-6xl lg:text-7xl">
            Automatic Football Highlights in Minutes
          </h1>
          <p className="mt-6 max-w-3xl text-pretty text-lg text-muted-foreground md:text-xl leading-relaxed">
            Transform 90-minute matches into broadcast-quality 10-15 minute highlights using advanced AI and computer
            vision. Powered by vision transformers and real-time object detection.
          </p>
          <div className="mt-10 flex flex-col items-start gap-4 sm:flex-row sm:items-center">
            <Link
              href="/chat"
              className="inline-flex h-12 items-center rounded-lg bg-primary px-8 text-base font-semibold text-primary-foreground transition-all hover:opacity-90 hover:shadow-lg active:scale-95"
            >
              Try Chat Interface
            </Link>
            <Link
              href="/upload"
              className="inline-flex h-12 items-center rounded-lg border border-primary/40 bg-primary/5 px-8 text-base font-semibold text-primary transition-all hover:bg-primary/10 hover:shadow-md"
            >
              Upload Match Video
            </Link>
          </div>
        </div>
      </div>
    </section>
  )
}
