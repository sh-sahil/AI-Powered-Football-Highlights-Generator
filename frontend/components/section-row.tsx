import { ContentCard } from "./content-card"

type Item = {
  id: number
  title: string
  meta?: string
}
type Props = {
  id?: string
  title: string
  items: Item[]
}

export function SectionRow({ id, title, items }: Props) {
  return (
    <section id={id} aria-label={title} className="py-6 md:py-8">
      <div className="mx-auto w-full max-w-7xl px-3 md:px-4">
        <h2 className="mb-4 text-xl font-bold tracking-tight md:mb-5 md:text-2xl">{title}</h2>
        <div className="grid auto-cols-[70%] grid-flow-col gap-3 overflow-x-auto scroll-smooth [scrollbar-width:none] md:auto-cols-[32%] md:gap-4 lg:auto-cols-[24%]">
          {items.map((item) => (
            <ContentCard key={item.id} title={item.title} meta={item.meta} />
          ))}
        </div>
      </div>
    </section>
  )
}
