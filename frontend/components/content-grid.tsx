import { SectionRow } from "./section-row"

const events = Array.from({ length: 10 }).map((_, i) => ({
  id: i + 1,
  title: `Tournament Event ${i + 1}`,
  meta: "Event",
}))
const replays = Array.from({ length: 10 }).map((_, i) => ({
  id: i + 1,
  title: `Replay • Match ${i + 1}`,
  meta: "Replay",
}))
const spotlights = Array.from({ length: 8 }).map((_, i) => ({
  id: i + 1,
  title: `KhelIQ Spotlight ${i + 1}`,
  meta: "Spotlight",
}))

export function ContentGrid() {
  return (
    <div className="pb-12">
      <SectionRow id="events" title="Events" items={events} />
      <SectionRow title="Replays" items={replays} />
      <SectionRow title="KhelIQ Spotlight" items={spotlights} />
    </div>
  )
}
