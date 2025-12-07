"use client"

import { Zap, MessageSquare, Video, Search } from "lucide-react"

export default function FeaturesSection() {
  const features = [
    {
      icon: Zap,
      title: "Automatic Event Detection",
      description:
        "Goals, fouls, free kicks, penalties, corners, tackles, and celebrations—detected and tagged automatically.",
    },
    {
      icon: MessageSquare,
      title: "Natural Language Querying",
      description:
        'Ask questions like "Show me all goals" or "Find the second free kick with celebrations" and get precise video segments instantly.',
    },
    {
      icon: Video,
      title: "Broadcast-Quality Output",
      description:
        "Professional highlight reels with precise timestamps and smooth transitions, ready to share or stream.",
    },
    {
      icon: Search,
      title: "Fast Video Extraction",
      description:
        "Extract specific moments in seconds—no scrubbing through hours of footage. Query what you need, get exactly that clip.",
    },
  ]

  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-muted/30">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold text-foreground mb-6">What You Get</h2>
          <p className="text-lg text-muted-foreground">Powerful features designed for grassroots football</p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {features.map((feature, index) => {
            const Icon = feature.icon
            return (
              <div
                key={index}
                className="bg-background rounded-xl p-8 border border-primary/10 hover:border-primary/30 transition-all duration-300 hover:shadow-lg hover:-translate-y-1"
              >
                <Icon className="w-12 h-12 text-primary mb-4" />
                <h3 className="text-xl font-bold text-foreground mb-3">{feature.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{feature.description}</p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
