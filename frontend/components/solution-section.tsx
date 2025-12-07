"use client"

import { Brain, Eye, Zap } from "lucide-react"

export default function SolutionSection() {
  const solutions = [
    {
      icon: Brain,
      title: "Deep Learning Detection",
      description:
        "Video Swin Transformer model identifies key moments—goals, fouls, tackles, penalties, and celebrations—with broadcast-level accuracy.",
    },
    {
      icon: Eye,
      title: "Vision-Language Analysis",
      description:
        "LLaVA-v1.6-Mistral-7B analyzes each frame to understand context, player positions, and game flow beyond simple object detection.",
    },
    {
      icon: Zap,
      title: "Intelligent Compilation",
      description:
        "Smart filtering eliminates replays and redundant clips, delivering only the most meaningful moments in chronological order.",
    },
  ]

  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-background">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold text-foreground mb-6">AI That Understands Football</h2>
          <p className="text-lg text-muted-foreground">
            How our system transforms raw footage into professional highlights
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {solutions.map((solution, index) => {
            const Icon = solution.icon
            return (
              <div
                key={index}
                className="group relative bg-gradient-to-br from-primary/5 to-primary/10 rounded-xl p-8 border border-primary/20 hover:border-primary/40 transition-all duration-300 hover:shadow-lg"
              >
                <div className="absolute inset-0 bg-gradient-to-br from-primary/0 to-primary/5 rounded-xl opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <div className="relative z-10">
                  <Icon className="w-14 h-14 text-primary mb-6 group-hover:scale-110 transition-transform duration-300" />
                  <h3 className="text-xl font-bold text-foreground mb-4">{solution.title}</h3>
                  <p className="text-muted-foreground leading-relaxed">{solution.description}</p>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
