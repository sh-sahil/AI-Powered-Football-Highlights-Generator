"use client"

import { Users, Building2, Zap } from "lucide-react"

export default function BenefitsSection() {
  const benefits = [
    {
      icon: Users,
      title: "Video Editors & Content Creators",
      description:
        'Skip the manual scrubbing. Query "all penalties" or "goals with celebrations" and extract precise clips in seconds—not hours.',
    },
    {
      icon: Building2,
      title: "Local Clubs & Academies",
      description:
        "Highlights were once a luxury reserved for matches with big budgets. Now, every match can have professional-quality coverage at a fraction of the cost.",
    },
    {
      icon: Zap,
      title: "KhelIQ Platform",
      description:
        "Power livestreams with instant highlights, automated match summaries, and searchable video archives—enhancing the entire grassroots sports ecosystem.",
    },
  ]

  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-muted/30">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold text-foreground mb-6">
            Making Professional Coverage Accessible
          </h2>
          <p className="text-lg text-muted-foreground">Who benefits from AI-powered highlights</p>
        </div>

        <div className="grid md:grid-cols-3 gap-8">
          {benefits.map((benefit, index) => {
            const Icon = benefit.icon
            return (
              <div
                key={index}
                className="group bg-background rounded-xl p-8 border border-primary/10 hover:border-primary/30 transition-all duration-300 hover:shadow-lg hover:-translate-y-2"
              >
                <Icon className="w-14 h-14 text-primary mb-6 group-hover:scale-110 transition-transform duration-300" />
                <h3 className="text-xl font-bold text-foreground mb-4">{benefit.title}</h3>
                <p className="text-muted-foreground leading-relaxed">{benefit.description}</p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
