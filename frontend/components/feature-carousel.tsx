"use client"

import { useState, useEffect } from "react"
import { ChevronLeft, ChevronRight, Zap, Search, MessageSquare, Film, Brain, Cpu } from "lucide-react"
import { Button } from "@/components/ui/button"

const features = [
  {
    title: "Automatic Event Detection",
    description: "AI-powered detection of goals, fouls, tackles, and celebrations",
    icon: Zap,
  },
  {
    title: "Player-Specific Search",
    description: "Query and find specific player performances and moments",
    icon: Search,
  },
  {
    title: "Intelligent Chatbot",
    description: "Ask questions about highlights, player appearances, and match summaries",
    icon: MessageSquare,
  },
  {
    title: "Broadcast Quality",
    description: "Condense 90-minute matches into 10-15 minute highlight reels",
    icon: Film,
  },
  {
    title: "Vision Transformers",
    description: "LLaVA-v1.6-Mistral-7B for advanced vision-language reasoning",
    icon: Brain,
  },
  {
    title: "Real-time Processing",
    description: "Optimized with 4-bit quantization for efficient GPU usage",
    icon: Cpu,
  },
]

export function FeatureCarousel() {
  const [current, setCurrent] = useState(0)
  const [autoScroll, setAutoScroll] = useState(true)

  useEffect(() => {
    if (!autoScroll) return

    const interval = setInterval(() => {
      setCurrent((prev) => (prev + 1) % features.length)
    }, 5000)

    return () => clearInterval(interval)
  }, [autoScroll])

  const next = () => {
    setCurrent((current + 1) % features.length)
    setAutoScroll(false)
  }

  const prev = () => {
    setCurrent((current - 1 + features.length) % features.length)
    setAutoScroll(false)
  }

  const getVisibleFeatures = () => {
    const visible = []
    for (let i = 0; i < 3; i++) {
      visible.push(features[(current + i) % features.length])
    }
    return visible
  }

  return (
    <div className="w-full">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        {getVisibleFeatures().map((feature, idx) => {
          const IconComponent = feature.icon
          return (
            <div
              key={idx}
              className="group relative bg-gradient-to-br from-primary/15 via-primary/5 to-background border border-primary/30 rounded-2xl p-8 hover:border-primary/60 transition-all duration-300 hover:shadow-xl hover:shadow-primary/20 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

              <div className="relative z-10">
                <div className="inline-flex p-3 rounded-xl bg-primary/20 group-hover:bg-primary/30 transition-colors mb-4">
                  <IconComponent className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-xl font-bold text-foreground mb-3 group-hover:text-primary transition-colors">
                  {feature.title}
                </h3>
                <p className="text-muted-foreground text-sm leading-relaxed">{feature.description}</p>
              </div>
            </div>
          )
        })}
      </div>

      <div className="flex justify-center items-center gap-6">
        <Button
          variant="outline"
          size="icon"
          onClick={prev}
          onMouseEnter={() => setAutoScroll(false)}
          onMouseLeave={() => setAutoScroll(true)}
          className="rounded-full bg-primary/10 border-primary/30 hover:bg-primary/20 hover:border-primary/50"
        >
          <ChevronLeft className="w-5 h-5" />
        </Button>

        <div className="flex items-center gap-2">
          {features.map((_, idx) => (
            <button
              key={idx}
              onClick={() => {
                setCurrent(idx)
                setAutoScroll(false)
              }}
              className={`transition-all duration-300 rounded-full ${
                idx === current ? "bg-primary w-8 h-2" : "bg-muted-foreground/30 w-2 h-2 hover:bg-muted-foreground/50"
              }`}
            />
          ))}
        </div>

        <Button
          variant="outline"
          size="icon"
          onClick={next}
          onMouseEnter={() => setAutoScroll(false)}
          onMouseLeave={() => setAutoScroll(true)}
          className="rounded-full bg-primary/10 border-primary/30 hover:bg-primary/20 hover:border-primary/50"
        >
          <ChevronRight className="w-5 h-5" />
        </Button>
      </div>
    </div>
  )
}
