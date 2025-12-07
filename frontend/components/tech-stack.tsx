"use client"

import { Brain, Zap, Cpu, Layers, Shield, Database } from "lucide-react"

export function TechStack() {
  const technologies = [
    {
      name: "LLaVA-v1.6-Mistral-7B",
      category: "Vision-Language Model",
      icon: Brain,
      description: "Advanced vision-language reasoning for event analysis",
    },
    {
      name: "YOLOv8n",
      category: "Object Detection",
      icon: Zap,
      description: "Real-time player and ball detection",
    },
    {
      name: "4-bit Quantization",
      category: "Optimization",
      icon: Cpu,
      description: "Efficient model compression for GPU",
    },
    {
      name: "Batch Processing",
      category: "Performance",
      icon: Layers,
      description: "Optimized frame processing pipeline",
    },
    {
      name: "Checkpoint Mechanism",
      category: "Fault Tolerance",
      icon: Shield,
      description: "Reliable recovery and resumption",
    },
    {
      name: "JSON Output",
      category: "Data Format",
      icon: Database,
      description: "Structured event tagging and export",
    },
  ]

  return (
    <div className="w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {technologies.map((tech, idx) => {
          const IconComponent = tech.icon
          return (
            <div
              key={idx}
              className="group relative bg-gradient-to-br from-primary/10 to-background border border-primary/20 rounded-2xl p-6 hover:border-primary/50 transition-all duration-300 hover:shadow-lg hover:shadow-primary/15 overflow-hidden"
            >
              <div className="absolute inset-0 bg-gradient-to-br from-primary/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />

              <div className="relative z-10">
                <div className="flex items-start justify-between mb-4">
                  <div className="inline-flex p-2.5 rounded-lg bg-primary/20 group-hover:bg-primary/30 transition-colors">
                    <IconComponent className="w-5 h-5 text-primary" />
                  </div>
                </div>
                <div className="text-xs font-semibold text-primary uppercase tracking-wider mb-2">{tech.category}</div>
                <h4 className="text-lg font-bold text-foreground mb-2 group-hover:text-primary transition-colors">
                  {tech.name}
                </h4>
                <p className="text-xs text-muted-foreground leading-relaxed">{tech.description}</p>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
