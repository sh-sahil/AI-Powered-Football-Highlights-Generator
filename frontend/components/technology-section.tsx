"use client"

import { Cpu, Layers } from "lucide-react"

export default function TechnologySection() {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-background">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold text-foreground mb-6">Built With Cutting-Edge AI</h2>
          <p className="text-lg text-muted-foreground">State-of-the-art models optimized for grassroots sports</p>
        </div>

        <div className="grid md:grid-cols-2 gap-12">
          <div className="bg-gradient-to-br from-primary/10 to-primary/5 rounded-xl p-10 border border-primary/20">
            <Cpu className="w-14 h-14 text-primary mb-6" />
            <h3 className="text-2xl font-bold text-foreground mb-6">Computer Vision</h3>
            <ul className="space-y-4">
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                <span className="text-muted-foreground">
                  <strong className="text-foreground">Video Swin Transformer</strong> for temporal action recognition
                </span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                <span className="text-muted-foreground">
                  <strong className="text-foreground">Multi-frame analysis</strong> for accurate event classification
                </span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                <span className="text-muted-foreground">
                  <strong className="text-foreground">Optimized</strong> for resource-constrained environments
                </span>
              </li>
            </ul>
          </div>

          <div className="bg-gradient-to-br from-primary/10 to-primary/5 rounded-xl p-10 border border-primary/20">
            <Layers className="w-14 h-14 text-primary mb-6" />
            <h3 className="text-2xl font-bold text-foreground mb-6">Vision-Language Models</h3>
            <ul className="space-y-4">
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                <span className="text-muted-foreground">
                  <strong className="text-foreground">LLaVA multimodal AI</strong> for contextual understanding
                </span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                <span className="text-muted-foreground">
                  <strong className="text-foreground">Natural language querying</strong> and summarization
                </span>
              </li>
              <li className="flex items-start gap-3">
                <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                <span className="text-muted-foreground">
                  <strong className="text-foreground">Frame-level scene</strong> comprehension
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
    </section>
  )
}
