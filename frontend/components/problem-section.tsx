"use client"

import { AlertCircle, Clock, DollarSign } from "lucide-react"

export default function ProblemSection() {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-muted/30">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-4xl sm:text-5xl font-bold text-foreground mb-6">
            Highlights Shouldn't Require a Hollywood Budget
          </h2>
          <p className="text-lg text-muted-foreground leading-relaxed">
            Professional highlight creation is expensive and time-consuming—often requiring 3-4 hours of manual editing
            per match. For grassroots football, this means most matches go undocumented, and exciting moments are lost
            forever.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <div className="bg-background rounded-xl p-6 border border-primary/10 hover:border-primary/30 transition-colors">
            <Clock className="w-12 h-12 text-primary mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Time Consuming</h3>
            <p className="text-muted-foreground text-sm">
              Video editors spend countless hours scrubbing through footage manually.
            </p>
          </div>

          <div className="bg-background rounded-xl p-6 border border-primary/10 hover:border-primary/30 transition-colors">
            <DollarSign className="w-12 h-12 text-primary mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Expensive</h3>
            <p className="text-muted-foreground text-sm">Local clubs can't afford professional production services.</p>
          </div>

          <div className="bg-background rounded-xl p-6 border border-primary/10 hover:border-primary/30 transition-colors">
            <AlertCircle className="w-12 h-12 text-primary mb-4" />
            <h3 className="font-semibold text-foreground mb-2">Lost Moments</h3>
            <p className="text-muted-foreground text-sm">
              Without highlights, great plays and player development go unnoticed.
            </p>
          </div>
        </div>

        <div className="bg-gradient-to-r from-primary/10 to-primary/5 rounded-xl p-8 border border-primary/20">
          <h3 className="text-2xl font-bold text-foreground mb-6">The Cost Reality</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div>
              <p className="text-primary font-semibold text-lg">₹5,000-15,000</p>
              <p className="text-muted-foreground text-sm">Manual editing per match</p>
            </div>
            <div>
              <p className="text-primary font-semibold text-lg">3-4 Hours</p>
              <p className="text-muted-foreground text-sm">Skilled work required</p>
            </div>
            <div>
              <p className="text-primary font-semibold text-lg">Zero Coverage</p>
              <p className="text-muted-foreground text-sm">Most grassroots matches</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
