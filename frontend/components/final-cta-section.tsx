"use client"

import Link from "next/link"
import { ArrowRight, MessageSquare, Upload } from "lucide-react"

export default function FinalCTASection() {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-gradient-to-br from-primary/10 via-background to-primary/5">
      <div className="max-w-4xl mx-auto text-center">
        <h2 className="text-4xl sm:text-5xl font-bold text-foreground mb-6">Ready to Transform Your Match Footage?</h2>
        <p className="text-xl text-muted-foreground mb-12">Start generating highlights in minutes, not hours.</p>

        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
          <Link
            href="/chat"
            className="inline-flex items-center justify-center px-8 py-4 bg-primary text-primary-foreground rounded-lg font-semibold hover:bg-primary/90 transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-xl"
          >
            <MessageSquare className="w-5 h-5 mr-2" />
            Chat with AI
            <ArrowRight className="w-5 h-5 ml-2" />
          </Link>
          <Link
            href="/upload"
            className="inline-flex items-center justify-center px-8 py-4 bg-secondary text-secondary-foreground rounded-lg font-semibold hover:bg-secondary/90 transition-all duration-300 border border-primary/20"
          >
            <Upload className="w-5 h-5 mr-2" />
            Upload Match Video
          </Link>
        </div>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-8 text-sm text-muted-foreground">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full" />
            Powered by KhelIQ
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full" />
            Broadcast-quality output
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full" />
            Automated & affordable
          </div>
        </div>
      </div>
    </section>
  )
}
