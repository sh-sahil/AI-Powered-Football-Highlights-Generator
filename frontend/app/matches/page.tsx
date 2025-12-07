"use client"

import { useState, useEffect } from "react"
import Link from "next/link"
import { SiteNav } from "@/components/site-nav"
import { SiteFooter } from "@/components/footer"
import { Play, Calendar, Loader2, AlertCircle, CheckCircle, Clock } from "lucide-react"

interface Match {
  id: string
  match_id: string
  title: string
  date: string
  description: string
  poster_path: string | null
  status: "uploaded" | "processing" | "completed" | "failed"
  created_at: string
}

export default function MatchesPage() {
  const [matches, setMatches] = useState<Match[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedStatus, setSelectedStatus] = useState<string>("all")

  useEffect(() => {
    fetchMatches()
  }, [selectedStatus])

  const fetchMatches = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const url = selectedStatus === "all" 
        ? "http://localhost:9000/api/matches"
        : `http://localhost:9000/api/matches?status=${selectedStatus}`
      
      const response = await fetch(url)
      const data = await response.json()

      if (data.success) {
        setMatches(data.matches)
      } else {
        setError(data.error || "Failed to load matches")
      }
    } catch (err) {
      setError("Failed to connect to server")
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const getStatusBadge = (status: string) => {
    const badges = {
      uploaded: { icon: Clock, text: "Uploaded", class: "bg-blue-500/10 text-blue-500 border-blue-500/20" },
      processing: { icon: Loader2, text: "Processing", class: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20" },
      completed: { icon: CheckCircle, text: "Completed", class: "bg-green-500/10 text-green-500 border-green-500/20" },
      failed: { icon: AlertCircle, text: "Failed", class: "bg-red-500/10 text-red-500 border-red-500/20" },
    }

    const badge = badges[status as keyof typeof badges] || badges.uploaded
    const Icon = badge.icon

    return (
      <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${badge.class}`}>
        <Icon className={`w-3.5 h-3.5 ${status === "processing" ? "animate-spin" : ""}`} />
        {badge.text}
      </span>
    )
  }

  const getPosterUrl = (match: Match) => {
    if (match.poster_path) {
      return `http://localhost:9000/api/media/${match.match_id}/${match.poster_path.split('/').pop()}`
    }
    return "/placeholder.svg"
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <SiteNav />
      
      <section className="border-b border-border/60 bg-gradient-to-b from-secondary/5 to-background">
        <div className="mx-auto max-w-7xl px-4 py-4 md:px-6 md:py-6">
          <p className="text-sm text-muted-foreground">
            Browse all uploaded matches and their AI-generated highlights
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-8 md:px-6 md:py-12">
        {/* Filter Tabs */}
        <div className="flex gap-2 mb-8 border-b border-border/60 pb-4">
          {["all", "uploaded", "processing", "completed", "failed"].map((status) => (
            <button
              key={status}
              onClick={() => setSelectedStatus(status)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                selectedStatus === status
                  ? "bg-primary text-primary-foreground"
                  : "bg-secondary/50 text-muted-foreground hover:bg-secondary"
              }`}
            >
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </button>
          ))}
        </div>

        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center py-20">
            <Loader2 className="w-12 h-12 animate-spin text-primary mb-4" />
            <p className="text-muted-foreground">Loading matches...</p>
          </div>
        )}

        {/* Error State */}
        {error && !isLoading && (
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6 text-center">
            <AlertCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
            <p className="text-destructive font-semibold mb-2">Failed to load matches</p>
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <button
              onClick={fetchMatches}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90"
            >
              Retry
            </button>
          </div>
        )}

        {/* Empty State */}
        {!isLoading && !error && matches.length === 0 && (
          <div className="text-center py-20">
            <div className="w-20 h-20 rounded-full bg-secondary/50 flex items-center justify-center mx-auto mb-4">
              <Play className="w-10 h-10 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2"></h3>
            <h3 className="text-xl font-semibold mb-2">No matches found</h3>
            <p className="text-muted-foreground mb-6">
              {selectedStatus === "all" 
                ? "Upload your first match to get started" 
                : `No matches with status: ${selectedStatus}`}
            </p>
            <Link
              href="/upload"
              className="inline-flex items-center px-6 py-3 bg-primary text-primary-foreground rounded-lg font-medium hover:opacity-90 transition-all"
            >
              Upload Match
            </Link>
          </div>
        )}

        {/* Matches Grid */}
        {!isLoading && !error && matches.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {matches.map((match) => (
              <Link
                key={match.id}
                href={`/matches/${match.match_id}`}
                className="group block bg-card border border-border/60 rounded-xl overflow-hidden hover:border-primary/50 hover:shadow-xl transition-all duration-300"
              >
                {/* Thumbnail */}
                <div className="relative aspect-video bg-secondary/20 overflow-hidden">
                  <img
                    src={getPosterUrl(match)}
                    alt={match.title}
                    className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 via-transparent to-transparent" />
                  
                  {/* Play Button Overlay */}
                  <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                    <div className="w-16 h-16 rounded-full bg-primary/90 flex items-center justify-center">
                      <Play className="w-8 h-8 text-primary-foreground ml-1" fill="currentColor" />
                    </div>
                  </div>

                  {/* Status Badge */}
                  <div className="absolute top-3 right-3">
                    {getStatusBadge(match.status)}
                  </div>
                </div>

                {/* Match Info */}
                <div className="p-4">
                  <h3 className="font-semibold text-lg mb-2 line-clamp-2 group-hover:text-primary transition-colors">
                    {match.title}
                  </h3>
                  
                  <div className="flex items-center gap-2 text-sm text-muted-foreground mb-3">
                    <Calendar className="w-4 h-4" />
                    <span>{new Date(match.date).toLocaleDateString()}</span>
                  </div>

                  {match.description && (
                    <p className="text-sm text-muted-foreground line-clamp-2 mb-3">
                      {match.description}
                    </p>
                  )}

                  <div className="text-xs text-muted-foreground">
                    Uploaded {new Date(match.created_at).toLocaleDateString()}
                  </div>
                </div>
              </Link>
            ))}
          </div>
        )}
      </section>

      <SiteFooter />
    </main>
  )
}