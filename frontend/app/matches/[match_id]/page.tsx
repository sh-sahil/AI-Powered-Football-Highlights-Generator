"use client"

import { useState, useEffect, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import { SiteNav } from "@/components/site-nav"
import { SiteFooter } from "@/components/footer"
import { ConfirmDialog } from "@/components/confirm-dialog"
import { 
  Play, Calendar, Loader2, AlertCircle, CheckCircle, 
  ArrowLeft, Download, Trash2, RefreshCw, Video 
} from "lucide-react"

interface EventClip {
  [key: string]: string[]
}

interface Match {
  id: string
  match_id: string
  title: string
  date: string
  description: string
  video_path: string
  poster_path: string | null
  status: string
  main_highlights: string | null
  event_clips: EventClip
  analysis_data: any
  created_at: string
  updated_at: string
}

interface ProgressUpdate {
  status: string
  progress: number
  message: string
  timestamp: string
}

export default function MatchDetailsPage() {
  const params = useParams()
  const router = useRouter()
  const matchId = params.match_id as string

  const [match, setMatch] = useState<Match | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [selectedClipType, setSelectedClipType] = useState<string>("main")
  const [currentVideoUrl, setCurrentVideoUrl] = useState<string>("")
  const [progressData, setProgressData] = useState<ProgressUpdate | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const [dialog, setDialog] = useState<{
    isOpen: boolean
    type: "success" | "error" | "warning" | "info" | "confirm"
    title: string
    message: string
    onConfirm?: () => void
  }>({
    isOpen: false,
    type: "info",
    title: "",
    message: ""
  })

  useEffect(() => {
    fetchMatchDetails()
    return () => {
      // Cleanup WebSocket on unmount
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [matchId])

  const connectWebSocket = () => {
    // Close existing connection
    if (wsRef.current) {
      wsRef.current.close()
    }

    // Create new WebSocket connection
    const ws = new WebSocket(`ws://localhost:9000/ws/progress/${matchId}`)
    
    ws.onopen = () => {
      console.log("WebSocket connected")
    }
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as ProgressUpdate
      setProgressData(data)
      
      // If analysis is complete or failed, refresh match details
      if (data.status === "completed" || data.status === "failed") {
        setTimeout(() => {
          fetchMatchDetails()
          setIsAnalyzing(false)
          setProgressData(null)
          if (ws) ws.close()
        }, 2000)
      }
    }
    
    ws.onerror = (error) => {
      console.error("WebSocket error:", error)
    }
    
    ws.onclose = () => {
      console.log("WebSocket disconnected")
    }
    
    wsRef.current = ws
  }

  const fetchMatchDetails = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const response = await fetch(`http://localhost:9000/api/matches/${matchId}`)
      const data = await response.json()

      if (data.success) {
        setMatch(data.match)
        // Set initial video URL
        if (data.match.main_highlights) {
          const highlightsPath = data.match.main_highlights.split(/[/\\]/).slice(-2).join('/')
          setCurrentVideoUrl(`http://localhost:9000/api/media/${matchId}/${highlightsPath}`)
        }
      } else {
        setError(data.error || "Failed to load match details")
      }
    } catch (err) {
      setError("Failed to connect to server")
      console.error(err)
    } finally {
      setIsLoading(false)
    }
  }

  const handleAnalyze = async () => {
    if (!match || match.status === "processing") return

    setIsAnalyzing(true)
    setProgressData({ status: "starting", progress: 0, message: "Initializing analysis...", timestamp: new Date().toISOString() })
    
    // Connect to WebSocket for real-time updates
    connectWebSocket()

    try {
      const response = await fetch(`http://localhost:9000/api/matches/${matchId}/analyze`, {
        method: "POST",
      })
      const data = await response.json()

      if (data.success) {
        setDialog({
          isOpen: true,
          type: "success",
          title: "Analysis Complete",
          message: "Match analysis completed successfully! The page will refresh."
        })
        setTimeout(() => {
          fetchMatchDetails()
        }, 1500)
      } else {
        setDialog({
          isOpen: true,
          type: "error",
          title: "Analysis Failed",
          message: data.error || "An error occurred during analysis."
        })
      }
    } catch (err) {
      setDialog({
        isOpen: true,
        type: "error",
        title: "Analysis Failed",
        message: "Failed to connect to the server. Please try again."
      })
      console.error(err)
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleDelete = async () => {
    setDialog({
      isOpen: true,
      type: "confirm",
      title: "Delete Match",
      message: "Are you sure you want to delete this match? This action cannot be undone.",
      onConfirm: async () => {
        try {
          const response = await fetch(`http://localhost:9000/api/matches/${matchId}`, {
            method: "DELETE",
          })
          const data = await response.json()

          if (data.success) {
            setDialog({
              isOpen: true,
              type: "success",
              title: "Match Deleted",
              message: "Match deleted successfully. Redirecting..."
            })
            setTimeout(() => {
              router.push("/matches")
            }, 1500)
          } else {
            setDialog({
              isOpen: true,
              type: "error",
              title: "Delete Failed",
              message: data.error || "Failed to delete match."
            })
          }
        } catch (err) {
          setDialog({
            isOpen: true,
            type: "error",
            title: "Delete Failed",
            message: "Failed to connect to the server."
          })
          console.error(err)
        }
      }
    })
  }

  const playClip = (clipPath: string) => {
    const relativePath = clipPath.split(/[/\\]/).slice(-2).join('/')
    setCurrentVideoUrl(`http://localhost:9000/api/media/${matchId}/${relativePath}`)
  }

  const downloadClip = (clipPath: string) => {
    const relativePath = clipPath.split(/[/\\]/).slice(-2).join('/')
    const url = `http://localhost:9000/api/media/${matchId}/${relativePath}`
    window.open(url, '_blank')
  }

  const watchOriginalVideo = () => {
    if (match?.video_path) {
      const videoFileName = match.video_path.split(/[/\\]/).pop()
      const url = `http://localhost:9000/api/media/${matchId}/${videoFileName}`
      window.open(url, '_blank')
    }
  }

  if (isLoading) {
    return (
      <main className="min-h-screen bg-background">
        <SiteNav />
        <div className="flex flex-col items-center justify-center py-20">
          <Loader2 className="w-12 h-12 animate-spin text-primary mb-4" />
          <p className="text-muted-foreground">Loading match details...</p>
        </div>
        <SiteFooter />
      </main>
    )
  }

  if (error || !match) {
    return (
      <main className="min-h-screen bg-background">
        <SiteNav />
        <div className="mx-auto max-w-4xl px-4 py-20">
          <div className="bg-destructive/10 border border-destructive/20 rounded-lg p-6 text-center">
            <AlertCircle className="w-12 h-12 text-destructive mx-auto mb-4" />
            <p className="text-destructive font-semibold mb-2">Failed to load match</p>
            <p className="text-sm text-muted-foreground mb-4">{error}</p>
            <button
              onClick={() => router.push("/matches")}
              className="px-4 py-2 bg-primary text-primary-foreground rounded-lg text-sm font-medium hover:opacity-90"
            >
              Back to Matches
            </button>
          </div>
        </div>
        <SiteFooter />
      </main>
    )
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <SiteNav />

      {/* Header Section */}
      <section className="border-b border-border/60 bg-gradient-to-b from-secondary/5 to-background">
        <div className="mx-auto max-w-7xl px-4 py-8 md:px-6 md:py-12">
          <button
            onClick={() => router.push("/matches")}
            className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground mb-4 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Matches
          </button>

          <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
            <div>
              <h1 className="text-4xl md:text-5xl font-bold mb-3">{match.title}</h1>
              <div className="flex items-center gap-4 text-muted-foreground">
                <div className="flex items-center gap-2">
                  <Calendar className="w-4 h-4" />
                  <span>{new Date(match.date).toLocaleDateString()}</span>
                </div>
                {match.status === "uploaded" && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-blue-500/10 text-blue-500 border-blue-500/20">
                    <AlertCircle className="w-3.5 h-3.5" />
                    Not Analyzed Yet
                  </span>
                )}
                {match.status === "processing" && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-yellow-500/10 text-yellow-500 border-yellow-500/20">
                    <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    Processing...
                  </span>
                )}
                {match.status === "completed" && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-green-500/10 text-green-500 border-green-500/20">
                    <CheckCircle className="w-3.5 h-3.5" />
                    Completed
                  </span>
                )}
                {match.status === "failed" && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border bg-red-500/10 text-red-500 border-red-500/20">
                    <AlertCircle className="w-3.5 h-3.5" />
                    Failed
                  </span>
                )}
              </div>
            </div>

            <div className="flex gap-2">
              {match.status === "uploaded" && (
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:opacity-90 disabled:opacity-50 transition-all"
                >
                  {isAnalyzing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Play className="w-4 h-4" />
                      Analyze Match
                    </>
                  )}
                </button>
              )}
              {(match.status === "completed" || match.status === "failed") && (
                <button
                  onClick={handleAnalyze}
                  disabled={isAnalyzing}
                  className="inline-flex items-center gap-2 px-4 py-2 bg-secondary text-foreground rounded-lg font-medium hover:bg-secondary/80 disabled:opacity-50 transition-all"
                >
                  <RefreshCw className="w-4 h-4" />
                  Re-analyze
                </button>
              )}
              <button
                onClick={watchOriginalVideo}
                className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 text-blue-500 border border-blue-500/20 rounded-lg font-medium hover:bg-blue-500/20 transition-all"
                title="Watch original uploaded video"
              >
                <Video className="w-4 h-4" />
                Original Video
              </button>
              <button
                onClick={handleDelete}
                className="inline-flex items-center gap-2 px-4 py-2 bg-destructive/10 text-destructive border border-destructive/20 rounded-lg font-medium hover:bg-destructive/20 transition-all"
              >
                <Trash2 className="w-4 h-4" />
                Delete
              </button>
            </div>
          </div>

          {match.description && (
            <p className="text-muted-foreground mt-4 max-w-3xl">{match.description}</p>
          )}

          {/* Real-time Progress Bar */}
          {progressData && isAnalyzing && (
            <div className="mt-6 bg-card border border-border/60 rounded-xl overflow-hidden">
              <div className="p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="relative">
                      <Loader2 className="w-5 h-5 animate-spin text-primary" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold">Processing Video</h3>
                      <p className="text-sm text-muted-foreground">{progressData.message || "Initializing..."}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold text-primary">{progressData.progress.toFixed(0)}%</div>
                    <div className="text-xs text-muted-foreground">Complete</div>
                  </div>
                </div>
                
                <div className="relative w-full h-2.5 bg-secondary/30 rounded-full overflow-hidden">
                  <div 
                    className="absolute top-0 left-0 h-full bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 transition-all duration-700 ease-out rounded-full"
                    style={{ width: `${progressData.progress}%` }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/30 to-transparent animate-shimmer" />
                  </div>
                </div>
                
                {progressData.status === "processing" && progressData.progress > 0 && progressData.progress < 70 && (
                  <div className="flex items-start gap-2 p-3 bg-blue-500/5 border border-blue-500/10 rounded-lg">
                    <div className="text-blue-500 text-lg">💡</div>
                    <p className="text-sm text-muted-foreground flex-1">
                      Processing long videos in <span className="font-medium text-foreground">50-minute chunks</span> for optimal performance and memory efficiency.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Main Content */}
      <section className="mx-auto max-w-7xl px-4 py-12 md:px-6 md:py-16">
        {match.status === "completed" && match.main_highlights ? (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Video Player */}
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-card border border-border/60 rounded-xl overflow-hidden">
                <video
                  key={currentVideoUrl}
                  controls
                  className="w-full aspect-video bg-black"
                  src={currentVideoUrl}
                >
                  Your browser does not support the video tag.
                </video>
              </div>

              {/* Event Statistics */}
              {match.analysis_data && (
                <div className="bg-card border border-border/60 rounded-xl p-6">
                  <h3 className="text-xl font-semibold mb-4">Event Statistics</h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {Object.entries(match.analysis_data.event_statistics || {}).map(([event, count]) => (
                      <div key={event} className="text-center p-3 bg-secondary/30 rounded-lg">
                        <div className="text-2xl font-bold text-primary">{count as number}</div>
                        <div className="text-sm text-muted-foreground capitalize">{event}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Verified Events Timeline */}
              {match.analysis_data?.verified_events && (
                <div className="bg-card border border-border/60 rounded-xl p-6">
                  <h3 className="text-xl font-semibold mb-4">Event Timeline</h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {match.analysis_data.verified_events.map((event: any) => (
                      <div
                        key={event.id}
                        className="flex items-center gap-4 p-3 bg-secondary/20 rounded-lg hover:bg-secondary/40 transition-colors"
                      >
                        <div className="flex-shrink-0 w-16 text-center">
                          <div className="text-sm font-semibold">{event.time_formatted}</div>
                        </div>
                        <div className="flex-1">
                          <div className="font-medium capitalize">{event.event_type}</div>
                          <div className="text-xs text-muted-foreground">
                            Confidence: {(event.confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                        {event.is_replay && (
                          <span className="text-xs px-2 py-1 bg-yellow-500/10 text-yellow-500 rounded">
                            Replay
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Clips Sidebar */}
            <div className="space-y-6">
              {/* Clip Type Selector */}
              <div className="bg-card border border-border/60 rounded-xl p-4">
                <h3 className="font-semibold mb-3">Highlights</h3>
                <div className="space-y-2">
                  <button
                    onClick={() => {
                      setSelectedClipType("main")
                      const highlightsPath = match.main_highlights!.split(/[/\\]/).slice(-2).join('/')
                      setCurrentVideoUrl(`http://localhost:9000/api/media/${matchId}/${highlightsPath}`)
                    }}
                    className={`w-full text-left px-4 py-3 rounded-lg transition-all ${
                      selectedClipType === "main"
                        ? "bg-primary text-primary-foreground"
                        : "bg-secondary/30 hover:bg-secondary/50"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-medium">Main Highlights</span>
                      <Play className="w-4 h-4" />
                    </div>
                  </button>
                </div>
              </div>

              {/* Event Clips */}
              {Object.keys(match.event_clips).length > 0 && (
                <div className="bg-card border border-border/60 rounded-xl p-4">
                  <h3 className="font-semibold mb-3">Event Clips</h3>
                  <div className="space-y-4">
                    {Object.entries(match.event_clips).map(([eventType, clips]) => (
                      <div key={eventType}>
                        <div className="text-sm font-medium capitalize mb-2 text-muted-foreground">
                          {eventType} ({clips.length})
                        </div>
                        <div className="space-y-1.5">
                          {clips.map((clip, index) => (
                            <div
                              key={clip}
                              className="flex items-center gap-2 p-2 bg-secondary/20 rounded-lg hover:bg-secondary/40 transition-colors group"
                            >
                              <button
                                onClick={() => playClip(clip)}
                                className="flex-1 text-left text-sm"
                              >
                                {eventType} #{index + 1}
                              </button>
                              <button
                                onClick={() => playClip(clip)}
                                className="p-1.5 hover:bg-primary/10 rounded transition-colors"
                                title="Play"
                              >
                                <Play className="w-4 h-4" />
                              </button>
                              <button
                                onClick={() => downloadClip(clip)}
                                className="p-1.5 hover:bg-primary/10 rounded transition-colors"
                                title="Download"
                              >
                                <Download className="w-4 h-4" />
                              </button>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : match.status === "processing" ? (
          <div className="text-center py-20">
            <Loader2 className="w-16 h-16 animate-spin text-primary mx-auto mb-4" />
            <h3 className="text-xl font-semibold mb-2">AI Analysis in Progress</h3>
            <p className="text-muted-foreground">
              This may take several minutes depending on video length...
            </p>
          </div>
        ) : (
          <div className="text-center py-20">
            <div className="w-20 h-20 rounded-full bg-secondary/50 flex items-center justify-center mx-auto mb-4">
              <Play className="w-10 h-10 text-muted-foreground" />
            </div>
            <h3 className="text-xl font-semibold mb-2">Match Not Analyzed Yet</h3>
            <p className="text-muted-foreground mb-6">
              Click the "Analyze Match" button above to start AI processing
            </p>
          </div>
        )}
      </section>

      <SiteFooter />
      
      <ConfirmDialog
        isOpen={dialog.isOpen}
        onClose={() => setDialog({ ...dialog, isOpen: false })}
        onConfirm={dialog.onConfirm}
        type={dialog.type}
        title={dialog.title}
        message={dialog.message}
        confirmText={dialog.type === "confirm" ? "Delete" : "OK"}
        cancelText="Cancel"
      />
    </main>
  )
}