"use client"

import { useState, useRef, useEffect } from "react"
import { SiteNav } from "@/components/site-nav"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Loader2 } from "lucide-react"
import AnalysisDisplay from "./components/AnalysisDisplay"

type Msg = {
  id: number
  role: "user" | "assistant"
  text: string
  data?: AnalysisResult
}

type Frame = {
  file: string
  global_timestamp: string
  rag_description: string
  active_events: string[]
  segment_name?: string
  full_description?: string
}

type Commentary = {
  global_timestamp: string
  full_description: string
}

type EventDetail = {
  sequence_number: number
  timestamp: string
  key_frame: string
  time_range: string
  duration_seconds: number
  frame_count: number
  commentary_count: number
  frame_files: string[]
  context_frames: Frame[]
  context_commentary: Commentary[]
  is_replay: boolean
  clip_url?: string
}

type AnalysisResult = {
  query: string
  query_type: "answer" | "commentary" | "all_commentary"
  event_type: string
  sequence_number?: number
  timestamp?: string
  key_frame?: string
  time_range?: string
  duration_seconds?: number
  frame_count?: number
  commentary_count?: number
  frame_files?: string[]
  context_frames?: Frame[]
  context_commentary?: Commentary[]
  response: string
  success: boolean
  is_replay?: boolean
  clip_url?: string
  total?: number
  all_events?: EventDetail[]
  error?: string
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Msg[]>([
    {
      id: 1,
      role: "assistant",
      text: "Hi! I'm your AI Football Match Analyst. Ask me about any match events.",
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null)
  const [selectedFrame, setSelectedFrame] = useState<Frame | null>(null)
  const listRef = useRef<HTMLDivElement>(null)

  // Fix image URLs by adding backend prefix
  const fixImageUrl = (url: string) => {
    if (url.startsWith('/')) {
      return `http://localhost:8000${url}`
    }
    return url
  }

  // Clean frame description by extracting only the response content
  const cleanFrameDescription = (description: string) => {
    // If it contains [/INST], extract everything after it
    if (description.includes('[/INST]')) {
      const parts = description.split('[/INST]')
      if (parts.length > 1) {
        let cleanDesc = parts[1].trim()
        
        // Extract scene description if it exists
        const sceneMatch = cleanDesc.match(/1\.\s*SCENE DESCRIPTION:\s*([\s\S]*?)(?=\n2\.|$)/)
        if (sceneMatch) {
          return sceneMatch[1].trim()
        }
        
        // If no scene description pattern, return the cleaned content
        return cleanDesc
      }
    }
    
    // If no [/INST] pattern, return original but clean up
    return description.replace(/\[INST\][\s\S]*?\[\/INST\]/g, '').trim()
  }

  // Fix all URLs in the data
  const fixAllUrls = (data: AnalysisResult) => {
    if (data.key_frame) {
      data.key_frame = fixImageUrl(data.key_frame)
    }
    if (data.clip_url) {
      data.clip_url = fixImageUrl(data.clip_url)
    }
    if (data.context_frames) {
      data.context_frames.forEach(frame => {
        if (frame.file) {
          frame.file = fixImageUrl(frame.file)
        }
        // Clean the frame description
        if (frame.rag_description) {
          frame.rag_description = cleanFrameDescription(frame.rag_description)
        }
        if (frame.full_description) {
          frame.full_description = cleanFrameDescription(frame.full_description)
        }
      })
    }
    if (data.all_events) {
      data.all_events.forEach(event => {
        if (event.key_frame) {
          event.key_frame = fixImageUrl(event.key_frame)
        }
        if (event.clip_url) {
          event.clip_url = fixImageUrl(event.clip_url)
        }
        if (event.context_frames) {
          event.context_frames.forEach(frame => {
            if (frame.file) {
              frame.file = fixImageUrl(frame.file)
            }
            // Clean the frame description
            if (frame.rag_description) {
              frame.rag_description = cleanFrameDescription(frame.rag_description)
            }
            if (frame.full_description) {
              frame.full_description = cleanFrameDescription(frame.full_description)
            }
          })
        }
      })
    }
    return data
  }

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" })
  }, [messages.length])

  async function send() {
    if (!input.trim() || isLoading) return

    const userMsg: Msg = { id: Date.now(), role: "user", text: input.trim() }
    setMessages((m) => [...m, userMsg])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(input.trim()),
      })

      const data: AnalysisResult = await response.json()

      if (data.error) {
        setMessages((m) => [
          ...m,
          {
            id: Date.now() + 1,
            role: "assistant",
            text: `❌ Error: ${data.error}`,
          },
        ])
      } else {
        // Fix all URLs before storing the data
        const fixedData = fixAllUrls(data)
        
        setMessages((m) => [
          ...m,
          {
            id: Date.now() + 1,
            role: "assistant",
            text: fixedData.response,
            data: fixedData,
          },
        ])
      }
    } catch (error) {
      setMessages((m) => [
        ...m,
        {
          id: Date.now() + 1,
          role: "assistant",
          text: `❌ Failed to connect to backend. Make sure the FastAPI server is running on http://localhost:8000`,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-background">
      <SiteNav />
      
      {/* Main Chat Container */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="container mx-auto px-4 py-2">
          </div>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-hidden">
          <div ref={listRef} className="h-full overflow-y-auto">
            <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
              {messages.map((m) => (
                <div
                  key={m.id}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                      m.role === "user"
                        ? "bg-primary text-primary-foreground ml-12"
                        : "bg-muted mr-12"
                    }`}
                  >
                    {/* Only show raw text if there's no analysis data */}
                    {!(m.role === "assistant" && m.data) && (
                      <div className="whitespace-pre-wrap text-sm leading-relaxed">{m.text}</div>
                    )}

                    {/* Render analysis results */}
                    {m.role === "assistant" && m.data && (
                      <AnalysisDisplay
                        data={m.data}
                        onVideoClick={setSelectedVideo}
                        onFrameClick={setSelectedFrame}
                      />
                    )}
                  </div>
                </div>
              ))}

              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-muted rounded-2xl px-4 py-3 mr-12">
                    <div className="flex items-center gap-2 text-sm">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span>Analyzing match footage...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
          <div className="max-w-4xl mx-auto px-4 py-4">
            <div className="flex gap-3 items-end">
              <div className="flex-1 relative">
                <Input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && send()}
                  placeholder="Ask about match events, players, or specific moments..."
                  disabled={isLoading}
                  className="min-h-[48px] pr-12 rounded-xl border-2 focus:border-primary transition-colors"
                  aria-label="Message"
                />
                <Button 
                  onClick={send} 
                  disabled={isLoading || !input.trim()}
                  size="sm"
                  className="absolute right-2 top-1/2 -translate-y-1/2 h-8 w-8 p-0 rounded-lg"
                >
                  {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "→"}
                </Button>
              </div>
            </div>
            <p className="text-xs text-muted-foreground mt-2 text-center">
              Ask about fouls, goals, celebrations, or any match events
            </p>
          </div>
        </div>
      </div>

      {/* Video Dialog */}
      <Dialog open={!!selectedVideo} onOpenChange={() => setSelectedVideo(null)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>Match Clip</DialogTitle>
          </DialogHeader>
          {selectedVideo && (
            <video
              src={selectedVideo}
              controls
              autoPlay
              className="w-full rounded-lg"
            />
          )}
        </DialogContent>
      </Dialog>

      {/* Frame Detail Dialog */}
      <Dialog open={!!selectedFrame} onOpenChange={() => setSelectedFrame(null)}>
        <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Frame Analysis - {selectedFrame?.global_timestamp}</DialogTitle>
          </DialogHeader>
          {selectedFrame && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Left: Image */}
              <div>
                <img
                  src={selectedFrame.file}
                  alt={`Frame at ${selectedFrame.global_timestamp}`}
                  className="w-full rounded-lg shadow-lg"
                />
                {selectedFrame.active_events.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {selectedFrame.active_events.map((event, idx) => (
                      <span
                        key={idx}
                        className="px-3 py-1 bg-primary text-primary-foreground rounded-full text-sm font-medium"
                      >
                        {event}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* Right: Description */}
              <div className="space-y-4">
                <div>
                  <h3 className="font-semibold text-lg mb-2">Frame Description</h3>
                  <p className="text-sm leading-relaxed">
                    {selectedFrame.full_description || selectedFrame.rag_description}
                  </p>
                </div>

                <div className="border-t pt-4">
                  <h3 className="font-semibold mb-2">Metadata</h3>
                  <dl className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Timestamp:</dt>
                      <dd className="font-mono">{selectedFrame.global_timestamp}</dd>
                    </div>
                    {selectedFrame.segment_name && (
                      <div className="flex justify-between">
                        <dt className="text-muted-foreground">Segment:</dt>
                        <dd>{selectedFrame.segment_name}</dd>
                      </div>
                    )}
                    <div className="flex justify-between">
                      <dt className="text-muted-foreground">Active Events:</dt>
                      <dd>{selectedFrame.active_events.length || "None"}</dd>
                    </div>
                  </dl>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

