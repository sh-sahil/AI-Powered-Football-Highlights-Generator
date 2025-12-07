"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { ChevronDown, ChevronUp, Play } from "lucide-react"

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

interface AnalysisDisplayProps {
  data: AnalysisResult
  onVideoClick: (url: string) => void
  onFrameClick: (frame: Frame) => void
}

// Component to format markdown-style text
function FormattedText({ text }: { text: string }) {
  // Split text into lines and process each line
  const lines = text.split('\n')
  
  return (
    <div className="space-y-3">
      {lines.map((line, index) => {
        // Handle headers (lines starting with **)
        if (line.startsWith('**') && line.endsWith('**')) {
          const headerText = line.slice(2, -2)
          return (
            <h3 key={index} className="text-lg font-bold text-primary mt-4 mb-2">
              {headerText}
            </h3>
          )
        }
        
        // Handle numbered sections (1. 2. 3. etc.)
        if (/^\d+\.\s\*\*/.test(line)) {
          const match = line.match(/^(\d+)\.\s\*\*([^*]+)\*\*:\s*(.*)/)
          if (match) {
            const [, number, title, content] = match
            return (
              <div key={index} className="mb-4">
                <h4 className="text-base font-semibold text-primary mb-2">
                  {number}. {title.toUpperCase()}
                </h4>
                {content && <p className="text-sm leading-relaxed ml-4">{content}</p>}
              </div>
            )
          }
        }
        
        // Handle regular paragraphs with inline formatting
        if (line.trim()) {
          // Process inline **bold** text
          const processedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
          
          return (
            <p 
              key={index} 
              className="text-sm leading-relaxed"
              dangerouslySetInnerHTML={{ __html: processedLine }}
            />
          )
        }
        
        // Empty lines for spacing
        return <div key={index} className="h-2" />
      })}
    </div>
  )
}

export default function AnalysisDisplay({
  data,
  onVideoClick,
  onFrameClick,
}: AnalysisDisplayProps) {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({})

  const toggleSection = (key: string) => {
    setExpandedSections((prev) => ({ ...prev, [key]: !prev[key] }))
  }

  // Handle "all_commentary" query type (multiple events)
  if (data.query_type === "all_commentary" && data.all_events) {
    return (
      <div className="mt-4 space-y-4">
        <div className="text-sm font-semibold text-muted-foreground">
          Found {data.total} {data.event_type}(s)
        </div>

        {/* Grid layout for multiple events - 2 columns on larger screens */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data.all_events.map((event, idx) => (
            <Card key={idx} className="bg-background/50 backdrop-blur border">
              <CardContent className="p-0">
                {/* Event Header */}
                <div className="px-4 py-3 border-b bg-muted/30">
                  <div className="flex items-center justify-between">
                    <h3 className="font-semibold text-base">
                      {data.event_type} #{event.sequence_number}
                    </h3>
                    <span className="text-sm text-muted-foreground font-mono">
                      {event.timestamp}
                    </span>
                  </div>
                </div>

                {/* Video Thumbnail */}
                {event.clip_url && (
                  <div className="relative group cursor-pointer" onClick={() => onVideoClick(event.clip_url!)}>
                    <img
                      src={event.key_frame}
                      alt={`${data.event_type} ${event.sequence_number}`}
                      className="w-full h-52 object-cover"
                    />
                    <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                      <Play className="h-16 w-16 text-white" />
                    </div>
                    {event.is_replay && (
                      <span className="absolute top-3 right-3 px-2 py-1 bg-yellow-500 text-black text-xs font-bold rounded">
                        POSSIBLE REPLAY
                      </span>
                    )}
                  </div>
                )}

                {/* Event Details */}
                <div className="px-4 py-3">
                  <Collapsible open={expandedSections[`event-${idx}`]} onOpenChange={() => toggleSection(`event-${idx}`)}>
                    <CollapsibleTrigger asChild>
                      <Button variant="outline" className="w-full justify-between text-xs">
                        <span className="truncate">
                          Details ({event.frame_count}f, {event.commentary_count}c)
                        </span>
                        {expandedSections[`event-${idx}`] ? <ChevronUp className="h-3 w-3 ml-2 flex-shrink-0" /> : <ChevronDown className="h-3 w-3 ml-2 flex-shrink-0" />}
                      </Button>
                    </CollapsibleTrigger>

                    <CollapsibleContent className="mt-3 space-y-3">
                      {/* Metadata */}
                      <div className="bg-muted/30 rounded-lg p-3">
                        <h5 className="font-medium text-xs mb-2 text-muted-foreground">Event Info</h5>
                        <div className="grid grid-cols-2 gap-2 text-xs">
                          <div>
                            <span className="text-muted-foreground">Duration:</span> {event.duration_seconds}s
                          </div>
                          <div>
                            <span className="text-muted-foreground">Range:</span> {event.time_range}
                          </div>
                        </div>
                      </div>

                      {/* Commentary */}
                      {event.context_commentary && event.context_commentary.length > 0 && (
                        <div className="bg-muted/30 rounded-lg p-3">
                          <h5 className="font-medium text-xs mb-2 text-muted-foreground">Commentary</h5>
                          <div className="space-y-1">
                            {event.context_commentary.slice(0, 2).map((comm, idx) => (
                              <div key={idx} className="text-xs">
                                <span className="font-mono text-muted-foreground">{comm.global_timestamp}</span>
                                <p className="mt-1">{comm.full_description}</p>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Frame Thumbnails */}
                      <div className="bg-muted/30 rounded-lg p-3">
                        <h5 className="font-medium text-xs mb-2 text-muted-foreground">Key Frames</h5>
                        <div className="grid grid-cols-4 gap-2">
                          {event.context_frames.slice(0, 8).map((frame, idx) => (
                            <div
                              key={idx}
                              className="group cursor-pointer relative"
                              onClick={() => onFrameClick(frame)}
                            >
                              <img
                                src={frame.file}
                                alt={`Frame ${idx + 1}`}
                                className="w-full h-16 object-cover rounded border-2 border-transparent hover:border-primary transition-colors"
                              />
                              {frame.active_events.length > 0 && (
                                <div className="absolute top-1 right-1">
                                  <span className="bg-red-500 text-white text-xs px-1 rounded">
                                    {frame.active_events.length}
                                  </span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </CollapsibleContent>
                  </Collapsible>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    )
  }

  // Handle single event (answer or commentary)
  return (
    <div className="mt-4 space-y-4">
      {/* Formatted Response Text */}
      <div className="bg-background/30 rounded-lg p-4 border">
        <FormattedText text={data.response} />
      </div>

      {/* Video Thumbnail */}
      {data.clip_url && data.key_frame && (
        <div className="relative group cursor-pointer" onClick={() => onVideoClick(data.clip_url!)}>
          <img
            src={data.key_frame}
            alt={`${data.event_type} ${data.sequence_number}`}
            className="w-full h-64 object-cover rounded-lg"
          />
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity rounded-lg">
            <Play className="h-16 w-16 text-white" />
          </div>
          {data.is_replay && (
            <span className="absolute top-2 right-2 px-2 py-1 bg-yellow-500 text-black text-xs font-bold rounded">
              POSSIBLE REPLAY
            </span>
          )}
        </div>
      )}

      {/* Event Details */}
      {data.context_frames && (
        <EventDetails
          event={{
            sequence_number: data.sequence_number!,
            timestamp: data.timestamp!,
            time_range: data.time_range!,
            duration_seconds: data.duration_seconds!,
            frame_count: data.frame_count!,
            commentary_count: data.commentary_count!,
            context_frames: data.context_frames,
            context_commentary: data.context_commentary || [],
            is_replay: data.is_replay || false,
          } as EventDetail}
          eventType={data.event_type}
          onFrameClick={onFrameClick}
          expanded={expandedSections["single-event"]}
          onToggle={() => toggleSection("single-event")}
        />
      )}
    </div>
  )
}

// Component for event details (frames + commentary)
function EventDetails({
  event,
  eventType,
  onFrameClick,
  expanded,
  onToggle,
}: {
  event: EventDetail
  eventType: string
  onFrameClick: (frame: Frame) => void
  expanded: boolean
  onToggle: () => void
}) {
  return (
    <Collapsible open={expanded} onOpenChange={onToggle}>
      <CollapsibleTrigger asChild>
        <Button variant="outline" className="w-full justify-between">
          <span>
            View Analysis Details ({event.frame_count} frames, {event.commentary_count} commentary)
          </span>
          {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
        </Button>
      </CollapsibleTrigger>

      <CollapsibleContent className="mt-4 space-y-4">
        {/* Metadata */}
        <Card>
          <CardContent className="p-4">
            <h4 className="font-semibold mb-3">Event Metadata</h4>
            <dl className="grid grid-cols-2 gap-2 text-sm">
              <div>
                <dt className="text-muted-foreground">Time Range:</dt>
                <dd className="font-mono">{event.time_range}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Duration:</dt>
                <dd>{event.duration_seconds}s</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Frames:</dt>
                <dd>{event.frame_count}</dd>
              </div>
              <div>
                <dt className="text-muted-foreground">Commentary:</dt>
                <dd>{event.commentary_count}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        {/* Commentary Section */}
        {event.context_commentary && event.context_commentary.length > 0 && (
          <Card>
            <CardContent className="p-4">
              <h4 className="font-semibold mb-3">Live Commentary</h4>
              <div className="space-y-2">
                {event.context_commentary.map((comm, idx) => (
                  <div key={idx} className="border-l-2 border-primary pl-3 py-1">
                    <div className="text-xs text-muted-foreground font-mono mb-1">
                      {comm.global_timestamp}
                    </div>
                    <p className="text-sm">{comm.full_description}</p>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Frames Section */}
        <Card>
          <CardContent className="p-4">
            <h4 className="font-semibold mb-3">Frame-by-Frame Analysis</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
              {event.context_frames.map((frame, idx) => (
                <div
                  key={idx}
                  className="group cursor-pointer relative"
                  onClick={() => onFrameClick(frame)}
                >
                  <img
                    src={frame.file}
                    alt={`Frame ${idx + 1}`}
                    className="w-full h-24 object-cover rounded-lg border-2 border-transparent hover:border-primary transition-colors"
                  />
                  <div className="absolute inset-0 bg-black/70 opacity-0 group-hover:opacity-100 transition-opacity rounded-lg flex items-center justify-center">
                    <span className="text-white text-xs font-semibold">
                      {frame.global_timestamp}
                    </span>
                  </div>
                  {frame.active_events.length > 0 && (
                    <div className="absolute top-1 right-1">
                      <span className="bg-red-500 text-white text-xs px-1 rounded">
                        {frame.active_events.length}
                      </span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </CollapsibleContent>
    </Collapsible>
  )
}