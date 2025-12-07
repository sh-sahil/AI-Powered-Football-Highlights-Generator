"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { SiteNav } from "@/components/site-nav"
import { SiteFooter } from "@/components/footer"
import { ConfirmDialog } from "@/components/confirm-dialog"
import { Upload, CheckCircle2, Loader2 } from "lucide-react"

export default function UploadPage() {
  const router = useRouter()
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [posterFile, setPosterFile] = useState<File | null>(null)
  const [videoUrl, setVideoUrl] = useState<string | null>(null)
  const [posterUrl, setPosterUrl] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [dialog, setDialog] = useState<{
    isOpen: boolean
    type: "success" | "error" | "warning" | "info"
    title: string
    message: string
  }>({
    isOpen: false,
    type: "info",
    title: "",
    message: ""
  })

  const handleVideoChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setVideoFile(file)
      setVideoUrl(URL.createObjectURL(file))
    }
  }

  const handlePosterChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      setPosterFile(file)
      setPosterUrl(URL.createObjectURL(file))
    }
  }

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (!videoFile) {
      setDialog({
        isOpen: true,
        type: "warning",
        title: "No Video Selected",
        message: "Please select a video file before uploading."
      })
      return
    }

    setIsUploading(true)
    setUploadProgress(0)

    try {
      const formData = new FormData(e.currentTarget)
      formData.append("video", videoFile)
      if (posterFile) {
        formData.append("poster", posterFile)
      }

      const response = await fetch("http://localhost:9000/api/matches/upload", {
        method: "POST",
        body: formData,
      })

      const data = await response.json()

      if (data.success) {
        setDialog({
          isOpen: true,
          type: "success",
          title: "Upload Successful",
          message: "Match uploaded successfully! Redirecting to matches page..."
        })
        setTimeout(() => {
          router.push("/matches")
        }, 1500)
      } else {
        setDialog({
          isOpen: true,
          type: "error",
          title: "Upload Failed",
          message: data.error || "An error occurred during upload."
        })
      }
    } catch (error) {
      console.error("Upload error:", error)
      setDialog({
        isOpen: true,
        type: "error",
        title: "Upload Failed",
        message: "Failed to connect to the server. Please try again."
      })
    } finally {
      setIsUploading(false)
    }
  }

  const handleReset = () => {
    setVideoFile(null)
    setPosterFile(null)
    setVideoUrl(null)
    setPosterUrl(null)
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <SiteNav />
      <section className="border-b border-border/60 bg-gradient-to-b from-secondary/5 to-background">
        <div className="mx-auto max-w-4xl px-4 py-4 md:px-6 md:py-6">
          <p className="text-sm text-muted-foreground">
            Upload your football match video and let our AI analyze it for highlights, player appearances, and key
            moments.
          </p>
        </div>
      </section>

      <section className="mx-auto max-w-4xl px-4 py-8 md:px-6 md:py-12">
        <form className="space-y-8" onSubmit={handleSubmit}>
          <div className="grid gap-6 md:grid-cols-2">
            <label className="space-y-2">
              <span className="text-sm font-semibold text-foreground">Match Title *</span>
              <input
                name="title"
                className="w-full rounded-lg border border-border/60 bg-card/40 px-4 py-3 text-sm transition-all focus:border-primary/50 focus:ring-1 focus:ring-primary/20"
                placeholder="e.g., Team A vs Team B - Final"
                required
                disabled={isUploading}
              />
            </label>
            <label className="space-y-2">
              <span className="text-sm font-semibold text-foreground">Match Date *</span>
              <input
                name="date"
                type="date"
                className="w-full rounded-lg border border-border/60 bg-card/40 px-4 py-3 text-sm transition-all focus:border-primary/50 focus:ring-1 focus:ring-primary/20"
                required
                disabled={isUploading}
              />
            </label>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            <label className="space-y-2">
              <span className="text-sm font-semibold text-foreground">Video File *</span>
              <input
                type="file"
                accept="video/*"
                className="w-full rounded-lg border border-border/60 bg-card/40 px-4 py-3 text-sm"
                onChange={handleVideoChange}
                required
                disabled={isUploading}
              />
              {videoUrl ? (
                <video
                  src={videoUrl}
                  controls
                  className="mt-3 aspect-video w-full rounded-lg border border-border/60 bg-black shadow-md"
                />
              ) : (
                <div className="mt-3 aspect-video w-full rounded-lg border-2 border-dashed border-border/40 bg-card/20 flex flex-col items-center justify-center gap-2">
                  <Upload className="w-6 h-6 text-muted-foreground" />
                  <p className="text-xs text-muted-foreground">Select a video file (MP4, WebM, etc.)</p>
                </div>
              )}
            </label>

            <label className="space-y-2">
              <span className="text-sm font-semibold text-foreground">Poster Image (Optional)</span>
              <input
                type="file"
                accept="image/*"
                className="w-full rounded-lg border border-border/60 bg-card/40 px-4 py-3 text-sm"
                onChange={handlePosterChange}
                disabled={isUploading}
              />
              {posterUrl ? (
                <img
                  src={posterUrl}
                  alt="Poster preview"
                  className="mt-3 aspect-video w-full rounded-lg border border-border/60 object-cover shadow-md"
                />
              ) : (
                <div className="mt-3 aspect-video w-full rounded-lg border-2 border-dashed border-border/40 bg-card/20 flex flex-col items-center justify-center gap-2">
                  <Upload className="w-6 h-6 text-muted-foreground" />
                  <p className="text-xs text-muted-foreground">Select a poster image (optional)</p>
                </div>
              )}
            </label>
          </div>

          <label className="space-y-2">
            <span className="text-sm font-semibold text-foreground">Match Description (Optional)</span>
            <textarea
              name="description"
              className="min-h-32 w-full rounded-lg border border-border/60 bg-card/40 px-4 py-3 text-sm transition-all focus:border-primary/50 focus:ring-1 focus:ring-primary/20"
              placeholder="Add details about the match, teams, tournament, etc..."
              disabled={isUploading}
            />
          </label>

          {isUploading && (
            <div className="bg-secondary/50 border border-secondary/30 rounded-lg p-4">
              <div className="flex items-center gap-3">
                <Loader2 className="w-5 h-5 animate-spin text-primary" />
                <div className="flex-1">
                  <p className="text-sm font-semibold text-foreground">Uploading match...</p>
                  <div className="mt-2 h-2 bg-secondary/30 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="flex flex-col items-start gap-3 sm:flex-row sm:items-center">
            <button
              type="submit"
              disabled={isUploading}
              className="inline-flex h-12 items-center rounded-lg bg-primary px-8 text-base font-semibold text-primary-foreground transition-all hover:opacity-90 hover:shadow-lg active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUploading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Uploading...
                </>
              ) : (
                "Upload Match"
              )}
            </button>
            <button
              type="button"
              onClick={handleReset}
              disabled={isUploading}
              className="inline-flex h-12 items-center rounded-lg border border-border/60 bg-background/60 px-8 text-base font-medium transition-all hover:bg-background/80 hover:shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Reset Form
            </button>
          </div>
        </form>
      </section>
      <SiteFooter />
      
      <ConfirmDialog
        isOpen={dialog.isOpen}
        onClose={() => setDialog({ ...dialog, isOpen: false })}
        type={dialog.type}
        title={dialog.title}
        message={dialog.message}
      />
    </main>
  )
}