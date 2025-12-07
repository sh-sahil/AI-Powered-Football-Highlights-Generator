"use client"

import { AlertCircle, CheckCircle, Info, XCircle } from "lucide-react"

interface ConfirmDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm?: () => void
  title: string
  message: string
  type?: "success" | "error" | "warning" | "info" | "confirm"
  confirmText?: string
  cancelText?: string
}

export function ConfirmDialog({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  type = "info",
  confirmText = "OK",
  cancelText = "Cancel"
}: ConfirmDialogProps) {
  if (!isOpen) return null

  const icons = {
    success: <CheckCircle className="w-12 h-12 text-green-500" />,
    error: <XCircle className="w-12 h-12 text-destructive" />,
    warning: <AlertCircle className="w-12 h-12 text-yellow-500" />,
    info: <Info className="w-12 h-12 text-blue-500" />,
    confirm: <AlertCircle className="w-12 h-12 text-yellow-500" />
  }

  const backgrounds = {
    success: "bg-green-500/10 border-green-500/20",
    error: "bg-destructive/10 border-destructive/20",
    warning: "bg-yellow-500/10 border-yellow-500/20",
    info: "bg-blue-500/10 border-blue-500/20",
    confirm: "bg-yellow-500/10 border-yellow-500/20"
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
      <div className="bg-card border border-border/60 rounded-xl shadow-2xl max-w-md w-full animate-in fade-in zoom-in duration-200">
        <div className={`border-b border-border/60 p-6 ${backgrounds[type]}`}>
          <div className="flex flex-col items-center text-center gap-3">
            {icons[type]}
            <h3 className="text-xl font-semibold text-foreground">{title}</h3>
          </div>
        </div>
        
        <div className="p-6">
          <p className="text-muted-foreground text-center mb-6">{message}</p>
          
          <div className="flex gap-3">
            {type === "confirm" && onConfirm ? (
              <>
                <button
                  onClick={onClose}
                  className="flex-1 px-4 py-2.5 rounded-lg border border-border/60 bg-background/60 font-medium hover:bg-background/80 transition-all"
                >
                  {cancelText}
                </button>
                <button
                  onClick={() => {
                    onConfirm()
                    onClose()
                  }}
                  className="flex-1 px-4 py-2.5 rounded-lg bg-primary text-primary-foreground font-medium hover:opacity-90 transition-all"
                >
                  {confirmText}
                </button>
              </>
            ) : (
              <button
                onClick={onClose}
                className="w-full px-4 py-2.5 rounded-lg bg-primary text-primary-foreground font-medium hover:opacity-90 transition-all"
              >
                {confirmText}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
