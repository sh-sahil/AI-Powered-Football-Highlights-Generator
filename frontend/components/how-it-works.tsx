"use client"

export function HowItWorks() {
  const steps = [
    {
      number: "01",
      title: "Video Upload",
      description: "Upload full-length football match videos",
    },
    {
      number: "02",
      title: "Frame Extraction",
      description: "System converts video into timestamped frames",
    },
    {
      number: "03",
      title: "AI Analysis",
      description: "YOLOv8n detects objects, LLaVA analyzes events",
    },
    {
      number: "04",
      title: "Event Tagging",
      description: "Key moments tagged and structured in JSON format",
    },
    {
      number: "05",
      title: "Highlight Generation",
      description: "Condense 90 minutes into 10-15 minute highlights",
    },
    {
      number: "06",
      title: "Interactive Review",
      description: "Query and explore highlights via chatbot interface",
    },
  ]

  return (
    <div className="w-full">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {steps.map((step, idx) => (
          <div key={idx} className="relative">
            <div className="flex flex-col h-full">
              <div className="text-6xl font-bold text-primary/20 mb-4">{step.number}</div>
              <h3 className="text-xl font-bold text-foreground mb-3">{step.title}</h3>
              <p className="text-muted-foreground text-sm leading-relaxed flex-grow">{step.description}</p>
            </div>
            {idx < steps.length - 1 && (
              <div className="hidden lg:block absolute top-12 -right-4 w-8 h-0.5 bg-gradient-to-r from-primary/50 to-transparent" />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
