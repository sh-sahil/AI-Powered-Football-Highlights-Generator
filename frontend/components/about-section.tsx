"use client"

export default function AboutSection() {
  return (
    <section className="py-20 px-4 sm:px-6 lg:px-8 bg-background">
      <div className="max-w-4xl mx-auto">
        <div className="text-center">
          <h2 className="text-4xl sm:text-5xl font-bold text-foreground mb-8">Digitizing Grassroots Football</h2>
          <p className="text-lg text-muted-foreground leading-relaxed mb-6">
            KhelIQ is transforming how local sports are experienced and shared. Our platform enables clubs, academies,
            and organizers across India to livestream matches, create digital player profiles, and now—generate
            professional highlights automatically.
          </p>
          <p className="text-lg text-muted-foreground leading-relaxed">
            This AI-powered system is part of our mission to give grassroots football the same production quality as
            professional leagues, making every match memorable and every player visible.
          </p>
        </div>
      </div>
    </section>
  )
}
