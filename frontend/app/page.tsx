import { SiteNav } from "@/components/site-nav"
import HeroSection from "@/components/hero-section"
import ProblemSection from "@/components/problem-section"
import SolutionSection from "@/components/solution-section"
import FeaturesSection from "@/components/features-section"
import TechnologySection from "@/components/technology-section"
import BenefitsSection from "@/components/benefits-section"
import AboutSection from "@/components/about-section"
import FinalCTASection from "@/components/final-cta-section"
import { SiteFooter } from "@/components/footer"

export default function Page() {
  return (
    <main className="min-h-screen bg-background text-foreground">
      <SiteNav />
      <HeroSection />
      <ProblemSection />
      <SolutionSection />
      <FeaturesSection />
      <TechnologySection />
      <BenefitsSection />
      <AboutSection />
      <FinalCTASection />
      <SiteFooter />
    </main>
  )
}
