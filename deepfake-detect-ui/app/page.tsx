"use client";

import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowRight, Layers, ScanFace, ShieldCheck } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Dropzone } from "@/components/upload/Dropzone";
import { ScenarioPicker } from "@/components/upload/ScenarioPicker";
import { Button } from "@/components/ui/button";
import { usePipelineStore } from "@/store/pipelineStore";

const STATS = [
  { icon: Layers, label: "6-stage pipeline", detail: "preprocess → fusion → explanation" },
  { icon: ShieldCheck, label: "Calibrated policy", detail: "approve / flag / block, false-suppression capped" },
  { icon: ScanFace, label: "Dual explanations", detail: "acoustic SHAP + visual Grad-CAM" },
];

export default function LandingPage() {
  const router = useRouter();
  const { file, runInference } = usePipelineStore();

  const handleRun = () => {
    if (!file) return;
    router.push("/review");
    void runInference();
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main className="relative flex-1 grid-glow-bg">
        <div className="mx-auto flex max-w-5xl flex-col items-center px-4 pb-24 pt-16 sm:px-6 sm:pt-24 lg:px-8">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, ease: "easeOut" }}
            className="mb-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-muted-foreground"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-approve" />
            Explainable multimodal deepfake detection
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.05 }}
            className="max-w-3xl text-balance text-center text-4xl font-semibold tracking-tight text-foreground sm:text-5xl"
          >
            Every decision, <span className="accent-gradient-text">explained down to the frame.</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 }}
            className="mt-5 max-w-xl text-center text-[15px] leading-relaxed text-muted-foreground"
          >
            Upload a clip to see the moderator review console: a calibrated authenticity
            score, visual/acoustic modality attribution, and frame-level saliency evidence
            for a decision you can actually defend.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.15 }}
            className="mt-10 grid w-full grid-cols-1 gap-3 sm:grid-cols-3"
          >
            {STATS.map(({ icon: Icon, label, detail }) => (
              <div key={label} className="glass-panel rounded-xl px-4 py-3">
                <Icon className="h-4 w-4 text-cyan" />
                <p className="mt-2 text-sm font-medium text-foreground">{label}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">{detail}</p>
              </div>
            ))}
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, ease: "easeOut", delay: 0.2 }}
            className="mt-12 w-full space-y-6"
          >
            <Dropzone />
            <ScenarioPicker />

            <div className="flex flex-col items-center gap-3 pt-2 sm:flex-row sm:justify-between">
              <p className="text-xs text-muted-foreground sm:max-w-sm">
                No trained checkpoint runs in-browser: real frames &amp; waveform are read from your
                file, scoring follows the scenario above for demonstration.
              </p>
              <Button
                size="lg"
                disabled={!file}
                onClick={handleRun}
                className="accent-gradient-bg h-11 w-full shrink-0 gap-2 border-0 px-6 font-medium text-background hover:opacity-90 sm:w-auto"
              >
                Run analysis
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
