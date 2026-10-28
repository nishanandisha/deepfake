"use client";

import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowRight, Ear, Eye, MapPin } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Dropzone } from "@/components/upload/Dropzone";
import { ScenarioPicker } from "@/components/upload/ScenarioPicker";
import { Button } from "@/components/ui/button";
import { usePipelineStore } from "@/store/pipelineStore";
import { useBackendStatus } from "@/hooks/useBackendStatus";
import { useEntrance } from "@/lib/motion";

const STATS = [
  {
    icon: Eye,
    label: "Checks the video",
    detail: "frame-by-frame, for face-swap and reenactment artefacts",
  },
  {
    icon: Ear,
    label: "Checks the audio",
    detail: "for cloned, vocoded and synthesised speech",
  },
  {
    icon: MapPin,
    label: "Pinpoints the fake",
    detail: "the exact spans that were altered, not just a yes/no",
  },
];

export default function LandingPage() {
  const router = useRouter();
  const { file, runInference } = usePipelineStore();
  const backend = useBackendStatus();
  const entrance = useEntrance();

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
            {...entrance(0)}
            className="t-meta mb-5 inline-flex items-center gap-2 rounded-full border border-border bg-secondary/40 px-3 py-1.5 font-medium uppercase tracking-[0.08em] text-muted-foreground"
          >
            <span className="h-1.5 w-1.5 rounded-full bg-brand shadow-[0_0_0_3px_color-mix(in_srgb,var(--brand)_22%,transparent)]" />
            Multimodal deepfake detection
          </motion.div>

          <motion.h1
            {...entrance(1)}
            className="t-display max-w-3xl text-balance text-center font-semibold text-foreground"
          >
            Is it real? And if not,{" "}
            <span className="accent-gradient-text">which part was faked?</span>
          </motion.h1>

          <motion.p
            {...entrance(2)}
            className="t-body mt-5 max-w-xl text-center text-muted-foreground"
          >
            Drop in a clip and DeepFake checks the picture and the sound separately, then
            shows you the exact seconds it believes were manipulated — so you can look at the
            evidence and judge for yourself.
          </motion.p>

          <motion.div
            {...entrance(3)}
            className="mt-12 grid w-full grid-cols-1 gap-3 sm:grid-cols-3"
          >
            {STATS.map(({ icon: Icon, label, detail }) => (
              <div
                key={label}
                className="glass-panel glass-panel-interactive rounded-xl p-4"
              >
                <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-brand/20 bg-brand/10">
                  <Icon className="h-4 w-4 text-brand" />
                </span>
                <p className="t-title mt-3 font-medium text-foreground">{label}</p>
                <p className="t-support mt-1 text-muted-foreground">{detail}</p>
              </div>
            ))}
          </motion.div>

          <motion.div {...entrance(4)} className="mt-10 w-full space-y-6">
            <Dropzone />

            {/* The scenario picker only steers the mock engine. With the trained
                model up it decides nothing, and showing it would imply the
                result had been dialled in by hand. */}
            {backend === "mock" && <ScenarioPicker />}

            <div className="flex flex-col items-center gap-3 pt-2 sm:flex-row sm:justify-between">
              <p className="t-support text-muted-foreground sm:max-w-sm">
                {backend === "model"
                  ? "Scored by the trained cross-attention fusion model running locally. Your file is sent to that local server and not stored."
                  : "The detection backend is offline, so results below are simulated from the scenario you pick — not model output."}
              </p>
              <Button
                size="lg"
                disabled={!file}
                onClick={handleRun}
                className="accent-gradient-bg h-11 w-full shrink-0 gap-2 border-0 px-6 font-medium text-foreground hover:opacity-90 sm:w-auto"
              >
                Scan for deepfakes
                <ArrowRight className="h-4 w-4" />
              </Button>
            </div>
          </motion.div>
        </div>
      </main>
    </div>
  );
}
