"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { AlertTriangle, ArrowLeft, RotateCcw } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { StageStepper } from "@/components/pipeline/StageStepper";
import { DecisionBadge } from "@/components/results/DecisionBadge";
import { ScoreGauge } from "@/components/results/ScoreGauge";
import { ModalitySplit } from "@/components/results/ModalitySplit";
import { AcousticShapChart } from "@/components/results/AcousticShapChart";
import { SaliencyFilmstrip } from "@/components/results/SaliencyFilmstrip";
import { WaveformPanel } from "@/components/results/WaveformPanel";
import { PolicyPanel } from "@/components/results/PolicyPanel";
import { NarrativeSummary } from "@/components/results/NarrativeSummary";
import { ModeratorActions } from "@/components/results/ModeratorActions";
import { Button } from "@/components/ui/button";
import { usePipelineStore } from "@/store/pipelineStore";

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] as const, delay },
});

export default function ReviewPage() {
  const router = useRouter();
  const { file, previewUrl, status, stageIndex, result, error, runInference, reset } = usePipelineStore();

  useEffect(() => {
    if (!file) {
      router.replace("/");
      return;
    }
    if (status === "idle") {
      void runInference();
    }
  }, [file, status, runInference, router]);

  if (!file) return null;

  const handleNewAnalysis = () => {
    reset();
    router.push("/");
  };

  return (
    <div className="flex min-h-screen flex-col">
      <Header
        right={
          <Button variant="ghost" size="sm" onClick={handleNewAnalysis} className="gap-1.5 text-muted-foreground">
            <RotateCcw className="h-3.5 w-3.5" />
            New analysis
          </Button>
        }
      />
      <div className="flex flex-1">
        <Sidebar activeSampleId={result?.sampleId} />
        <main className="flex-1 grid-glow-bg">
          <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6 lg:px-8">
            {(status === "running" || status === "idle") && (
              <div className="flex min-h-[60vh] flex-col items-center justify-center gap-6 text-center">
                {/* Deliberately NOT autoPlay/loop: the browser hardware-decodes
                    on the same GPU that PyTorch is using for inference, and on a
                    4GB card that contention crashed the renderer
                    (STATUS_ACCESS_VIOLATION). preload="metadata" shows a still
                    first frame without decoding the whole stream. */}
                {previewUrl && file.type.startsWith("video/") && (
                  <video
                    src={previewUrl}
                    muted
                    playsInline
                    preload="metadata"
                    className="h-32 w-56 rounded-xl border border-white/10 object-cover opacity-70"
                  />
                )}
                <div>
                  <h1 className="text-lg font-medium text-foreground">Analyzing {file.name}</h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Running the six-stage pipeline against the selected scenario…
                  </p>
                </div>
                <StageStepper stageIndex={stageIndex} />
              </div>
            )}

            {status === "error" && (
              <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-center">
                <AlertTriangle className="h-8 w-8 text-block" />
                <p className="text-sm text-foreground">{error ?? "Something went wrong during analysis."}</p>
                <Button variant="outline" onClick={() => void runInference()} className="gap-1.5">
                  <ArrowLeft className="h-3.5 w-3.5" />
                  Try again
                </Button>
              </div>
            )}

            {status === "done" && result && (
              <div className="space-y-5 pb-16">
                <motion.div {...fadeUp(0)} className="glass-panel flex flex-col gap-6 rounded-2xl p-6 sm:flex-row sm:items-center">
                  <ScoreGauge cScore={result.cScore} decision={result.decision} />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-mono text-xs text-muted-foreground">{result.fileName}</p>
                    <div className="mt-2">
                      <DecisionBadge decision={result.decision} />
                    </div>
                    <p className="mt-3 text-xs text-muted-foreground">
                      Sample <span className="font-mono">{result.sampleId}</span> &middot; analyzed{" "}
                      {new Date(result.createdAt).toLocaleTimeString()}
                    </p>
                  </div>
                </motion.div>

                <motion.div {...fadeUp(0.05)}>
                  <NarrativeSummary result={result} />
                </motion.div>

                <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                  <motion.div {...fadeUp(0.1)}>
                    <ModalitySplit result={result} />
                  </motion.div>
                  <motion.div {...fadeUp(0.15)}>
                    <AcousticShapChart entries={result.acousticShap} />
                  </motion.div>
                </div>

                <motion.div {...fadeUp(0.2)}>
                  <SaliencyFilmstrip frames={result.visualSaliency} />
                </motion.div>

                <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                  <motion.div {...fadeUp(0.25)}>
                    <WaveformPanel waveform={result.waveform} />
                  </motion.div>
                  <motion.div {...fadeUp(0.3)}>
                    <PolicyPanel result={result} />
                  </motion.div>
                </div>

                <motion.div {...fadeUp(0.35)}>
                  <ModeratorActions sampleId={result.sampleId} systemDecision={result.decision} />
                </motion.div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
