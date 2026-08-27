"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { AlertTriangle, ArrowLeft, RotateCcw } from "lucide-react";
import { Header } from "@/components/layout/Header";
import { Sidebar } from "@/components/layout/Sidebar";
import { StageStepper } from "@/components/pipeline/StageStepper";
import { ResultsView } from "@/components/results/ResultsView";
import { Button } from "@/components/ui/button";
import { usePipelineStore } from "@/store/pipelineStore";

export default function ReviewPage() {
  const router = useRouter();
  const { file, previewUrl, status, stageIndex, result, error, runInference, reset } =
    usePipelineStore();

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
          <Button
            variant="ghost"
            size="sm"
            onClick={handleNewAnalysis}
            className="gap-1.5 text-muted-foreground"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            New scan
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
                    className="h-32 w-56 rounded-xl border border-border object-cover opacity-70"
                  />
                )}
                <div>
                  <h1 className="text-lg font-medium text-foreground">Scanning {file.name}</h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Checking the video and the audio track for signs of manipulation…
                  </p>
                </div>
                <StageStepper stageIndex={stageIndex} />
              </div>
            )}

            {status === "error" && (
              <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 text-center">
                <AlertTriangle className="h-8 w-8 text-deepfake" />
                <p className="text-sm text-foreground">
                  {error ?? "Something went wrong during the scan."}
                </p>
                <Button variant="outline" onClick={() => void runInference()} className="gap-1.5">
                  <ArrowLeft className="h-3.5 w-3.5" />
                  Try again
                </Button>
              </div>
            )}

            {status === "done" && result && <ResultsView key={result.sampleId} result={result} />}
          </div>
        </main>
      </div>
    </div>
  );
}
