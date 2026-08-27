"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { VerdictBadge } from "@/components/results/VerdictBadge";
import { ScoreGauge } from "@/components/results/ScoreGauge";
import { ModalitySplit } from "@/components/results/ModalitySplit";
import { AcousticShapChart } from "@/components/results/AcousticShapChart";
import { TamperTimeline } from "@/components/results/TamperTimeline";
import { SegmentReview } from "@/components/results/SegmentReview";
import { SaliencyFilmstrip } from "@/components/results/SaliencyFilmstrip";
import { WaveformPanel } from "@/components/results/WaveformPanel";
import { PolicyPanel } from "@/components/results/PolicyPanel";
import { NarrativeSummary } from "@/components/results/NarrativeSummary";
import { ReviewDecision } from "@/components/results/ReviewDecision";
import {
  buildTamperSegments,
  clipDuration,
  deepfakeScore,
  isAudioOnly,
  verdictFor,
  VERDICT_BLURB,
} from "@/lib/analysis/localization";
import type { InferenceResult, SegmentJudgement } from "@/lib/types";

const fadeUp = (delay = 0) => ({
  initial: { opacity: 0, y: 14 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.45, ease: [0.16, 1, 0.3, 1] as const, delay },
});

/**
 * Mounted with `key={result.sampleId}`, so a new scan starts with a clean
 * selection and no carried-over judgements without an effect to reset them.
 */
export function ResultsView({ result }: { result: InferenceResult }) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [judgements, setJudgements] = useState<Record<string, SegmentJudgement>>({});

  const segments = useMemo(() => buildTamperSegments(result), [result]);
  const verdict = verdictFor(result);
  const audioOnly = isAudioOnly(result);

  return (
    <div className="space-y-5 pb-16">
      <motion.div
        {...fadeUp(0)}
        className="glass-panel flex flex-col gap-6 rounded-2xl p-6 sm:flex-row sm:items-center"
      >
        <ScoreGauge score={deepfakeScore(result)} verdict={verdict} />
        <div className="min-w-0 flex-1">
          <p className="truncate font-mono text-xs text-muted-foreground">{result.fileName}</p>
          <div className="mt-2">
            <VerdictBadge verdict={verdict} size="lg" />
          </div>
          <p className="mt-3 text-[13px] leading-relaxed text-muted-foreground">
            {VERDICT_BLURB[verdict]}
          </p>
          <p className="mt-2 font-mono text-[11px] text-muted-foreground">
            {result.sampleId} · scanned {new Date(result.createdAt).toLocaleTimeString()}
          </p>
        </div>
      </motion.div>

      <motion.div {...fadeUp(0.05)}>
        <NarrativeSummary result={result} segments={segments} />
      </motion.div>

      <motion.div {...fadeUp(0.1)}>
        <TamperTimeline
          segments={segments}
          duration={clipDuration(result)}
          selectedId={selectedId}
          onSelect={(segment) => setSelectedId((id) => (id === segment.id ? null : segment.id))}
          audioOnly={audioOnly}
        />
      </motion.div>

      <motion.div {...fadeUp(0.15)}>
        <SegmentReview
          segments={segments}
          selectedId={selectedId}
          onSelect={setSelectedId}
          judgements={judgements}
          onJudge={(id, judgement) => setJudgements((prev) => ({ ...prev, [id]: judgement }))}
        />
      </motion.div>

      <motion.div {...fadeUp(0.2)}>
        <ReviewDecision
          sampleId={result.sampleId}
          systemVerdict={verdict}
          segments={segments}
          judgements={judgements}
        />
      </motion.div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <motion.div {...fadeUp(0.25)}>
          <ModalitySplit result={result} />
        </motion.div>
        <motion.div {...fadeUp(0.3)}>
          <AcousticShapChart entries={result.acousticShap} />
        </motion.div>
      </div>

      {!audioOnly && (
        <motion.div {...fadeUp(0.35)}>
          <SaliencyFilmstrip frames={result.visualSaliency} segments={segments} />
        </motion.div>
      )}

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <motion.div {...fadeUp(0.4)}>
          <WaveformPanel waveform={result.waveform} segments={segments} />
        </motion.div>
        <motion.div {...fadeUp(0.45)}>
          <PolicyPanel result={result} />
        </motion.div>
      </div>
    </div>
  );
}
