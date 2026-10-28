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
import { cn } from "@/lib/utils";
import { useEntrance } from "@/lib/motion";

/**
 * Mounted with `key={result.sampleId}`, so a new scan starts with a clean
 * selection and no carried-over judgements without an effect to reset them.
 */
export function ResultsView({ result }: { result: InferenceResult }) {
  const entrance = useEntrance();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [judgements, setJudgements] = useState<Record<string, SegmentJudgement>>({});

  const segments = useMemo(() => buildTamperSegments(result), [result]);
  const verdict = verdictFor(result);
  const audioOnly = isAudioOnly(result);
  // Audio-only uploads carry no named-feature attribution (WavLM embeddings
  // have no descriptors to name), so the panel and its grid column both go.
  const hasAcousticShap = result.acousticShap.length > 0;

  return (
    <div className="space-y-4 pb-16 sm:space-y-5">
      <motion.div
        {...entrance(0)}
        className="glass-panel glass-panel-raised flex flex-col gap-6 rounded-2xl p-6 sm:flex-row sm:items-center"
      >
        <ScoreGauge score={deepfakeScore(result)} verdict={verdict} />
        <div className="min-w-0 flex-1">
          <p className="t-meta truncate font-mono text-muted-foreground">{result.fileName}</p>
          <div className="mt-2">
            <VerdictBadge verdict={verdict} size="lg" />
          </div>
          <p className="t-support mt-3 text-muted-foreground">{VERDICT_BLURB[verdict]}</p>
          <p className="t-meta nums mt-2 font-mono text-muted-foreground/80">
            {result.sampleId} · scanned {new Date(result.createdAt).toLocaleTimeString()}
          </p>
        </div>
      </motion.div>

      <motion.div {...entrance(1)} className="-mt-1 sm:-mt-1.5">
        <NarrativeSummary result={result} segments={segments} />
      </motion.div>

      <motion.div {...entrance(2)}>
        <TamperTimeline
          segments={segments}
          duration={clipDuration(result)}
          selectedId={selectedId}
          onSelect={(segment) => setSelectedId((id) => (id === segment.id ? null : segment.id))}
          audioOnly={audioOnly}
        />
      </motion.div>

      <motion.div {...entrance(3)}>
        <SegmentReview
          segments={segments}
          selectedId={selectedId}
          onSelect={setSelectedId}
          judgements={judgements}
          onJudge={(id, judgement) => setJudgements((prev) => ({ ...prev, [id]: judgement }))}
        />
      </motion.div>

      <motion.div {...entrance(4)}>
        <ReviewDecision
          sampleId={result.sampleId}
          systemVerdict={verdict}
          segments={segments}
          judgements={judgements}
        />
      </motion.div>

      <div className={cn("grid grid-cols-1 gap-5", hasAcousticShap && "md:grid-cols-2")}>
        <motion.div {...entrance(5)}>
          <ModalitySplit result={result} />
        </motion.div>
        {hasAcousticShap && (
          <motion.div {...entrance(6)}>
            <AcousticShapChart
              entries={result.acousticShap}
              source={result.acousticShapSource}
            />
          </motion.div>
        )}
      </div>

      {!audioOnly && (
        <motion.div {...entrance(7)}>
          <SaliencyFilmstrip frames={result.visualSaliency} segments={segments} />
        </motion.div>
      )}

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
        <motion.div {...entrance(8)}>
          <WaveformPanel waveform={result.waveform} segments={segments} />
        </motion.div>
        <motion.div {...entrance(9)}>
          <PolicyPanel result={result} />
        </motion.div>
      </div>
    </div>
  );
}
