"use client";

import { motion } from "framer-motion";
import { Eye, Ear } from "lucide-react";
import type { InferenceResult } from "@/lib/types";

function Bar({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: typeof Eye;
  label: string;
  value: number;
  color: string;
}) {
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="flex items-center gap-1.5 text-xs font-medium text-foreground">
          <Icon className="h-3.5 w-3.5" style={{ color }} />
          {label}
        </span>
        <span className="font-mono text-xs text-muted-foreground">{Math.round(value * 100)}%</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          animate={{ width: `${value * 100}%` }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
    </div>
  );
}

export function ModalitySplit({ result }: { result: InferenceResult }) {
  const visualPct = result.gate;
  const acousticPct = 1 - result.gate;

  return (
    <div className="glass-panel rounded-2xl p-5">
      <h3 className="text-sm font-semibold text-foreground">Modality attribution</h3>
      <p className="mt-1 text-xs text-muted-foreground">
        Learned gate weighting from cross-modal fusion — which stream drove the fused decision.
      </p>

      <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full">
        <motion.div
          className="h-full bg-cyan"
          initial={{ width: 0 }}
          animate={{ width: `${visualPct * 100}%` }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
        <motion.div
          className="h-full bg-violet"
          initial={{ width: 0 }}
          animate={{ width: `${acousticPct * 100}%` }}
          transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
        />
      </div>
      <div className="mt-2 flex justify-between font-mono text-[11px] text-muted-foreground">
        <span className="text-cyan">visual {Math.round(visualPct * 100)}%</span>
        <span className="text-violet">acoustic {Math.round(acousticPct * 100)}%</span>
      </div>

      <div className="mt-5 space-y-4 border-t border-white/[0.06] pt-4">
        <p className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Per-branch fake probability
        </p>
        <Bar icon={Eye} label="Visual branch" value={result.yHatVisual} color="var(--cyan)" />
        <Bar icon={Ear} label="Acoustic branch" value={result.yHatAcoustic} color="var(--violet)" />
      </div>
    </div>
  );
}
