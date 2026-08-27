"use client";

import { useEffect, useState } from "react";
import { animate } from "framer-motion";
import type { Verdict } from "@/lib/types";

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

const COLOR: Record<Verdict, string> = {
  deepfake: "var(--deepfake)",
  uncertain: "var(--uncertain)",
  authentic: "var(--authentic)",
};

/**
 * Shows manipulation likelihood, not authenticity — the number now reads in
 * the same direction as the verdict beside it (high = deepfake).
 */
export function ScoreGauge({ score, verdict }: { score: number; verdict: Verdict }) {
  const [display, setDisplay] = useState(0);
  const [dashOffset, setDashOffset] = useState(CIRCUMFERENCE);

  useEffect(() => {
    const controls = animate(0, score, {
      duration: 1.1,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (value) => {
        setDisplay(value);
        setDashOffset(CIRCUMFERENCE * (1 - value));
      },
    });
    return () => controls.stop();
  }, [score]);

  const color = COLOR[verdict];

  return (
    <div className="relative flex h-40 w-40 shrink-0 items-center justify-center">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle cx="64" cy="64" r={RADIUS} fill="none" stroke="var(--border)" strokeWidth="8" />
        <circle
          cx="64"
          cy="64"
          r={RADIUS}
          fill="none"
          stroke={color}
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={CIRCUMFERENCE}
          strokeDashoffset={dashOffset}
          style={{ filter: `drop-shadow(0 0 6px color-mix(in srgb, ${color} 55%, transparent))` }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-mono text-3xl font-semibold tabular-nums text-foreground">
          {Math.round(display * 100)}%
        </span>
        <span className="mt-0.5 max-w-[5.5rem] text-center text-[10px] uppercase leading-tight tracking-wider text-muted-foreground">
          deepfake likelihood
        </span>
      </div>
    </div>
  );
}
