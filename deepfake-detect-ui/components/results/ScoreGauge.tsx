"use client";

import { useEffect, useState } from "react";
import { animate } from "framer-motion";
import type { Decision } from "@/lib/types";

const RADIUS = 54;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

const COLOR: Record<Decision, string> = {
  approve: "var(--approve)",
  flag: "var(--flag)",
  block: "var(--block)",
};

export function ScoreGauge({ cScore, decision }: { cScore: number; decision: Decision }) {
  const [display, setDisplay] = useState(0);
  const [dashOffset, setDashOffset] = useState(CIRCUMFERENCE);

  useEffect(() => {
    const controls = animate(0, cScore, {
      duration: 1.1,
      ease: [0.16, 1, 0.3, 1],
      onUpdate: (value) => {
        setDisplay(value);
        setDashOffset(CIRCUMFERENCE * (1 - value));
      },
    });
    return () => controls.stop();
  }, [cScore]);

  const color = COLOR[decision];

  return (
    <div className="relative flex h-40 w-40 shrink-0 items-center justify-center">
      <svg viewBox="0 0 128 128" className="h-full w-full -rotate-90">
        <circle
          cx="64"
          cy="64"
          r={RADIUS}
          fill="none"
          stroke="var(--border)"
          strokeWidth="8"
        />
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
          style={{ filter: `drop-shadow(0 0 6px color-mix(in oklch, ${color} 55%, transparent))` }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="font-mono text-3xl font-semibold tabular-nums text-foreground">
          {Math.round(display * 100)}
        </span>
        <span className="mt-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
          authenticity
        </span>
      </div>
    </div>
  );
}
