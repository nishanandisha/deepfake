"use client";

import { Bar, BarChart, Cell, ResponsiveContainer, XAxis, YAxis } from "recharts";
import type { AcousticShapEntry } from "@/lib/types";

const FRIENDLY_NAMES: Record<string, string> = {
  f0: "F0 (pitch)",
  voicing_confidence: "Voicing confidence",
  spectral_centroid: "Spectral centroid",
  spectral_bandwidth: "Spectral bandwidth",
  spectral_rolloff: "Spectral roll-off",
  spectral_flatness: "Spectral flatness",
  zero_crossing_rate: "Zero-crossing rate",
  short_time_energy: "Short-time energy",
};

function friendlyName(feature: string): string {
  if (FRIENDLY_NAMES[feature]) return FRIENDLY_NAMES[feature];
  const mfccMatch = feature.match(/^mfcc(_delta2?)?_(\d+)$/);
  if (mfccMatch) {
    const kind = mfccMatch[1] === "_delta2" ? "MFCC ΔΔ" : mfccMatch[1] === "_delta" ? "MFCC Δ" : "MFCC";
    return `${kind} ${mfccMatch[2]}`;
  }
  return feature;
}

export function AcousticShapChart({ entries }: { entries: AcousticShapEntry[] }) {
  const data = [...entries]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .map((e) => ({ ...e, name: friendlyName(e.feature) }));

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.value)), 0.01);

  return (
    <div className="glass-panel rounded-2xl p-5">
      <h3 className="text-sm font-semibold text-foreground">Acoustic feature attribution</h3>
      <p className="mt-1 text-xs text-muted-foreground">
        KernelSHAP over named descriptors — positive pushes toward fake, negative toward authentic.
      </p>
      <div className="mt-4 h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 4, right: 16, top: 4, bottom: 4 }}>
            <XAxis
              type="number"
              domain={[-maxAbs * 1.15, maxAbs * 1.15]}
              tick={{ fill: "var(--muted-foreground)", fontSize: 10 }}
              axisLine={{ stroke: "var(--border)" }}
              tickLine={false}
              tickFormatter={(v: number) => v.toFixed(2)}
              tickCount={5}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={118}
              tick={{ fill: "var(--muted-foreground)", fontSize: 11 }}
              axisLine={false}
              tickLine={false}
            />
            <Bar dataKey="value" radius={3} barSize={12}>
              {data.map((entry) => (
                <Cell
                  key={entry.feature}
                  fill={entry.value >= 0 ? "var(--block)" : "var(--approve)"}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-center gap-4 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-block/85" /> toward fake
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-approve/85" /> toward authentic
        </span>
      </div>
    </div>
  );
}
