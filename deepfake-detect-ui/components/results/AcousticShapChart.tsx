"use client";

import { Bar, BarChart, Cell, ResponsiveContainer, XAxis, YAxis } from "recharts";
import type { AcousticShapEntry } from "@/lib/types";
import { friendlyName } from "@/lib/analysis/featureNames";

export function AcousticShapChart({
  entries,
  source,
}: {
  entries: AcousticShapEntry[];
  source?: string;
}) {
  // Nothing to attribute: the WavLM audio model has no named descriptors, so
  // an audio-only upload legitimately returns []. Render nothing rather than
  // an empty card -- the same choice `hasVideo` already drives for the video
  // timeline and saliency filmstrip.
  if (entries.length === 0) return null;

  const legacy = source === "legacy-acoustic-branch";

  const data = [...entries]
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .map((e) => ({ ...e, name: friendlyName(e.feature) }));

  const maxAbs = Math.max(...data.map((d) => Math.abs(d.value)), 0.01);

  return (
    <div className="glass-panel rounded-2xl p-5">
      <h3 className="t-title font-semibold text-foreground">Acoustic feature attribution</h3>
      <p className="t-support mt-1 text-muted-foreground">
        KernelSHAP over named descriptors — positive pushes toward fake, negative toward authentic.
      </p>
      {legacy && (
        <p className="mt-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-2.5 py-1.5 text-[11px] leading-snug text-amber-200/90">
          Advisory — from the legacy 68-feature acoustic branch, not the WavLM
          model that set this verdict. Read it as what that older branch keyed
          on, not as the reason for the decision.
        </p>
      )}
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
                  fill={entry.value >= 0 ? "var(--deepfake)" : "var(--authentic)"}
                  fillOpacity={0.85}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="flex items-center justify-center gap-4 text-[11px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-deepfake/85" /> toward fake
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2 w-2 rounded-sm bg-authentic/85" /> toward authentic
        </span>
      </div>
    </div>
  );
}
