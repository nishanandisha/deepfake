"use client";

import { useState } from "react";
import { toast } from "sonner";
import { CircleCheck, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { SegmentJudgement, TamperSegment, Verdict } from "@/lib/types";
import { VERDICT_LABEL } from "@/lib/analysis/localization";
import { cn } from "@/lib/utils";

type FinalCall = "deepfake" | "authentic";

const ACTIONS: { call: FinalCall; label: string; icon: typeof CircleCheck; active: string; idle: string }[] = [
  {
    call: "deepfake",
    label: "It's a deepfake",
    icon: ShieldAlert,
    active: "border-deepfake/60 bg-deepfake/15 text-deepfake",
    idle: "hover:border-deepfake/50 hover:text-deepfake",
  },
  {
    call: "authentic",
    label: "It's genuine",
    icon: CircleCheck,
    active: "border-authentic/60 bg-authentic/15 text-authentic",
    idle: "hover:border-authentic/50 hover:text-authentic",
  },
];

export function ReviewDecision({
  sampleId,
  systemVerdict,
  segments,
  judgements,
}: {
  sampleId: string;
  systemVerdict: Verdict;
  segments: TamperSegment[];
  judgements: Record<string, SegmentJudgement>;
}) {
  const [finalCall, setFinalCall] = useState<FinalCall | null>(null);

  const confirmed = segments.filter((s) => judgements[s.id] === "agree").length;
  const rejected = segments.filter((s) => judgements[s.id] === "disagree").length;
  const reviewed = confirmed + rejected;

  const handle = (call: FinalCall) => {
    setFinalCall(call);
    const agreesWithModel =
      (call === "deepfake" && systemVerdict === "deepfake") ||
      (call === "authentic" && systemVerdict === "authentic");
    toast(agreesWithModel ? "Model confirmed" : "Model overridden", {
      description: `${sampleId.slice(0, 18)} recorded as ${
        call === "deepfake" ? "a deepfake" : "genuine"
      }${reviewed > 0 ? ` · ${confirmed}/${segments.length} regions confirmed` : ""}.`,
    });
  };

  return (
    <div className="glass-panel flex flex-col gap-4 rounded-2xl p-5 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <h3 className="text-sm font-semibold text-foreground">Your verdict</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {finalCall
            ? `Recorded as ${finalCall === "deepfake" ? "a deepfake" : "genuine"}.`
            : `The model says "${VERDICT_LABEL[systemVerdict]}". Confirm it or override it.`}
        </p>
        {segments.length > 0 && (
          <p className="mt-1.5 font-mono text-[11px] text-muted-foreground">
            {reviewed === 0
              ? `${segments.length} region${segments.length === 1 ? "" : "s"} awaiting your review`
              : `${confirmed} confirmed · ${rejected} rejected · ${segments.length - reviewed} unreviewed`}
          </p>
        )}
      </div>

      <div className="flex shrink-0 gap-2">
        {ACTIONS.map(({ call, label, icon: Icon, active, idle }) => (
          <Button
            key={call}
            variant="outline"
            onClick={() => handle(call)}
            className={cn(
              "gap-1.5 border-border bg-secondary/30 text-foreground",
              finalCall === call ? active : idle
            )}
          >
            <Icon className="h-3.5 w-3.5" />
            {label}
          </Button>
        ))}
      </div>
    </div>
  );
}
