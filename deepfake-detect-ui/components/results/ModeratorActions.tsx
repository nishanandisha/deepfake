"use client";

import { useState } from "react";
import { toast } from "sonner";
import { CheckCircle2, OctagonAlert, ShieldAlert } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Decision } from "@/lib/types";
import { cn } from "@/lib/utils";

const ACTIONS: { decision: Decision; label: string; icon: typeof CheckCircle2; className: string }[] = [
  { decision: "approve", label: "Approve", icon: CheckCircle2, className: "hover:border-approve/50 hover:text-approve" },
  { decision: "flag", label: "Escalate", icon: ShieldAlert, className: "hover:border-flag/50 hover:text-flag" },
  { decision: "block", label: "Block", icon: OctagonAlert, className: "hover:border-block/50 hover:text-block" },
];

export function ModeratorActions({ sampleId, systemDecision }: { sampleId: string; systemDecision: Decision }) {
  const [override, setOverride] = useState<Decision | null>(null);

  const handle = (decision: Decision) => {
    setOverride(decision);
    const isOverride = decision !== systemDecision;
    toast(isOverride ? "Decision overridden" : "Decision confirmed", {
      description: `Sample ${sampleId.slice(0, 18)} marked as "${decision}"${isOverride ? " (moderator override)" : ""}.`,
    });
  };

  return (
    <div className="glass-panel flex flex-col gap-3 rounded-2xl p-5 sm:flex-row sm:items-center sm:justify-between">
      <div>
        <h3 className="text-sm font-semibold text-foreground">Moderator action</h3>
        <p className="mt-0.5 text-xs text-muted-foreground">
          {override ? `Recorded: ${override}` : "Confirm or override the system decision."}
        </p>
      </div>
      <div className="flex gap-2">
        {ACTIONS.map(({ decision, label, icon: Icon, className }) => (
          <Button
            key={decision}
            variant="outline"
            onClick={() => handle(decision)}
            className={cn(
              "gap-1.5 border-white/10 bg-white/[0.02] text-foreground",
              className,
              override === decision && "border-cyan/50 bg-cyan/5 text-cyan"
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
