import { CheckCircle2, OctagonAlert, ShieldAlert } from "lucide-react";
import type { Decision } from "@/lib/types";
import { cn } from "@/lib/utils";

const CONFIG: Record<Decision, { label: string; icon: typeof CheckCircle2; className: string }> = {
  approve: {
    label: "Approve",
    icon: CheckCircle2,
    className: "border-approve/40 bg-approve/10 text-approve",
  },
  flag: {
    label: "Flag for review",
    icon: ShieldAlert,
    className: "border-flag/40 bg-flag/10 text-flag",
  },
  block: {
    label: "Block",
    icon: OctagonAlert,
    className: "border-block/40 bg-block/10 text-block",
  },
};

export function DecisionBadge({ decision, className }: { decision: Decision; className?: string }) {
  const { label, icon: Icon, className: variantClassName } = CONFIG[decision];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border px-3.5 py-1.5 text-sm font-medium",
        variantClassName,
        className
      )}
    >
      <Icon className="h-4 w-4" />
      {label}
    </span>
  );
}
