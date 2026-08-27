import { CircleCheck, ShieldAlert, TriangleAlert } from "lucide-react";
import type { Verdict } from "@/lib/types";
import { VERDICT_LABEL } from "@/lib/analysis/localization";
import { cn } from "@/lib/utils";

const CONFIG: Record<Verdict, { icon: typeof CircleCheck; className: string }> = {
  deepfake: {
    icon: ShieldAlert,
    className: "border-deepfake/45 bg-deepfake/12 text-deepfake",
  },
  uncertain: {
    icon: TriangleAlert,
    className: "border-uncertain/45 bg-uncertain/12 text-uncertain",
  },
  authentic: {
    icon: CircleCheck,
    className: "border-authentic/45 bg-authentic/12 text-authentic",
  },
};

export function VerdictBadge({
  verdict,
  size = "md",
  className,
}: {
  verdict: Verdict;
  size?: "sm" | "md" | "lg";
  className?: string;
}) {
  const { icon: Icon, className: variantClassName } = CONFIG[verdict];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-2 rounded-full border font-semibold",
        size === "sm" && "px-2.5 py-1 text-xs",
        size === "md" && "px-3.5 py-1.5 text-sm",
        size === "lg" && "px-4 py-2 text-base",
        variantClassName,
        className
      )}
    >
      <Icon className={cn(size === "lg" ? "h-5 w-5" : "h-4 w-4")} />
      {VERDICT_LABEL[verdict]}
    </span>
  );
}
