import Link from "next/link";
import { ShieldHalf } from "lucide-react";
import { BackendBadge } from "@/components/layout/BackendBadge";
import type { ReactNode } from "react";

export function Header({ right }: { right?: ReactNode }) {
  return (
    <header className="sticky top-0 z-30 border-b border-white/[0.06] bg-background/70 backdrop-blur-lg">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg accent-gradient-bg">
            <ShieldHalf className="h-4.5 w-4.5 text-background" strokeWidth={2.25} />
          </span>
          <span className="text-[15px] font-semibold tracking-tight text-foreground">
            Aegis
            <span className="ml-2 font-mono text-[11px] font-normal text-muted-foreground">
              moderator console
            </span>
          </span>
        </Link>
        <div className="flex items-center gap-3">
          <BackendBadge />
          {right}
        </div>
      </div>
    </header>
  );
}
