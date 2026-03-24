"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart2, Scale, Upload, Zap } from "lucide-react";

const TABS = [
  { href: "/analyze",   label: "Analyze",  icon: BarChart2 },
  { href: "/upload",    label: "Upload",   icon: Upload    },
  { href: "/compare",   label: "Compare",  icon: Scale     },
  { href: "/influence", label: "Influence",icon: Zap       },
] as const;

export function MobileNav() {
  const pathname = usePathname();

  return (
    <nav
      className="fixed bottom-0 left-0 right-0 z-40 flex h-16 items-center border-t bg-[var(--cream-50)] px-2 lg:hidden"
      style={{ borderColor: "var(--border-warm)" }}
    >
      {TABS.map(({ href, label, icon: Icon }) => {
        const active = pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            className="flex flex-1 flex-col items-center justify-center gap-0.5 rounded-xl py-1 text-[10px] font-medium transition-colors"
            style={{ color: active ? "var(--crimson-700)" : "var(--ink-400)" }}
          >
            <Icon size={20} />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
