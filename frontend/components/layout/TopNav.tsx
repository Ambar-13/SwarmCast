"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { ShieldCheck } from "lucide-react";
import { CommandPalette } from "@/components/ui/CommandPalette";
import { NAV_TABS } from "@/lib/nav-tabs";

export function TopNav() {
  const pathname = usePathname();
  const [paletteOpen, setPaletteOpen] = useState(false);

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, []);

  return (
    <>
      <header
        className="sticky top-0 z-40 flex h-14 items-center border-b px-4 lg:px-8"
        style={{
          borderColor: "var(--border-warm)",
          /* Animated gradient mesh — very subtle warm motion behind nav */
          background: `
            radial-gradient(ellipse 40% 80% at 20% 50%, rgba(250,240,220,0.9) 0%, transparent 70%),
            radial-gradient(ellipse 35% 70% at 75% 30%, rgba(253,250,246,0.95) 0%, transparent 70%),
            radial-gradient(ellipse 50% 100% at 50% 100%, rgba(244,237,224,0.6) 0%, transparent 70%),
            var(--cream-50)
          `,
          backgroundSize: "200% 200%",
          animation: "meshDrift 12s ease-in-out infinite",
          backdropFilter: "blur(0px)",
        }}
      >
        {/* Logo */}
        <Link href="/analyze" className="mr-6 flex flex-shrink-0 items-center gap-2.5">
          <div
            className="flex h-8 w-8 items-center justify-center rounded-xl"
            style={{ background: "var(--crimson-700)" }}
          >
            <ShieldCheck size={16} className="text-white" />
          </div>
          <span className="hidden text-sm font-semibold sm:block" style={{ color: "var(--ink-900)" }}>
            Swarmcast
          </span>
        </Link>

        {/* Tabs */}
        <nav className="flex gap-0.5">
          {NAV_TABS.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className="relative flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150"
                style={{
                  color:      active ? "var(--crimson-700)" : "var(--ink-500)",
                  background: active ? "var(--crimson-50)"  : "transparent",
                }}
              >
                <Icon size={15} />
                <span className="hidden sm:inline">{label}</span>
                {active && (
                  <span
                    className="absolute bottom-0 left-3 right-3 h-0.5 rounded-full"
                    style={{ background: "var(--crimson-700)" }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        {/* Right — search trigger */}
        <div className="ml-auto">
          <button
            onClick={() => setPaletteOpen(true)}
            className="hidden items-center gap-2 rounded-lg border px-3 py-1.5 text-xs transition-colors hover:bg-[var(--cream-200)] lg:flex"
            style={{
              color:       "var(--ink-400)",
              borderColor: "var(--border-warm)",
              background:  "var(--cream-100)",
            }}
          >
            <span>Search…</span>
            <kbd
              className="rounded border px-1 py-0.5 text-[10px]"
              style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
            >
              ⌘K
            </kbd>
          </button>
        </div>
      </header>

      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  );
}
