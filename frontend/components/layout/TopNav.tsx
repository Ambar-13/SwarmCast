"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
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
        <Link href="/analyze" className="mr-6 flex flex-shrink-0 items-center gap-2">
          <svg viewBox="62 6 116 110" width="32" height="32" xmlns="http://www.w3.org/2000/svg">
            {/* Antenna */}
            <line x1="120" y1="20" x2="120" y2="44" stroke="#A82020" strokeWidth="2" strokeLinecap="round"/>
            <circle cx="120" cy="14" r="6" fill="#A82020"/>
            <circle cx="120" cy="12" r="2" fill="#FAE8E8" opacity="0.6"/>
            {/* Head */}
            <rect x="80" y="44" width="80" height="68" rx="18" fill="#A82020"/>
            {/* Ear nubs */}
            <rect x="66" y="62" width="14" height="24" rx="7" fill="#A82020"/>
            <rect x="160" y="62" width="14" height="24" rx="7" fill="#8B1A1A"/>
            {/* Eyes */}
            <rect x="92" y="56" width="20" height="20" rx="6" fill="#FAF6EF"/>
            <rect x="128" y="56" width="20" height="20" rx="6" fill="#FAF6EF"/>
            <circle cx="102" cy="66" r="5.5" fill="#6B1414"/>
            <circle cx="138" cy="66" r="5.5" fill="#6B1414"/>
            <circle cx="104" cy="63" r="2" fill="white"/>
            <circle cx="140" cy="63" r="2" fill="white"/>
            {/* Mouth */}
            <rect x="96" y="88" width="48" height="11" rx="5.5" fill="#8B1A1A"/>
            <rect x="101" y="92" width="10" height="3" rx="1.5" fill="#FAE8E8" opacity="0.55"/>
            <rect x="115" y="92" width="14" height="3" rx="1.5" fill="#FAE8E8" opacity="0.55"/>
            <rect x="133" y="92" width="7" height="3" rx="1.5" fill="#FAE8E8" opacity="0.55"/>
          </svg>
          <span className="hidden text-sm font-semibold sm:block" style={{ color: "var(--ink-900)" }}>
            SwarmCast
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
