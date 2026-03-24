# Warm Paper UI Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the dark glassmorphism theme with an Awwwards-quality warm cream + dark crimson editorial UI, add count-up animations, bento grid results layout, 5 new simulation components, top navigation, and mobile nav — all in strict TypeScript with no `any` types.

**Architecture:** Rewrite `globals.css` CSS variables first (everything downstream inherits them), then replace `Sidebar` → `TopNav`, rebuild all components with warm tokens, add new viz components, overhaul all 4 pages.

**Tech Stack:** Next.js 14 App Router, React 18, TanStack Query 5, Zustand 4.5, Recharts 2.12, Tailwind CSS 3.4, TypeScript 5.4 strict, JetBrains Mono + Inter via next/font.

**Working directory:** `/Users/ambar/Downloads/policylab_m3/frontend/`

---

## Phase 1 — Design Foundation

### Task 1: Rewrite `globals.css` — warm color tokens + component classes

**Files:**
- Modify: `app/globals.css`

**Step 1: Replace the entire file**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  /* Surfaces */
  --cream-50:   #FDFAF6;
  --cream-100:  #FAF6EF;
  --cream-200:  #F4EDE0;
  --cream-300:  #E8DDD0;
  --cream-400:  #D8CBBF;

  /* Text */
  --ink-900:    #1A1208;
  --ink-700:    #3D2E1E;
  --ink-500:    #6B5744;
  --ink-400:    #8C7B68;
  --ink-300:    #A69485;

  /* Accent — dark crimson */
  --crimson-700: #8B1A1A;
  --crimson-600: #A82020;
  --crimson-500: #C0392B;
  --crimson-100: #FAE8E8;
  --crimson-50:  #FDF4F4;

  /* Semantic */
  --success:    #15803D;
  --warning:    #B45309;
  --danger:     #DC2626;

  /* Chart palette */
  --chart-compliance:  #8B1A1A;
  --chart-relocation:  #C2410C;
  --chart-investment:  #1D6344;
  --chart-enforcement: #1E3A8A;
  --chart-evasion:     #6B21A8;

  /* Borders */
  --border-warm:   rgba(60, 40, 20, 0.10);
  --border-strong: rgba(60, 40, 20, 0.20);

  /* Shadows */
  --shadow-card:   0 1px 3px rgba(26,18,8,0.06), 0 4px 16px rgba(26,18,8,0.04);
  --shadow-raised: 0 2px 8px rgba(26,18,8,0.08), 0 8px 32px rgba(26,18,8,0.06);
  --shadow-focus:  0 0 0 3px rgba(139,26,26,0.18);
}

html {
  color-scheme: light;
}

body {
  color: var(--ink-900);
  background-color: var(--cream-50);
  font-family: var(--font-inter), sans-serif;
}

* {
  scrollbar-color: rgba(139,100,80,0.28) transparent;
}
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
  background: rgba(139,100,80,0.25);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
::selection {
  background: rgba(139,26,26,0.16);
  color: var(--ink-900);
}

@layer base {
  h1, h2, h3, h4 { letter-spacing: -0.03em; }
}

@layer components {
  /* Cards */
  .card-warm {
    @apply rounded-2xl border bg-[var(--cream-100)];
    border-color: var(--border-warm);
    box-shadow: var(--shadow-card);
  }
  .card-raised {
    @apply rounded-2xl border bg-[var(--cream-100)];
    border-color: var(--border-warm);
    box-shadow: var(--shadow-raised);
  }
  .card-tinted {
    @apply rounded-2xl border bg-[var(--crimson-50)];
    border-color: var(--crimson-100);
    box-shadow: var(--shadow-card);
  }

  /* Buttons */
  .btn-primary {
    @apply inline-flex items-center justify-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold text-white transition-all duration-150 active:scale-[0.97] disabled:cursor-not-allowed disabled:opacity-50;
    background: var(--crimson-700);
  }
  .btn-primary:hover:not(:disabled) { background: var(--crimson-600); }

  .btn-secondary {
    @apply inline-flex items-center justify-center gap-2 rounded-xl border px-4 py-2.5 text-sm font-semibold transition-all duration-150 active:scale-[0.97];
    background: var(--cream-200);
    border-color: var(--border-warm);
    color: var(--ink-700);
  }
  .btn-secondary:hover { background: var(--cream-300); }

  /* Form inputs */
  .input-warm {
    @apply w-full rounded-xl border px-3 py-2 text-sm outline-none transition-all duration-150;
    background: var(--cream-200);
    border-color: var(--border-warm);
    color: var(--ink-900);
  }
  .input-warm::placeholder { color: var(--ink-400); }
  .input-warm:focus {
    border-color: var(--crimson-700);
    box-shadow: var(--shadow-focus);
  }
  .select-warm {
    @apply input-warm cursor-pointer;
  }

  /* Tag / badge */
  .tag-badge {
    @apply inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] font-semibold;
  }

  /* Typography helpers */
  .kicker {
    @apply text-[11px] font-semibold uppercase tracking-[0.22em];
    color: var(--ink-400);
  }
  .metric-num {
    font-family: var(--font-jetbrains-mono), monospace;
    font-variant-numeric: tabular-nums;
  }
  /* Backward compat aliases for existing components */
  .section-kicker { @apply kicker; }
  .metric-value   { @apply metric-num; }
  .data-card      { @apply card-warm p-4; }
  .panel-shell    { @apply card-raised; }
  .panel-shell-soft { @apply card-warm; }

  /* Divider */
  .divider-warm { @apply border-t; border-color: var(--border-warm); }
}

/* Animations */
@layer utilities {
  .text-balance { text-wrap: balance; }
  .font-mono-num { @apply metric-num; }
}

@keyframes slideUpFade {
  from { opacity: 0; transform: translateY(12px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmerWarm {
  0%   { background-position: -400px 0; }
  100% { background-position:  400px 0; }
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -30
```
Expected: 0 errors (CSS changes don't affect TS)

**Step 3: Commit**
```bash
git add app/globals.css
git commit -m "style: replace dark theme with warm paper design tokens"
```

---

### Task 2: Build `TopNav.tsx` — sticky warm top navigation

**Files:**
- Create: `components/layout/TopNav.tsx`

**Step 1: Write the component**

```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart2, Scale, Upload, Zap, ShieldCheck } from "lucide-react";

const NAV_TABS = [
  { href: "/analyze",  label: "Analyze",  icon: BarChart2 },
  { href: "/upload",   label: "Upload",   icon: Upload    },
  { href: "/compare",  label: "Compare",  icon: Scale     },
  { href: "/influence",label: "Influence",icon: Zap       },
] as const;

export function TopNav() {
  const pathname = usePathname();

  return (
    <header
      className="sticky top-0 z-40 flex h-14 items-center border-b bg-[var(--cream-50)] px-4 lg:px-8"
      style={{ borderColor: "var(--border-warm)" }}
    >
      {/* Logo */}
      <Link href="/analyze" className="mr-8 flex items-center gap-2.5 flex-shrink-0">
        <div
          className="flex h-8 w-8 items-center justify-center rounded-xl"
          style={{ background: "var(--crimson-700)" }}
        >
          <ShieldCheck size={16} className="text-white" />
        </div>
        <span className="text-sm font-semibold" style={{ color: "var(--ink-900)" }}>
          PolicyLab
        </span>
      </Link>

      {/* Tab nav */}
      <nav className="flex gap-1">
        {NAV_TABS.map(({ href, label, icon: Icon }) => {
          const active = pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className="relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150"
              style={{
                color: active ? "var(--crimson-700)" : "var(--ink-500)",
                background: active ? "var(--crimson-50)" : "transparent",
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

      <div className="ml-auto flex items-center gap-2">
        <kbd
          className="hidden rounded border px-2 py-0.5 text-[11px] lg:inline-flex"
          style={{
            color: "var(--ink-400)",
            borderColor: "var(--border-warm)",
            background: "var(--cream-200)",
          }}
        >
          ⌘K
        </kbd>
      </div>
    </header>
  );
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```
Expected: 0 errors

**Step 3: Commit**
```bash
git add components/layout/TopNav.tsx
git commit -m "feat: add TopNav sticky header with warm theme"
```

---

### Task 3: Build `MobileNav.tsx` — bottom tab bar for small screens

**Files:**
- Create: `components/layout/MobileNav.tsx`

**Step 1: Write the component**

```tsx
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
            style={{
              color: active ? "var(--crimson-700)" : "var(--ink-400)",
            }}
          >
            <Icon size={20} />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 3: Commit**
```bash
git add components/layout/MobileNav.tsx
git commit -m "feat: add MobileNav bottom tab bar"
```

---

### Task 4: Rewrite `AppShell.tsx` — use TopNav, remove Sidebar

**Files:**
- Modify: `components/layout/AppShell.tsx`
- Delete: `components/layout/Sidebar.tsx` (after AppShell update)

**Step 1: Rewrite AppShell**

```tsx
import { TopNav } from "@/components/layout/TopNav";
import { MobileNav } from "@/components/layout/MobileNav";

interface Props {
  children: React.ReactNode;
}

export function AppShell({ children }: Props) {
  return (
    <div className="min-h-screen" style={{ background: "var(--cream-50)" }}>
      <TopNav />
      <main className="pb-20 lg:pb-8">{children}</main>
      <MobileNav />
    </div>
  );
}
```

**Step 2: Delete Sidebar.tsx**
```bash
rm /Users/ambar/Downloads/policylab_m3/frontend/components/layout/Sidebar.tsx
```

**Step 3: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```
Expected: 0 errors (Sidebar was not imported anywhere except AppShell)

**Step 4: Commit**
```bash
git add components/layout/AppShell.tsx
git rm components/layout/Sidebar.tsx
git commit -m "refactor: replace Sidebar with TopNav in AppShell"
```

---

## Phase 2 — UI Primitives

### Task 5: Build `AnimatedCounter.tsx` — count-up hook + component

**Files:**
- Create: `components/ui/AnimatedCounter.tsx`

**Step 1: Write the component**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

export function useCountUp(
  target: number,
  duration = 900,
  formatter: (v: number) => string = (v) => v.toFixed(0),
): string {
  const [display, setDisplay] = useState(formatter(0));
  const startRef = useRef<number | null>(null);
  const frameRef = useRef<number>(0);
  const prevTargetRef = useRef<number>(0);

  useEffect(() => {
    const from = prevTargetRef.current;
    prevTargetRef.current = target;
    startRef.current = null;

    function step(ts: number) {
      if (startRef.current === null) startRef.current = ts;
      const elapsed = ts - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const value = from + (target - from) * easeOutCubic(progress);
      setDisplay(formatter(value));
      if (progress < 1) {
        frameRef.current = requestAnimationFrame(step);
      }
    }

    cancelAnimationFrame(frameRef.current);
    frameRef.current = requestAnimationFrame(step);
    return () => cancelAnimationFrame(frameRef.current);
  }, [target, duration, formatter]);

  return display;
}

interface Props {
  value: number;
  duration?: number;
  formatter?: (v: number) => string;
  className?: string;
}

export function AnimatedCounter({ value, duration = 900, formatter, className }: Props) {
  const display = useCountUp(value, duration, formatter);
  return <span className={className}>{display}</span>;
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 3: Commit**
```bash
git add components/ui/AnimatedCounter.tsx
git commit -m "feat: add AnimatedCounter with easeOutCubic count-up"
```

---

### Task 6: Build `CommandPalette.tsx` — Cmd+K overlay

**Files:**
- Create: `components/ui/CommandPalette.tsx`

**Step 1: Write the component**

```tsx
"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { fetchPresets } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { BarChart2, Scale, Search, Upload, Zap } from "lucide-react";

const PAGES = [
  { label: "Analyze policy", href: "/analyze",   icon: BarChart2 },
  { label: "Upload document", href: "/upload",   icon: Upload    },
  { label: "Compare policies", href: "/compare", icon: Scale     },
  { label: "Influence test", href: "/influence", icon: Zap       },
] as const;

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
}

export function CommandPalette({ open, onClose }: CommandPaletteProps) {
  const router = useRouter();
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const { data } = useQuery({
    queryKey: queryKeys.presets(),
    queryFn: fetchPresets,
    staleTime: Infinity,
  });

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setQuery("");
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Close on Escape
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!open) return null;

  const q = query.toLowerCase();

  const matchedPages = PAGES.filter((p) => p.label.toLowerCase().includes(q));
  const matchedPresets = (data?.presets ?? []).filter(
    (p) => p.label.toLowerCase().includes(q) || p.slug.includes(q),
  );

  function navigate(href: string) {
    router.push(href);
    onClose();
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-24"
      style={{ background: "rgba(26,18,8,0.4)", backdropFilter: "blur(4px)" }}
      onClick={onClose}
    >
      <div
        className="card-raised w-full max-w-lg overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search input */}
        <div
          className="flex items-center gap-3 border-b px-4 py-3"
          style={{ borderColor: "var(--border-warm)" }}
        >
          <Search size={16} style={{ color: "var(--ink-400)" }} />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search pages or presets…"
            className="flex-1 bg-transparent text-sm outline-none"
            style={{ color: "var(--ink-900)" }}
          />
          <kbd
            className="rounded border px-1.5 py-0.5 text-[10px]"
            style={{
              color: "var(--ink-400)",
              borderColor: "var(--border-warm)",
              background: "var(--cream-200)",
            }}
          >
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div className="max-h-96 overflow-y-auto p-2">
          {matchedPages.length > 0 && (
            <div className="mb-2">
              <p className="kicker px-2 py-1.5">Pages</p>
              {matchedPages.map(({ href, label, icon: Icon }) => (
                <button
                  key={href}
                  onClick={() => navigate(href)}
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors hover:bg-[var(--cream-200)]"
                  style={{ color: "var(--ink-700)" }}
                >
                  <Icon size={15} style={{ color: "var(--crimson-700)" }} />
                  {label}
                </button>
              ))}
            </div>
          )}
          {matchedPresets.length > 0 && (
            <div>
              <p className="kicker px-2 py-1.5">Presets</p>
              {matchedPresets.map((p) => (
                <button
                  key={p.slug}
                  onClick={() => navigate(`/analyze`)}
                  className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition-colors hover:bg-[var(--cream-200)]"
                  style={{ color: "var(--ink-700)" }}
                >
                  <span
                    className="h-2 w-2 flex-shrink-0 rounded-full"
                    style={{ background: "var(--crimson-700)" }}
                  />
                  {p.label}
                </button>
              ))}
            </div>
          )}
          {matchedPages.length === 0 && matchedPresets.length === 0 && (
            <p className="px-3 py-6 text-center text-sm" style={{ color: "var(--ink-400)" }}>
              No results for &ldquo;{query}&rdquo;
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 2: Wire Cmd+K in TopNav** — add to `components/layout/TopNav.tsx`:

Replace the existing file with this updated version that adds the palette:

```tsx
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { BarChart2, Scale, Upload, Zap, ShieldCheck } from "lucide-react";
import { CommandPalette } from "@/components/ui/CommandPalette";

const NAV_TABS = [
  { href: "/analyze",   label: "Analyze",  icon: BarChart2 },
  { href: "/upload",    label: "Upload",   icon: Upload    },
  { href: "/compare",   label: "Compare",  icon: Scale     },
  { href: "/influence", label: "Influence",icon: Zap       },
] as const;

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
        className="sticky top-0 z-40 flex h-14 items-center border-b bg-[var(--cream-50)] px-4 lg:px-8"
        style={{ borderColor: "var(--border-warm)" }}
      >
        <Link href="/analyze" className="mr-8 flex items-center gap-2.5 flex-shrink-0">
          <div
            className="flex h-8 w-8 items-center justify-center rounded-xl"
            style={{ background: "var(--crimson-700)" }}
          >
            <ShieldCheck size={16} className="text-white" />
          </div>
          <span className="hidden text-sm font-semibold sm:block" style={{ color: "var(--ink-900)" }}>
            PolicyLab
          </span>
        </Link>

        <nav className="flex gap-1">
          {NAV_TABS.map(({ href, label, icon: Icon }) => {
            const active = pathname.startsWith(href);
            return (
              <Link
                key={href}
                href={href}
                className="relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors duration-150"
                style={{
                  color: active ? "var(--crimson-700)" : "var(--ink-500)",
                  background: active ? "var(--crimson-50)" : "transparent",
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

        <div className="ml-auto">
          <button
            onClick={() => setPaletteOpen(true)}
            className="hidden items-center gap-2 rounded-lg border px-3 py-1.5 text-xs transition-colors hover:bg-[var(--cream-200)] lg:flex"
            style={{ color: "var(--ink-400)", borderColor: "var(--border-warm)", background: "var(--cream-100)" }}
          >
            <span>Search…</span>
            <kbd className="rounded border px-1 py-0.5 text-[10px]"
              style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}>
              ⌘K
            </kbd>
          </button>
        </div>
      </header>
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />
    </>
  );
}
```

**Step 3: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 4: Commit**
```bash
git add components/ui/CommandPalette.tsx components/layout/TopNav.tsx
git commit -m "feat: add CommandPalette (Cmd+K) wired into TopNav"
```

---

## Phase 3 — Form Components (Warm Theme)

### Task 7: Restyle `PresetSelector.tsx`, `SimConfigPanel.tsx`, `EpistemicBadge.tsx`

**Files:**
- Modify: `components/policy/PresetSelector.tsx`
- Modify: `components/simulation/SimConfigPanel.tsx`
- Modify: `components/simulation/EpistemicBadge.tsx`

**Step 1: Rewrite PresetSelector.tsx**

```tsx
"use client";

import { useQuery } from "@tanstack/react-query";
import { fetchPresets } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import type { PresetPolicy } from "@/lib/types";

interface Props {
  onSelect: (preset: PresetPolicy) => void;
  selectedSlug?: string;
}

export function PresetSelector({ onSelect, selectedSlug }: Props) {
  const { data, isLoading, error } = useQuery({
    queryKey: queryKeys.presets(),
    queryFn: fetchPresets,
    staleTime: Infinity,
  });

  if (isLoading) {
    return <div className="kicker animate-pulse">Loading presets…</div>;
  }
  if (error || !data) {
    return <div className="text-xs" style={{ color: "var(--danger)" }}>Failed to load presets</div>;
  }

  return (
    <div>
      <label className="kicker mb-1.5 block">Load a preset</label>
      <select
        className="select-warm"
        value={selectedSlug ?? ""}
        onChange={(e) => {
          const preset = data.presets.find((p) => p.slug === e.target.value);
          if (preset) onSelect(preset);
        }}
      >
        <option value="">Select a starting policy…</option>
        {data.presets.map((p) => (
          <option key={p.slug} value={p.slug}>{p.label}</option>
        ))}
      </select>
    </div>
  );
}
```

**Step 2: Rewrite SimConfigPanel.tsx**

```tsx
"use client";

import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { SimConfigRequest } from "@/lib/types";

interface Props {
  config: SimConfigRequest;
  onChange: (patch: Partial<SimConfigRequest>) => void;
}

function Row({ label, description, children }: {
  label: string; description: string; children: React.ReactNode;
}) {
  return (
    <div
      className="grid gap-2 rounded-xl border px-3 py-3 md:grid-cols-[1fr_auto] md:items-center"
      style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
    >
      <div>
        <p className="text-sm font-medium" style={{ color: "var(--ink-700)" }}>{label}</p>
        <p className="mt-0.5 text-xs leading-5" style={{ color: "var(--ink-400)" }}>{description}</p>
      </div>
      {children}
    </div>
  );
}

function NumInput({ value, min, max, step, onChange }: {
  value: number; min?: number; max?: number; step?: number;
  onChange: (v: number) => void;
}) {
  return (
    <input
      type="number"
      value={value}
      min={min}
      max={max}
      step={step}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="input-warm w-28"
    />
  );
}

export function SimConfigPanel({ config, onChange }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="card-warm overflow-hidden">
      <button
        className="flex w-full items-center gap-2 px-4 py-3 text-sm font-medium transition-colors"
        style={{ color: open ? "var(--ink-700)" : "var(--ink-400)" }}
        onClick={() => setOpen((o) => !o)}
      >
        {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
        Advanced simulation config
      </button>
      {open && (
        <div className="space-y-2 border-t px-4 pb-4 pt-3" style={{ borderColor: "var(--border-warm)" }}>
          <Row label="Population size" description="Number of simulated actors.">
            <NumInput value={config.n_population} min={100} max={20000} step={100}
              onChange={(v) => onChange({ n_population: v })} />
          </Row>
          <Row label="Rounds" description="Decision cycles the simulation runs.">
            <NumInput value={config.num_rounds} min={1} max={64} step={1}
              onChange={(v) => onChange({ num_rounds: v })} />
          </Row>
          <Row label="Seed" description="Deterministic seed for reproducibility.">
            <NumInput value={config.seed} min={0} step={1}
              onChange={(v) => onChange({ seed: v })} />
          </Row>
          <Row label="Spillover factor" description="How strongly effects propagate across the system.">
            <NumInput value={config.spillover_factor} min={0} max={2} step={0.05}
              onChange={(v) => onChange({ spillover_factor: v })} />
          </Row>
          <Row label="Compute cost factor" description="Relative compute burden on covered entities.">
            <NumInput value={config.compute_cost_factor} min={0} max={10} step={0.1}
              onChange={(v) => onChange({ compute_cost_factor: v })} />
          </Row>
          <Row label="HK epsilon" description="Sensitivity term in population response dynamics.">
            <NumInput value={config.hk_epsilon} min={0} max={2} step={0.05}
              onChange={(v) => onChange({ hk_epsilon: v })} />
          </Row>
        </div>
      )}
    </div>
  );
}
```

**Step 3: Rewrite EpistemicBadge.tsx**

First read what EpistemicBadge currently looks like:
```bash
cat /Users/ambar/Downloads/policylab_m3/frontend/components/simulation/EpistemicBadge.tsx
```

Then replace with warm colors:

```tsx
import type { EpistemicTier } from "@/lib/types";

const TIER_STYLES: Record<EpistemicTier, { color: string; bg: string; border: string }> = {
  GROUNDED:    { color: "#15803D", bg: "#f0fdf4", border: "rgba(21,128,61,0.2)"   },
  DIRECTIONAL: { color: "#B45309", bg: "#fffbeb", border: "rgba(180,83,9,0.2)"    },
  ASSUMED:     { color: "#DC2626", bg: "#fef2f2", border: "rgba(220,38,38,0.2)"   },
};

interface Props {
  tier: EpistemicTier;
  small?: boolean;
}

export function EpistemicBadge({ tier, small = false }: Props) {
  const s = TIER_STYLES[tier];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border font-semibold ${small ? "px-2 py-0.5 text-[10px]" : "px-2.5 py-1 text-xs"}`}
      style={{ color: s.color, background: s.bg, borderColor: s.border }}
    >
      <span className="h-1.5 w-1.5 rounded-full flex-shrink-0" style={{ background: s.color }} />
      {tier}
    </span>
  );
}
```

**Step 4: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 5: Commit**
```bash
git add components/policy/PresetSelector.tsx components/simulation/SimConfigPanel.tsx components/simulation/EpistemicBadge.tsx
git commit -m "style: restyle form + config + epistemic badge to warm theme"
```

---

### Task 8: Restyle `PolicyParamPanel.tsx` and `EvidencePackSection.tsx`

**Files:**
- Modify: `components/policy/PolicyParamPanel.tsx`
- Modify: `components/simulation/EvidencePackSection.tsx`

**Step 1: Update PolicyParamPanel.tsx** — replace all dark CSS variable references with warm equivalents. Key replacements:
- `var(--border-subtle)` → `var(--border-warm)`
- `bg-[rgba(255,255,255,0.03)]` → `bg-[var(--cream-200)]`
- `text-[var(--ink-100)]` → keep using `ink` variables (they're redefined as warm now)
- `focus:border-[rgba(91,124,255,0.42)]` → `focus:border-[var(--crimson-700)]`
- `bg-[rgba(255,255,255,0.02)]` → `bg-[var(--cream-100)]`

Replace `InputBase` className:
```tsx
className={`w-full rounded-xl border px-3 py-2 text-sm outline-none transition-all focus:border-[var(--crimson-700)] focus:shadow-[var(--shadow-focus)] ${className ?? ""}`}
style={{ background: "var(--cream-200)", borderColor: "var(--border-warm)", color: "var(--ink-900)" }}
```

Replace `SelectBase` className — same pattern.

Replace `StaticValue`:
```tsx
<div className="rounded-xl border px-3 py-2 text-sm"
  style={{ background: "var(--cream-100)", borderColor: "var(--border-warm)", color: "var(--ink-700)" }}>
  {children}
</div>
```

Replace `Label`:
```tsx
<label className="kicker mb-1.5 block">{children}</label>
```

Replace the outer grid `bg-[rgba(255,255,255,0.02)]` wrapping the SeverityGauge:
```tsx
className="flex items-center justify-center rounded-2xl border p-2"
style={{ borderColor: "var(--border-warm)", background: "var(--cream-200)" }}
```

**Step 2: Rewrite EvidencePackSection.tsx**

```tsx
"use client";

import { useState, useEffect } from "react";
import { startEvidencePack, pollEvidencePack } from "@/lib/api-client";
import { useAnalyzeStore } from "@/lib/store";
import type { EvidencePackRequest } from "@/lib/types";

export function EvidencePackSection() {
  const store = useAnalyzeStore();
  const [jobId, setJobId] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId || status === "complete" || status === "error") return;
    const interval = setInterval(async () => {
      try {
        const job = await pollEvidencePack(jobId);
        setProgress(job.progress);
        setStatus(job.status);
        if (job.status === "complete" && job.result) {
          store.setEvidencePackResult(job.result);
          clearInterval(interval);
        } else if (job.status === "error") {
          setError(job.error ?? "Unknown error");
          clearInterval(interval);
        }
      } catch { clearInterval(interval); }
    }, 1200);
    return () => clearInterval(interval);
  }, [jobId, status, store]);

  async function handleStart() {
    setError(null); setProgress(0); setStatus("queued");
    const req: EvidencePackRequest = {
      policy_name: store.policyName,
      policy_description: store.policyDescription,
      base_severity: store.policySeverity,
      sim_config: {
        n_population: store.simConfig.n_population,
        num_rounds: store.simConfig.num_rounds,
        seed: store.simConfig.seed,
        compute_cost_factor: store.simConfig.compute_cost_factor,
      },
      ensemble_size: 3,
    };
    try {
      const { job_id } = await startEvidencePack(req);
      setJobId(job_id);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus(null);
    }
  }

  const hasResult = Boolean(store.evidencePackResult);

  return (
    <div className="card-warm p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-sm font-semibold" style={{ color: "var(--ink-700)" }}>
            Confidence bands
          </p>
          <p className="mt-0.5 text-xs leading-5" style={{ color: "var(--ink-400)" }}>
            9 ensemble runs (3 severity × 3 seeds) applied to the chart.
          </p>
        </div>
        {!hasResult && status !== "running" && status !== "queued" && (
          <button onClick={handleStart} className="btn-primary flex-shrink-0 px-3 py-1.5 text-xs">
            Build →
          </button>
        )}
        {hasResult && (
          <span className="text-xs font-medium" style={{ color: "var(--success)" }}>
            ✓ Applied
          </span>
        )}
      </div>

      {(status === "running" || status === "queued") && (
        <div className="space-y-1.5">
          <div className="flex justify-between text-[11px]" style={{ color: "var(--ink-400)" }}>
            <span>{status === "queued" ? "Queued…" : "Running ensemble…"}</span>
            <span>{(progress * 100).toFixed(0)}%</span>
          </div>
          <div className="h-1.5 overflow-hidden rounded-full" style={{ background: "var(--cream-300)" }}>
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${progress * 100}%`, background: "var(--crimson-700)" }}
            />
          </div>
        </div>
      )}
      {error && (
        <p className="text-xs" style={{ color: "var(--danger)" }}>{error}</p>
      )}
    </div>
  );
}
```

**Step 3: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 4: Commit**
```bash
git add components/policy/PolicyParamPanel.tsx components/simulation/EvidencePackSection.tsx
git commit -m "style: warm theme for PolicyParamPanel + EvidencePackSection"
```

---

## Phase 4 — New Simulation Components

### Task 9: Build `PopulationSummaryBar.tsx` — stacked horizontal bar

**Files:**
- Create: `components/simulation/PopulationSummaryBar.tsx`

**Step 1: Write the component**

```tsx
import type { AnimatedCounter } from "@/components/ui/AnimatedCounter";
import { formatPct } from "@/lib/format";

interface Props {
  complianceRate: number;
  relocationRate: number;
  evasionRate: number;
  everLobbyedRate: number;
}

interface Segment {
  label: string;
  value: number;
  color: string;
  bg: string;
}

export function PopulationSummaryBar({
  complianceRate,
  relocationRate,
  evasionRate,
  everLobbyedRate,
}: Props) {
  const segments: Segment[] = [
    { label: "Compliant",  value: complianceRate,  color: "#15803D", bg: "#f0fdf4" },
    { label: "Relocated",  value: relocationRate,  color: "#B45309", bg: "#fffbeb" },
    { label: "Evading",    value: evasionRate,     color: "#DC2626", bg: "#fef2f2" },
    { label: "Lobbied",    value: everLobbyedRate, color: "#1E3A8A", bg: "#eff6ff" },
  ];

  const total = segments.reduce((s, seg) => s + seg.value, 0);
  const rest = Math.max(0, 1 - total);

  return (
    <div className="card-warm p-5 space-y-4">
      <div>
        <p className="kicker">Population breakdown</p>
        <p className="mt-1.5 text-lg font-semibold" style={{ color: "var(--ink-900)" }}>
          End-state behaviour distribution
        </p>
      </div>

      {/* Stacked bar */}
      <div className="flex h-10 w-full overflow-hidden rounded-xl" style={{ background: "var(--cream-300)" }}>
        {segments.map((seg) => (
          <div
            key={seg.label}
            className="h-full transition-all duration-700"
            style={{ width: `${seg.value * 100}%`, background: seg.color }}
            title={`${seg.label}: ${formatPct(seg.value)}`}
          />
        ))}
        {rest > 0.001 && (
          <div
            className="h-full flex-1"
            style={{ background: "var(--cream-300)" }}
            title={`Other: ${formatPct(rest)}`}
          />
        )}
      </div>

      {/* Legend */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {segments.map((seg) => (
          <div
            key={seg.label}
            className="rounded-xl border px-3 py-2.5"
            style={{ background: seg.bg, borderColor: `${seg.color}33` }}
          >
            <p className="text-[10px] font-semibold uppercase tracking-[0.18em]"
              style={{ color: seg.color }}>{seg.label}</p>
            <p className="metric-num mt-1 text-xl font-bold" style={{ color: seg.color }}>
              {formatPct(seg.value)}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 3: Commit**
```bash
git add components/simulation/PopulationSummaryBar.tsx
git commit -m "feat: add PopulationSummaryBar stacked breakdown"
```

---

### Task 10: Build `NetworkStatsPanel.tsx`

**Files:**
- Create: `components/simulation/NetworkStatsPanel.tsx`

**Step 1: Write the component**

```tsx
import { AnimatedCounter } from "@/components/ui/AnimatedCounter";

interface Props {
  networkStatistics: Record<string, number>;
  networkHubs: Record<string, unknown>[];
}

export function NetworkStatsPanel({ networkStatistics, networkHubs }: Props) {
  const meanDegree = networkStatistics.mean_degree ?? 0;
  const clustering = networkStatistics.clustering_coefficient ?? 0;
  const density = networkStatistics.density ?? 0;

  return (
    <div className="card-warm p-5 space-y-4">
      <div>
        <p className="kicker">Network topology</p>
        <p className="mt-1.5 text-lg font-semibold" style={{ color: "var(--ink-900)" }}>
          Influence graph structure
        </p>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {[
          { label: "Mean degree", value: meanDegree, fmt: (v: number) => v.toFixed(1) },
          { label: "Clustering", value: clustering, fmt: (v: number) => v.toFixed(3) },
          { label: "Density",    value: density,    fmt: (v: number) => v.toFixed(3) },
        ].map(({ label, value, fmt }) => (
          <div
            key={label}
            className="rounded-xl border px-3 py-3 text-center"
            style={{ background: "var(--cream-200)", borderColor: "var(--border-warm)" }}
          >
            <p className="kicker text-[10px]">{label}</p>
            <p className="metric-num mt-2 text-2xl font-bold" style={{ color: "var(--ink-900)" }}>
              <AnimatedCounter value={value} formatter={fmt} />
            </p>
          </div>
        ))}
      </div>

      {networkHubs.length > 0 && (
        <div>
          <p className="kicker mb-2">Top network hubs</p>
          <div className="space-y-1">
            {networkHubs.slice(0, 3).map((hub, i) => (
              <div
                key={i}
                className="flex items-center justify-between rounded-lg border px-3 py-2 text-sm"
                style={{ borderColor: "var(--border-warm)", background: "var(--cream-100)" }}
              >
                <span style={{ color: "var(--ink-700)" }}>
                  Hub {String(hub.node ?? i)}
                </span>
                <span className="metric-num font-semibold" style={{ color: "var(--crimson-700)" }}>
                  deg {String(hub.degree ?? "—")}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 3: Commit**
```bash
git add components/simulation/NetworkStatsPanel.tsx
git commit -m "feat: add NetworkStatsPanel with animated topology metrics"
```

---

### Task 11: Build `JurisdictionFlowDiagram.tsx` — SVG Sankey

**Files:**
- Create: `components/simulation/JurisdictionFlowDiagram.tsx`

**Step 1: Write the component**

```tsx
"use client";

import { useMemo } from "react";

interface Props {
  jurisdictionSummary: Record<string, { company_count?: number; burden?: number }>;
  sourceJurisdiction?: string;
}

interface FlowNode {
  id: string;
  label: string;
  count: number;
  x: number;
  y: number;
  r: number;
}

const DEST_COLORS: Record<string, string> = {
  US: "#1E3A8A", UK: "#15803D", Singapore: "#B45309",
  UAE: "#6B21A8", EU: "#8B1A1A",
};

export function JurisdictionFlowDiagram({ jurisdictionSummary, sourceJurisdiction = "EU" }: Props) {
  const destinations = useMemo(() => {
    return Object.entries(jurisdictionSummary)
      .filter(([k]) => k !== `source_${sourceJurisdiction}` && !k.startsWith("source_"))
      .map(([key, val]) => ({
        label: key.replace("dest_", "").replace("_companies", ""),
        count: val.company_count ?? 0,
      }))
      .filter((d) => d.count > 0)
      .sort((a, b) => b.count - a.count);
  }, [jurisdictionSummary, sourceJurisdiction]);

  const sourceCount = Object.values(jurisdictionSummary)
    .reduce((s, v) => s + (v.company_count ?? 0), 0);

  if (destinations.length === 0) return null;

  const W = 480;
  const H = Math.max(200, destinations.length * 64 + 40);
  const cx = W / 2;
  const maxCount = Math.max(...destinations.map((d) => d.count));

  const srcX = 80;
  const srcY = H / 2;
  const destX = W - 80;

  const nodes: FlowNode[] = destinations.map((d, i) => {
    const step = H / (destinations.length + 1);
    return {
      id: d.label,
      label: d.label,
      count: d.count,
      x: destX,
      y: step * (i + 1),
      r: 10 + (d.count / maxCount) * 20,
    };
  });

  return (
    <div className="card-warm p-5 space-y-3">
      <div>
        <p className="kicker">Jurisdiction flow</p>
        <p className="mt-1.5 text-lg font-semibold" style={{ color: "var(--ink-900)" }}>
          Relocation destinations from {sourceJurisdiction}
        </p>
      </div>
      <div className="overflow-x-auto">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ maxHeight: 280 }}>
          {/* Flows */}
          {nodes.map((node) => {
            const strokeW = Math.max(1.5, (node.count / maxCount) * 14);
            const color = DEST_COLORS[node.label] ?? "#8B1A1A";
            const cp1x = srcX + (destX - srcX) * 0.42;
            const cp2x = srcX + (destX - srcX) * 0.58;
            const d = `M ${srcX} ${srcY} C ${cp1x} ${srcY}, ${cp2x} ${node.y}, ${node.x} ${node.y}`;
            return (
              <path
                key={node.id}
                d={d}
                fill="none"
                stroke={color}
                strokeWidth={strokeW}
                strokeOpacity={0.45}
                strokeLinecap="round"
              />
            );
          })}

          {/* Source node */}
          <circle cx={srcX} cy={srcY} r={28} fill="var(--crimson-100)" stroke="var(--crimson-700)" strokeWidth={2} />
          <text x={srcX} y={srcY - 6} textAnchor="middle" fontSize={10} fontWeight={700}
            fill="var(--crimson-700)">{sourceJurisdiction}</text>
          <text x={srcX} y={srcY + 8} textAnchor="middle" fontSize={9} fill="var(--ink-500)">
            source
          </text>

          {/* Destination nodes */}
          {nodes.map((node) => {
            const color = DEST_COLORS[node.label] ?? "#8B1A1A";
            return (
              <g key={node.id}>
                <circle cx={node.x} cy={node.y} r={node.r} fill={`${color}22`}
                  stroke={color} strokeWidth={2} />
                <text x={node.x} y={node.y - 4} textAnchor="middle" fontSize={10}
                  fontWeight={700} fill={color}>{node.label}</text>
                <text x={node.x} y={node.y + 9} textAnchor="middle" fontSize={9}
                  fill="var(--ink-500)">{node.count}</text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 3: Commit**
```bash
git add components/simulation/JurisdictionFlowDiagram.tsx
git commit -m "feat: add JurisdictionFlowDiagram SVG Sankey"
```

---

## Phase 5 — Updated Simulation Components

### Task 12: Rewrite `RoundSummaryChart.tsx` — warm palette + chart styles

**Files:**
- Modify: `components/simulation/RoundSummaryChart.tsx`
- Modify: `lib/constants.ts` — update CHART_COLORS

**Step 1: Update CHART_COLORS in `lib/constants.ts`**

Find the CHART_COLORS object and replace:
```ts
export const CHART_COLORS = {
  compliance:  "var(--chart-compliance)",
  relocation:  "var(--chart-relocation)",
  investment:  "var(--chart-investment)",
  enforcement: "var(--chart-enforcement)",
  evasion:     "var(--chart-evasion)",
} as const;
```

**Step 2: Update RoundSummaryChart.tsx** — replace dark hardcoded colors and styles

Key changes:
- Toggle button inactive: `background: var(--cream-200)`, `borderColor: var(--border-warm)`, `color: var(--ink-400)`
- Chart container: `background: var(--cream-100)`, `borderColor: var(--border-warm)`
- `CartesianGrid stroke`: `"var(--border-warm)"`
- XAxis/YAxis tick `fill`: `"var(--ink-400)"`
- Tooltip `contentStyle`: `background: "var(--cream-50)"`, `border: "1px solid var(--border-warm)"`, `borderRadius: 16`, `boxShadow: "var(--shadow-raised)"`
- Tooltip `labelStyle`: `color: "var(--ink-700)"`
- `activeDot stroke`: `"var(--cream-50)"`
- Metric colors now read from CSS variables (pass the string `"var(--chart-compliance)"` etc.)

Since Recharts requires literal color strings for `stroke`, use the hex values directly from the design tokens (not CSS vars):

```ts
const METRICS = [
  { key: "compliance_rate",         label: "Compliance", color: "#8B1A1A", note: "Policy adherence"    },
  { key: "relocation_rate",         label: "Relocation", color: "#B45309", note: "Jurisdiction flight"  },
  { key: "ai_investment_index",     label: "Investment", color: "#1D6344", note: "Capital sentiment"    },
  { key: "enforcement_contact_rate",label: "Enforcement",color: "#1E3A8A", note: "Regulatory contact"   },
] as const;
```

Update inactive toggle button style to warm:
```tsx
: {
    borderColor: "var(--border-warm)",
    background: "var(--cream-200)",
    color: "var(--ink-400)",
  }
```

Update chart wrapper:
```tsx
<div className="overflow-hidden rounded-2xl border p-3 md:p-4"
  style={{ background: "var(--cream-100)", borderColor: "var(--border-warm)" }}>
```

**Step 3: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 4: Commit**
```bash
git add components/simulation/RoundSummaryChart.tsx lib/constants.ts
git commit -m "style: warm chart palette for RoundSummaryChart"
```

---

### Task 13: Rewrite `FinalStocksGrid.tsx` — warm cards + AnimatedCounter

**Files:**
- Modify: `components/simulation/FinalStocksGrid.tsx`

**Step 1: Rewrite the file**

```tsx
"use client";

import { AnimatedCounter } from "@/components/ui/AnimatedCounter";
import { formatDelta, formatPct } from "@/lib/format";

interface StatCardProps {
  label: string;
  value: number;
  formatter: (v: number) => string;
  note: string;
  delta?: number | null;
  index: number;
}

function StatCard({ label, value, formatter, note, delta, index }: StatCardProps) {
  const positive = delta !== null && delta !== undefined && delta >= 0;

  return (
    <div
      className="card-warm p-4"
      style={{
        animation: "slideUpFade 380ms ease both",
        animationDelay: `${index * 55}ms`,
      }}
    >
      <div className="flex items-start justify-between gap-2">
        <p className="kicker">{label}</p>
        {delta !== null && delta !== undefined && (
          <span
            className="metric-num rounded-full border px-2 py-0.5 text-[11px] font-semibold flex-shrink-0"
            style={{
              color: positive ? "var(--success)" : "var(--danger)",
              borderColor: positive ? "rgba(21,128,61,0.2)" : "rgba(220,38,38,0.2)",
              background: positive ? "#f0fdf4" : "#fef2f2",
            }}
          >
            {formatDelta(delta)}
          </span>
        )}
      </div>
      <p className="metric-num mt-3 text-[32px] font-bold leading-none" style={{ color: "var(--ink-900)" }}>
        <AnimatedCounter value={value} formatter={formatter} />
      </p>
      <p className="mt-2 text-sm leading-5" style={{ color: "var(--ink-400)" }}>{note}</p>
    </div>
  );
}

interface Props {
  finalStocks: Record<string, number>;
  finalPopulationSummary: Record<string, number>;
  compareStocks?: Record<string, number>;
  comparePopulation?: Record<string, number>;
}

export function FinalStocksGrid({ finalStocks, finalPopulationSummary, compareStocks, comparePopulation }: Props) {
  const delta = (key: string, base: Record<string, number>, comp?: Record<string, number>) =>
    comp ? (comp[key] ?? 0) - (base[key] ?? 0) : null;

  return (
    <section className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
      <StatCard index={0} label="Compliance rate"
        value={finalPopulationSummary.compliance_rate ?? 0}
        formatter={formatPct}
        note="Share of regulated entities in compliance at close."
        delta={delta("compliance_rate", finalPopulationSummary, comparePopulation)} />
      <StatCard index={1} label="Relocation rate"
        value={finalPopulationSummary.relocation_rate ?? 0}
        formatter={formatPct}
        note="Fraction that shifted to another jurisdiction."
        delta={delta("relocation_rate", finalPopulationSummary, comparePopulation)} />
      <StatCard index={2} label="Evasion rate"
        value={finalPopulationSummary.evasion_rate ?? 0}
        formatter={formatPct}
        note="Population share sustaining evasion at end-state."
        delta={delta("evasion_rate", finalPopulationSummary, comparePopulation)} />
      <StatCard index={3} label="Public trust"
        value={finalStocks.public_trust ?? 0}
        formatter={(v) => v.toFixed(1)}
        note="Aggregate trust stock at close of run."
        delta={compareStocks ? (compareStocks.public_trust ?? 0) - (finalStocks.public_trust ?? 0) : null} />
      <StatCard index={4} label="Domestic firms"
        value={finalStocks.domestic_companies ?? 0}
        formatter={(v) => v.toFixed(0)}
        note="Domestic company count remaining in jurisdiction."
        delta={compareStocks ? (compareStocks.domestic_companies ?? 0) - (finalStocks.domestic_companies ?? 0) : null} />
    </section>
  );
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 3: Commit**
```bash
git add components/simulation/FinalStocksGrid.tsx
git commit -m "feat: warm cards + AnimatedCounter in FinalStocksGrid"
```

---

### Task 14: Rewrite `SMMTable.tsx` and `SimLoadingSkeleton.tsx` — warm styles

**Files:**
- Modify: `components/simulation/SMMTable.tsx`
- Modify: `components/simulation/SimLoadingSkeleton.tsx`

**Step 1: Update SMMTable.tsx**

Replace all dark token references:
- Table container outer: `card-warm overflow-hidden` (remove dark bg)
- `thead bg-[rgba(255,255,255,0.02)]` → `bg-[var(--cream-200)]`
- `border-b border-[var(--border-subtle)]` → `style={{ borderColor: "var(--border-warm)" }}`
- th text: `text-[var(--ink-500)]` → `kicker`
- td label: `text-[var(--ink-300)]` → `style={{ color: "var(--ink-700)" }}`
- td value: `text-[var(--ink-100)]` → `style={{ color: "var(--ink-900)" }}`
- SMM distance box: `card-warm px-4 py-3`

**Step 2: Rewrite SimLoadingSkeleton.tsx**

```tsx
export function SimLoadingSkeleton() {
  return (
    <div className="card-warm flex min-h-[600px] flex-col gap-5 p-6 md:p-8">
      <div className="space-y-5">
        {/* Header skeleton */}
        <div className="space-y-2.5">
          <div className="h-3 w-20 rounded-full" style={{ background: "var(--cream-300)",
            animation: "shimmerWarm 1.4s infinite linear",
            backgroundImage: "linear-gradient(90deg, var(--cream-300) 25%, var(--cream-400) 50%, var(--cream-300) 75%)",
            backgroundSize: "800px 100%" }} />
          <div className="h-7 w-72 rounded-xl" style={{ background: "var(--cream-300)" }} />
          <div className="h-4 w-full max-w-sm rounded-full" style={{ background: "var(--cream-200)" }} />
        </div>

        {/* Stat cards skeleton */}
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">
          {Array.from({ length: 5 }).map((_, i) => (
            <div key={i} className="h-28 rounded-2xl" style={{ background: "var(--cream-200)" }} />
          ))}
        </div>

        {/* Charts skeleton */}
        <div className="grid gap-4 xl:grid-cols-[1.1fr_0.9fr]">
          <div className="h-80 rounded-2xl" style={{ background: "var(--cream-200)" }} />
          <div className="h-80 rounded-2xl" style={{ background: "var(--cream-200)" }} />
        </div>
      </div>
      <p className="text-center text-sm" style={{ color: "var(--ink-400)" }}>
        Simulating scenario dynamics…
      </p>
    </div>
  );
}
```

**Step 3: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 4: Commit**
```bash
git add components/simulation/SMMTable.tsx components/simulation/SimLoadingSkeleton.tsx
git commit -m "style: warm theme for SMMTable + SimLoadingSkeleton"
```

---

### Task 15: Rewrite `ResultsPanel.tsx` — bento grid + new components

**Files:**
- Modify: `components/simulation/ResultsPanel.tsx`

**Step 1: Rewrite with bento layout + all new components**

```tsx
import { FinalStocksGrid } from "@/components/simulation/FinalStocksGrid";
import { RoundSummaryChart } from "@/components/simulation/RoundSummaryChart";
import { SMMTable } from "@/components/simulation/SMMTable";
import { PopulationSummaryBar } from "@/components/simulation/PopulationSummaryBar";
import { NetworkStatsPanel } from "@/components/simulation/NetworkStatsPanel";
import { JurisdictionFlowDiagram } from "@/components/simulation/JurisdictionFlowDiagram";
import { AnimatedCounter } from "@/components/ui/AnimatedCounter";
import { formatDurationMs, formatPct } from "@/lib/format";
import type { SimulateResponse, ConfidenceBand } from "@/lib/types";

interface Props {
  result: SimulateResponse;
  comparisonResult?: SimulateResponse;
  complianceBands?: ConfidenceBand[];
}

export function ResultsPanel({ result, comparisonResult, complianceBands }: Props) {
  const pop = result.final_population_summary;
  const stocks = result.final_stocks;

  return (
    <div className="space-y-4">
      {/* Verdict header */}
      <div
        className="card-raised overflow-hidden p-6 md:p-8"
        style={{ animation: "slideUpFade 380ms ease both" }}
      >
        <div className="flex flex-col gap-5 md:flex-row md:items-start md:justify-between">
          <div className="space-y-3 max-w-2xl">
            <p className="kicker">Simulation verdict</p>
            <h2 className="text-3xl font-bold leading-tight" style={{ color: "var(--ink-900)" }}>
              {result.policy_name}
            </h2>
            <p className="text-base leading-7" style={{ color: "var(--ink-500)" }}>
              Final compliance settled at{" "}
              <span className="metric-num font-bold" style={{ color: "var(--ink-900)" }}>
                {formatPct(pop.compliance_rate ?? 0)}
              </span>
              , with relocation at{" "}
              <span className="metric-num font-bold" style={{ color: "var(--ink-900)" }}>
                {formatPct(pop.relocation_rate ?? 0)}
              </span>
              . Public trust finished at{" "}
              <span className="metric-num font-bold" style={{ color: "var(--ink-900)" }}>
                {(stocks.public_trust ?? 0).toFixed(1)}
              </span>
              , AI investment at{" "}
              <span className="metric-num font-bold" style={{ color: "var(--ink-900)" }}>
                {(stocks.ai_investment_index ?? 0).toFixed(1)}
              </span>
              .
            </p>
          </div>

          <div className="flex flex-shrink-0 gap-3">
            {[
              { label: "Trust", value: stocks.public_trust ?? 0, fmt: (v: number) => v.toFixed(1) },
              { label: "Investment", value: stocks.ai_investment_index ?? 0, fmt: (v: number) => v.toFixed(1) },
            ].map(({ label, value, fmt }) => (
              <div key={label} className="card-warm px-4 py-3 text-center min-w-[90px]">
                <p className="kicker text-[10px]">{label}</p>
                <p className="metric-num mt-2 text-2xl font-bold" style={{ color: "var(--ink-900)" }}>
                  <AnimatedCounter value={value} formatter={fmt} />
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* 5-column stat cards */}
      <FinalStocksGrid
        finalStocks={stocks}
        finalPopulationSummary={pop}
        compareStocks={comparisonResult?.final_stocks}
        comparePopulation={comparisonResult?.final_population_summary}
      />

      {/* Population breakdown */}
      <PopulationSummaryBar
        complianceRate={pop.compliance_rate ?? 0}
        relocationRate={pop.relocation_rate ?? 0}
        evasionRate={pop.evasion_rate ?? 0}
        everLobbyedRate={pop.ever_lobbied_rate ?? 0}
      />

      {/* Charts row */}
      <div className="grid gap-4 xl:grid-cols-[1.15fr_0.85fr]">
        <div className="card-warm p-5 md:p-6">
          <RoundSummaryChart roundSummaries={result.round_summaries} bands={complianceBands} />
        </div>
        <div className="card-warm p-5 md:p-6">
          <SMMTable moments={result.simulated_moments} distanceToGdpr={result.smm_distance_to_gdpr} />
        </div>
      </div>

      {/* Network + Jurisdiction row */}
      <div className="grid gap-4 xl:grid-cols-[0.5fr_1fr]">
        <NetworkStatsPanel
          networkStatistics={result.network_statistics}
          networkHubs={result.network_hubs}
        />
        <JurisdictionFlowDiagram jurisdictionSummary={result.jurisdiction_summary} />
      </div>

      {/* Run metadata footer */}
      <div
        className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border px-4 py-3 text-sm"
        style={{ borderColor: "var(--border-warm)", background: "var(--cream-100)", color: "var(--ink-400)" }}
      >
        <span>Ran in <span className="metric-num font-semibold" style={{ color: "var(--ink-700)" }}>{formatDurationMs(result.run_metadata.duration_ms)}</span></span>
        <span>seed <span className="metric-num font-semibold" style={{ color: "var(--ink-700)" }}>{result.run_metadata.seed}</span></span>
        <span>n=<span className="metric-num font-semibold" style={{ color: "var(--ink-700)" }}>{result.run_metadata.n_population}</span></span>
      </div>
    </div>
  );
}
```

**Step 2: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -30
```

**Step 3: Commit**
```bash
git add components/simulation/ResultsPanel.tsx
git commit -m "feat: bento results layout with all new simulation components"
```

---

## Phase 6 — Pages

### Task 16: Rewrite `/analyze/page.tsx` — remove hero, direct work layout

**Files:**
- Modify: `app/analyze/page.tsx`

**Step 1: Rewrite the page**

```tsx
"use client";

import { AppShell } from "@/components/layout/AppShell";
import { PolicyParamPanel } from "@/components/policy/PolicyParamPanel";
import { PresetSelector } from "@/components/policy/PresetSelector";
import { SimConfigPanel } from "@/components/simulation/SimConfigPanel";
import { SimLoadingSkeleton } from "@/components/simulation/SimLoadingSkeleton";
import { ResultsPanel } from "@/components/simulation/ResultsPanel";
import { EvidencePackSection } from "@/components/simulation/EvidencePackSection";
import { simulate } from "@/lib/api-client";
import { useAnalyzeStore } from "@/lib/store";
import type { PresetPolicy } from "@/lib/types";
import { ArrowRight, BarChart2 } from "lucide-react";

export default function AnalyzePage() {
  const store = useAnalyzeStore();

  function handlePreset(preset: PresetPolicy) {
    store.setFromSpec(preset.spec);
  }

  async function handleRun() {
    store.setLoading(true);
    store.setError(null);
    try {
      const result = await simulate({
        policy_name: store.policyName,
        policy_description: store.policyDescription,
        policy_severity: store.policySeverity,
        config: store.simConfig,
      });
      store.setResult(result);
    } catch (e) {
      store.setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <AppShell>
      <div className="mx-auto max-w-[1560px] px-4 py-6 lg:px-8">
        {/* Page header */}
        <div className="mb-6 flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl flex-shrink-0"
            style={{ background: "var(--crimson-50)", border: "1px solid var(--crimson-100)" }}>
            <BarChart2 size={17} style={{ color: "var(--crimson-700)" }} />
          </div>
          <div>
            <h1 className="text-xl font-bold" style={{ color: "var(--ink-900)" }}>
              Policy Simulator
            </h1>
            <p className="text-sm" style={{ color: "var(--ink-400)" }}>
              Configure a bill and launch a population simulation
            </p>
          </div>
        </div>

        {/* Two-panel layout */}
        <div className="flex gap-5 items-start lg:flex-row flex-col">
          {/* Left: policy authoring */}
          <aside className="w-full lg:w-[380px] lg:flex-shrink-0 space-y-4">
            <div className="card-warm p-5 space-y-4">
              <div>
                <p className="kicker mb-1">Load preset</p>
                <PresetSelector onSelect={handlePreset} selectedSlug={undefined} />
              </div>
              <div className="divider-warm" />
              <PolicyParamPanel
                name={store.policyName}
                description={store.policyDescription}
                severity={store.policySeverity}
                penaltyType={store.penaltyType ?? "none"}
                penaltyCapUsd={store.penaltyCapUsd ?? null}
                computeThresholdFlops={store.computeThresholdFlops ?? null}
                enforcementMechanism={store.enforcementMechanism ?? "none"}
                gracePeriodMonths={store.gracePeriodMonths ?? 0}
                scope={store.scope ?? "all"}
                onChangeName={(v) => store.setPolicy(v, store.policyDescription, store.policySeverity)}
                onChangeDescription={(v) => store.setPolicy(store.policyName, v, store.policySeverity)}
                onChangeSeverity={(v) => store.setPolicy(store.policyName, store.policyDescription, v)}
                onChangePenaltyType={(v) => store.setPenaltyType?.(v)}
                onChangeEnforcementMechanism={(v) => store.setEnforcementMechanism?.(v)}
                onChangeGracePeriodMonths={(v) => store.setGracePeriodMonths?.(v)}
                onChangeScope={(v) => store.setScope?.(v)}
              />
            </div>

            <SimConfigPanel config={store.simConfig} onChange={(p) => store.setSimConfig(p)} />

            <button
              onClick={handleRun}
              disabled={store.isLoading}
              className="btn-primary w-full py-3 text-base"
            >
              {store.isLoading ? "Simulating…" : "Run simulation"}
              <ArrowRight size={16} className="transition-transform group-hover:translate-x-0.5" />
            </button>

            {store.error && (
              <p className="rounded-xl border px-3 py-2.5 text-sm"
                style={{ color: "var(--danger)", borderColor: "rgba(220,38,38,0.2)", background: "#fef2f2" }}>
                {store.error}
              </p>
            )}

            {store.result && <EvidencePackSection />}
          </aside>

          {/* Right: results */}
          <section className="min-w-0 flex-1">
            {store.isLoading && <SimLoadingSkeleton />}

            {!store.isLoading && store.result && (
              <ResultsPanel
                result={store.result}
                complianceBands={store.evidencePackResult?.bands?.compliance_rate}
              />
            )}

            {!store.isLoading && !store.result && !store.error && (
              <div className="card-warm flex min-h-[500px] flex-col items-center justify-center gap-5 p-8 text-center">
                <div className="flex h-16 w-16 items-center justify-center rounded-2xl"
                  style={{ background: "var(--crimson-50)", border: "1px solid var(--crimson-100)" }}>
                  <BarChart2 size={28} style={{ color: "var(--crimson-700)" }} />
                </div>
                <div className="space-y-2">
                  <h3 className="text-xl font-semibold" style={{ color: "var(--ink-900)" }}>
                    Ready to simulate
                  </h3>
                  <p className="max-w-sm text-sm leading-6" style={{ color: "var(--ink-400)" }}>
                    Configure your policy parameters on the left and click{" "}
                    <strong style={{ color: "var(--ink-700)" }}>Run simulation</strong> to see compliance
                    trajectories, relocation pressure, and calibration moments.
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-3 w-full max-w-xs">
                  {[
                    ["Population", `${store.simConfig.n_population.toLocaleString()}`],
                    ["Rounds", `${store.simConfig.num_rounds}`],
                    ["Severity", store.policySeverity.toFixed(1)],
                  ].map(([label, value]) => (
                    <div key={label} className="card-warm px-3 py-2.5 text-center">
                      <p className="kicker text-[10px]">{label}</p>
                      <p className="metric-num mt-1 text-lg font-bold" style={{ color: "var(--ink-900)" }}>{value}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </div>
      </div>
    </AppShell>
  );
}
```

**Important:** The `PolicyParamPanel` props referencing `store.penaltyType`, `store.penaltyCapUsd`, etc. need to exist in the store. Check `lib/store.ts` — if those fields don't exist in `useAnalyzeStore`, use safe fallbacks: `store.penaltyType ?? "none" as PenaltyType`. The `onChangePenaltyType` etc. are optional props so passing `undefined` is fine if the store doesn't have setters.

**Step 2: Run TypeScript check and fix any errors**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -40
```

Fix any type errors before committing.

**Step 3: Commit**
```bash
git add app/analyze/page.tsx
git commit -m "feat: redesign analyze page — compact header, direct two-panel layout"
```

---

### Task 17: Rewrite `/compare/page.tsx` — warm bento grid with delta highlights

**Files:**
- Modify: `app/compare/page.tsx`

**Step 1: Read current file first**
```bash
cat /Users/ambar/Downloads/policylab_m3/frontend/app/compare/page.tsx
```

**Step 2: Key changes to make**

Replace all dark token references:
- All `bg-[rgba(255,255,255,...)]` → `bg-[var(--cream-100)]` or `bg-[var(--cream-200)]`
- All `border-[var(--border-subtle)]` → `style={{ borderColor: "var(--border-warm)" }}`
- All `text-[var(--ink-100)]` → `style={{ color: "var(--ink-900)" }}`
- All `text-[var(--ink-300/400)]` → `style={{ color: "var(--ink-500/400)" }}`
- Section headers: add `.kicker` class
- Policy slot cards: use `.card-warm p-4`
- "Add policy" button: use `.btn-secondary`
- "Run compare" button: use `.btn-primary`
- Per-slot chart wrapper: use `.card-warm p-4`
- Delta cells: green tint = `background: "#f0fdf4"`, red = `background: "#fef2f2"`
- Error text: `style={{ color: "var(--danger)" }}`

**Step 3: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 4: Commit**
```bash
git add app/compare/page.tsx
git commit -m "style: warm theme for compare page"
```

---

### Task 18: Rewrite `/influence/page.tsx` and `/upload/page.tsx` — warm theme

**Files:**
- Modify: `app/influence/page.tsx`
- Modify: `app/upload/page.tsx`

**Step 1: Read both files first**
```bash
cat /Users/ambar/Downloads/policylab_m3/frontend/app/influence/page.tsx
cat /Users/ambar/Downloads/policylab_m3/frontend/app/upload/page.tsx
```

**Step 2: Apply the same warm token replacements as Compare page**

For `/influence/page.tsx`, key additions:
- The resilience score card: use `.card-tinted` (crimson-tinted) if score < 0.4, `.card-warm` otherwise
- `ResilienceScoreCard` value: large metric-num text in crimson-700 if low, success if high
- Injection config sliders: wrap in `.card-warm p-4`
- Chart container: `.card-warm p-4`

For `/upload/page.tsx`, key additions:
- Dropzone border: warm dashed border, on drag-over changes to `var(--crimson-700)`
- Dropzone background: `var(--cream-100)` default, `var(--crimson-50)` on hover/drag
- API key input: `.input-warm`
- Extraction accordion rows: `.card-warm` wrapping
- Confidence bars: fill color = `var(--crimson-700)`, track = `var(--cream-300)`

**Step 3: Run TypeScript check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1 | head -20
```

**Step 4: Commit**
```bash
git add app/influence/page.tsx app/upload/page.tsx
git commit -m "style: warm theme for influence + upload pages"
```

---

## Phase 7 — Final Verification

### Task 19: Full TypeScript check + production build

**Step 1: TypeScript strict check**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npx tsc --noEmit 2>&1
```
Expected: 0 errors. Fix any that appear.

**Step 2: Production build**
```bash
cd /Users/ambar/Downloads/policylab_m3/frontend && npm run build 2>&1
```
Expected: All 4 routes compile successfully.

**Step 3: If the API is running, do a quick smoke test**
```bash
# Start API if needed
cd /Users/ambar/Downloads/policylab_m3/api && uvicorn main:app --port 8000 &
# Start frontend
cd /Users/ambar/Downloads/policylab_m3/frontend && npm run dev &
# Check presets load
curl http://localhost:8000/presets | python3 -m json.tool | head -20
```

**Step 4: Final commit**
```bash
cd /Users/ambar/Downloads/policylab_m3
git add -A
git commit -m "feat: complete Warm Paper UI overhaul — cream/crimson theme, TopNav, AnimatedCounter, JurisdictionFlow, PopulationBar, NetworkStats, CommandPalette, bento results"
```

---

## Summary of All Files Changed

**New files:**
- `components/layout/TopNav.tsx`
- `components/layout/MobileNav.tsx`
- `components/ui/AnimatedCounter.tsx`
- `components/ui/CommandPalette.tsx`
- `components/simulation/PopulationSummaryBar.tsx`
- `components/simulation/NetworkStatsPanel.tsx`
- `components/simulation/JurisdictionFlowDiagram.tsx`

**Deleted:**
- `components/layout/Sidebar.tsx`

**Modified:**
- `app/globals.css`
- `app/layout.tsx` (remove Sidebar import if present)
- `app/analyze/page.tsx`
- `app/compare/page.tsx`
- `app/influence/page.tsx`
- `app/upload/page.tsx`
- `components/layout/AppShell.tsx`
- `components/policy/PresetSelector.tsx`
- `components/policy/PolicyParamPanel.tsx`
- `components/simulation/SimConfigPanel.tsx`
- `components/simulation/EpistemicBadge.tsx`
- `components/simulation/EvidencePackSection.tsx`
- `components/simulation/ResultsPanel.tsx`
- `components/simulation/FinalStocksGrid.tsx`
- `components/simulation/RoundSummaryChart.tsx`
- `components/simulation/SMMTable.tsx`
- `components/simulation/SimLoadingSkeleton.tsx`
- `lib/constants.ts`
