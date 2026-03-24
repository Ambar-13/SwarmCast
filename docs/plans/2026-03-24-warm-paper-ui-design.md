# PolicyLab "Warm Paper" UI Overhaul — Design Document
**Date:** 2026-03-24
**Status:** Approved
**Theme:** Cream + Dark Crimson (light, editorial, Awwwards-quality)

---

## 1. Design Philosophy

Replace the current dark glassmorphism theme with a warm editorial aesthetic inspired by:
- Claude's cream/off-white + warm accent palette
- Awwwards 2025 dominant patterns: warm neutrals, fluid typography, bento grids, scroll-driven reveals
- Linear.app's polish standard: every state matters, every number animates, nothing feels static

**Competitor context:** MiroFish is a dark dashboard "vibe coded in 10 days" — PolicyLab should feel like a premium research instrument, not a startup demo. The warm paper aesthetic signals credibility, academic seriousness, and considered design.

---

## 2. Color System

```css
/* Page surfaces */
--cream-50:   #FDFAF6   /* page background */
--cream-100:  #FAF6EF   /* card surfaces */
--cream-200:  #F4EDE0   /* inputs, hover states */
--cream-300:  #E8DDD0   /* borders, dividers */

/* Text */
--ink-900:    #1A1208   /* primary text — warm near-black */
--ink-700:    #3D2E1E   /* secondary text */
--ink-500:    #6B5744   /* tertiary text */
--ink-400:    #8C7B68   /* muted/placeholder */

/* Primary accent */
--crimson-700: #8B1A1A  /* buttons, active states */
--crimson-600: #A82020  /* hover */
--crimson-500: #C0392B  /* active/pressed */
--crimson-100: #FAE8E8  /* tint backgrounds */
--crimson-50:  #FDF4F4  /* very light tint */

/* Semantic */
--success:    #15803D   /* GROUNDED epistemic tier */
--warning:    #B45309   /* DIRECTIONAL epistemic tier */
--danger:     #DC2626   /* ASSUMED epistemic tier */

/* Chart palette (warm) */
--chart-compliance:  #8B1A1A   /* crimson */
--chart-relocation:  #B45309   /* amber */
--chart-investment:  #1D6344   /* forest green */
--chart-enforcement: #1E3A8A   /* deep blue */
--chart-evasion:     #6B21A8   /* deep purple */
```

---

## 3. Typography

- **Primary font:** Inter (variable, weights 300–800)
- **Monospace/numbers:** JetBrains Mono (tabular-nums for all metrics)
- **Display headings:** `clamp(28px, 4vw, 52px)`, `letter-spacing: -0.03em`, `line-height: 1.05`
- **Body:** 15px / 1.7 line-height / ink-700
- **Kickers:** 11px, 600 weight, `letter-spacing: 0.22em`, uppercase, ink-400

---

## 4. Component Library

### CSS Classes (globals.css)

```
.card-warm      — cream-100 bg, warm border, soft warm shadow
.card-raised    — card-warm + stronger shadow (for key metrics)
.card-tinted    — crimson-50 bg, crimson-100 border (highlighted)
.btn-primary    — crimson-700 bg, white text, rounded-xl, scale(0.97) active
.btn-secondary  — cream-200 bg, ink-700 text, rounded-xl
.tag-badge      — rounded-full, px-3 py-1, 11px font
.kicker         — 11px uppercase tracking-[0.22em] ink-400
.metric-num     — JetBrains Mono tabular-nums
.input-warm     — cream-200 bg, cream-300 border, rounded-xl, ink-900 text
.select-warm    — same as input-warm
.divider-warm   — 1px solid cream-300
```

### Shadow tokens
```
--shadow-card:  0 1px 3px rgba(26,18,8,0.06), 0 4px 16px rgba(26,18,8,0.04)
--shadow-raised: 0 2px 8px rgba(26,18,8,0.08), 0 8px 32px rgba(26,18,8,0.06)
--shadow-focus: 0 0 0 3px rgba(139,26,26,0.18)
```

---

## 5. Layout Architecture

### Top Navigation Bar (replaces sidebar)
- Height: 56px, sticky
- Background: cream-50 + 1px warm bottom border
- Left: PolicyLab logo/wordmark
- Center: Tab navigation — Analyze · Upload · Compare · Influence
- Right: Keyboard shortcut hint (`⌘K`), settings icon
- Active tab: crimson-700 underline (2px), crimson-700 text
- Mobile: hidden, replaced by bottom MobileNav

### Page Layout
- `max-w-[1560px]` centered, `px-6 lg:px-10`
- Vertical rhythm: 24px gap between sections
- Left panel (policy form): `w-[380px]` fixed width on lg+
- Right panel (results): `flex-1 min-w-0`

### Bento Grid (results)
- 12-column CSS grid
- Metric stat cards: span 2 cols (6 per row on xl)
- Round summary chart: span 7 cols
- SMM table: span 5 cols
- Jurisdiction flow: span 12 cols (full width)
- Population summary bar: span 12 cols

---

## 6. Animations

All implemented with CSS + tiny React hooks. No animation library.

### Count-up (`AnimatedCounter.tsx`)
- Hook: `useCountUp(target: number, duration = 900, formatter?)`
- Easing: `easeOutCubic = t => 1 - Math.pow(1-t, 3)`
- Triggers: when `target` changes from 0 or undefined to a real value
- Used on: every stat card number, compliance %, relocation %

### Card stagger
- CSS: `@keyframes slideUpFade { from { opacity:0; transform:translateY(12px) } to { opacity:1; transform:none } }`
- Each card in a grid gets `style={{ animationDelay: \`${index * 55}ms\` }}`
- Duration: 380ms, easeOutCubic

### Chart draw-in
- `stroke-dasharray + stroke-dashoffset` on Recharts line paths
- CSS: `@keyframes drawLine { from { stroke-dashoffset: var(--path-len) } to { stroke-dashoffset: 0 } }`
- Duration: 700ms, ease

### Button feedback
- `active:scale-[0.97]` via Tailwind

### Skeleton shimmer
- Warm tones: animates between `cream-200` and `cream-300`

---

## 7. New Components

| Component | Location | Purpose |
|---|---|---|
| `TopNav.tsx` | `components/layout/` | Sticky top nav, tab routing, Cmd+K trigger |
| `MobileNav.tsx` | `components/layout/` | Bottom tab bar for mobile |
| `AnimatedCounter.tsx` | `components/ui/` | Count-up number with formatter |
| `CommandPalette.tsx` | `components/ui/` | Cmd+K overlay, fuzzy search presets + navigation |
| `JurisdictionFlowDiagram.tsx` | `components/simulation/` | SVG Sankey: source → destination flows |
| `PopulationSummaryBar.tsx` | `components/simulation/` | Stacked horiz bar: compliant/relocated/evading |
| `NetworkStatsPanel.tsx` | `components/simulation/` | Mean degree + clustering cards |
| `ComplianceAreaChart.tsx` | `components/simulation/` | Area chart with confidence band gradient |
| `RelocationLineChart.tsx` | `components/simulation/` | Per-jurisdiction relocation lines |

---

## 8. Pages

### `/analyze` — Command Center
- **Remove** hero marketing section (replaced by compact header strip)
- Immediate two-panel layout on load
- Left: 380px aside — PresetSelector → PolicyParamPanel → SimConfigPanel → Run button → EvidencePackSection
- Right: empty state (bento preview) or results bento grid
- Policy name shown as large display heading in results header

### `/compare` — Policy Compare
- Top: slot manager (add up to 4 policies, X to remove)
- Bento grid: 2×2 area charts (shared Y axis), delta table below
- Best value per row: warm green tint; worst: warm red tint

### `/influence` — Influence Test
- Large SVG resilience score ring (0–1, color-banded, animated fill on result)
- Side-by-side compliance trajectory (baseline vs injected) with ReferenceLine at injection start
- Injection config: sliders with live value display

### `/upload` — Document Ingest
- Full-width dropzone with warm dashed border, drag-over state turns crimson
- Progress ring during processing
- Extracted fields accordion with confidence bars (warm fill)
- Auto-runs simulation after extraction

---

## 9. Removed / Changed

| Old | New |
|---|---|
| Dark glassmorphism (black bg, blue/cyan glow) | Warm paper (cream bg, crimson accent) |
| Left sidebar (304px) | Top navigation bar (56px) |
| Marketing hero section on analyze page | Compact page header + immediate work area |
| Neon chart colors (indigo, cyan, orange) | Warm chart palette (crimson, amber, forest green, deep blue) |
| `panel-shell` dark gradient | `card-warm` warm surface |
| `--accent-indigo`, `--accent-cyan` | `--crimson-700`, `--crimson-600` |
| `--surface-950` dark bg | `--cream-50` warm bg |

---

## 10. Files Changed

**Modified:**
- `app/globals.css` — full color + component token rewrite
- `app/layout.tsx` — wrap with TopNav, remove sidebar
- `app/analyze/page.tsx` — remove hero, bento results layout
- `app/compare/page.tsx` — bento grid, delta highlights
- `app/influence/page.tsx` — resilience ring, better layout
- `app/upload/page.tsx` — warm dropzone, confidence bars
- `components/layout/AppShell.tsx` — remove sidebar, top-nav layout
- `components/layout/Sidebar.tsx` — DELETED (replaced by TopNav)
- `components/policy/PolicyParamPanel.tsx` — warm color tokens
- `components/policy/PresetSelector.tsx` — warm styles
- `components/simulation/SimConfigPanel.tsx` — warm styles
- `components/simulation/ResultsPanel.tsx` — bento grid layout
- `components/simulation/FinalStocksGrid.tsx` — warm cards + AnimatedCounter
- `components/simulation/RoundSummaryChart.tsx` — warm chart palette, draw-in animation
- `components/simulation/SMMTable.tsx` — warm table styles
- `components/simulation/SimLoadingSkeleton.tsx` — warm shimmer
- `components/simulation/EpistemicBadge.tsx` — warm tier colors
- `components/simulation/EvidencePackSection.tsx` — warm styles
- `lib/constants.ts` — update CHART_COLORS
- `lib/format.ts` — update severityColor, epistemicColor

**New:**
- `components/layout/TopNav.tsx`
- `components/layout/MobileNav.tsx`
- `components/ui/AnimatedCounter.tsx`
- `components/ui/CommandPalette.tsx`
- `components/simulation/JurisdictionFlowDiagram.tsx`
- `components/simulation/PopulationSummaryBar.tsx`
- `components/simulation/NetworkStatsPanel.tsx`
- `components/simulation/ComplianceAreaChart.tsx`
- `components/simulation/RelocationLineChart.tsx`
