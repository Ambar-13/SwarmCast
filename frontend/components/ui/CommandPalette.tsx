"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { Search } from "lucide-react";
import { fetchPresets } from "@/lib/api-client";
import { queryKeys } from "@/lib/query-keys";
import { NAV_TABS } from "@/lib/nav-tabs";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function CommandPalette({ open, onClose }: Props) {
  const router   = useRouter();
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  const { data } = useQuery({
    queryKey: queryKeys.presets(),
    queryFn:  fetchPresets,
    staleTime: Infinity,
  });

  // Focus input when opened
  useEffect(() => {
    if (open) {
      setQuery("");
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  // Escape closes
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  if (!open) return null;

  const q = query.toLowerCase();
  const matchedPages   = NAV_TABS.filter((t) => t.label.toLowerCase().includes(q));
  const matchedPresets = (data?.presets ?? []).filter(
    (p) => p.label.toLowerCase().includes(q) || p.slug.includes(q),
  );

  function navigate(href: string) {
    router.push(href);
    onClose();
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-20"
      style={{ background: "rgba(26,18,8,0.38)", backdropFilter: "blur(4px)" }}
      onClick={onClose}
    >
      <div
        className="card-raised w-full max-w-lg overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Search row */}
        <div
          className="flex items-center gap-3 border-b px-4 py-3"
          style={{ borderColor: "var(--border-warm)" }}
        >
          <Search size={15} style={{ color: "var(--ink-400)" }} className="flex-shrink-0" />
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search pages or presets…"
            className="flex-1 bg-transparent text-sm outline-none"
            style={{ color: "var(--ink-900)" }}
          />
          <kbd
            className="flex-shrink-0 rounded border px-1.5 py-0.5 text-[10px]"
            style={{
              color:       "var(--ink-400)",
              borderColor: "var(--border-warm)",
              background:  "var(--cream-200)",
            }}
          >
            ESC
          </kbd>
        </div>

        {/* Results */}
        <div className="max-h-80 overflow-y-auto p-2">
          {matchedPages.length > 0 && (
            <section className="mb-3">
              <p className="kicker px-2 py-1.5">Pages</p>
              {matchedPages.map(({ href, label, icon: Icon }) => (
                <button
                  key={href}
                  onClick={() => navigate(href)}
                  className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors hover:bg-[var(--cream-200)]"
                  style={{ color: "var(--ink-700)" }}
                >
                  <Icon size={15} style={{ color: "var(--ink-400)" }} />
                  {label}
                </button>
              ))}
            </section>
          )}

          {matchedPresets.length > 0 && (
            <section>
              <p className="kicker px-2 py-1.5">Presets</p>
              {matchedPresets.map((p) => (
                <button
                  key={p.slug}
                  onClick={() => navigate("/analyze")}
                  className="flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition-colors hover:bg-[var(--cream-200)]"
                  style={{ color: "var(--ink-700)" }}
                >
                  <span
                    className="h-2 w-2 flex-shrink-0 rounded-full"
                    style={{ background: "var(--ink-400)" }}
                  />
                  {p.label}
                </button>
              ))}
            </section>
          )}

          {matchedPages.length === 0 && matchedPresets.length === 0 && (
            <p
              className="px-3 py-8 text-center text-sm"
              style={{ color: "var(--ink-400)" }}
            >
              No results for &ldquo;{query}&rdquo;
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
