"use client";

import { useEffect, useRef, useState } from "react";

const SCRAMBLE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*";

function randomChar() {
  return SCRAMBLE_CHARS[Math.floor(Math.random() * SCRAMBLE_CHARS.length)];
}

interface Props {
  text: string;
  className?: string;
  style?: React.CSSProperties;
  /** Duration of the full scramble animation in ms */
  duration?: number;
  /** Delay before animation starts, in ms */
  delay?: number;
  /** Re-trigger the animation when this changes */
  animKey?: string | number;
}

/**
 * Characters scramble through random glyphs then lock into place left-to-right.
 * Used on primary headings to create an "information resolving" aesthetic.
 */
export function ScrambleText({
  text,
  className,
  style,
  duration = 900,
  delay = 0,
  animKey,
}: Props) {
  // Initialize with real text so server HTML and initial client HTML match.
  // The scramble kicks off only after mount (client-only).
  const [display, setDisplay] = useState(text);
  const rafRef = useRef<number | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(false);

  useEffect(() => {
    // Cancel any in-flight animation
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    if (timeoutRef.current) clearTimeout(timeoutRef.current);

    // Skip the very first render so SSR text is shown briefly before scrambling
    if (!mountedRef.current) {
      mountedRef.current = true;
    }

    const chars = text.split("");
    const revealed = new Array(chars.length).fill(false);
    let startTime: number | null = null;

    function step(now: number) {
      if (startTime === null) startTime = now;
      const elapsed = now - startTime;
      const t = Math.min(elapsed / duration, 1);

      // Reveal characters left-to-right
      const revealCount = Math.floor(t * chars.length);
      for (let i = 0; i < revealCount; i++) revealed[i] = true;

      const scrambled = chars.map((c, i) => {
        if (c === " " || revealed[i]) return c;
        return randomChar();
      });

      setDisplay(scrambled.join(""));

      if (t < 1) {
        rafRef.current = requestAnimationFrame(step);
      } else {
        setDisplay(text);
      }
    }

    timeoutRef.current = setTimeout(() => {
      rafRef.current = requestAnimationFrame(step);
    }, delay);

    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text, duration, delay, animKey]);

  return (
    <span className={className} style={{ ...style, fontVariantNumeric: "tabular-nums" }}>
      {display}
    </span>
  );
}
