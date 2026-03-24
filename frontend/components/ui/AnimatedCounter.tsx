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
  const startRef    = useRef<number | null>(null);
  const frameRef    = useRef<number>(0);
  const prevTarget  = useRef<number>(0);

  useEffect(() => {
    const from = prevTarget.current;
    prevTarget.current = target;
    startRef.current = null;

    function step(ts: number) {
      if (startRef.current === null) startRef.current = ts;
      const elapsed  = ts - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      const value    = from + (target - from) * easeOutCubic(progress);
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

interface CounterProps {
  value: number;
  duration?: number;
  formatter?: (v: number) => string;
  className?: string;
}

export function AnimatedCounter({ value, duration = 900, formatter, className }: CounterProps) {
  const display = useCountUp(value, duration, formatter);
  return <span className={className}>{display}</span>;
}
