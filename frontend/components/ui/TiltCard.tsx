"use client";

import { useRef, type ReactNode, type CSSProperties } from "react";

interface Props {
  children: ReactNode;
  className?: string;
  style?: CSSProperties;
  /** Max tilt angle in degrees (applied symmetrically on X and Y axes) */
  maxTilt?: number;
  /** Scale factor applied on hover */
  hoverScale?: number;
}

/**
 * Wrapper that applies a subtle 3D perspective tilt following the cursor.
 * Creates physical depth on data cards without any animation library.
 */
export function TiltCard({
  children,
  className,
  style,
  maxTilt = 6,
  hoverScale = 1.015,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);

  function handleMove(e: React.MouseEvent<HTMLDivElement>) {
    const el = ref.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    // Normalized -0.5 → +0.5
    const nx = (e.clientX - rect.left) / rect.width - 0.5;
    const ny = (e.clientY - rect.top) / rect.height - 0.5;
    el.style.transform = `perspective(700px) rotateX(${-ny * maxTilt}deg) rotateY(${nx * maxTilt}deg) scale(${hoverScale})`;
    el.style.transition = "transform 0.08s ease-out";
  }

  function handleLeave() {
    const el = ref.current;
    if (!el) return;
    el.style.transform = "";
    el.style.transition = "transform 0.5s cubic-bezier(0.23, 1, 0.32, 1)";
  }

  return (
    <div
      ref={ref}
      className={className}
      style={{ ...style, transformStyle: "preserve-3d", willChange: "transform" }}
      onMouseMove={handleMove}
      onMouseLeave={handleLeave}
    >
      {children}
    </div>
  );
}
