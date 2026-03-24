// components/ui/CommandPalette.tsx — stub, will be replaced in Task 4
"use client";
interface Props { open: boolean; onClose: () => void; }
export function CommandPalette({ open, onClose }: Props) {
  if (!open) return null;
  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center px-4 pt-24"
      style={{ background: "rgba(26,18,8,0.4)" }}
      onClick={onClose}
    >
      <div className="card-raised w-full max-w-lg p-6 text-center text-sm" style={{ color: "var(--ink-400)" }}>
        Command palette coming soon — press Escape to close
      </div>
    </div>
  );
}
