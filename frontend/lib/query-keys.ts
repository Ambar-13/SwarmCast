/** TanStack Query key factories. */

export const queryKeys = {
  presets: () => ["presets"] as const,
  evidencePack: (jobId: string) => ["evidence-pack", jobId] as const,
};
