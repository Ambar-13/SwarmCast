import { API_BASE } from "@/lib/constants";
import type {
  PresetPolicy,
  SimulateRequest,
  SimulateResponse,
  InjectRequest,
  InjectionResult,
  CompareRequest,
  CompareResponse,
  EvidencePackRequest,
  JobStatus,
  UploadResponse,
} from "@/lib/types";

async function apiFetch<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  }
  return res.json() as Promise<T>;
}

export async function fetchPresets(): Promise<{ presets: PresetPolicy[] }> {
  return apiFetch("/presets");
}

export async function simulate(req: SimulateRequest): Promise<SimulateResponse> {
  return apiFetch("/simulate", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function simulateUpload(
  file: File,
  apiKey?: string,
  model?: string,
): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  if (apiKey) form.append("api_key", apiKey);
  if (model) form.append("model", model);

  const res = await fetch(`${API_BASE}/simulate/upload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  }
  return res.json();
}

export async function comparePolicies(
  req: CompareRequest,
): Promise<CompareResponse> {
  return apiFetch("/compare", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function runInjection(req: InjectRequest): Promise<InjectionResult> {
  return apiFetch("/inject", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function startEvidencePack(
  req: EvidencePackRequest,
): Promise<{ job_id: string; status: string; estimated_seconds: number }> {
  return apiFetch("/evidence-pack", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function pollEvidencePack(jobId: string): Promise<JobStatus> {
  return apiFetch(`/evidence-pack/${jobId}`);
}
