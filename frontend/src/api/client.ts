const RAW_API_BASE = import.meta.env.VITE_API_BASE_URL ?? "/v1";
const API_BASE = RAW_API_BASE.endsWith("/v1") ? RAW_API_BASE : `${RAW_API_BASE.replace(/\/+$/, "")}/v1`;
const API_ORIGIN = API_BASE.endsWith("/v1") ? API_BASE.slice(0, -3) : API_BASE;

export class ApiError extends Error {
  status: number;
  detail: unknown;

  constructor(status: number, message: string, detail: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

function normalizeErrorMessage(status: number, detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) return detail.trim();
  if (detail && typeof detail === "object") {
    const obj = detail as Record<string, unknown>;
    if (typeof obj.message === "string" && obj.message.trim()) return obj.message.trim();
    if (typeof obj.detail === "string" && obj.detail.trim()) return obj.detail.trim();
  }
  return `Request failed (HTTP ${status})`;
}

async function throwApiError(res: Response): Promise<never> {
  let detail: unknown = null;
  const contentType = res.headers.get("content-type") || "";
  try {
    if (contentType.includes("application/json")) {
      const payload = await res.json();
      detail = payload?.detail ?? payload;
    } else {
      detail = await res.text();
    }
  } catch {
    detail = null;
  }
  throw new ApiError(res.status, normalizeErrorMessage(res.status, detail), detail);
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}

export type LayerInfo = {
  id: string;
  name: string;
  available: boolean;
  unit?: string;
  source?: string;
};

export type RoutePlanRequest = {
  timestamp: string;
  start: { lat: number; lon: number };
  goal: { lat: number; lon: number };
  policy: {
    objective: string;
    blocked_sources: string[];
    caution_mode: string;
    corridor_bias: number;
    smoothing: boolean;
    planner?: string;
  };
};

export type DynamicRoutePlanRequest = {
  timestamps: string[];
  start: { lat: number; lon: number };
  goal: { lat: number; lon: number };
  advance_steps: number;
  policy: {
    objective: string;
    blocked_sources: string[];
    caution_mode: string;
    corridor_bias: number;
    smoothing: boolean;
    planner?: string;
  };
};

export type RoutePlanResponse = {
  route_geojson: {
    type: "Feature";
    geometry: { type: "LineString"; coordinates: [number, number][] };
    properties: Record<string, unknown>;
  };
  explain: {
    distance_km: number;
    distance_nm: number;
    caution_len_km: number;
    corridor_alignment: number;
    [key: string]: unknown;
  };
  gallery_id: string;
  progress_id?: string;
  resolved?: {
    requested_date?: string;
    requested_hour?: number;
    progress_id?: string;
    used_timestamp?: string;
    source?: string;
    note?: string;
  };
  latest_meta?: Record<string, unknown>;
};

export type GalleryItem = {
  id: string;
  created_at: string;
  timestamp: string;
  layers: string[];
  start: { lat: number; lon: number };
  goal: { lat: number; lon: number };
  distance_km: number;
  caution_len_km: number;
  corridor_bias: number;
  model_version?: string;
  route_geojson?: {
    type: "Feature";
    geometry: { type: "LineString"; coordinates: [number, number][] };
    properties: Record<string, unknown>;
  };
  explain?: Record<string, unknown>;
  action?: {
    type?: string;
    timestamp?: string;
    start_input?: { lat: number; lon: number };
    goal_input?: { lat: number; lon: number };
    policy?: {
      objective?: string;
      blocked_sources?: string[];
      caution_mode?: string;
      corridor_bias?: number;
      smoothing?: boolean;
    };
  };
  result?: {
    status?: string;
    distance_km?: number;
    distance_nm?: number;
    caution_len_km?: number;
    corridor_alignment?: number;
    route_points?: number;
    raw_points?: number;
    smoothed_points?: number;
    start_adjusted?: boolean;
    goal_adjusted?: boolean;
    blocked_ratio?: number;
  };
  timeline?: Array<{ event?: string; status?: string; [key: string]: unknown }>;
  [key: string]: unknown;
};

export type InferResponse = {
  pred_layer: string;
  timestamp: string;
  output_file: string;
  stats: {
    shape: number[];
    class_hist: { safe: number; caution: number; blocked: number };
    class_ratio: { safe: number; caution: number; blocked: number };
    cache_hit: boolean;
    model_version: string;
    model_summary?: string;
    device?: string;
    uncertainty_file?: string;
    uncertainty_mean?: number | null;
    uncertainty_p90?: number | null;
  };
};

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    await throwApiError(res);
  }
  return res.json() as Promise<T>;
}

export async function getDatasets() {
  return apiFetch<{ dataset: { months: string[] } }>("/datasets");
}

export async function getDatasetsQuality(sampleLimit = 80) {
  return apiFetch<{
    summary: {
      status: "pass" | "warn" | "fail" | string;
      timestamp_count: number;
      first_timestamp?: string;
      last_timestamp?: string;
      issues_count?: number;
    };
    checks: Array<{ name: string; status: string; detail?: Record<string, unknown> }>;
    issues: string[];
  }>(`/datasets/quality?sample_limit=${encodeURIComponent(String(sampleLimit))}`);
}

export async function getTimestamps(month?: string) {
  const query = month ? `?month=${encodeURIComponent(month)}` : "";
  return apiFetch<{ timestamps: string[] }>(`/timestamps${query}`);
}

export async function getLayers(timestamp: string) {
  return apiFetch<{ timestamp: string; layers: LayerInfo[] }>(`/layers?timestamp=${encodeURIComponent(timestamp)}`);
}

export async function planRoute(payload: RoutePlanRequest) {
  return apiFetch<RoutePlanResponse>("/route/plan", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function planDynamicRoute(payload: DynamicRoutePlanRequest) {
  return apiFetch<RoutePlanResponse>("/route/plan/dynamic", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function planLatestRoute(payload: {
  date: string;
  hour?: number;
  force_refresh?: boolean;
  progress_id?: string;
  start: { lat: number; lon: number };
  goal: { lat: number; lon: number };
  policy: {
    objective: string;
    blocked_sources: string[];
    caution_mode: string;
    corridor_bias: number;
    smoothing: boolean;
    planner?: string;
  };
}) {
  return apiFetch<RoutePlanResponse>("/latest/plan", {
    method: "POST",
    body: JSON.stringify({
      date: payload.date,
      hour: payload.hour ?? 12,
      force_refresh: payload.force_refresh ?? true,
      progress_id: payload.progress_id,
      start: payload.start,
      goal: payload.goal,
      policy: payload.policy,
    }),
  });
}

export type CopernicusConfigPayload = {
  username?: string;
  password?: string;
  ice_dataset_id?: string;
  wave_dataset_id?: string;
  wind_dataset_id?: string;
  ice_var?: string;
  ice_thick_var?: string;
  wave_var?: string;
  wind_u_var?: string;
  wind_v_var?: string;
};

export async function setCopernicusConfig(payload: CopernicusConfigPayload) {
  return apiFetch<{
    ok: boolean;
    configured: boolean;
    datasets: Record<string, string>;
    variables: Record<string, string>;
  }>("/latest/copernicus/config", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getCopernicusConfig() {
  return apiFetch<{
    configured: boolean;
    username_set: boolean;
    password_set: boolean;
    datasets: Record<string, string>;
    variables: Record<string, string>;
  }>("/latest/copernicus/config");
}

export async function getLatestStatus(timestamp: string) {
  return apiFetch<{
    timestamp: string;
    has_latest_meta: boolean;
    meta: Record<string, unknown>;
  }>(`/latest/status?timestamp=${encodeURIComponent(timestamp)}`);
}

export type LatestProgress = {
  progress_id: string;
  exists: boolean;
  status: "running" | "completed" | "failed" | "not_found" | string;
  phase: string;
  message: string;
  percent: number;
  error?: string | null;
  updated_at?: string;
};

export async function getLatestProgress(progressId: string) {
  return apiFetch<LatestProgress>(`/latest/progress?progress_id=${encodeURIComponent(progressId)}`);
}

export async function runInference(payload: { timestamp: string; model_version?: string }) {
  return apiFetch<InferResponse>("/infer", {
    method: "POST",
    body: JSON.stringify({
      timestamp: payload.timestamp,
      model_version: payload.model_version ?? "unet_v1",
    }),
  });
}

export async function getGalleryList() {
  return apiFetch<{ items: GalleryItem[] }>("/gallery/list");
}

export async function getGalleryItem(galleryId: string) {
  return apiFetch<GalleryItem>(`/gallery/${encodeURIComponent(galleryId)}`);
}

export async function deleteGalleryItem(galleryId: string) {
  const res = await fetch(`${API_BASE}/gallery/${encodeURIComponent(galleryId)}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    await throwApiError(res);
  }
}

export async function uploadGalleryImage(galleryId: string, imageBase64: string) {
  const res = await fetch(`${API_BASE}/gallery/${encodeURIComponent(galleryId)}/image`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image_base64: imageBase64 }),
  });
  if (!res.ok) {
    await throwApiError(res);
  }
}

export type AisBacktestMetrics = {
  timestamp: string;
  source: string;
  route_point_count: number;
  route_inside_grid_ratio: number;
  route_mean_heat: number;
  route_p90_heat: number;
  global_mean_heat: number;
  global_p90_heat: number;
  top10pct_hit_rate: number;
  top25pct_hit_rate: number;
  median_or_higher_hit_rate: number;
  alignment_norm_0_1: number;
  alignment_zscore: number;
};

export type ActiveReviewRun = {
  run_id: string;
  created_at: string;
  candidate_count: number;
  top_k: number;
  mapping_count: number;
  accepted_count: number;
  needs_revision_count: number;
  summary_file: string;
};

export type ActiveReviewItem = {
  rank: number;
  timestamp: string;
  score: number;
  uncertainty_score: number;
  route_impact_score: number;
  class_balance_score: number;
  pred_caution_ratio: number;
  dominant_factor: string;
  explain_json: string;
  explain_png: string;
  explanation: Record<string, unknown>;
  decision?: {
    decision?: "accepted" | "needs_revision" | string;
    note?: string;
    updated_at?: string;
  };
};

export type AnnotationPoint = { lat: number; lon: number };
export type AnnotationOperation = {
  id?: string;
  mode: "add" | "erase";
  shape?: "polygon" | "stroke";
  radius_cells?: number;
  points: AnnotationPoint[];
};

export type AnnotationPatchResponse = {
  timestamp: string;
  updated_at?: string;
  patch_file: string;
  caution_file: string;
  y_class_file: string;
  operations: AnnotationOperation[];
  stats: {
    shape: number[];
    blocked_pixels: number;
    caution_pixels: number;
    caution_ratio: number;
    operations_count: number;
    has_patch_file?: boolean;
  };
};

export async function runAisBacktest(payload: { gallery_id?: string; timestamp?: string; route_geojson?: unknown }) {
  return apiFetch<{ metrics: AisBacktestMetrics; gallery_id?: string; note?: string }>("/eval/ais/backtest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getActiveReviewRuns() {
  return apiFetch<{ runs: ActiveReviewRun[] }>("/active/review/runs");
}

export async function getActiveReviewItems(runId?: string, limit = 20) {
  const params = new URLSearchParams();
  if (runId) params.set("run_id", runId);
  params.set("limit", String(limit));
  return apiFetch<{ run_id: string; items: ActiveReviewItem[]; count: number }>(`/active/review/items?${params.toString()}`);
}

export async function postActiveReviewDecision(payload: {
  run_id: string;
  timestamp: string;
  decision: "accepted" | "needs_revision";
  note?: string;
}) {
  return apiFetch<{
    ok: boolean;
    run_id: string;
    timestamp: string;
    decision: string;
    state_file: string;
    updated_at: string;
  }>("/active/review/decision", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function getAnnotationPatch(timestamp: string) {
  return apiFetch<AnnotationPatchResponse>(`/annotation/workspace/patch?timestamp=${encodeURIComponent(timestamp)}`);
}

export async function saveAnnotationPatch(payload: {
  timestamp: string;
  operations: AnnotationOperation[];
  note?: string;
  author?: string;
}) {
  return apiFetch<AnnotationPatchResponse>("/annotation/workspace/patch", {
    method: "POST",
    body: JSON.stringify({
      timestamp: payload.timestamp,
      operations: payload.operations,
      note: payload.note ?? "",
      author: payload.author ?? "web",
    }),
  });
}

export function getGalleryImageUrl(galleryId: string): string {
  return `${API_BASE}/gallery/${encodeURIComponent(galleryId)}/image.png`;
}

export function getApiOrigin(): string {
  return API_ORIGIN;
}
