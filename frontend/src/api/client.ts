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
    risk_mode?: "conservative" | "balanced" | "aggressive" | string;
    risk_weight_scale?: number;
    risk_constraint_mode?: "none" | "chance" | "cvar" | string;
    risk_budget?: number;
    confidence_level?: number;
    return_candidates?: boolean;
    candidate_limit?: number;
    dynamic_risk_switch_enabled?: boolean;
    dynamic_risk_budget_km?: number;
    dynamic_risk_warn_ratio?: number;
    dynamic_risk_hard_ratio?: number;
    dynamic_risk_warn_mode?: "conservative" | "balanced" | "aggressive" | string;
    dynamic_risk_hard_mode?: "conservative" | "balanced" | "aggressive" | string;
    dynamic_risk_switch_min_interval?: number;
    vessel_profile_id?: string;
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
    risk_mode?: "conservative" | "balanced" | "aggressive" | string;
    risk_weight_scale?: number;
    risk_constraint_mode?: "none" | "chance" | "cvar" | string;
    risk_budget?: number;
    confidence_level?: number;
    return_candidates?: boolean;
    candidate_limit?: number;
    uncertainty_uplift?: boolean;
    uncertainty_uplift_scale?: number;
    dynamic_replan_mode?: "always" | "on_event" | string;
    replan_blocked_ratio?: number;
    replan_risk_spike?: number;
    replan_corridor_min?: number;
    replan_max_skip_steps?: number;
    dynamic_risk_switch_enabled?: boolean;
    dynamic_risk_budget_km?: number;
    dynamic_risk_warn_ratio?: number;
    dynamic_risk_hard_ratio?: number;
    dynamic_risk_warn_mode?: "conservative" | "balanced" | "aggressive" | string;
    dynamic_risk_hard_mode?: "conservative" | "balanced" | "aggressive" | string;
    dynamic_risk_switch_min_interval?: number;
    vessel_profile_id?: string;
  };
};

export type VesselProfile = {
  id: string;
  name: string;
  category: string;
  description: string;
  ice_class: string;
  draft_m: number;
  min_safe_depth_m: number;
  default_policy: {
    risk_mode: string;
    risk_weight_scale: number;
    risk_budget: number;
    confidence_level: number;
    corridor_bias_multiplier: number;
  };
};

export type DynamicExecutionEntry = {
  step: number;
  timestamp: string;
  update_mode?: string;
  triggered_replan?: boolean;
  trigger_reasons?: string[];
  moved_edges?: number;
  moved_distance_km?: number;
  step_effective_cost_km?: number;
  step_risk_extra_km?: number;
  cumulative_distance_km?: number;
  cumulative_risk_extra_km?: number;
  replan_runtime_ms?: number;
  cumulative_replan_runtime_ms?: number;
  segment_coordinates?: [number, number][];
  segment_start?: [number, number];
  segment_end?: [number, number];
  candidate_coordinates?: [number, number][];
};

export type DynamicReplanEntry = {
  step: number;
  timestamp: string;
  runtime_ms?: number;
  update_runtime_ms?: number;
  moved_distance_km?: number;
  step_effective_cost_km?: number;
  step_risk_extra_km?: number;
  changed_cells_total?: number;
  changed_edge_count?: number;
  update_mode?: string;
  triggered_replan?: boolean;
  trigger_reasons?: string[];
};

export type RouteCandidate = {
  id: string;
  label: string;
  strategy: string;
  status: "ok" | "failed" | string;
  distance_km?: number;
  risk_exposure?: number;
  caution_len_km?: number;
  corridor_score?: number;
  pareto_rank?: number | null;
  pareto_frontier?: boolean;
  pareto_order?: number | null;
  pareto_score?: number | null;
  planner?: string;
  risk_mode?: string;
  caution_mode?: string;
  error?: string;
  route_geojson?: {
    type: "Feature";
    geometry: { type: "LineString"; coordinates: [number, number][] };
    properties: Record<string, unknown>;
  };
  explain?: Record<string, unknown>;
  policy?: Record<string, unknown>;
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
  candidates?: RouteCandidate[];
  gallery_id: string;
  progress_id?: string;
  resolved?: {
    requested_date?: string;
    requested_hour?: number;
    progress_id?: string;
    used_timestamp?: string;
    source?: string;
    note?: string;
    dynamic?: {
      enabled?: boolean;
      mode?: string;
      requested_window?: number;
      requested_advance_steps?: number;
      used_timestamps?: string[];
      note?: string;
    };
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

export type GalleryRiskReport = {
  report_version: string;
  generated_at: string;
  gallery_id: string;
  timestamp: string;
  summary: Record<string, unknown>;
  risk: Record<string, unknown>;
  strategy: Record<string, unknown>;
  candidate_comparison: {
    count: number;
    ok_count: number;
    pareto_summary?: Record<string, unknown>;
    items: Array<Record<string, unknown>>;
  };
  explain: Record<string, unknown>;
  compliance?: ComplianceNoticesPayload;
};

export type ComplianceNotice = {
  id: string;
  severity: "low" | "medium" | "high" | string;
  messages: {
    en?: string;
    zh?: string;
    [key: string]: string | undefined;
  };
};

export type ComplianceNoticesPayload = {
  version: string;
  context: "workspace" | "export" | string;
  generated_at: string;
  notices: ComplianceNotice[];
  data_freshness: {
    timestamp?: string;
    source?: string;
    materialized_at?: string | null;
    age_hours?: number | null;
    status?: "fresh" | "stale" | "outdated" | "unknown" | string;
    hint?: { en?: string; zh?: string; [key: string]: string | undefined };
    [key: string]: unknown;
  };
  source_credibility: {
    level?: "normal" | "medium_risk" | "high_risk" | string;
    summary?: { healthy?: number; degraded?: number; blocked?: number };
    updated_at?: string;
    hint?: { en?: string; zh?: string; [key: string]: string | undefined };
    sources?: Record<string, unknown>;
    [key: string]: unknown;
  };
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

export async function getVesselProfiles() {
  return apiFetch<{ default_profile_id: string; profiles: VesselProfile[] }>("/vessels/profiles");
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
  dynamic_replan_enabled?: boolean;
  dynamic_window?: number;
  dynamic_advance_steps?: number;
  start: { lat: number; lon: number };
  goal: { lat: number; lon: number };
  policy: {
    objective: string;
    blocked_sources: string[];
    caution_mode: string;
    corridor_bias: number;
    smoothing: boolean;
    planner?: string;
    risk_mode?: "conservative" | "balanced" | "aggressive" | string;
    risk_weight_scale?: number;
    risk_constraint_mode?: "none" | "chance" | "cvar" | string;
    risk_budget?: number;
    confidence_level?: number;
    return_candidates?: boolean;
    candidate_limit?: number;
    dynamic_risk_switch_enabled?: boolean;
    dynamic_risk_budget_km?: number;
    dynamic_risk_warn_ratio?: number;
    dynamic_risk_hard_ratio?: number;
    dynamic_risk_warn_mode?: "conservative" | "balanced" | "aggressive" | string;
    dynamic_risk_hard_mode?: "conservative" | "balanced" | "aggressive" | string;
    dynamic_risk_switch_min_interval?: number;
    vessel_profile_id?: string;
  };
}) {
  return apiFetch<RoutePlanResponse>("/latest/plan", {
    method: "POST",
    body: JSON.stringify({
      date: payload.date,
      hour: payload.hour ?? 12,
      force_refresh: payload.force_refresh ?? true,
      progress_id: payload.progress_id,
      dynamic_replan_enabled: payload.dynamic_replan_enabled ?? false,
      dynamic_window: payload.dynamic_window ?? 6,
      dynamic_advance_steps: payload.dynamic_advance_steps ?? 12,
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

export async function getDeletedGalleryList() {
  return apiFetch<{ items: GalleryItem[] }>("/gallery/deleted");
}

export async function getGalleryItem(galleryId: string) {
  return apiFetch<GalleryItem>(`/gallery/${encodeURIComponent(galleryId)}`);
}

export async function getGalleryRiskReport(galleryId: string) {
  return apiFetch<GalleryRiskReport>(`/gallery/${encodeURIComponent(galleryId)}/risk-report`);
}

export async function getComplianceNotices(payload: { context: "workspace" | "export"; timestamp?: string }) {
  const params = new URLSearchParams();
  params.set("context", payload.context);
  if (payload.timestamp) params.set("timestamp", payload.timestamp);
  return apiFetch<ComplianceNoticesPayload>(`/compliance/notices?${params.toString()}`);
}

export async function getGalleryReportTemplate(
  galleryId: string,
  format: "json" | "csv" | "markdown" = "json"
): Promise<Record<string, unknown> | string> {
  const res = await fetch(`${API_BASE}/gallery/${encodeURIComponent(galleryId)}/report-template?format=${encodeURIComponent(format)}`);
  if (!res.ok) {
    await throwApiError(res);
  }
  if (format === "json") {
    return (await res.json()) as Record<string, unknown>;
  }
  return res.text();
}

export async function deleteGalleryItem(galleryId: string, softDelete = true) {
  const query = `?soft_delete=${softDelete ? "true" : "false"}`;
  const res = await fetch(`${API_BASE}/gallery/${encodeURIComponent(galleryId)}${query}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    await throwApiError(res);
  }
}

export async function restoreGalleryItem(galleryId: string) {
  const res = await fetch(`${API_BASE}/gallery/${encodeURIComponent(galleryId)}/restore`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    await throwApiError(res);
  }
  return (await res.json()) as { ok: boolean; gallery_id: string };
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
