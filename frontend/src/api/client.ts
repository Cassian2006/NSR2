const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/v1";
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

export async function runAisBacktest(payload: { gallery_id?: string; timestamp?: string; route_geojson?: unknown }) {
  return apiFetch<{ metrics: AisBacktestMetrics; gallery_id?: string; note?: string }>("/eval/ais/backtest", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function getGalleryImageUrl(galleryId: string): string {
  return `${API_BASE}/gallery/${encodeURIComponent(galleryId)}/image.png`;
}

export function getApiOrigin(): string {
  return API_ORIGIN;
}
