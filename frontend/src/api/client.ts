const API_BASE = import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000/v1";

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

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`HTTP ${res.status}: ${detail}`);
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
