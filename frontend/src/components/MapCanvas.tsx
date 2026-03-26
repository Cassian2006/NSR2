import { useMemo, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Pane,
  Polyline,
  Rectangle,
  TileLayer,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";
import { useEffect, useRef } from "react";

import { getApiOrigin } from "../api/client";
import type { GridBounds } from "../api/client";
import { useLanguage } from "../contexts/LanguageContext";

interface MapCanvasProps {
  timestamp: string;
  tileRevision?: number;
  layoutKey?: string;
  layers: {
    bathymetry: { enabled: boolean; opacity: number };
    riskMean: { enabled: boolean; opacity: number };
    aisHeatmap: { enabled: boolean; opacity: number };
    unetZones: { enabled: boolean; opacity: number };
    unetUncertainty: { enabled: boolean; opacity: number };
    ice: { enabled: boolean; opacity: number };
    wave: { enabled: boolean; opacity: number };
    wind: { enabled: boolean; opacity: number };
  };
  showRoute: boolean;
  gridBounds?: GridBounds | null;
  routeGeojson?: {
    geometry?: { coordinates?: [number, number][] };
    properties?: Record<string, unknown> & {
      display_coordinates?: [number, number][];
      feasible_smoothed_coordinates?: [number, number][];
      raw_coordinates?: [number, number][];
    };
  };
  secondaryRouteGeojsons?: Array<{
    geometry?: { coordinates?: [number, number][] };
    properties?: Record<string, unknown> & {
      display_coordinates?: [number, number][];
      feasible_smoothed_coordinates?: [number, number][];
      raw_coordinates?: [number, number][];
    };
  }>;
  selectedRouteKey?: string;
  start?: { lat: number; lon: number };
  goal?: { lat: number; lon: number };
  onMapClick?: (lat: number, lon: number) => void;
  replayOverlay?: {
    executedCoordinates?: [number, number][];
    currentSegment?: [number, number][];
    candidateSegment?: [number, number][];
  };
}

const API_ORIGIN = getApiOrigin();

function toLeafletBounds(bounds?: GridBounds | null): LatLngBoundsExpression {
  const valid =
    bounds &&
    Number.isFinite(bounds.lat_min) &&
    Number.isFinite(bounds.lat_max) &&
    Number.isFinite(bounds.lon_min) &&
    Number.isFinite(bounds.lon_max) &&
    bounds.lat_min < bounds.lat_max &&
    bounds.lon_min < bounds.lon_max;
  return [
    [valid ? bounds.lat_min : 60, valid ? bounds.lon_min : 20],
    [valid ? bounds.lat_max : 80, valid ? bounds.lon_max : 180],
  ];
}

function normalizeRoutePoint(
  point: [number, number],
  bounds?: GridBounds | null
): [number, number] {
  const [a, b] = point;
  const latMin = bounds?.lat_min ?? 60;
  const latMax = bounds?.lat_max ?? 80;
  const lonMin = bounds?.lon_min ?? 20;
  const lonMax = bounds?.lon_max ?? 180;
  const aLooksLikeLon = a >= lonMin && a <= lonMax;
  const bLooksLikeLat = b >= latMin && b <= latMax;
  const aLooksLikeLat = a >= latMin && a <= latMax;
  const bLooksLikeLon = b >= lonMin && b <= lonMax;

  if (aLooksLikeLon && bLooksLikeLat) {
    return [b, a];
  }
  if (aLooksLikeLat && bLooksLikeLon) {
    return [a, b];
  }
  return [b, a];
}

function normalizeRouteCoords(
  coords: [number, number][],
  bounds?: GridBounds | null
): [number, number][] {
  return coords.map((point) => normalizeRoutePoint(point, bounds));
}

function RasterTileLayer({
  layerId,
  enabled,
  opacity,
  timestamp,
  tileRevision,
  zIndex,
}: {
  layerId: string;
  enabled: boolean;
  opacity: number;
  timestamp: string;
  tileRevision: number;
  zIndex: number;
}) {
  if (!enabled || !timestamp) return null;
  const paneName = `overlay-${layerId}`;
  const rev = `${timestamp}-${tileRevision}`;
  const url = `${API_ORIGIN}/v1/tiles/${layerId}/{z}/{x}/{y}.png?timestamp=${encodeURIComponent(timestamp)}&v=${encodeURIComponent(rev)}`;
  return (
    <Pane name={paneName} style={{ zIndex }}>
      <TileLayer
        key={`${layerId}-${rev}`}
        pane={paneName}
        url={url}
        opacity={Math.max(0, Math.min(1, opacity / 100))}
        tileSize={256}
        noWrap
        updateWhenIdle
        crossOrigin="anonymous"
      />
    </Pane>
  );
}

function MapEvents({
  onMapClick,
  onMouseMove,
}: {
  onMapClick?: (lat: number, lon: number) => void;
  onMouseMove: (lat: number, lon: number) => void;
}) {
  useMapEvents({
    click: (e) => {
      if (onMapClick) onMapClick(e.latlng.lat, e.latlng.lng);
    },
    mousemove: (e) => onMouseMove(e.latlng.lat, e.latlng.lng),
  });
  return null;
}

function MapResizeGuard({ layoutKey }: { layoutKey: string }) {
  const map = useMap();

  useEffect(() => {
    const refresh = () => {
      map.invalidateSize({ pan: false });
    };
    const t1 = window.setTimeout(refresh, 0);
    const t2 = window.setTimeout(refresh, 180);
    const t3 = window.setTimeout(refresh, 520);
    let attempts = 0;
    const retry = window.setInterval(() => {
      attempts += 1;
      const size = map.getSize();
      refresh();
      if ((size.x > 0 && size.y > 0) || attempts >= 8) {
        window.clearInterval(retry);
      }
    }, 350);
    map.whenReady(() => {
      refresh();
    });
    window.addEventListener("resize", refresh, { passive: true });
    window.addEventListener("orientationchange", refresh, { passive: true });
    const viewport = window.visualViewport;
    viewport?.addEventListener("resize", refresh);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.clearTimeout(t3);
      window.clearInterval(retry);
      window.removeEventListener("resize", refresh);
      window.removeEventListener("orientationchange", refresh);
      viewport?.removeEventListener("resize", refresh);
    };
  }, [layoutKey, map]);

  return null;
}

function RouteFocusGuard({
  routeLatLng,
  selectedRouteKey,
}: {
  routeLatLng: [number, number][];
  selectedRouteKey: string;
}) {
  const map = useMap();
  const lastKeyRef = useRef<string>("");

  useEffect(() => {
    if (routeLatLng.length < 2) return;
    if (lastKeyRef.current === selectedRouteKey) return;
    lastKeyRef.current = selectedRouteKey;
    map.fitBounds(routeLatLng, {
      padding: [28, 28],
      maxZoom: 6,
      animate: true,
      duration: 0.45,
    });
  }, [map, routeLatLng, selectedRouteKey]);

  return null;
}

export default function MapCanvas({
  timestamp,
  tileRevision = 0,
  layoutKey = "auto",
  layers,
  gridBounds,
  showRoute,
  routeGeojson,
  secondaryRouteGeojsons,
  selectedRouteKey = "requested",
  start,
  goal,
  onMapClick,
  replayOverlay,
}: MapCanvasProps) {
  const { t } = useLanguage();
  const [mousePos, setMousePos] = useState({ lat: 79.234, lon: 45.678 });
  const mouseRafRef = useRef<number | null>(null);
  const pendingMouseRef = useRef<{ lat: number; lon: number } | null>(null);

  useEffect(() => {
    return () => {
      if (mouseRafRef.current !== null) {
        window.cancelAnimationFrame(mouseRafRef.current);
        mouseRafRef.current = null;
      }
    };
  }, []);

  const handleMouseMove = (lat: number, lon: number) => {
    pendingMouseRef.current = { lat, lon };
    if (mouseRafRef.current !== null) return;
    mouseRafRef.current = window.requestAnimationFrame(() => {
      mouseRafRef.current = null;
      const next = pendingMouseRef.current;
      if (next) {
        setMousePos(next);
      }
    });
  };

  const routeLatLng = useMemo(() => {
    const rawCoords = routeGeojson?.properties?.raw_coordinates;
    const feasibleCoords = routeGeojson?.properties?.feasible_smoothed_coordinates;
    const coords =
      rawCoords && rawCoords.length >= 2
        ? rawCoords
        : feasibleCoords && feasibleCoords.length >= 2
          ? feasibleCoords
          : routeGeojson?.geometry?.coordinates ?? [];
    return normalizeRouteCoords(coords, gridBounds);
  }, [gridBounds, routeGeojson]);

  const secondaryRoutesLatLng = useMemo(
    () =>
      (secondaryRouteGeojsons ?? [])
        .map((feature) => {
          const rawCoords = feature?.properties?.raw_coordinates;
          const feasibleCoords = feature?.properties?.feasible_smoothed_coordinates;
          const coords =
            rawCoords && rawCoords.length >= 2
              ? rawCoords
              : feasibleCoords && feasibleCoords.length >= 2
                ? feasibleCoords
                : feature?.geometry?.coordinates ?? [];
          return normalizeRouteCoords(coords, gridBounds);
        })
        .filter((coords) => coords.length >= 2),
    [gridBounds, secondaryRouteGeojsons]
  );

  const replayExecutedLatLng = useMemo(
    () => normalizeRouteCoords(replayOverlay?.executedCoordinates ?? [], gridBounds),
    [gridBounds, replayOverlay?.executedCoordinates]
  );
  const replayCurrentLatLng = useMemo(
    () => normalizeRouteCoords(replayOverlay?.currentSegment ?? [], gridBounds),
    [gridBounds, replayOverlay?.currentSegment]
  );
  const replayCandidateLatLng = useMemo(
    () => normalizeRouteCoords(replayOverlay?.candidateSegment ?? [], gridBounds),
    [gridBounds, replayOverlay?.candidateSegment]
  );
  const mapBounds = useMemo(() => toLeafletBounds(gridBounds), [gridBounds]);

  return (
    <div className="absolute inset-0">
      <MapContainer
        key={`map-${layoutKey}-${timestamp}`}
        bounds={mapBounds}
        className="h-full w-full"
        zoomSnap={0.25}
        minZoom={1}
        maxZoom={8}
        worldCopyJump={false}
        preferCanvas
      >
        <MapResizeGuard layoutKey={`${layoutKey}:${timestamp}`} />
        {showRoute && routeLatLng.length >= 2 ? <RouteFocusGuard routeLatLng={routeLatLng} selectedRouteKey={selectedRouteKey} /> : null}
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
          noWrap
        />

        <Pane name="aoi-frame" style={{ zIndex: 390 }}>
          <Rectangle
            bounds={mapBounds}
            pathOptions={{
              color: "#0ea5e9",
              weight: 2,
              opacity: 0.95,
              fill: false,
              dashArray: "8 6",
            }}
          />
        </Pane>

        <MapEvents
          onMapClick={onMapClick}
          onMouseMove={handleMouseMove}
        />

        <RasterTileLayer layerId="bathy" enabled={layers.bathymetry.enabled} opacity={layers.bathymetry.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={300} />
        <RasterTileLayer layerId="ice" enabled={layers.ice.enabled} opacity={layers.ice.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={320} />
        <RasterTileLayer layerId="wave" enabled={layers.wave.enabled} opacity={layers.wave.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={330} />
        <RasterTileLayer layerId="wind" enabled={layers.wind.enabled} opacity={layers.wind.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={340} />
        <RasterTileLayer layerId="risk_mean" enabled={layers.riskMean.enabled} opacity={layers.riskMean.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={350} />
        <RasterTileLayer layerId="ais_heatmap" enabled={layers.aisHeatmap.enabled} opacity={layers.aisHeatmap.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={360} />
        <RasterTileLayer layerId="unet_uncertainty" enabled={layers.unetUncertainty.enabled} opacity={layers.unetUncertainty.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={370} />
        <RasterTileLayer layerId="unet_pred" enabled={layers.unetZones.enabled} opacity={layers.unetZones.opacity} timestamp={timestamp} tileRevision={tileRevision ?? 0} zIndex={380} />

        {showRoute
          ? secondaryRoutesLatLng.map((coords, index) => (
              <Pane key={`secondary-route-${index}`} name={`secondary-route-pane-${index}`} style={{ zIndex: 385 }}>
                <Polyline
                  positions={coords}
                  pathOptions={{
                    color: "#6b7280",
                    weight: 3,
                    opacity: 0.38,
                    dashArray: "10 8",
                    lineCap: "round",
                    lineJoin: "round",
                    className: "map-route-secondary",
                  }}
                />
                <CircleMarker
                  center={coords[Math.max(0, Math.floor(coords.length / 2))]}
                  radius={10}
                  pathOptions={{ color: "#475569", weight: 1.5, fillColor: "#f8fafc", fillOpacity: 0.92 }}
                >
                  <Tooltip permanent direction="center" offset={[0, 0]} opacity={1}>
                    <span className="text-[10px] font-semibold text-slate-700">#{index + 1}</span>
                  </Tooltip>
                </CircleMarker>
              </Pane>
            ))
          : null}
        {showRoute && routeLatLng.length >= 2 ? (
          <Polyline
            positions={routeLatLng}
            pathOptions={{
              color: "#1e40af",
              weight: 4.5,
              opacity: 0.98,
              lineCap: "round",
              lineJoin: "round",
              className: "map-route-primary",
            }}
          />
        ) : null}
        {replayCandidateLatLng.length >= 2 ? (
          <Polyline
            positions={replayCandidateLatLng}
            pathOptions={{ color: "#f59e0b", weight: 3, opacity: 0.95, dashArray: "7 5" }}
          />
        ) : null}
        {replayExecutedLatLng.length >= 2 ? (
          <Polyline positions={replayExecutedLatLng} pathOptions={{ color: "#16a34a", weight: 4, opacity: 0.95 }} />
        ) : null}
        {replayCurrentLatLng.length >= 2 ? (
          <Polyline positions={replayCurrentLatLng} pathOptions={{ color: "#0ea5e9", weight: 5, opacity: 0.98 }} />
        ) : null}
        {start ? <CircleMarker center={[start.lat, start.lon]} radius={6} pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#10b981", fillOpacity: 1 }} /> : null}
        {goal ? <CircleMarker center={[goal.lat, goal.lon]} radius={6} pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#ef4444", fillOpacity: 1 }} /> : null}
      </MapContainer>

      <div className="pointer-events-none absolute bottom-4 left-4 rounded-lg bg-white/95 px-3 py-2 shadow-md backdrop-blur-sm">
        <div className="text-xs text-muted-foreground">{t("workspace.mousePosition")}</div>
        <div className="text-sm font-mono">
          北纬 {mousePos.lat.toFixed(3)}°, 东经 {mousePos.lon.toFixed(3)}°
        </div>
      </div>
    </div>
  );
}
