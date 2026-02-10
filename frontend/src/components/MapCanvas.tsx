import { useCallback, useEffect, useMemo, useState } from "react";
import {
  CircleMarker,
  ImageOverlay,
  MapContainer,
  Polyline,
  TileLayer,
  useMap,
  useMapEvents,
} from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

import { getApiOrigin } from "../api/client";
import { useLanguage } from "../contexts/LanguageContext";

interface MapCanvasProps {
  timestamp: string;
  layers: {
    bathymetry: { enabled: boolean; opacity: number };
    aisHeatmap: { enabled: boolean; opacity: number };
    unetZones: { enabled: boolean; opacity: number };
    ice: { enabled: boolean; opacity: number };
    wave: { enabled: boolean; opacity: number };
    wind: { enabled: boolean; opacity: number };
  };
  showRoute: boolean;
  routeGeojson?: {
    geometry?: { coordinates?: [number, number][] };
  };
  start?: { lat: number; lon: number };
  goal?: { lat: number; lon: number };
  onMapClick?: (lat: number, lon: number) => void;
}

type OverlayState = {
  url: string;
  bounds: LatLngBoundsExpression;
};

const API_ORIGIN = getApiOrigin();
const INITIAL_BOUNDS: LatLngBoundsExpression = [
  [66, -180],
  [86, 180],
];

function OverlayLayer({
  layerId,
  enabled,
  opacity,
  timestamp,
  zIndex,
}: {
  layerId: string;
  enabled: boolean;
  opacity: number;
  timestamp: string;
  zIndex: number;
}) {
  const map = useMap();
  const [overlay, setOverlay] = useState<OverlayState | null>(null);

  const refresh = useCallback(() => {
    if (!enabled || !timestamp) {
      setOverlay(null);
      return;
    }
    const b = map.getBounds();
    const size = map.getSize();
    const width = Math.max(256, Math.round(size.x));
    const height = Math.max(256, Math.round(size.y));
    const bbox = [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()].map((v) => v.toFixed(6)).join(",");
    const url =
      `${API_ORIGIN}/v1/overlay/${layerId}.png` +
      `?timestamp=${encodeURIComponent(timestamp)}` +
      `&bbox=${encodeURIComponent(bbox)}` +
      `&size=${width},${height}` +
      `&v=${encodeURIComponent(`${timestamp}-${b.toBBoxString()}-${width}x${height}`)}`;
    setOverlay({
      url,
      bounds: [
        [b.getSouth(), b.getWest()],
        [b.getNorth(), b.getEast()],
      ],
    });
  }, [enabled, layerId, map, timestamp]);

  useMapEvents({
    moveend: refresh,
    zoomend: refresh,
    resize: refresh,
  });

  // Trigger first load and timestamp/layer changes.
  useEffect(() => {
    refresh();
  }, [refresh]);

  if (!enabled || !overlay) return null;
  return <ImageOverlay url={overlay.url} bounds={overlay.bounds} opacity={Math.max(0, Math.min(1, opacity / 100))} zIndex={zIndex} />;
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

export default function MapCanvas({ timestamp, layers, showRoute, routeGeojson, start, goal, onMapClick }: MapCanvasProps) {
  const { t } = useLanguage();
  const [mousePos, setMousePos] = useState({ lat: 79.234, lon: 45.678 });

  const routeLatLng = useMemo(() => {
    const coords = routeGeojson?.geometry?.coordinates ?? [];
    return coords.map(([lon, lat]) => [lat, lon] as [number, number]);
  }, [routeGeojson]);

  return (
    <div className="absolute inset-0">
      <MapContainer
        bounds={INITIAL_BOUNDS}
        className="h-full w-full"
        zoomSnap={0.25}
        minZoom={1}
        maxZoom={8}
        worldCopyJump={false}
      >
        <TileLayer
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          attribution="&copy; OpenStreetMap contributors"
        />

        <MapEvents
          onMapClick={onMapClick}
          onMouseMove={(lat, lon) => {
            setMousePos({ lat, lon });
          }}
        />

        <OverlayLayer layerId="bathy" enabled={layers.bathymetry.enabled} opacity={layers.bathymetry.opacity} timestamp={timestamp} zIndex={300} />
        <OverlayLayer layerId="ice" enabled={layers.ice.enabled} opacity={layers.ice.opacity} timestamp={timestamp} zIndex={320} />
        <OverlayLayer layerId="wave" enabled={layers.wave.enabled} opacity={layers.wave.opacity} timestamp={timestamp} zIndex={330} />
        <OverlayLayer layerId="wind" enabled={layers.wind.enabled} opacity={layers.wind.opacity} timestamp={timestamp} zIndex={340} />
        <OverlayLayer layerId="ais_heatmap" enabled={layers.aisHeatmap.enabled} opacity={layers.aisHeatmap.opacity} timestamp={timestamp} zIndex={360} />
        <OverlayLayer layerId="unet_pred" enabled={layers.unetZones.enabled} opacity={layers.unetZones.opacity} timestamp={timestamp} zIndex={380} />

        {showRoute && routeLatLng.length >= 2 ? (
          <Polyline positions={routeLatLng} pathOptions={{ color: "#1e40af", weight: 4, opacity: 0.95 }} />
        ) : null}
        {start ? <CircleMarker center={[start.lat, start.lon]} radius={6} pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#10b981", fillOpacity: 1 }} /> : null}
        {goal ? <CircleMarker center={[goal.lat, goal.lon]} radius={6} pathOptions={{ color: "#ffffff", weight: 2, fillColor: "#ef4444", fillOpacity: 1 }} /> : null}
      </MapContainer>

      <div className="pointer-events-none absolute bottom-4 left-4 rounded-lg bg-white/95 px-3 py-2 shadow-md backdrop-blur-sm">
        <div className="text-xs text-muted-foreground">{t("workspace.mousePosition")}</div>
        <div className="text-sm font-mono">
          {mousePos.lat.toFixed(3)} degN, {mousePos.lon.toFixed(3)} degE
        </div>
      </div>
    </div>
  );
}
