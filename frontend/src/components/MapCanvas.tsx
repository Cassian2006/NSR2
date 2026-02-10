import { useMemo, useState } from "react";
import {
  CircleMarker,
  MapContainer,
  Pane,
  Polyline,
  TileLayer,
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

const API_ORIGIN = getApiOrigin();
const INITIAL_BOUNDS: LatLngBoundsExpression = [
  [66, -180],
  [86, 180],
];

function RasterTileLayer({
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
  if (!enabled || !timestamp) return null;
  const paneName = `overlay-${layerId}`;
  const url = `${API_ORIGIN}/v1/tiles/${layerId}/{z}/{x}/{y}.png?timestamp=${encodeURIComponent(timestamp)}&v=${encodeURIComponent(
    timestamp
  )}`;
  return (
    <Pane name={paneName} style={{ zIndex }}>
      <TileLayer
        key={`${layerId}-${timestamp}`}
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
          noWrap
        />

        <MapEvents
          onMapClick={onMapClick}
          onMouseMove={(lat, lon) => {
            setMousePos({ lat, lon });
          }}
        />

        <RasterTileLayer layerId="bathy" enabled={layers.bathymetry.enabled} opacity={layers.bathymetry.opacity} timestamp={timestamp} zIndex={300} />
        <RasterTileLayer layerId="ice" enabled={layers.ice.enabled} opacity={layers.ice.opacity} timestamp={timestamp} zIndex={320} />
        <RasterTileLayer layerId="wave" enabled={layers.wave.enabled} opacity={layers.wave.opacity} timestamp={timestamp} zIndex={330} />
        <RasterTileLayer layerId="wind" enabled={layers.wind.enabled} opacity={layers.wind.opacity} timestamp={timestamp} zIndex={340} />
        <RasterTileLayer layerId="ais_heatmap" enabled={layers.aisHeatmap.enabled} opacity={layers.aisHeatmap.opacity} timestamp={timestamp} zIndex={360} />
        <RasterTileLayer layerId="unet_pred" enabled={layers.unetZones.enabled} opacity={layers.unetZones.opacity} timestamp={timestamp} zIndex={380} />

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
