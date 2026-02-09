import { useMemo, useRef, useState } from "react";

interface MapCanvasProps {
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

type Bounds = {
  minLat: number;
  maxLat: number;
  minLon: number;
  maxLon: number;
};

const BASE_BOUNDS: Bounds = {
  minLat: 66,
  maxLat: 86,
  minLon: -30,
  maxLon: 180,
};

const SVG_W = 1000;
const SVG_H = 600;

function expandBounds(bounds: Bounds, lat: number, lon: number): Bounds {
  return {
    minLat: Math.min(bounds.minLat, lat),
    maxLat: Math.max(bounds.maxLat, lat),
    minLon: Math.min(bounds.minLon, lon),
    maxLon: Math.max(bounds.maxLon, lon),
  };
}

function toCanvas(bounds: Bounds, lat: number, lon: number): [number, number] {
  const lonSpan = Math.max(1e-6, bounds.maxLon - bounds.minLon);
  const latSpan = Math.max(1e-6, bounds.maxLat - bounds.minLat);
  const x = ((lon - bounds.minLon) / lonSpan) * SVG_W;
  const y = SVG_H - ((lat - bounds.minLat) / latSpan) * SVG_H;
  return [x, y];
}

export default function MapCanvas({ layers, showRoute, routeGeojson, start, goal, onMapClick }: MapCanvasProps) {
  const canvasRef = useRef<HTMLDivElement>(null);
  const [mousePos, setMousePos] = useState({ lat: 79.234, lon: 45.678, x: 142, y: 67 });

  const routeCoordinates = useMemo(() => routeGeojson?.geometry?.coordinates ?? [], [routeGeojson]);

  const bounds = useMemo(() => {
    let b = { ...BASE_BOUNDS };
    if (start) b = expandBounds(b, start.lat, start.lon);
    if (goal) b = expandBounds(b, goal.lat, goal.lon);
    routeCoordinates.forEach(([lon, lat]) => {
      b = expandBounds(b, lat, lon);
    });
    const latPad = (b.maxLat - b.minLat) * 0.1;
    const lonPad = (b.maxLon - b.minLon) * 0.1;
    return {
      minLat: b.minLat - latPad,
      maxLat: b.maxLat + latPad,
      minLon: b.minLon - lonPad,
      maxLon: b.maxLon + lonPad,
    };
  }, [goal, routeCoordinates, start]);

  const routePath = useMemo(() => {
    if (!routeCoordinates.length) return "";
    const points = routeCoordinates.map(([lon, lat]) => toCanvas(bounds, lat, lon));
    const [head, ...tail] = points;
    return `M ${head[0].toFixed(2)} ${head[1].toFixed(2)} ` + tail.map(([x, y]) => `L ${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
  }, [bounds, routeCoordinates]);

  const startPoint = start ? toCanvas(bounds, start.lat, start.lon) : null;
  const goalPoint = goal ? toCanvas(bounds, goal.lat, goal.lon) : null;

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const lon = bounds.minLon + (x / rect.width) * (bounds.maxLon - bounds.minLon);
    const lat = bounds.maxLat - (y / rect.height) * (bounds.maxLat - bounds.minLat);
    const gridX = Math.floor((x / rect.width) * 200);
    const gridY = Math.floor((y / rect.height) * 100);
    setMousePos({ lat, lon, x: gridX, y: gridY });
  };

  const handleClick = () => {
    if (onMapClick) onMapClick(mousePos.lat, mousePos.lon);
  };

  return (
    <div ref={canvasRef} className="absolute inset-0 bg-gradient-to-br from-slate-100 to-slate-200 cursor-crosshair" onMouseMove={handleMouseMove} onClick={handleClick}>
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage: `
            linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px)
          `,
          backgroundSize: "50px 50px",
        }}
      />

      {layers.bathymetry.enabled && <div className="absolute inset-0 bg-red-500" style={{ opacity: layers.bathymetry.opacity / 400 }} />}
      {layers.unetZones.enabled && (
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-1/3 h-full bg-green-500" style={{ opacity: layers.unetZones.opacity / 300 }} />
          <div className="absolute top-0 right-0 w-1/3 h-full bg-amber-500" style={{ opacity: layers.unetZones.opacity / 300 }} />
        </div>
      )}
      {layers.aisHeatmap.enabled && (
        <div className="absolute inset-0 bg-gradient-to-r from-blue-400 via-cyan-400 to-yellow-400" style={{ opacity: layers.aisHeatmap.opacity / 400 }} />
      )}
      {layers.ice.enabled && <div className="absolute inset-0 bg-gradient-to-b from-blue-100 to-transparent" style={{ opacity: layers.ice.opacity / 200 }} />}

      {showRoute && (
        <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${SVG_W} ${SVG_H}`} preserveAspectRatio="none">
          {routePath ? <path d={routePath} stroke="#1e40af" strokeWidth="4" fill="none" strokeLinecap="round" strokeLinejoin="round" /> : null}
          {startPoint ? <circle cx={startPoint[0]} cy={startPoint[1]} r="8" fill="#10b981" stroke="white" strokeWidth="2" /> : null}
          {goalPoint ? <circle cx={goalPoint[0]} cy={goalPoint[1]} r="8" fill="#ef4444" stroke="white" strokeWidth="2" /> : null}
        </svg>
      )}

      <div className="absolute bottom-4 left-4 bg-white/95 backdrop-blur-sm px-3 py-2 rounded-lg shadow-md pointer-events-none">
        <div className="text-xs text-muted-foreground">Mouse Position</div>
        <div className="text-sm font-mono">
          {mousePos.lat.toFixed(3)} degN, {mousePos.lon.toFixed(3)} degE
        </div>
        <div className="text-xs text-muted-foreground">Grid: [{mousePos.x}, {mousePos.y}]</div>
      </div>
    </div>
  );
}
