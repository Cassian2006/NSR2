import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { AlertCircle, CheckCircle2, Cpu, Navigation } from "lucide-react";
import { toast } from "sonner";

import {
  getErrorMessage,
  getLayers,
  getTimestamps,
  planRoute,
  runInference,
  uploadGalleryImage,
  type InferResponse,
  type RoutePlanResponse,
} from "../api/client";
import CoordinateInput from "../components/CoordinateInput";
import LayerToggle from "../components/LayerToggle";
import LegendCard from "../components/LegendCard";
import MapCanvas from "../components/MapCanvas";
import StatCard from "../components/StatCard";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { Label } from "../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Slider } from "../components/ui/slider";
import { useLanguage } from "../contexts/LanguageContext";

type LayerState = {
  enabled: boolean;
  opacity: number;
};

type LayerStates = {
  bathymetry: LayerState;
  aisHeatmap: LayerState;
  unetZones: LayerState;
  ice: LayerState;
  wave: LayerState;
  wind: LayerState;
};

type LayoutMode = "auto" | "desktop" | "mobile";

const DEFAULT_LAYERS: LayerStates = {
  bathymetry: { enabled: true, opacity: 80 },
  aisHeatmap: { enabled: true, opacity: 60 },
  unetZones: { enabled: true, opacity: 70 },
  ice: { enabled: false, opacity: 50 },
  wave: { enabled: false, opacity: 50 },
  wind: { enabled: false, opacity: 50 },
};

const AVAILABILITY_DEFAULT = {
  bathy: true,
  ais_heatmap: true,
  unet_pred: false,
  ice: true,
  wave: true,
  wind: true,
};

export default function MapWorkspace() {
  const { t } = useLanguage();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const queryTimestamp = searchParams.get("timestamp") ?? "";

  const [timestampOptions, setTimestampOptions] = useState<string[]>([]);
  const [timestamp, setTimestamp] = useState(queryTimestamp);
  const [startLat, setStartLat] = useState("70.5000");
  const [startLon, setStartLon] = useState("30.0000");
  const [goalLat, setGoalLat] = useState("72.0000");
  const [goalLon, setGoalLon] = useState("150.0000");

  const [layers, setLayers] = useState<LayerStates>(DEFAULT_LAYERS);
  const [availability, setAvailability] = useState(AVAILABILITY_DEFAULT);

  const [safetyPolicy, setSafetyPolicy] = useState("blocked-bathy-unet");
  const [cautionHandling, setCautionHandling] = useState("tiebreaker");
  const [corridorBias, setCorridorBias] = useState([20]);
  const [routeResult, setRouteResult] = useState<RoutePlanResponse | null>(null);
  const [planning, setPlanning] = useState(false);
  const [pickTarget, setPickTarget] = useState<"start" | "goal" | null>(null);
  const [inferring, setInferring] = useState(false);
  const [inferResult, setInferResult] = useState<InferResponse | null>(null);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("auto");
  const [isWideViewport, setIsWideViewport] = useState(false);
  const mapCaptureRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const mql = window.matchMedia("(min-width: 1024px)");
    const apply = () => setIsWideViewport(mql.matches);
    apply();
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", apply);
      return () => mql.removeEventListener("change", apply);
    }
    mql.addListener(apply);
    return () => mql.removeListener(apply);
  }, []);

  useEffect(() => {
    let active = true;
    async function loadTimestamps() {
      try {
        const res = await getTimestamps();
        if (!active) return;
        setTimestampOptions(res.timestamps);
        setTimestamp((prev) => {
          if (prev && res.timestamps.includes(prev)) return prev;
          return queryTimestamp && res.timestamps.includes(queryTimestamp) ? queryTimestamp : res.timestamps[0] ?? "";
        });
      } catch (error) {
        console.warn("timestamps api unavailable", error);
      }
    }
    loadTimestamps();
    return () => {
      active = false;
    };
  }, [queryTimestamp]);

  const refreshLayerAvailability = useCallback(async (ts: string) => {
    const res = await getLayers(ts);
    const nextAvailability = {
      bathy: res.layers.find((l) => l.id === "bathy")?.available ?? false,
      ais_heatmap: res.layers.find((l) => l.id === "ais_heatmap")?.available ?? false,
      unet_pred: res.layers.find((l) => l.id === "unet_pred")?.available ?? false,
      ice: res.layers.find((l) => l.id === "ice")?.available ?? false,
      wave: res.layers.find((l) => l.id === "wave")?.available ?? false,
      wind: res.layers.find((l) => l.id === "wind")?.available ?? false,
    };
    setAvailability(nextAvailability);
  }, []);

  useEffect(() => {
    if (!timestamp) return;
    let active = true;
    async function loadLayerAvailability() {
      try {
        await refreshLayerAvailability(timestamp);
      } catch (error) {
        if (!active) return;
        console.warn("layers api unavailable", error);
      }
    }
    loadLayerAvailability();
    return () => {
      active = false;
    };
  }, [refreshLayerAvailability, timestamp]);

  const handleLayerToggle = (layer: keyof LayerStates, enabled: boolean) => {
    setLayers((prev) => ({
      ...prev,
      [layer]: { ...prev[layer], enabled },
    }));
  };

  const handleOpacityChange = (layer: keyof LayerStates, opacity: number) => {
    setLayers((prev) => ({
      ...prev,
      [layer]: { ...prev[layer], opacity },
    }));
  };

  const routeMetrics = routeResult?.explain;

  const routeSummary = useMemo(
    () => ({
      distanceKm: routeMetrics?.distance_km ?? 0,
      distanceNm: routeMetrics?.distance_nm ?? 0,
      cautionPct:
        routeMetrics && routeMetrics.distance_km > 0
          ? ((routeMetrics.caution_len_km / routeMetrics.distance_km) * 100).toFixed(1)
          : "0.0",
      safePct:
        routeMetrics && routeMetrics.distance_km > 0
          ? (100 - (routeMetrics.caution_len_km / routeMetrics.distance_km) * 100).toFixed(1)
          : "100.0",
      alignment: routeMetrics?.corridor_alignment ?? 0,
    }),
    [routeMetrics]
  );

  const handleOpenLatestGallery = () => {
    if (!routeResult?.gallery_id) return;
    navigate(`/export?gallery=${encodeURIComponent(routeResult.gallery_id)}`);
  };

  const captureAndUploadGalleryImage = useCallback(async (galleryId: string) => {
    if (!mapCaptureRef.current) return;
    try {
      const { default: html2canvas } = await import("html2canvas");
      await new Promise((resolve) => setTimeout(resolve, 450));
      const canvas = await html2canvas(mapCaptureRef.current, {
        useCORS: true,
        allowTaint: false,
        backgroundColor: "#f8fafc",
        scale: Math.min(2, window.devicePixelRatio || 1),
      });
      const dataUrl = canvas.toDataURL("image/png");
      await uploadGalleryImage(galleryId, dataUrl);
      toast.success(t("toast.galleryShotUpdated"), { duration: 1800 });
    } catch (error) {
      // Keep planning success even if screenshot capture fails.
      console.warn("gallery screenshot upload failed", error);
      toast.warning(t("toast.galleryShotFallback"));
    }
  }, [t]);

  const handlePlanRoute = async () => {
    const startLatNum = Number.parseFloat(startLat);
    const startLonNum = Number.parseFloat(startLon);
    const goalLatNum = Number.parseFloat(goalLat);
    const goalLonNum = Number.parseFloat(goalLon);
    if ([startLatNum, startLonNum, goalLatNum, goalLonNum].some((v) => Number.isNaN(v))) {
      toast.error(t("toast.invalidCoords"));
      return;
    }
    if (!timestamp) {
      toast.error(t("toast.tsRequired"));
      return;
    }
    setPlanning(true);
    toast.loading(t("toast.planning"), { id: "plan-route" });
    try {
      const cautionMode = safetyPolicy === "strict" ? "strict" : cautionHandling === "tiebreaker" ? "tie_breaker" : cautionHandling;
      const blockedSources =
        safetyPolicy === "blocked-bathy-only"
          ? ["bathy"]
          : safetyPolicy === "strict"
            ? ["bathy", "unet_blocked", "unet_caution"]
            : ["bathy", "unet_blocked"];
      const response = await planRoute({
        timestamp,
        start: { lat: startLatNum, lon: startLonNum },
        goal: { lat: goalLatNum, lon: goalLonNum },
        policy: {
          objective: "shortest_distance_under_safety",
          blocked_sources: blockedSources,
          caution_mode: cautionMode,
          corridor_bias: corridorBias[0] / 100,
          smoothing: true,
        },
      });
      setRouteResult(response);
      const startAdjusted = Boolean(response.explain?.["start_adjusted"]);
      const goalAdjusted = Boolean(response.explain?.["goal_adjusted"]);
      if (startAdjusted || goalAdjusted) {
        toast.warning("Start/Goal adjusted to nearest navigable cell", { id: "plan-adjusted" });
      }
      toast.success(`${t("toast.success")} (Gallery: ${response.gallery_id})`, { id: "plan-route" });
      void captureAndUploadGalleryImage(response.gallery_id);
    } catch (error) {
      toast.error(`${t("toast.planFailed")}: ${getErrorMessage(error)}`, { id: "plan-route" });
    } finally {
      setPlanning(false);
    }
  };

  const handleRunInference = async () => {
    if (!timestamp) {
      toast.error(t("toast.tsRequired"));
      return;
    }
    setInferring(true);
    toast.loading(t("toast.inferRunning"), { id: "run-infer" });
    try {
      const res = await runInference({ timestamp, model_version: "unet_v1" });
      setInferResult(res);
      await refreshLayerAvailability(timestamp);
      toast.success(`${t("toast.inferDone")} (${res.stats.cache_hit ? "cache hit" : "fresh run"})`, { id: "run-infer" });
    } catch (error) {
      toast.error(`${t("toast.inferFailed")}: ${getErrorMessage(error)}`, { id: "run-infer" });
    } finally {
      setInferring(false);
    }
  };

  const handlePickStart = () => {
    setPickTarget("start");
    toast.info(t("toast.pickStart"));
  };

  const handlePickGoal = () => {
    setPickTarget("goal");
    toast.info(t("toast.pickGoal"));
  };

  const handleMapClick = (lat: number, lon: number) => {
    if (pickTarget === "start") {
      setStartLat(lat.toFixed(4));
      setStartLon(lon.toFixed(4));
      setPickTarget(null);
      toast.success(`${t("workspace.startPoint")}: ${lat.toFixed(4)} degN, ${lon.toFixed(4)} degE`);
      return;
    }
    if (pickTarget === "goal") {
      setGoalLat(lat.toFixed(4));
      setGoalLon(lon.toFixed(4));
      setPickTarget(null);
      toast.success(`${t("workspace.goalPoint")}: ${lat.toFixed(4)} degN, ${lon.toFixed(4)} degE`);
      return;
    }
    toast.success(`${t("toast.mapClicked")} ${lat.toFixed(4)} degN, ${lon.toFixed(4)} degE`);
  };

  const plannedCoords = routeResult?.route_geojson?.geometry?.coordinates ?? [];
  const routedStart = plannedCoords.length ? { lat: plannedCoords[0][1], lon: plannedCoords[0][0] } : null;
  const routedGoal = plannedCoords.length ? { lat: plannedCoords[plannedCoords.length - 1][1], lon: plannedCoords[plannedCoords.length - 1][0] } : null;
  const mapStart = routedStart ?? { lat: Number.parseFloat(startLat) || 0, lon: Number.parseFloat(startLon) || 0 };
  const mapGoal = routedGoal ?? { lat: Number.parseFloat(goalLat) || 0, lon: Number.parseFloat(goalLon) || 0 };
  const useDesktopLayout = layoutMode === "desktop" || (layoutMode === "auto" && isWideViewport);

  const pageStyle: CSSProperties = {
    minHeight: 0,
    overflow: useDesktopLayout ? "hidden" : "auto",
  };
  const shellStyle: CSSProperties = {
    display: "flex",
    flexDirection: useDesktopLayout ? "row" : "column",
    minHeight: "100%",
    height: useDesktopLayout ? "100%" : "auto",
  };
  const leftPaneStyle: CSSProperties = useDesktopLayout
    ? { width: 320, maxHeight: "none", minHeight: 0, borderBottomWidth: 0, borderRightWidth: 1 }
    : { width: "100%", maxHeight: "46vh", borderBottomWidth: 1, borderRightWidth: 0 };
  const mapPaneStyle: CSSProperties = useDesktopLayout
    ? { minHeight: 0, height: "100%", flex: 1 }
    : { minHeight: "56vh", height: "56vh", width: "100%", flex: "0 0 auto" };
  const rightPaneStyle: CSSProperties = useDesktopLayout
    ? { width: 360, maxHeight: "none", minHeight: 0, borderTopWidth: 0, borderLeftWidth: 1 }
    : { width: "100%", maxHeight: "42vh", borderTopWidth: 1, borderLeftWidth: 0 };
  const floatingLegendStyle: CSSProperties = {
    position: "absolute",
    bottom: useDesktopLayout ? 16 : 8,
    right: useDesktopLayout ? 16 : 8,
    maxWidth: "70vw",
    display: useDesktopLayout ? "block" : "none",
  };

  return (
    <div className="h-full overflow-auto bg-gradient-to-br from-gray-50 to-slate-100" style={pageStyle}>
      <div style={shellStyle}>
      <div className="bg-white border border-purple-200 flex flex-col shadow-lg" style={leftPaneStyle}>
        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="p-4 space-y-6">
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="mb-2 text-xs text-slate-600">Layout</div>
              <div className="flex gap-2">
                <Button
                  type="button"
                  variant={layoutMode === "auto" ? "default" : "outline"}
                  className="flex-1"
                  onClick={() => setLayoutMode("auto")}
                >
                  Auto
                </Button>
                <Button
                  type="button"
                  variant={layoutMode === "desktop" ? "default" : "outline"}
                  className="flex-1"
                  onClick={() => setLayoutMode("desktop")}
                >
                  Desktop
                </Button>
                <Button
                  type="button"
                  variant={layoutMode === "mobile" ? "default" : "outline"}
                  className="flex-1"
                  onClick={() => setLayoutMode("mobile")}
                >
                  Mobile
                </Button>
              </div>
            </div>

            <div>
              <h3 className="mb-3 text-purple-900 flex items-center gap-2">
                <div className="w-1 h-5 bg-purple-600 rounded-full"></div>
                {t("workspace.scenario")}
              </h3>
              <div className="space-y-3">
                <div className="space-y-2">
                  <Label htmlFor="timestamp">{t("scenario.timestamp")}</Label>
                  <Select value={timestamp} onValueChange={setTimestamp} disabled={!timestampOptions.length}>
                    <SelectTrigger id="timestamp" className="w-full">
                      <SelectValue placeholder={timestampOptions.length ? "" : "Loading..."} />
                    </SelectTrigger>
                    <SelectContent>
                      {timestampOptions.slice(0, 200).map((ts) => (
                        <SelectItem key={ts} value={ts}>
                          {ts}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <CoordinateInput
                  label={t("workspace.startPoint")}
                  lat={startLat}
                  lon={startLon}
                  onLatChange={setStartLat}
                  onLonChange={setStartLon}
                  onPickFromMap={handlePickStart}
                />

                <CoordinateInput
                  label={t("workspace.goalPoint")}
                  lat={goalLat}
                  lon={goalLon}
                  onLatChange={setGoalLat}
                  onLonChange={setGoalLon}
                  onPickFromMap={handlePickGoal}
                />
              </div>
            </div>

            <div>
              <h3 className="mb-3 text-indigo-900 flex items-center gap-2">
                <div className="w-1 h-5 bg-indigo-600 rounded-full"></div>
                {t("workspace.layers")}
              </h3>
              <Card className="border-indigo-200">
                <CardContent className="p-3">
                  <LayerToggle
                    name={`${t("workspace.layer.bathymetry")} ${availability.bathy ? "" : "(missing)"}`}
                    enabled={layers.bathymetry.enabled}
                    opacity={layers.bathymetry.opacity}
                    onToggle={(enabled) => handleLayerToggle("bathymetry", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("bathymetry", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.ais")} ${availability.ais_heatmap ? "" : "(missing)"}`}
                    enabled={layers.aisHeatmap.enabled}
                    opacity={layers.aisHeatmap.opacity}
                    onToggle={(enabled) => handleLayerToggle("aisHeatmap", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("aisHeatmap", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.unet")} ${availability.unet_pred ? "" : "(missing)"}`}
                    enabled={layers.unetZones.enabled}
                    opacity={layers.unetZones.opacity}
                    onToggle={(enabled) => handleLayerToggle("unetZones", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("unetZones", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.ice")} ${availability.ice ? "" : "(missing)"}`}
                    enabled={layers.ice.enabled}
                    opacity={layers.ice.opacity}
                    onToggle={(enabled) => handleLayerToggle("ice", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("ice", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.wave")} ${availability.wave ? "" : "(missing)"}`}
                    enabled={layers.wave.enabled}
                    opacity={layers.wave.opacity}
                    onToggle={(enabled) => handleLayerToggle("wave", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("wave", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.wind")} ${availability.wind ? "" : "(missing)"}`}
                    enabled={layers.wind.enabled}
                    opacity={layers.wind.opacity}
                    onToggle={(enabled) => handleLayerToggle("wind", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("wind", opacity)}
                  />
                </CardContent>
              </Card>
            </div>

            <LegendCard
              title={t("workspace.legend")}
              items={[
                { color: "#10b981", label: t("workspace.legend.safe"), description: t("workspace.legend.safe.desc") },
                { color: "#f59e0b", label: t("workspace.legend.caution"), description: t("workspace.legend.caution.desc") },
                { color: "#ef4444", label: t("workspace.legend.blocked"), description: t("workspace.legend.blocked.desc") },
                { color: "rgba(59, 130, 246, 0.6)", label: t("workspace.legend.ais"), description: t("workspace.legend.ais.desc") },
              ]}
            />

            <div>
              <h3 className="mb-3 text-green-900 flex items-center gap-2">
                <div className="w-1 h-5 bg-green-600 rounded-full"></div>
                {t("workspace.planning")}
              </h3>
              <Card className="border-green-200 bg-green-50/30">
                <CardContent className="p-4 space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="safety-policy">{t("workspace.safetyPolicy")}</Label>
                    <Select value={safetyPolicy} onValueChange={setSafetyPolicy}>
                      <SelectTrigger id="safety-policy">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="blocked-bathy-unet">{t("workspace.safetyPolicy.opt1")}</SelectItem>
                        <SelectItem value="blocked-bathy-only">{t("workspace.safetyPolicy.opt2")}</SelectItem>
                        <SelectItem value="strict">{t("workspace.safetyPolicy.opt3")}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="caution-handling">{t("workspace.cautionHandling")}</Label>
                    <Select value={cautionHandling} onValueChange={setCautionHandling}>
                      <SelectTrigger id="caution-handling">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tiebreaker">{t("workspace.cautionHandling.opt1")}</SelectItem>
                        <SelectItem value="budget">{t("workspace.cautionHandling.opt2")}</SelectItem>
                        <SelectItem value="minimize">{t("workspace.cautionHandling.opt3")}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>{t("workspace.corridorBias")}</Label>
                    <div className="flex items-center gap-3">
                      <Slider value={corridorBias} onValueChange={setCorridorBias} min={0} max={100} step={5} className="flex-1" />
                      <span className="text-sm text-muted-foreground w-12 text-right">{corridorBias[0] / 100}</span>
                    </div>
                  </div>

                  <Button onClick={handlePlanRoute} className="w-full gap-2 bg-green-600 hover:bg-green-700" size="lg" disabled={planning}>
                    <Navigation className="size-4" />
                    {planning ? t("workspace.planRoute.loading") : t("workspace.planRoute")}
                  </Button>
                  <Button
                    onClick={handleRunInference}
                    variant="outline"
                    className="w-full gap-2"
                    size="lg"
                    disabled={inferring}
                  >
                    <Cpu className="size-4" />
                    {inferring ? t("workspace.infer.loading") : t("workspace.infer.run")}
                  </Button>
                  {inferResult ? (
                    <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-xs text-slate-700">
                      <div className="font-medium mb-1">{t("workspace.infer.latest")}</div>
                      <div>safe: {(inferResult.stats.class_ratio.safe * 100).toFixed(1)}%</div>
                      <div>caution: {(inferResult.stats.class_ratio.caution * 100).toFixed(1)}%</div>
                      <div>blocked: {(inferResult.stats.class_ratio.blocked * 100).toFixed(1)}%</div>
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      <div className="relative" style={mapPaneStyle} ref={mapCaptureRef}>
        <MapCanvas
          timestamp={timestamp}
          layers={layers}
          showRoute={Boolean(routeResult)}
          onMapClick={handleMapClick}
          routeGeojson={routeResult?.route_geojson}
          start={mapStart}
          goal={mapGoal}
        />
        {pickTarget ? (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 rounded-full bg-black/70 px-4 py-2 text-xs text-white">
            {pickTarget === "start" ? t("workspace.pick.start.hint") : t("workspace.pick.goal.hint")}
          </div>
        ) : null}

        <div style={floatingLegendStyle}>
          <LegendCard
              title="Active Layers"
            items={[
              ...(layers.unetZones.enabled
                ? [
                    { color: "#10b981", label: "SAFE" },
                    { color: "#f59e0b", label: "CAUTION" },
                    { color: "#ef4444", label: "BLOCKED" },
                  ]
                : []),
              ...(layers.aisHeatmap.enabled ? [{ color: "#3b82f6", label: "AIS Traffic" }] : []),
            ]}
          />
        </div>
      </div>

      <div className="bg-white border border-border flex flex-col" style={rightPaneStyle}>
        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="p-4 space-y-6">
            <div>
              <h3 className="mb-3">{t("summary.title")}</h3>
              {routeResult ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <StatCard label={t("summary.distance")} value={routeSummary.distanceKm.toFixed(1)} unit="km" />
                    <StatCard label={t("summary.distance")} value={routeSummary.distanceNm.toFixed(1)} unit="nm" />
                  </div>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <StatCard label={t("summary.safe")} value={routeSummary.safePct} unit="%" variant="success" />
                    <StatCard label={t("summary.caution")} value={routeSummary.cautionPct} unit="%" variant="warning" />
                  </div>
                  <StatCard label={t("summary.alignment")} value={routeSummary.alignment.toFixed(2)} variant="success" />
                  <div className="p-3 rounded-lg border border-green-200 bg-green-50 flex items-start gap-2">
                    <CheckCircle2 className="size-4 text-green-600 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-green-800">
                      <div className="font-medium mb-1">{t("summary.noViolations")}</div>
                      <div className="text-xs">{t("summary.noViolations.desc")}</div>
                    </div>
                  </div>
                  {routeResult.gallery_id ? (
                    <Button variant="outline" className="w-full" onClick={handleOpenLatestGallery}>
                      {t("workspace.openGallery")} ({routeResult.gallery_id})
                    </Button>
                  ) : null}
                </div>
              ) : (
                <div className="p-8 text-center border-2 border-dashed border-border rounded-lg">
                  <Navigation className="size-8 text-muted-foreground mx-auto mb-2" />
                  <p className="text-sm text-muted-foreground">{t("summary.planToSee")}</p>
                </div>
              )}
            </div>

            {routeResult && (
              <div>
                <h3 className="mb-3">{t("explain.title")}</h3>
                <Card>
                  <CardContent className="pt-4 space-y-2 text-sm">
                    <div className="flex gap-2">
                      <div className="text-green-600 mt-0.5">OK</div>
                      <div>{t("explain.reason1")}</div>
                    </div>
                    <div className="flex gap-2">
                      <div className="text-green-600 mt-0.5">OK</div>
                      <div>{t("explain.reason2")}</div>
                    </div>
                    <div className="flex gap-2">
                      <div className="text-blue-600 mt-0.5">-&gt;</div>
                      <div>
                        {t("explain.reason3")} ({(corridorBias[0] / 100).toFixed(2)})
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <div className="text-amber-600 mt-0.5">
                        <AlertCircle className="size-3" />
                      </div>
                      <div>
                        {routeSummary.cautionPct}% {t("explain.reason4")}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </div>
      </div>
    </div>
  );
}
