import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { AlertCircle, CheckCircle2, Cpu, Navigation } from "lucide-react";
import { toast } from "sonner";

import {
  getErrorMessage,
  getLayers,
  getLatestProgress,
  getLatestStatus,
  getTimestamps,
  getCopernicusConfig,
  planLatestRoute,
  planRoute,
  runInference,
  setCopernicusConfig,
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
import { Progress } from "../components/ui/progress";
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
  unetUncertainty: LayerState;
  ice: LayerState;
  wave: LayerState;
  wind: LayerState;
};

type LayoutMode = "auto" | "desktop" | "mobile";

const DEFAULT_LAYERS: LayerStates = {
  bathymetry: { enabled: true, opacity: 80 },
  aisHeatmap: { enabled: true, opacity: 60 },
  unetZones: { enabled: true, opacity: 70 },
  unetUncertainty: { enabled: false, opacity: 65 },
  ice: { enabled: false, opacity: 50 },
  wave: { enabled: false, opacity: 50 },
  wind: { enabled: false, opacity: 50 },
};

const AVAILABILITY_DEFAULT = {
  bathy: true,
  ais_heatmap: true,
  unet_pred: false,
  unet_uncertainty: false,
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
  const [latestPlanning, setLatestPlanning] = useState(false);
  const [tileRevision, setTileRevision] = useState(0);
  const [pickTarget, setPickTarget] = useState<"start" | "goal" | null>(null);
  const [inferring, setInferring] = useState(false);
  const [inferResult, setInferResult] = useState<InferResponse | null>(null);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("auto");
  const [isWideViewport, setIsWideViewport] = useState(false);
  const [plannerMode, setPlannerMode] = useState("dstar_lite");
  const [latestDate, setLatestDate] = useState(() => {
    const now = new Date();
    const yyyy = now.getUTCFullYear();
    const mm = String(now.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(now.getUTCDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`;
  });
  const mapCaptureRef = useRef<HTMLDivElement | null>(null);
  const [copernicusForm, setCopernicusForm] = useState({
    username: "",
    password: "",
    ice_dataset_id: "",
    wave_dataset_id: "",
    wind_dataset_id: "",
  });
  const [copernicusConfigured, setCopernicusConfigured] = useState(false);
  const [latestMeta, setLatestMeta] = useState<Record<string, unknown> | null>(null);
  const [latestProgress, setLatestProgress] = useState({
    progressId: "",
    status: "idle",
    phase: "idle",
    message: "",
    percent: 0,
    error: "",
    visible: false,
  });
  const latestProgressTimerRef = useRef<number | null>(null);

  const stopLatestProgressPolling = useCallback(() => {
    if (latestProgressTimerRef.current !== null) {
      window.clearInterval(latestProgressTimerRef.current);
      latestProgressTimerRef.current = null;
    }
  }, []);

  const normalizedTsToUi = useCallback((value: string) => {
    const match = /^(\d{4}-\d{2}-\d{2})_(\d{2})$/.exec(value.trim());
    if (!match) return value;
    return `${match[1]}-${match[2]}:00`;
  }, []);

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
    return () => {
      stopLatestProgressPolling();
    };
  }, [stopLatestProgressPolling]);

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

  useEffect(() => {
    let active = true;
    async function loadCopernicusStatus() {
      try {
        const res = await getCopernicusConfig();
        if (!active) return;
        setCopernicusConfigured(res.configured);
        setCopernicusForm((prev) => ({
          ...prev,
          ice_dataset_id: String(res.datasets?.ice_dataset_id ?? ""),
          wave_dataset_id: String(res.datasets?.wave_dataset_id ?? ""),
          wind_dataset_id: String(res.datasets?.wind_dataset_id ?? ""),
        }));
      } catch {
        // keep local defaults
      }
    }
    loadCopernicusStatus();
    return () => {
      active = false;
    };
  }, []);

  const refreshLayerAvailability = useCallback(async (ts: string) => {
    const res = await getLayers(ts);
    const nextAvailability = {
      bathy: res.layers.find((l) => l.id === "bathy")?.available ?? false,
      ais_heatmap: res.layers.find((l) => l.id === "ais_heatmap")?.available ?? false,
      unet_pred: res.layers.find((l) => l.id === "unet_pred")?.available ?? false,
      unet_uncertainty: res.layers.find((l) => l.id === "unet_uncertainty")?.available ?? false,
      ice: res.layers.find((l) => l.id === "ice")?.available ?? false,
      wave: res.layers.find((l) => l.id === "wave")?.available ?? false,
      wind: res.layers.find((l) => l.id === "wind")?.available ?? false,
    };
    setAvailability(nextAvailability);
    setTileRevision((prev) => prev + 1);
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

  const latestPhaseText = useMemo(() => {
    const phase = latestProgress.phase || "unknown";
    const labels: Record<string, string> = {
      idle: "空闲",
      init: "初始化",
      prepare: "准备模板",
      download: "下载网格",
      merge: "合并通道",
      materialize: "写入快照",
      snapshot: "快照下载",
      resolve: "解析时间片",
      plan: "路径规划",
      done: "完成",
      error: "失败",
      unknown: "未知",
    };
    return labels[phase] ?? phase;
  }, [latestProgress.phase]);

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
          planner: plannerMode,
        },
      });
      setRouteResult(response);
      const startAdjusted = Boolean(response.explain?.["start_adjusted"]);
      const goalAdjusted = Boolean(response.explain?.["goal_adjusted"]);
      if (startAdjusted || goalAdjusted) {
        toast.warning("起点/终点已自动调整到最近可航行网格", { id: "plan-adjusted" });
      }
      toast.success(`${t("toast.success")}（记录：${response.gallery_id}）`, { id: "plan-route" });
      void captureAndUploadGalleryImage(response.gallery_id);
    } catch (error) {
      toast.error(`${t("toast.planFailed")}: ${getErrorMessage(error)}`, { id: "plan-route" });
    } finally {
      setPlanning(false);
    }
  };

  const handlePlanLatestRoute = async () => {
    const startLatNum = Number.parseFloat(startLat);
    const startLonNum = Number.parseFloat(startLon);
    const goalLatNum = Number.parseFloat(goalLat);
    const goalLonNum = Number.parseFloat(goalLon);
    if ([startLatNum, startLonNum, goalLatNum, goalLonNum].some((v) => Number.isNaN(v))) {
      toast.error(t("toast.invalidCoords"));
      return;
    }
    if (!latestDate) {
      toast.error("请选择最新预测日期");
      return;
    }
    const progressId = `latest-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
    stopLatestProgressPolling();
    setLatestProgress({
      progressId,
      status: "running",
      phase: "init",
      message: "正在准备最新预测流程...",
      percent: 1,
      error: "",
      visible: true,
    });
    setLatestPlanning(true);
    setPlanning(true);
    toast.loading("正在使用最新预报进行规划...", { id: "plan-latest-route" });
    const pollOnce = async () => {
      try {
        const p = await getLatestProgress(progressId);
        setLatestProgress((prev) => ({
          ...prev,
          progressId,
          status: p.status,
          phase: p.phase ?? prev.phase,
          message: p.message ?? prev.message,
          percent: Number.isFinite(p.percent) ? p.percent : prev.percent,
          error: typeof p.error === "string" ? p.error : "",
          visible: true,
        }));
        if (p.status === "completed" || p.status === "failed") {
          stopLatestProgressPolling();
        }
      } catch {
        // Keep UI responsive even if one polling round fails.
      }
    };
    void pollOnce();
    latestProgressTimerRef.current = window.setInterval(() => {
      void pollOnce();
    }, 1200);
    try {
      const cautionMode = safetyPolicy === "strict" ? "strict" : cautionHandling === "tiebreaker" ? "tie_breaker" : cautionHandling;
      const blockedSources =
        safetyPolicy === "blocked-bathy-only"
          ? ["bathy"]
          : safetyPolicy === "strict"
            ? ["bathy", "unet_blocked", "unet_caution"]
            : ["bathy", "unet_blocked"];
      const response = await planLatestRoute({
        date: latestDate,
        hour: 12,
        progress_id: progressId,
        start: { lat: startLatNum, lon: startLonNum },
        goal: { lat: goalLatNum, lon: goalLonNum },
        policy: {
          objective: "shortest_distance_under_safety",
          blocked_sources: blockedSources,
          caution_mode: cautionMode,
          corridor_bias: corridorBias[0] / 100,
          smoothing: true,
          planner: plannerMode,
        },
      });
      stopLatestProgressPolling();
      setLatestProgress((prev) => ({
        ...prev,
        progressId,
        status: "completed",
        phase: "done",
        message: "最新预测完成",
        percent: 100,
        error: "",
        visible: true,
      }));
      setRouteResult(response);
      if (response.latest_meta && Object.keys(response.latest_meta).length > 0) {
        setLatestMeta(response.latest_meta);
      }
      const usedTimestamp = response.resolved?.used_timestamp;
      if (usedTimestamp) {
        const uiTs = normalizedTsToUi(usedTimestamp);
        setTimestamp(uiTs);
        setTimestampOptions((prev) => (prev.includes(uiTs) ? prev : [uiTs, ...prev]));
        await refreshLayerAvailability(uiTs);
        try {
          const st = await getLatestStatus(usedTimestamp);
          if (st.has_latest_meta) {
            setLatestMeta(st.meta);
          }
        } catch {
          // best-effort status load
        }
      }
      setLayers((prev) => ({
        ...prev,
        ice: { ...prev.ice, enabled: true },
        wave: { ...prev.wave, enabled: true },
        wind: { ...prev.wind, enabled: true },
        unetZones: { ...prev.unetZones, enabled: true },
        unetUncertainty: { ...prev.unetUncertainty, enabled: true },
      }));
      toast.success(
        `最新航线已就绪（${response.resolved?.source ?? "未知来源"} -> ${response.resolved?.used_timestamp ?? "无"}）`,
        { id: "plan-latest-route" }
      );
    } catch (error) {
      stopLatestProgressPolling();
      setLatestProgress((prev) => ({
        ...prev,
        progressId,
        status: "failed",
        phase: "error",
        message: "最新预测失败",
        error: getErrorMessage(error),
        visible: true,
      }));
      toast.error(`最新预测失败：${getErrorMessage(error)}`, { id: "plan-latest-route" });
    } finally {
      setLatestPlanning(false);
      setPlanning(false);
    }
  };

  const handleSaveCopernicusConfig = async () => {
    try {
      const res = await setCopernicusConfig({
        username: copernicusForm.username || undefined,
        password: copernicusForm.password || undefined,
        ice_dataset_id: copernicusForm.ice_dataset_id || undefined,
        wave_dataset_id: copernicusForm.wave_dataset_id || undefined,
        wind_dataset_id: copernicusForm.wind_dataset_id || undefined,
      });
      setCopernicusConfigured(res.configured);
      toast.success(res.configured ? "Copernicus 配置已就绪" : "Copernicus 配置已保存（信息不完整）");
      setCopernicusForm((prev) => ({ ...prev, password: "" }));
    } catch (error) {
      toast.error(`保存 Copernicus 配置失败: ${getErrorMessage(error)}`);
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
      toast.success(`${t("toast.inferDone")} (${res.stats.cache_hit ? "命中缓存" : "实时推理"})`, { id: "run-infer" });
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
      toast.success(`${t("workspace.startPoint")}：北纬 ${lat.toFixed(4)}°，东经 ${lon.toFixed(4)}°`);
      return;
    }
    if (pickTarget === "goal") {
      setGoalLat(lat.toFixed(4));
      setGoalLon(lon.toFixed(4));
      setPickTarget(null);
      toast.success(`${t("workspace.goalPoint")}：北纬 ${lat.toFixed(4)}°，东经 ${lon.toFixed(4)}°`);
      return;
    }
    toast.success(`${t("toast.mapClicked")} 北纬 ${lat.toFixed(4)}°，东经 ${lon.toFixed(4)}°`);
  };

  const plannedCoords = routeResult?.route_geojson?.geometry?.coordinates ?? [];
  const routedStart = plannedCoords.length ? { lat: plannedCoords[0][1], lon: plannedCoords[0][0] } : null;
  const routedGoal = plannedCoords.length ? { lat: plannedCoords[plannedCoords.length - 1][1], lon: plannedCoords[plannedCoords.length - 1][0] } : null;
  const mapStart = routedStart ?? { lat: Number.parseFloat(startLat) || 0, lon: Number.parseFloat(startLon) || 0 };
  const mapGoal = routedGoal ?? { lat: Number.parseFloat(goalLat) || 0, lon: Number.parseFloat(goalLon) || 0 };
  const useDesktopLayout = layoutMode === "desktop" || (layoutMode === "auto" && isWideViewport);
  const mapLayoutKey = useDesktopLayout ? "desktop" : "mobile";

  const pageStyle: CSSProperties = useDesktopLayout
    ? {
        minHeight: "100dvh",
        height: "100dvh",
        overflow: "hidden",
      }
    : {
        minHeight: "100dvh",
        height: "auto",
        overflowX: "hidden",
        overflowY: "auto",
      };
  const shellStyle: CSSProperties = {
    display: "flex",
    flexDirection: useDesktopLayout ? "row" : "column",
    minHeight: useDesktopLayout ? "100%" : "auto",
    height: useDesktopLayout ? "100%" : "auto",
    gap: useDesktopLayout ? 0 : 8,
    alignItems: useDesktopLayout ? "stretch" : "center",
  };
  const leftPaneStyle: CSSProperties = useDesktopLayout
    ? { width: 320, maxHeight: "none", minHeight: 0, borderBottomWidth: 0, borderRightWidth: 1, order: 1 }
    : { width: "min(100%, 1100px)", maxHeight: "none", minHeight: 0, borderBottomWidth: 1, borderRightWidth: 0, order: 2 };
  const mapPaneStyle: CSSProperties = useDesktopLayout
    ? { minHeight: 520, height: "100%", flex: 1, width: 0, order: 2 }
    : {
        minHeight: 300,
        height: "clamp(320px, 54dvh, 620px)",
        maxHeight: "66dvh",
        width: "min(100%, 1100px)",
        flex: "0 0 auto",
        order: 1,
      };
  const rightPaneStyle: CSSProperties = useDesktopLayout
    ? { width: 360, maxHeight: "none", minHeight: 0, borderTopWidth: 0, borderLeftWidth: 1, order: 3 }
    : { width: "min(100%, 1100px)", maxHeight: "none", minHeight: 0, borderTopWidth: 1, borderLeftWidth: 0, order: 3 };
  const floatingLegendStyle: CSSProperties = {
    position: "absolute",
    bottom: useDesktopLayout ? 16 : 8,
    right: useDesktopLayout ? 16 : 8,
    maxWidth: "70vw",
    display: useDesktopLayout ? "block" : "none",
  };

  return (
    <div className="relative h-full bg-gradient-to-br from-gray-50 to-slate-100" style={pageStyle}>
      <div className="absolute right-3 top-3 z-50 rounded-lg border border-slate-200 bg-white/95 p-2 shadow-md backdrop-blur-sm">
        <div className="mb-1 text-[11px] text-slate-600">布局</div>
        <div className="flex gap-1">
          <Button
            type="button"
            size="sm"
            variant={layoutMode === "auto" ? "default" : "outline"}
            onClick={() => setLayoutMode("auto")}
          >
            自动
          </Button>
          <Button
            type="button"
            size="sm"
            variant={layoutMode === "desktop" ? "default" : "outline"}
            onClick={() => setLayoutMode("desktop")}
          >
            桌面
          </Button>
          <Button
            type="button"
            size="sm"
            variant={layoutMode === "mobile" ? "default" : "outline"}
            onClick={() => setLayoutMode("mobile")}
          >
            移动
          </Button>
        </div>
      </div>
      <div style={shellStyle}>
      <div className="bg-white border border-purple-200 flex flex-col shadow-lg" style={leftPaneStyle}>
        <div className="flex-1 min-h-0 overflow-y-auto">
          <div className="p-4 space-y-6">
            <div>
              <h3 className="mb-3 text-purple-900 flex items-center gap-2">
                <div className="w-1 h-5 bg-purple-600 rounded-full"></div>
                {t("workspace.scenario")}
              </h3>
              <div className="space-y-3">
                <div className="space-y-2">
                  <Label>布局模式</Label>
                  <div className="flex flex-wrap gap-2">
                    <Button type="button" size="sm" variant={layoutMode === "auto" ? "default" : "outline"} onClick={() => setLayoutMode("auto")}>
                      自动
                    </Button>
                    <Button type="button" size="sm" variant={layoutMode === "desktop" ? "default" : "outline"} onClick={() => setLayoutMode("desktop")}>
                      桌面
                    </Button>
                    <Button type="button" size="sm" variant={layoutMode === "mobile" ? "default" : "outline"} onClick={() => setLayoutMode("mobile")}>
                      移动
                    </Button>
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="timestamp">{t("scenario.timestamp")}</Label>
                  <Select value={timestamp} onValueChange={setTimestamp} disabled={!timestampOptions.length}>
                    <SelectTrigger id="timestamp" className="w-full">
                      <SelectValue placeholder={timestampOptions.length ? "" : "加载中..."} />
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
                    name={`${t("workspace.layer.bathymetry")} ${availability.bathy ? "" : "（缺失）"}`}
                    enabled={layers.bathymetry.enabled}
                    opacity={layers.bathymetry.opacity}
                    onToggle={(enabled) => handleLayerToggle("bathymetry", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("bathymetry", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.ais")} ${availability.ais_heatmap ? "" : "（缺失）"}`}
                    enabled={layers.aisHeatmap.enabled}
                    opacity={layers.aisHeatmap.opacity}
                    onToggle={(enabled) => handleLayerToggle("aisHeatmap", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("aisHeatmap", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.unet")} ${availability.unet_pred ? "" : "（缺失）"}`}
                    enabled={layers.unetZones.enabled}
                    opacity={layers.unetZones.opacity}
                    onToggle={(enabled) => handleLayerToggle("unetZones", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("unetZones", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.uncertainty")} ${availability.unet_uncertainty ? "" : "（缺失）"}`}
                    enabled={layers.unetUncertainty.enabled}
                    opacity={layers.unetUncertainty.opacity}
                    onToggle={(enabled) => handleLayerToggle("unetUncertainty", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("unetUncertainty", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.ice")} ${availability.ice ? "" : "（缺失）"}`}
                    enabled={layers.ice.enabled}
                    opacity={layers.ice.opacity}
                    onToggle={(enabled) => handleLayerToggle("ice", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("ice", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.wave")} ${availability.wave ? "" : "（缺失）"}`}
                    enabled={layers.wave.enabled}
                    opacity={layers.wave.opacity}
                    onToggle={(enabled) => handleLayerToggle("wave", enabled)}
                    onOpacityChange={(opacity) => handleOpacityChange("wave", opacity)}
                  />
                  <LayerToggle
                    name={`${t("workspace.layer.wind")} ${availability.wind ? "" : "（缺失）"}`}
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

                  <div className="space-y-2">
                    <Label>规划算法</Label>
                    <Select value={plannerMode} onValueChange={setPlannerMode}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="astar">A*（基线）</SelectItem>
                        <SelectItem value="any_angle">Any-Angle / Theta*（更直线路径）</SelectItem>
                        <SelectItem value="hybrid_astar">Hybrid A*（考虑航向连续）</SelectItem>
                        <SelectItem value="dstar_lite">D* Lite（静态环境）</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <Button onClick={handlePlanRoute} className="w-full gap-2 bg-green-600 hover:bg-green-700" size="lg" disabled={planning}>
                    <Navigation className="size-4" />
                    {planning ? t("workspace.planRoute.loading") : t("workspace.planRoute")}
                  </Button>
                  <div className="rounded-md border border-slate-200 bg-white p-3 space-y-2">
                    <Label htmlFor="latest-date">最新预报日期</Label>
                    <input
                      id="latest-date"
                      type="date"
                      value={latestDate}
                      onChange={(e) => setLatestDate(e.target.value)}
                      className="h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    />
                    <Button onClick={handlePlanLatestRoute} variant="outline" className="w-full">
                      拉取最新数据并规划
                    </Button>
                    {(latestPlanning || latestProgress.visible) && latestProgress.progressId ? (
                      <div className="rounded-md border border-slate-200 bg-slate-50 p-2 space-y-1">
                        <div className="flex items-center justify-between text-[11px] text-slate-600">
                          <span>{latestPhaseText}</span>
                          <span>{Math.max(0, Math.min(100, Math.round(latestProgress.percent)))}%</span>
                        </div>
                        <Progress value={Math.max(0, Math.min(100, latestProgress.percent))} />
                        <div className="text-[11px] text-slate-700">{latestProgress.message || "进行中..."}</div>
                        {latestProgress.error ? (
                          <div className="text-[11px] text-red-600">{latestProgress.error}</div>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                  <div className="rounded-md border border-slate-200 bg-white p-3 space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Copernicus 账户</Label>
                      <span className={`text-xs ${copernicusConfigured ? "text-green-600" : "text-amber-600"}`}>
                        {copernicusConfigured ? "已配置" : "未就绪"}
                      </span>
                    </div>
                    <input
                      type="text"
                      placeholder="账号"
                      value={copernicusForm.username}
                      onChange={(e) => setCopernicusForm((prev) => ({ ...prev, username: e.target.value }))}
                      className="h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    />
                    <input
                      type="password"
                      placeholder="密码"
                      value={copernicusForm.password}
                      onChange={(e) => setCopernicusForm((prev) => ({ ...prev, password: e.target.value }))}
                      className="h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    />
                    <input
                      type="text"
                      placeholder="海冰数据集 ID"
                      value={copernicusForm.ice_dataset_id}
                      onChange={(e) => setCopernicusForm((prev) => ({ ...prev, ice_dataset_id: e.target.value }))}
                      className="h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    />
                    <input
                      type="text"
                      placeholder="海浪数据集 ID"
                      value={copernicusForm.wave_dataset_id}
                      onChange={(e) => setCopernicusForm((prev) => ({ ...prev, wave_dataset_id: e.target.value }))}
                      className="h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    />
                    <input
                      type="text"
                      placeholder="风场数据集 ID"
                      value={copernicusForm.wind_dataset_id}
                      onChange={(e) => setCopernicusForm((prev) => ({ ...prev, wind_dataset_id: e.target.value }))}
                      className="h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    />
                    <Button onClick={handleSaveCopernicusConfig} variant="outline" className="w-full">
                      保存 Copernicus 配置
                    </Button>
                  </div>
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
                      <div>安全：{(inferResult.stats.class_ratio.safe * 100).toFixed(1)}%</div>
                      <div>谨慎：{(inferResult.stats.class_ratio.caution * 100).toFixed(1)}%</div>
                      <div>禁行：{(inferResult.stats.class_ratio.blocked * 100).toFixed(1)}%</div>
                      {typeof inferResult.stats.uncertainty_mean === "number" ? (
                        <div>平均不确定性：{inferResult.stats.uncertainty_mean.toFixed(3)}</div>
                      ) : null}
                      {typeof inferResult.stats.uncertainty_p90 === "number" ? (
                        <div>P90 不确定性：{inferResult.stats.uncertainty_p90.toFixed(3)}</div>
                      ) : null}
                    </div>
                  ) : null}
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>

      <div className="relative" style={mapPaneStyle} ref={mapCaptureRef}>
        <div className="relative h-full w-full min-h-[360px] overflow-hidden rounded-xl border border-slate-200 bg-white shadow-md">
          <MapCanvas
            timestamp={timestamp}
            tileRevision={tileRevision}
            layoutKey={mapLayoutKey}
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
                title="当前图层"
              items={[
                ...(layers.unetZones.enabled
                  ? [
                      { color: "#10b981", label: "安全" },
                      { color: "#f59e0b", label: "谨慎" },
                      { color: "#ef4444", label: "禁行" },
                    ]
                  : []),
                ...(layers.unetUncertainty.enabled ? [{ color: "rgba(220, 38, 38, 0.75)", label: "U-Net 不确定性" }] : []),
                ...(layers.aisHeatmap.enabled ? [{ color: "#3b82f6", label: "AIS 航道热力" }] : []),
              ]}
            />
          </div>
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
                  {latestMeta ? (
                    <div className="rounded-lg border border-sky-200 bg-sky-50 p-3 text-xs text-slate-700 space-y-1">
                      <div className="font-medium text-sky-900">最新网格来源</div>
                      <div>来源：{String(latestMeta.source ?? "无")}</div>
                      <div>请求时间：{String(latestMeta.requested_timestamp ?? "无")}</div>
                      <div>落盘时间：{String(latestMeta.materialized_at ?? "无")}</div>
                      <div className="text-sky-800 font-medium pt-1">实时图层统计</div>
                      <div>海冰浓度：{String((latestMeta as any)?.stats?.ice_conc ? JSON.stringify((latestMeta as any).stats.ice_conc) : "无")}</div>
                      <div>浪高：{String((latestMeta as any)?.stats?.wave_hs ? JSON.stringify((latestMeta as any).stats.wave_hs) : "无")}</div>
                      <div>风场 u10/v10：{String((latestMeta as any)?.stats?.wind_u10 && (latestMeta as any)?.stats?.wind_v10 ? `${JSON.stringify((latestMeta as any).stats.wind_u10)} / ${JSON.stringify((latestMeta as any).stats.wind_v10)}` : "无")}</div>
                    </div>
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
                  <CardContent className="pt-4 space-y-3 text-sm">
                    <div className="grid grid-cols-1 gap-2 text-xs text-slate-700">
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        规划器：{String(routeResult.explain?.planner ?? "unknown")}
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        有效代价：{Number(routeResult.explain?.route_cost_effective_km ?? routeResult.explain?.distance_km ?? 0).toFixed(3)} km
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        代价拆解：基础 {Number(routeResult.explain?.route_cost_base_km ?? routeResult.explain?.distance_km ?? 0).toFixed(3)} / 谨慎附加{" "}
                        {Number(routeResult.explain?.route_cost_caution_extra_km ?? 0).toFixed(3)} / 航道折扣{" "}
                        {Number(routeResult.explain?.route_cost_corridor_discount_km ?? 0).toFixed(3)} km
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        风险贴边比例：{(Number(routeResult.explain?.adjacent_blocked_ratio ?? 0) * 100).toFixed(1)}%
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        谨慎网格占比：{(Number(routeResult.explain?.caution_cell_ratio ?? 0) * 100).toFixed(1)}%
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        走廊贴合分位：P50 {Number(routeResult.explain?.corridor_alignment_p50 ?? routeResult.explain?.corridor_alignment ?? 0).toFixed(3)} / P90{" "}
                        {Number(routeResult.explain?.corridor_alignment_p90 ?? routeResult.explain?.corridor_alignment ?? 0).toFixed(3)}
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        起终点修正：start {Number(routeResult.explain?.start_adjust_km ?? 0).toFixed(3)} / goal {Number(routeResult.explain?.goal_adjust_km ?? 0).toFixed(3)} km
                      </div>
                      {Array.isArray((routeResult.explain as any)?.dynamic_replans) ? (
                        <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                          动态重规划：{((routeResult.explain as any).dynamic_replans as unknown[]).length} 次，累计{" "}
                          {Number(routeResult.explain?.replan_runtime_ms_total ?? 0).toFixed(1)} ms，均值{" "}
                          {Number(routeResult.explain?.replan_runtime_ms_mean ?? 0).toFixed(1)} ms
                        </div>
                      ) : null}
                    </div>

                    <div className="flex gap-2">
                      <div className="text-green-600 mt-0.5">通过</div>
                      <div>{t("explain.reason1")}</div>
                    </div>
                    <div className="flex gap-2">
                      <div className="text-green-600 mt-0.5">通过</div>
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

