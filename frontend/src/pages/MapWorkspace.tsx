import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { useNavigate, useSearchParams } from "react-router";
import { AlertCircle, CheckCircle2, Cpu, Navigation } from "lucide-react";
import { toast } from "sonner";

import {
  type ComplianceNoticesPayload,
  type DynamicExecutionEntry,
  type DynamicReplanEntry,
  getComplianceNotices,
  getErrorMessage,
  getLayers,
  getVesselProfiles,
  getLatestProgress,
  getLatestStatus,
  getTimestamps,
  getCopernicusConfig,
  planDynamicRoute,
  planLatestRoute,
  planRoute,
  runInference,
  setCopernicusConfig,
  uploadGalleryImage,
  type InferResponse,
  type RouteCandidate,
  type RoutePlanResponse,
  type VesselProfile,
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

function toLineCoords(value: unknown): [number, number][] {
  if (!Array.isArray(value)) return [];
  const coords: [number, number][] = [];
  for (const item of value) {
    if (!Array.isArray(item) || item.length < 2) continue;
    const lon = Number(item[0]);
    const lat = Number(item[1]);
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) continue;
    coords.push([lon, lat]);
  }
  return coords;
}

function toNumberOrNull(value: unknown): number | null {
  const num = Number(value);
  return Number.isFinite(num) ? num : null;
}

export default function MapWorkspace() {
  const { t, language } = useLanguage();
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
  const [dynamicPlanning, setDynamicPlanning] = useState(false);
  const [tileRevision, setTileRevision] = useState(0);
  const [pickTarget, setPickTarget] = useState<"start" | "goal" | null>(null);
  const [inferring, setInferring] = useState(false);
  const [inferResult, setInferResult] = useState<InferResponse | null>(null);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>("auto");
  const [isWideViewport, setIsWideViewport] = useState(false);
  const [plannerMode, setPlannerMode] = useState("dstar_lite");
  const [vesselProfileId, setVesselProfileId] = useState("arc7_lng");
  const [vesselProfiles, setVesselProfiles] = useState<VesselProfile[]>([]);
  const [riskMode, setRiskMode] = useState("balanced");
  const [riskWeightScale, setRiskWeightScale] = useState([100]);
  const [riskConstraintMode, setRiskConstraintMode] = useState("none");
  const [riskBudget, setRiskBudget] = useState([100]);
  const [riskConfidence, setRiskConfidence] = useState([90]);
  const [returnCandidates, setReturnCandidates] = useState(true);
  const [candidateLimit, setCandidateLimit] = useState([3]);
  const [dynamicWindow, setDynamicWindow] = useState([8]);
  const [dynamicAdvanceSteps, setDynamicAdvanceSteps] = useState([12]);
  const [dynamicRiskSwitchEnabled, setDynamicRiskSwitchEnabled] = useState(true);
  const [dynamicRiskBudgetKm, setDynamicRiskBudgetKm] = useState([3]);
  const [dynamicRiskWarnRatio, setDynamicRiskWarnRatio] = useState([70]);
  const [dynamicRiskHardRatio, setDynamicRiskHardRatio] = useState([100]);
  const [dynamicRiskWarnMode, setDynamicRiskWarnMode] = useState("conservative");
  const [dynamicRiskHardMode, setDynamicRiskHardMode] = useState("conservative");
  const [dynamicRiskSwitchMinInterval, setDynamicRiskSwitchMinInterval] = useState([1]);
  const [replayStepIndex, setReplayStepIndex] = useState(0);
  const [replayPlaying, setReplayPlaying] = useState(false);
  const [replaySpeed, setReplaySpeed] = useState("1");
  const [selectedCandidateId, setSelectedCandidateId] = useState<string>("requested");
  const [latestDate, setLatestDate] = useState(() => {
    const now = new Date();
    const yyyy = now.getUTCFullYear();
    const mm = String(now.getUTCMonth() + 1).padStart(2, "0");
    const dd = String(now.getUTCDate()).padStart(2, "0");
    return `${yyyy}-${mm}-${dd}`;
  });
  const [latestDynamicReplanEnabled, setLatestDynamicReplanEnabled] = useState(true);
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
  const [complianceNotices, setComplianceNotices] = useState<ComplianceNoticesPayload | null>(null);
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
    async function loadVesselProfiles() {
      try {
        const payload = await getVesselProfiles();
        if (!active) return;
        const profiles = Array.isArray(payload.profiles) ? payload.profiles : [];
        setVesselProfiles(profiles);
        if (payload.default_profile_id && profiles.some((item) => item.id === payload.default_profile_id)) {
          setVesselProfileId(payload.default_profile_id);
        }
      } catch (error) {
        if (!active) return;
        console.warn("vessel profiles api unavailable", error);
      }
    }
    loadVesselProfiles();
    return () => {
      active = false;
    };
  }, []);

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

  useEffect(() => {
    let active = true;
    async function loadComplianceNotices() {
      try {
        const payload = await getComplianceNotices({
          context: "workspace",
          timestamp: timestamp || undefined,
        });
        if (!active) return;
        setComplianceNotices(payload);
      } catch (error) {
        if (!active) return;
        console.warn("compliance notices unavailable", error);
        setComplianceNotices(null);
      }
    }
    loadComplianceNotices();
    return () => {
      active = false;
    };
  }, [timestamp]);

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

  const routeCandidates = useMemo(() => routeResult?.candidates ?? [], [routeResult]);
  const activeCandidate: RouteCandidate | null = useMemo(() => {
    if (!routeCandidates.length) return null;
    const matched = routeCandidates.find((c) => c.id === selectedCandidateId && c.status === "ok");
    if (matched) return matched;
    return routeCandidates.find((c) => c.status === "ok") ?? null;
  }, [routeCandidates, selectedCandidateId]);

  const activeRouteGeojson = activeCandidate?.route_geojson ?? routeResult?.route_geojson ?? null;
  const routeMetrics = (activeCandidate?.explain as RoutePlanResponse["explain"] | undefined) ?? routeResult?.explain;
  const dynamicExecutionLog = useMemo(() => {
    const raw = (routeMetrics?.dynamic_execution_log ?? []) as unknown[];
    if (!Array.isArray(raw)) return [] as DynamicExecutionEntry[];
    return raw as DynamicExecutionEntry[];
  }, [routeMetrics]);
  const dynamicReplans = useMemo(() => {
    const raw = (routeMetrics?.dynamic_replans ?? []) as unknown[];
    if (!Array.isArray(raw)) return [] as DynamicReplanEntry[];
    return raw as DynamicReplanEntry[];
  }, [routeMetrics]);
  const replayStepCount = dynamicExecutionLog.length;
  const replayStep = replayStepCount ? Math.max(0, Math.min(replayStepIndex, replayStepCount - 1)) : 0;
  const replayCurrentEntry = replayStepCount ? dynamicExecutionLog[replayStep] : null;

  const replayOverlay = useMemo(() => {
    if (!replayStepCount) return null;
    const executed: [number, number][] = [];
    for (let idx = 0; idx <= replayStep; idx += 1) {
      const seg = toLineCoords(dynamicExecutionLog[idx]?.segment_coordinates);
      if (seg.length < 2) continue;
      if (!executed.length) {
        executed.push(...seg);
      } else {
        executed.push(...seg.slice(1));
      }
    }
    const currentSegment = toLineCoords(replayCurrentEntry?.segment_coordinates);
    const candidateSegment = toLineCoords(replayCurrentEntry?.candidate_coordinates);
    return {
      executedCoordinates: executed,
      currentSegment,
      candidateSegment,
    };
  }, [dynamicExecutionLog, replayCurrentEntry, replayStep, replayStepCount]);

  const replayEvents = useMemo(() => {
    let prevCost: number | null = null;
    return dynamicReplans
      .filter((item) => Boolean(item?.triggered_replan))
      .map((item) => {
        const currentCost = toNumberOrNull(item?.step_effective_cost_km);
        const gain = currentCost !== null && prevCost !== null ? prevCost - currentCost : null;
        prevCost = currentCost ?? prevCost;
        return {
          step: Number(item?.step ?? 0),
          timestamp: String(item?.timestamp ?? ""),
          reasons: Array.isArray(item?.trigger_reasons) ? item.trigger_reasons : [],
          gainKm: gain,
          updateMode: String(item?.update_mode ?? ""),
          runtimeMs: toNumberOrNull(item?.runtime_ms),
        };
      });
  }, [dynamicReplans]);

  useEffect(() => {
    if (!routeCandidates.length) {
      setSelectedCandidateId("requested");
      return;
    }
    const preferred = routeCandidates.find((c) => c.id === "requested" && c.status === "ok");
    const fallback = routeCandidates.find((c) => c.status === "ok");
    setSelectedCandidateId((preferred ?? fallback ?? routeCandidates[0]).id);
  }, [routeCandidates]);

  useEffect(() => {
    if (!replayStepCount) {
      setReplayStepIndex(0);
      setReplayPlaying(false);
      return;
    }
    setReplayStepIndex((prev) => Math.max(0, Math.min(prev, replayStepCount - 1)));
  }, [replayStepCount]);

  useEffect(() => {
    if (!replayPlaying || replayStepCount <= 1) return;
    const speed = Math.max(0.25, Number.parseFloat(replaySpeed) || 1);
    const intervalMs = Math.max(120, Math.round(900 / speed));
    const timer = window.setInterval(() => {
      setReplayStepIndex((prev) => {
        if (prev >= replayStepCount - 1) {
          setReplayPlaying(false);
          return replayStepCount - 1;
        }
        return prev + 1;
      });
    }, intervalMs);
    return () => window.clearInterval(timer);
  }, [replayPlaying, replaySpeed, replayStepCount]);

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
  const riskConstraintModeResult = String((routeMetrics as any)?.risk_constraint_mode ?? "none");
  const riskConstraintSatisfiedResult = Boolean((routeMetrics as any)?.risk_constraint_satisfied ?? true);
  const riskBudgetUsagePctResult = Number((routeMetrics as any)?.risk_budget_usage ?? 0) * 100;
  const riskConstraintMetricResult = Number((routeMetrics as any)?.risk_constraint_metric ?? 0);
  const safetyBanner = useMemo(() => {
    if (!routeMetrics) return null;
    if (riskConstraintModeResult === "none") {
      return {
        tone: "neutral" as const,
        title: t("summary.riskConstraint.none.title"),
        desc: t("summary.riskConstraint.none.desc"),
      };
    }
    if (riskConstraintSatisfiedResult) {
      return {
        tone: "pass" as const,
        title: t("summary.riskConstraint.pass.title"),
        desc: `${t("summary.riskConstraint.usage")} ${riskBudgetUsagePctResult.toFixed(1)}%，${t("summary.riskConstraint.metric")} ${riskConstraintMetricResult.toFixed(4)}`,
      };
    }
    return {
      tone: "fail" as const,
      title: t("summary.riskConstraint.fail.title"),
      desc: `${t("summary.riskConstraint.usage")} ${riskBudgetUsagePctResult.toFixed(1)}%，${t("summary.riskConstraint.metric")} ${riskConstraintMetricResult.toFixed(4)}`,
    };
  }, [riskBudgetUsagePctResult, riskConstraintMetricResult, riskConstraintModeResult, riskConstraintSatisfiedResult, routeMetrics, t]);

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

  const freshnessStatusKey =
    complianceNotices?.data_freshness?.status && ["fresh", "stale", "outdated", "unknown"].includes(String(complianceNotices.data_freshness.status))
      ? String(complianceNotices.data_freshness.status)
      : "unknown";
  const freshnessHint =
    complianceNotices?.data_freshness?.hint?.[language] ??
    complianceNotices?.data_freshness?.hint?.en ??
    "";
  const sourceHealthHint =
    complianceNotices?.source_credibility?.hint?.[language] ??
    complianceNotices?.source_credibility?.hint?.en ??
    "";

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
          risk_mode: riskMode,
          risk_weight_scale: riskWeightScale[0] / 100,
          risk_constraint_mode: riskConstraintMode,
          risk_budget: riskBudget[0] / 100,
          confidence_level: riskConfidence[0] / 100,
          return_candidates: returnCandidates,
          candidate_limit: candidateLimit[0],
          vessel_profile_id: vesselProfileId,
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

  const handlePlanDynamicRoute = async () => {
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

    const startIndex = timestampOptions.indexOf(timestamp);
    if (startIndex < 0) {
      toast.error("当前时间片不在可用时序集合中，无法执行时序回放");
      return;
    }
    const windowSize = Math.max(2, dynamicWindow[0]);
    const dynamicTimestamps = timestampOptions.slice(startIndex, startIndex + windowSize);
    if (dynamicTimestamps.length < 2) {
      toast.error("可用后续时间片不足，无法启动时序重规划");
      return;
    }

    setDynamicPlanning(true);
    setPlanning(true);
    setReplayPlaying(false);
    toast.loading("正在运行时序数字孪生回放...", { id: "plan-dynamic-route" });
    try {
      const cautionMode = safetyPolicy === "strict" ? "strict" : cautionHandling === "tiebreaker" ? "tie_breaker" : cautionHandling;
      const blockedSources =
        safetyPolicy === "blocked-bathy-only"
          ? ["bathy"]
          : safetyPolicy === "strict"
            ? ["bathy", "unet_blocked", "unet_caution"]
            : ["bathy", "unet_blocked"];
      const response = await planDynamicRoute({
        timestamps: dynamicTimestamps,
        start: { lat: startLatNum, lon: startLonNum },
        goal: { lat: goalLatNum, lon: goalLonNum },
        advance_steps: Math.max(1, dynamicAdvanceSteps[0]),
        policy: {
          objective: "shortest_distance_under_safety",
          blocked_sources: blockedSources,
          caution_mode: cautionMode,
          corridor_bias: corridorBias[0] / 100,
          smoothing: true,
          planner: plannerMode,
          risk_mode: riskMode,
          risk_weight_scale: riskWeightScale[0] / 100,
          risk_constraint_mode: riskConstraintMode,
          risk_budget: riskBudget[0] / 100,
          confidence_level: riskConfidence[0] / 100,
          return_candidates: false,
          candidate_limit: candidateLimit[0],
          dynamic_replan_mode: "on_event",
          replan_blocked_ratio: 0.002,
          replan_risk_spike: 0.05,
          replan_corridor_min: 0.05,
          replan_max_skip_steps: 2,
          dynamic_risk_switch_enabled: dynamicRiskSwitchEnabled,
          dynamic_risk_budget_km: Math.max(0, dynamicRiskBudgetKm[0]),
          dynamic_risk_warn_ratio: Math.max(0, dynamicRiskWarnRatio[0] / 100),
          dynamic_risk_hard_ratio: Math.max(dynamicRiskWarnRatio[0], dynamicRiskHardRatio[0]) / 100,
          dynamic_risk_warn_mode: dynamicRiskWarnMode,
          dynamic_risk_hard_mode: dynamicRiskHardMode,
          dynamic_risk_switch_min_interval: Math.max(1, dynamicRiskSwitchMinInterval[0]),
          vessel_profile_id: vesselProfileId,
        },
      });
      setRouteResult(response);
      const logLen = Array.isArray((response.explain as any)?.dynamic_execution_log)
        ? ((response.explain as any).dynamic_execution_log as unknown[]).length
        : 0;
      setReplayStepIndex(Math.max(0, logLen - 1));
      toast.success(`时序回放完成：${dynamicTimestamps.length} 个时间片`, { id: "plan-dynamic-route" });
      void captureAndUploadGalleryImage(response.gallery_id);
    } catch (error) {
      toast.error(`时序回放失败：${getErrorMessage(error)}`, { id: "plan-dynamic-route" });
    } finally {
      setDynamicPlanning(false);
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
        dynamic_replan_enabled: latestDynamicReplanEnabled,
        dynamic_window: Math.max(2, dynamicWindow[0]),
        dynamic_advance_steps: Math.max(1, dynamicAdvanceSteps[0]),
        start: { lat: startLatNum, lon: startLonNum },
        goal: { lat: goalLatNum, lon: goalLonNum },
        policy: {
          objective: "shortest_distance_under_safety",
          blocked_sources: blockedSources,
          caution_mode: cautionMode,
          corridor_bias: corridorBias[0] / 100,
          smoothing: true,
          planner: plannerMode,
          risk_mode: riskMode,
          risk_weight_scale: riskWeightScale[0] / 100,
          risk_constraint_mode: riskConstraintMode,
          risk_budget: riskBudget[0] / 100,
          confidence_level: riskConfidence[0] / 100,
          return_candidates: false,
          candidate_limit: candidateLimit[0],
          dynamic_risk_switch_enabled: dynamicRiskSwitchEnabled,
          dynamic_risk_budget_km: Math.max(0, dynamicRiskBudgetKm[0]),
          dynamic_risk_warn_ratio: Math.max(0, dynamicRiskWarnRatio[0] / 100),
          dynamic_risk_hard_ratio: Math.max(dynamicRiskWarnRatio[0], dynamicRiskHardRatio[0]) / 100,
          dynamic_risk_warn_mode: dynamicRiskWarnMode,
          dynamic_risk_hard_mode: dynamicRiskHardMode,
          dynamic_risk_switch_min_interval: Math.max(1, dynamicRiskSwitchMinInterval[0]),
          vessel_profile_id: vesselProfileId,
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

  const handleReplayStepPrev = () => {
    setReplayPlaying(false);
    setReplayStepIndex((prev) => Math.max(0, prev - 1));
  };

  const handleReplayStepNext = () => {
    setReplayPlaying(false);
    setReplayStepIndex((prev) => Math.min(Math.max(0, replayStepCount - 1), prev + 1));
  };

  const plannedCoords = activeRouteGeojson?.geometry?.coordinates ?? [];
  const routedStart = plannedCoords.length ? { lat: plannedCoords[0][1], lon: plannedCoords[0][0] } : null;
  const routedGoal = plannedCoords.length ? { lat: plannedCoords[plannedCoords.length - 1][1], lon: plannedCoords[plannedCoords.length - 1][0] } : null;
  const mapStart = routedStart ?? { lat: Number.parseFloat(startLat) || 0, lon: Number.parseFloat(startLon) || 0 };
  const mapGoal = routedGoal ?? { lat: Number.parseFloat(goalLat) || 0, lon: Number.parseFloat(goalLon) || 0 };
  const dynamicWindowPreview = useMemo(() => {
    const startIndex = timestampOptions.indexOf(timestamp);
    if (startIndex < 0) return [] as string[];
    return timestampOptions.slice(startIndex, startIndex + Math.max(2, dynamicWindow[0]));
  }, [dynamicWindow, timestamp, timestampOptions]);
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

            {complianceNotices ? (
              <Card className="border-amber-200 bg-amber-50/50">
                <CardContent className="p-3 space-y-2 text-xs">
                  <div className="font-medium text-amber-900">{t("compliance.title")}</div>
                  <div className="text-amber-800">{t("compliance.researchOnly")}</div>
                  <div className="rounded border border-amber-200 bg-white p-2 text-slate-700 space-y-1">
                    <div>
                      {t("compliance.freshness")}：{t(`compliance.status.${freshnessStatusKey}`)}
                      {typeof complianceNotices.data_freshness?.age_hours === "number"
                        ? ` (${complianceNotices.data_freshness.age_hours.toFixed(1)}h)`
                        : ""}
                    </div>
                    {freshnessHint ? <div>{freshnessHint}</div> : null}
                    <div>
                      {t("compliance.sourceHealth")}：{sourceHealthHint || "-"}
                    </div>
                    {complianceNotices.source_credibility?.updated_at ? (
                      <div className="text-[11px] text-slate-500">
                        {t("compliance.updatedAt")}：{String(complianceNotices.source_credibility.updated_at)}
                      </div>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
            ) : null}

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
                    <Label>船型预设（北极典型）</Label>
                    <Select value={vesselProfileId} onValueChange={setVesselProfileId}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {vesselProfiles.length ? (
                          vesselProfiles.map((item) => (
                            <SelectItem key={item.id} value={item.id}>
                              {item.name}
                            </SelectItem>
                          ))
                        ) : (
                          <SelectItem value="arc7_lng">Arc7 LNG Carrier</SelectItem>
                        )}
                      </SelectContent>
                    </Select>
                    {vesselProfiles.length ? (
                      (() => {
                        const current = vesselProfiles.find((item) => item.id === vesselProfileId);
                        if (!current) return null;
                        return (
                          <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-2 text-xs text-slate-700 space-y-1">
                            <div>冰级：{current.ice_class}</div>
                            <div>吃水：{Number(current.draft_m).toFixed(1)} m</div>
                            <div>建议最小水深：{Number(current.min_safe_depth_m).toFixed(1)} m</div>
                            <div>默认风险模式：{current.default_policy?.risk_mode ?? "-"}</div>
                          </div>
                        );
                      })()
                    ) : null}
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

                  <div className="space-y-2">
                    <Label>风险模式</Label>
                    <Select value={riskMode} onValueChange={setRiskMode}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="conservative">Conservative（保守）</SelectItem>
                        <SelectItem value="balanced">Balanced（均衡）</SelectItem>
                        <SelectItem value="aggressive">Aggressive（激进）</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>风险权重缩放</Label>
                    <div className="flex items-center gap-3">
                      <Slider value={riskWeightScale} onValueChange={setRiskWeightScale} min={0} max={300} step={5} className="flex-1" />
                      <span className="text-sm text-muted-foreground w-12 text-right">{(riskWeightScale[0] / 100).toFixed(2)}</span>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>风险约束</Label>
                    <Select value={riskConstraintMode} onValueChange={setRiskConstraintMode}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">无</SelectItem>
                        <SelectItem value="chance">Chance</SelectItem>
                        <SelectItem value="cvar">CVaR</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {riskConstraintMode !== "none" ? (
                    <>
                      <div className="space-y-2">
                        <Label>风险预算</Label>
                        <div className="flex items-center gap-3">
                          <Slider value={riskBudget} onValueChange={setRiskBudget} min={1} max={100} step={1} className="flex-1" />
                          <span className="text-sm text-muted-foreground w-12 text-right">{(riskBudget[0] / 100).toFixed(2)}</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>置信水平</Label>
                        <div className="flex items-center gap-3">
                          <Slider value={riskConfidence} onValueChange={setRiskConfidence} min={50} max={99} step={1} className="flex-1" />
                          <span className="text-sm text-muted-foreground w-12 text-right">{(riskConfidence[0] / 100).toFixed(2)}</span>
                        </div>
                      </div>
                    </>
                  ) : null}

                  <div className="rounded-md border border-slate-200 bg-white px-3 py-2 space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <Label htmlFor="return-candidates">多候选路线</Label>
                      <input
                        id="return-candidates"
                        type="checkbox"
                        checked={returnCandidates}
                        onChange={(e) => setReturnCandidates(e.target.checked)}
                        className="h-4 w-4"
                      />
                    </div>
                    {returnCandidates ? (
                      <div className="space-y-1">
                        <Label>候选数量</Label>
                        <div className="flex items-center gap-3">
                          <Slider value={candidateLimit} onValueChange={setCandidateLimit} min={1} max={6} step={1} className="flex-1" />
                          <span className="text-sm text-muted-foreground w-8 text-right">{candidateLimit[0]}</span>
                        </div>
                      </div>
                    ) : null}
                  </div>

                  <Button onClick={handlePlanRoute} className="w-full gap-2 bg-green-600 hover:bg-green-700" size="lg" disabled={planning}>
                    <Navigation className="size-4" />
                    {planning ? t("workspace.planRoute.loading") : t("workspace.planRoute")}
                  </Button>
                  <div className="rounded-md border border-emerald-200 bg-emerald-50 p-3 space-y-3">
                    <div className="text-sm font-medium text-emerald-900">时序数字孪生回放</div>
                    <div className="space-y-1">
                      <Label>时间片窗口</Label>
                      <div className="flex items-center gap-3">
                        <Slider value={dynamicWindow} onValueChange={setDynamicWindow} min={2} max={24} step={1} className="flex-1" />
                        <span className="text-sm text-muted-foreground w-8 text-right">{dynamicWindow[0]}</span>
                      </div>
                    </div>
                    <div className="space-y-1">
                      <Label>每步推进边数</Label>
                      <div className="flex items-center gap-3">
                        <Slider value={dynamicAdvanceSteps} onValueChange={setDynamicAdvanceSteps} min={1} max={48} step={1} className="flex-1" />
                        <span className="text-sm text-muted-foreground w-8 text-right">{dynamicAdvanceSteps[0]}</span>
                      </div>
                    </div>
                    <div className="rounded-md border border-emerald-200 bg-white px-2 py-2 space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <Label htmlFor="dynamic-risk-switch-enabled">时序风险策略切换</Label>
                        <input
                          id="dynamic-risk-switch-enabled"
                          type="checkbox"
                          checked={dynamicRiskSwitchEnabled}
                          onChange={(e) => setDynamicRiskSwitchEnabled(e.target.checked)}
                          className="h-4 w-4"
                        />
                      </div>
                      {dynamicRiskSwitchEnabled ? (
                        <>
                          <div className="space-y-1">
                            <Label>风险预算（km）</Label>
                            <div className="flex items-center gap-3">
                              <Slider value={dynamicRiskBudgetKm} onValueChange={setDynamicRiskBudgetKm} min={0} max={20} step={0.5} className="flex-1" />
                              <span className="text-sm text-muted-foreground w-10 text-right">{dynamicRiskBudgetKm[0].toFixed(1)}</span>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <Label>预警阈值</Label>
                            <div className="flex items-center gap-3">
                              <Slider value={dynamicRiskWarnRatio} onValueChange={setDynamicRiskWarnRatio} min={10} max={200} step={5} className="flex-1" />
                              <span className="text-sm text-muted-foreground w-10 text-right">{dynamicRiskWarnRatio[0]}%</span>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <Label>硬阈值</Label>
                            <div className="flex items-center gap-3">
                              <Slider value={dynamicRiskHardRatio} onValueChange={setDynamicRiskHardRatio} min={10} max={250} step={5} className="flex-1" />
                              <span className="text-sm text-muted-foreground w-10 text-right">{dynamicRiskHardRatio[0]}%</span>
                            </div>
                          </div>
                          <div className="space-y-1">
                            <Label>预警阶段策略</Label>
                            <Select value={dynamicRiskWarnMode} onValueChange={setDynamicRiskWarnMode}>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="conservative">Conservative</SelectItem>
                                <SelectItem value="balanced">Balanced</SelectItem>
                                <SelectItem value="aggressive">Aggressive</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-1">
                            <Label>硬阈值阶段策略</Label>
                            <Select value={dynamicRiskHardMode} onValueChange={setDynamicRiskHardMode}>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="conservative">Conservative</SelectItem>
                                <SelectItem value="balanced">Balanced</SelectItem>
                                <SelectItem value="aggressive">Aggressive</SelectItem>
                              </SelectContent>
                            </Select>
                          </div>
                          <div className="space-y-1">
                            <Label>切换最小间隔（步）</Label>
                            <div className="flex items-center gap-3">
                              <Slider
                                value={dynamicRiskSwitchMinInterval}
                                onValueChange={setDynamicRiskSwitchMinInterval}
                                min={1}
                                max={8}
                                step={1}
                                className="flex-1"
                              />
                              <span className="text-sm text-muted-foreground w-8 text-right">{dynamicRiskSwitchMinInterval[0]}</span>
                            </div>
                          </div>
                        </>
                      ) : null}
                    </div>
                    <div className="rounded-md border border-emerald-200 bg-white px-2 py-1 text-[11px] text-emerald-900">
                      窗口范围：{dynamicWindowPreview[0] ?? "无"} → {dynamicWindowPreview[dynamicWindowPreview.length - 1] ?? "无"}（共{" "}
                      {dynamicWindowPreview.length} 个时间片）
                    </div>
                    <Button
                      onClick={handlePlanDynamicRoute}
                      className="w-full gap-2 bg-emerald-600 hover:bg-emerald-700"
                      size="lg"
                      disabled={planning || dynamicPlanning}
                    >
                      <Navigation className="size-4" />
                      {dynamicPlanning ? "时序回放计算中..." : "执行时序重规划回放"}
                    </Button>
                  </div>
                  <div className="rounded-md border border-slate-200 bg-white p-3 space-y-2">
                    <Label htmlFor="latest-date">最新预报日期</Label>
                    <input
                      id="latest-date"
                      type="date"
                      value={latestDate}
                      onChange={(e) => setLatestDate(e.target.value)}
                      className="h-9 w-full rounded-md border border-slate-300 px-2 text-sm"
                    />
                    <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-2 space-y-2">
                      <div className="flex items-center justify-between gap-2">
                        <Label htmlFor="latest-dynamic-enabled">启用时序重规划联动</Label>
                        <input
                          id="latest-dynamic-enabled"
                          type="checkbox"
                          checked={latestDynamicReplanEnabled}
                          onChange={(e) => setLatestDynamicReplanEnabled(e.target.checked)}
                          className="h-4 w-4"
                        />
                      </div>
                      {latestDynamicReplanEnabled ? (
                        <div className="text-[11px] text-slate-600">
                          使用窗口 {Math.max(2, dynamicWindow[0])}、步长 {Math.max(1, dynamicAdvanceSteps[0])} 进行 latest + 时序重规划。
                        </div>
                      ) : (
                        <div className="text-[11px] text-slate-600">关闭后仅执行 latest 单时刻静态规划。</div>
                      )}
                    </div>
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
            showRoute={Boolean(activeRouteGeojson)}
            onMapClick={handleMapClick}
            routeGeojson={activeRouteGeojson ?? undefined}
            start={mapStart}
            goal={mapGoal}
            replayOverlay={replayOverlay ?? undefined}
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
                ...(replayOverlay ? [{ color: "#16a34a", label: "已执行段" }] : []),
                ...(replayOverlay ? [{ color: "#0ea5e9", label: "当前段" }] : []),
                ...(replayOverlay ? [{ color: "#f59e0b", label: "候选重规划段" }] : []),
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
                  {routeCandidates.length ? (
                    <div className="space-y-2 rounded-lg border border-slate-200 bg-slate-50 p-3">
                      <div className="text-sm font-medium text-slate-800">候选路线对比（distance / risk / caution / corridor）</div>
                      <div className="space-y-2">
                        {routeCandidates.map((c) => {
                          const isActive = c.id === selectedCandidateId && c.status === "ok";
                          return (
                            <button
                              key={c.id}
                              type="button"
                              disabled={c.status !== "ok"}
                              onClick={() => c.status === "ok" && setSelectedCandidateId(c.id)}
                              className={`w-full rounded-md border px-2 py-2 text-left text-xs transition ${
                                isActive ? "border-blue-500 bg-blue-50" : "border-slate-200 bg-white hover:border-slate-300"
                              } ${c.status !== "ok" ? "opacity-60 cursor-not-allowed" : ""}`}
                            >
                              <div className="flex items-center justify-between gap-2">
                                <span className="font-medium">{c.label}</span>
                                <span>{c.status === "ok" ? `Pareto#${c.pareto_rank ?? "-"}` : "failed"}</span>
                              </div>
                              {c.status === "ok" ? (
                                <div className="mt-1 grid grid-cols-2 gap-1 text-slate-700">
                                  <div>d={Number(c.distance_km ?? 0).toFixed(2)} km</div>
                                  <div>risk={Number(c.risk_exposure ?? 0).toFixed(3)}</div>
                                  <div>caution={Number(c.caution_len_km ?? 0).toFixed(2)} km</div>
                                  <div>corridor={Number(c.corridor_score ?? 0).toFixed(3)}</div>
                                </div>
                              ) : (
                                <div className="mt-1 text-red-600">{c.error ?? "candidate failed"}</div>
                              )}
                            </button>
                          );
                        })}
                      </div>
                      {routeMetrics?.pareto_summary ? (
                        <div className="text-xs text-slate-600">
                          前沿解数量：{Number((routeMetrics.pareto_summary as any)?.frontier_count ?? 0)} / 候选总数：
                          {Number((routeMetrics.pareto_summary as any)?.candidate_count ?? routeCandidates.length)}
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {activeCandidate?.status === "ok" ? (
                    <div className="rounded-lg border border-indigo-200 bg-indigo-50 p-3 text-xs text-indigo-900">
                      <div className="font-medium">为什么当前选中这条路线</div>
                      <div className="mt-1">
                        {activeCandidate.pareto_frontier
                          ? `该路线位于 Pareto 前沿（rank=${activeCandidate.pareto_rank ?? 1}），在距离与风险之间为非支配解。`
                          : `该路线 Pareto 排名 ${activeCandidate.pareto_rank ?? "-"}，可在候选卡片中切换到更短或更低风险方案。`}
                      </div>
                    </div>
                  ) : null}

                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <StatCard label={t("summary.distance")} value={routeSummary.distanceKm.toFixed(1)} unit="km" />
                    <StatCard label={t("summary.distance")} value={routeSummary.distanceNm.toFixed(1)} unit="nm" />
                  </div>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <StatCard label={t("summary.safe")} value={routeSummary.safePct} unit="%" variant="success" />
                    <StatCard label={t("summary.caution")} value={routeSummary.cautionPct} unit="%" variant="warning" />
                  </div>
                  <StatCard label={t("summary.alignment")} value={routeSummary.alignment.toFixed(2)} variant="success" />
                  {safetyBanner ? (
                    <div
                      className={`p-3 rounded-lg border flex items-start gap-2 ${
                        safetyBanner.tone === "pass"
                          ? "border-green-200 bg-green-50"
                          : safetyBanner.tone === "fail"
                            ? "border-red-200 bg-red-50"
                            : "border-slate-200 bg-slate-50"
                      }`}
                    >
                      {safetyBanner.tone === "pass" ? (
                        <CheckCircle2 className="size-4 text-green-600 mt-0.5 flex-shrink-0" />
                      ) : (
                        <AlertCircle
                          className={`size-4 mt-0.5 flex-shrink-0 ${
                            safetyBanner.tone === "fail" ? "text-red-600" : "text-slate-600"
                          }`}
                        />
                      )}
                      <div
                        className={`text-sm ${
                          safetyBanner.tone === "pass"
                            ? "text-green-800"
                            : safetyBanner.tone === "fail"
                              ? "text-red-800"
                              : "text-slate-700"
                        }`}
                      >
                        <div className="font-medium mb-1">{safetyBanner.title}</div>
                        <div className="text-xs">{safetyBanner.desc}</div>
                      </div>
                    </div>
                  ) : null}
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
                  {routeResult?.resolved?.dynamic ? (
                    <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-900 space-y-1">
                      <div className="font-medium">latest × 时序重规划</div>
                      <div>模式：{String(routeResult.resolved.dynamic.mode ?? "unknown")}</div>
                      <div>窗口：{Number(routeResult.resolved.dynamic.requested_window ?? 0)}</div>
                      <div>步长：{Number(routeResult.resolved.dynamic.requested_advance_steps ?? 0)}</div>
                      <div>时间片：{Array.isArray(routeResult.resolved.dynamic.used_timestamps) ? routeResult.resolved.dynamic.used_timestamps.join(" -> ") : "无"}</div>
                      {routeResult.resolved.dynamic.note ? <div>备注：{routeResult.resolved.dynamic.note}</div> : null}
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

            {routeResult && dynamicExecutionLog.length > 0 ? (
              <div>
                <h3 className="mb-3">时序回放面板</h3>
                <Card>
                  <CardContent className="pt-4 space-y-3 text-sm">
                    <div className="rounded-md border border-slate-200 bg-slate-50 px-3 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <div className="text-xs text-slate-600">
                          Step {replayStep + 1}/{replayStepCount}
                        </div>
                        <div className="text-xs text-slate-600">{String(replayCurrentEntry?.timestamp ?? "-")}</div>
                      </div>
                      <input
                        type="range"
                        min={0}
                        max={Math.max(0, replayStepCount - 1)}
                        value={replayStep}
                        onChange={(e) => {
                          setReplayPlaying(false);
                          setReplayStepIndex(Number.parseInt(e.target.value, 10) || 0);
                        }}
                        className="mt-2 h-2 w-full"
                      />
                    </div>

                    <div className="flex flex-wrap items-center gap-2">
                      <Button type="button" size="sm" variant="outline" onClick={() => setReplayStepIndex(0)}>
                        复位
                      </Button>
                      <Button type="button" size="sm" variant="outline" onClick={handleReplayStepPrev}>
                        上一步
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => {
                          if (replayStep >= replayStepCount - 1) {
                            setReplayStepIndex(0);
                          }
                          setReplayPlaying((prev) => !prev);
                        }}
                      >
                        {replayPlaying ? "暂停" : "播放"}
                      </Button>
                      <Button type="button" size="sm" variant="outline" onClick={handleReplayStepNext}>
                        下一步
                      </Button>
                      <Select value={replaySpeed} onValueChange={setReplaySpeed}>
                        <SelectTrigger className="h-8 w-24">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="0.5">0.5x</SelectItem>
                          <SelectItem value="1">1x</SelectItem>
                          <SelectItem value="2">2x</SelectItem>
                          <SelectItem value="4">4x</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="grid grid-cols-2 gap-2 text-xs text-slate-700">
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        当前段距离：{Number(replayCurrentEntry?.moved_distance_km ?? 0).toFixed(3)} km
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        当前段风险附加：{Number(replayCurrentEntry?.step_risk_extra_km ?? 0).toFixed(3)} km
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        累计距离：{Number(replayCurrentEntry?.cumulative_distance_km ?? 0).toFixed(3)} km
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        累计重规划耗时：{Number(replayCurrentEntry?.cumulative_replan_runtime_ms ?? 0).toFixed(1)} ms
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-2 text-xs text-slate-700">
                      <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1">
                        风险切换次数：{Number((routeMetrics as any)?.dynamic_risk_switch_count ?? 0)}
                      </div>
                      <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1">
                        风险预算占用：{(Number((routeMetrics as any)?.dynamic_risk_budget_usage_final ?? 0) * 100).toFixed(1)}%
                      </div>
                      <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1">
                        切换增益总计：{Number((routeMetrics as any)?.dynamic_risk_switch_gain_total ?? 0).toFixed(3)}
                      </div>
                      <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1">
                        预算保护步数：{Number((routeMetrics as any)?.dynamic_risk_budget_protection_steps ?? 0)}
                      </div>
                    </div>

                    <div className="rounded-md border border-slate-200 bg-white p-2 text-xs space-y-2">
                      <div className="font-medium text-slate-800">重规划事件（原因与增益）</div>
                      {replayEvents.length ? (
                        replayEvents.map((event) => (
                          <div key={`${event.step}-${event.timestamp}`} className="rounded border border-slate-200 bg-slate-50 px-2 py-2">
                            <div className="flex items-center justify-between gap-2">
                              <span className="font-medium">
                                Step {event.step} · {event.updateMode || "unknown"}
                              </span>
                              <span className={event.gainKm !== null && event.gainKm >= 0 ? "text-green-700" : "text-amber-700"}>
                                增益 {event.gainKm === null ? "-" : `${event.gainKm.toFixed(3)} km`}
                              </span>
                            </div>
                            <div className="mt-1 text-slate-600">触发原因：{event.reasons.length ? event.reasons.join(", ") : "无"}</div>
                            <div className="mt-1 text-slate-500">
                              {event.timestamp} · {event.runtimeMs === null ? "-" : `${event.runtimeMs.toFixed(1)} ms`}
                            </div>
                          </div>
                        ))
                      ) : (
                        <div className="text-slate-500">当前结果未触发事件型重规划。</div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </div>
            ) : null}

            {routeResult && (
              <div>
                <h3 className="mb-3">{t("explain.title")}</h3>
                <Card>
                  <CardContent className="pt-4 space-y-3 text-sm">
                    <div className="grid grid-cols-1 gap-2 text-xs text-slate-700">
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        规划器：{String(routeMetrics?.planner ?? "unknown")}
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        船型：{String((routeMetrics as any)?.vessel_profile?.name ?? (routeMetrics as any)?.vessel_profile?.id ?? vesselProfileId)}
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        有效代价：{Number(routeMetrics?.route_cost_effective_km ?? routeMetrics?.distance_km ?? 0).toFixed(3)} km
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        代价拆解：基础 {Number(routeMetrics?.route_cost_base_km ?? routeMetrics?.distance_km ?? 0).toFixed(3)} / 谨慎附加{" "}
                        {Number(routeMetrics?.route_cost_caution_extra_km ?? 0).toFixed(3)} / 航道折扣{" "}
                        {Number(routeMetrics?.route_cost_corridor_discount_km ?? 0).toFixed(3)} km
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        风险贴边比例：{(Number(routeMetrics?.adjacent_blocked_ratio ?? 0) * 100).toFixed(1)}%
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        谨慎网格占比：{(Number(routeMetrics?.caution_cell_ratio ?? 0) * 100).toFixed(1)}%
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        走廊贴合分位：P50 {Number(routeMetrics?.corridor_alignment_p50 ?? routeMetrics?.corridor_alignment ?? 0).toFixed(3)} / P90{" "}
                        {Number(routeMetrics?.corridor_alignment_p90 ?? routeMetrics?.corridor_alignment ?? 0).toFixed(3)}
                      </div>
                      <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                        起终点修正：start {Number(routeMetrics?.start_adjust_km ?? 0).toFixed(3)} / goal {Number(routeMetrics?.goal_adjust_km ?? 0).toFixed(3)} km
                      </div>
                      {Array.isArray((routeMetrics as any)?.dynamic_replans) ? (
                        <div className="rounded-md border border-slate-200 bg-slate-50 px-2 py-1">
                          动态重规划：{((routeMetrics as any).dynamic_replans as unknown[]).length} 次，累计{" "}
                          {Number(routeMetrics?.replan_runtime_ms_total ?? 0).toFixed(1)} ms，均值{" "}
                          {Number(routeMetrics?.replan_runtime_ms_mean ?? 0).toFixed(1)} ms
                        </div>
                      ) : null}
                      {Boolean((routeMetrics as any)?.dynamic_risk_switch_enabled) ? (
                        <div className="rounded-md border border-emerald-200 bg-emerald-50 px-2 py-1 text-emerald-900">
                          风险预算切换：{Number((routeMetrics as any)?.dynamic_risk_switch_count ?? 0)} 次，预算占用{" "}
                          {(Number((routeMetrics as any)?.dynamic_risk_budget_usage_final ?? 0) * 100).toFixed(1)}%，切换增益{" "}
                          {Number((routeMetrics as any)?.dynamic_risk_switch_gain_total ?? 0).toFixed(3)}
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

