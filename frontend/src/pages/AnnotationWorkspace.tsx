import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import { toast } from "sonner";
import { CircleMarker, MapContainer, Pane, Polygon, Polyline, Rectangle, TileLayer, useMap, useMapEvents } from "react-leaflet";
import type { LatLngBoundsExpression } from "leaflet";
import "leaflet/dist/leaflet.css";

import {
  getActiveReviewItems,
  getActiveReviewRuns,
  getAnnotationPatch,
  getApiOrigin,
  getErrorMessage,
  getLayers,
  getTimestamps,
  saveAnnotationPatch,
  type ActiveReviewItem,
  type ActiveReviewRun,
  type AnnotationOperation,
} from "../api/client";
import { useLanguage } from "../contexts/LanguageContext";
import { Button } from "../components/ui/button";
import { Card, CardContent } from "../components/ui/card";
import { Label } from "../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { Slider } from "../components/ui/slider";

const API_ORIGIN = getApiOrigin();
const INITIAL_BOUNDS: LatLngBoundsExpression = [
  [60, 20],
  [80, 180],
];
const AOI_BOUNDS: LatLngBoundsExpression = [
  [60, 20],
  [80, 180],
];

type LayerToggleState = { enabled: boolean; opacity: number };
type LayerStates = {
  bathy: LayerToggleState;
  ais: LayerToggleState;
  unet: LayerToggleState;
  caution: LayerToggleState;
};

const DEFAULT_LAYERS: LayerStates = {
  bathy: { enabled: true, opacity: 70 },
  ais: { enabled: true, opacity: 55 },
  unet: { enabled: false, opacity: 60 },
  caution: { enabled: true, opacity: 90 },
};

function layerUrl(layerId: string, timestamp: string, tileRevision: number): string {
  const rev = `${timestamp}-${tileRevision}`;
  return `${API_ORIGIN}/v1/tiles/${layerId}/{z}/{x}/{y}.png?timestamp=${encodeURIComponent(timestamp)}&v=${encodeURIComponent(rev)}`;
}

function RasterLayer({
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
  const paneName = `annotation-overlay-${layerId}`;
  return (
    <Pane name={paneName} style={{ zIndex }}>
      <TileLayer
        key={`${layerId}-${timestamp}-${tileRevision}`}
        pane={paneName}
        url={layerUrl(layerId, timestamp, tileRevision)}
        opacity={Math.max(0, Math.min(1, opacity / 100))}
        tileSize={256}
        noWrap
        crossOrigin="anonymous"
      />
    </Pane>
  );
}

function MapResizeGuard({ resizeKey }: { resizeKey: string }) {
  const map = useMap();
  useEffect(() => {
    const refresh = () => map.invalidateSize({ pan: false });
    const t1 = window.setTimeout(refresh, 0);
    const t2 = window.setTimeout(refresh, 220);
    window.addEventListener("resize", refresh);
    window.addEventListener("orientationchange", refresh);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
      window.removeEventListener("resize", refresh);
      window.removeEventListener("orientationchange", refresh);
    };
  }, [map, resizeKey]);
  return null;
}

function DrawEvents({
  enabled,
  tool,
  onPolygonClick,
  onStrokePreview,
  onStrokeEnd,
}: {
  enabled: boolean;
  tool: "polygon" | "brush";
  onPolygonClick: (lat: number, lon: number) => void;
  onStrokePreview: (points: Array<{ lat: number; lon: number }>) => void;
  onStrokeEnd: (points: Array<{ lat: number; lon: number }>) => void;
}) {
  const drawingRef = useRef(false);
  const pointsRef = useRef<Array<{ lat: number; lon: number }>>([]);
  useMapEvents({
    click: (e) => {
      if (!enabled || tool !== "polygon") return;
      onPolygonClick(e.latlng.lat, e.latlng.lng);
    },
    mousedown: (e) => {
      if (!enabled || tool !== "brush") return;
      drawingRef.current = true;
      pointsRef.current = [{ lat: e.latlng.lat, lon: e.latlng.lng }];
      onStrokePreview(pointsRef.current);
    },
    mousemove: (e) => {
      if (!drawingRef.current || tool !== "brush") return;
      pointsRef.current = [...pointsRef.current, { lat: e.latlng.lat, lon: e.latlng.lng }];
      onStrokePreview(pointsRef.current);
    },
    mouseup: () => {
      if (!drawingRef.current || tool !== "brush") return;
      drawingRef.current = false;
      if (pointsRef.current.length >= 2) {
        onStrokeEnd(pointsRef.current);
      }
      pointsRef.current = [];
      onStrokePreview([]);
    },
  });
  return null;
}

function buildOperationId(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `op_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
}

export default function AnnotationWorkspace() {
  const { t } = useLanguage();
  const [timestampOptions, setTimestampOptions] = useState<string[]>([]);
  const [timestamp, setTimestamp] = useState("");
  const [layerState, setLayerState] = useState<LayerStates>(DEFAULT_LAYERS);
  const [tileRevision, setTileRevision] = useState(0);
  const [operations, setOperations] = useState<AnnotationOperation[]>([]);
  const [draftPoints, setDraftPoints] = useState<Array<{ lat: number; lon: number }>>([]);
  const [drawMode, setDrawMode] = useState<"add" | "erase">("add");
  const [drawTool, setDrawTool] = useState<"polygon" | "brush">("polygon");
  const [brushRadiusCells, setBrushRadiusCells] = useState([2]);
  const [drawingEnabled, setDrawingEnabled] = useState(false);
  const [saveNote, setSaveNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [loadingPatch, setLoadingPatch] = useState(false);
  const [availability, setAvailability] = useState<Record<string, boolean>>({
    bathy: true,
    ais_heatmap: true,
    unet_pred: true,
    caution_mask: true,
  });
  const [patchStats, setPatchStats] = useState<{
    caution_pixels: number;
    caution_ratio: number;
    operations_count: number;
  } | null>(null);

  const [reviewRuns, setReviewRuns] = useState<ActiveReviewRun[]>([]);
  const [reviewRunId, setReviewRunId] = useState("");
  const [reviewItems, setReviewItems] = useState<ActiveReviewItem[]>([]);
  const [reviewLoading, setReviewLoading] = useState(false);
  const [layoutMode, setLayoutMode] = useState<"desktop" | "mobile">("desktop");
  const [isWideViewport, setIsWideViewport] = useState(false);
  const mapResizeKeyRef = useRef(0);

  useEffect(() => {
    const mql = window.matchMedia("(min-width: 1100px)");
    const update = () => {
      setIsWideViewport(mql.matches);
      setLayoutMode(mql.matches ? "desktop" : "mobile");
    };
    update();
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", update);
      return () => mql.removeEventListener("change", update);
    }
    mql.addListener(update);
    return () => mql.removeListener(update);
  }, []);

  const useDesktopLayout = layoutMode === "desktop" && isWideViewport;
  const pageStyle: CSSProperties = useDesktopLayout
    ? { height: "calc(100dvh - 72px)", minHeight: 640 }
    : { minHeight: "calc(100dvh - 72px)" };

  const shellStyle: CSSProperties = useDesktopLayout
    ? { display: "grid", gridTemplateColumns: "360px 1fr", gap: 0, height: "100%" }
    : { display: "flex", flexDirection: "column", minHeight: "100%" };

  const sideStyle: CSSProperties = useDesktopLayout
    ? { borderRightWidth: 1, maxHeight: "100%", overflowY: "auto" }
    : { borderBottomWidth: 1, maxHeight: "40dvh", overflowY: "auto" };

  const mapStyle: CSSProperties = useDesktopLayout
    ? { position: "relative", minHeight: 0 }
    : { position: "relative", minHeight: "58dvh" };

  const refreshLayers = useCallback(async (ts: string) => {
    const res = await getLayers(ts);
    const layerMap: Record<string, boolean> = {};
    for (const item of res.layers) layerMap[item.id] = !!item.available;
    setAvailability((prev) => ({ ...prev, ...layerMap }));
    setTileRevision((v) => v + 1);
  }, []);

  const refreshPatch = useCallback(async (ts: string) => {
    setLoadingPatch(true);
    try {
      const res = await getAnnotationPatch(ts);
      setOperations(res.operations ?? []);
      setPatchStats({
        caution_pixels: Number(res.stats?.caution_pixels ?? 0),
        caution_ratio: Number(res.stats?.caution_ratio ?? 0),
        operations_count: Number(res.stats?.operations_count ?? 0),
      });
      setTileRevision((v) => v + 1);
    } catch (error) {
      toast.error(`读取标注补丁失败: ${getErrorMessage(error)}`);
      setOperations([]);
      setPatchStats(null);
    } finally {
      setLoadingPatch(false);
    }
  }, []);

  useEffect(() => {
    let active = true;
    async function loadTimestamps() {
      try {
        const res = await getTimestamps();
        if (!active) return;
        setTimestampOptions(res.timestamps);
        setTimestamp((prev) => prev || res.timestamps[0] || "");
      } catch (error) {
        toast.error(`加载时间片失败: ${getErrorMessage(error)}`);
      }
    }
    void loadTimestamps();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadReviewRuns() {
      try {
        const res = await getActiveReviewRuns();
        if (!active) return;
        setReviewRuns(res.runs);
        setReviewRunId((prev) => prev || res.runs[0]?.run_id || "");
      } catch {
        setReviewRuns([]);
        setReviewRunId("");
      }
    }
    void loadReviewRuns();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadReviewItems() {
      if (!reviewRunId) {
        setReviewItems([]);
        return;
      }
      setReviewLoading(true);
      try {
        const res = await getActiveReviewItems(reviewRunId, 20);
        if (!active) return;
        setReviewItems(res.items);
      } catch {
        if (!active) return;
        setReviewItems([]);
      } finally {
        if (active) setReviewLoading(false);
      }
    }
    void loadReviewItems();
    return () => {
      active = false;
    };
  }, [reviewRunId]);

  useEffect(() => {
    if (!timestamp) return;
    void refreshLayers(timestamp);
    void refreshPatch(timestamp);
    setDraftPoints([]);
  }, [timestamp, refreshLayers, refreshPatch]);

  const appendDraftPoint = useCallback((lat: number, lon: number) => {
    setDraftPoints((prev) => [...prev, { lat, lon }]);
  }, []);

  const handleStrokeEnd = useCallback(
    (points: Array<{ lat: number; lon: number }>) => {
      if (!drawingEnabled || drawTool !== "brush") return;
      if (points.length < 2) return;
      const op: AnnotationOperation = {
        id: buildOperationId(),
        mode: drawMode,
        shape: "stroke",
        radius_cells: brushRadiusCells[0] ?? 2,
        points,
      };
      setOperations((prev) => [...prev, op]);
      setDraftPoints([]);
      toast.success(drawMode === "add" ? "已添加画笔 caution" : "已添加画笔擦除");
    },
    [brushRadiusCells, drawMode, drawTool, drawingEnabled]
  );

  const finalizeDraft = useCallback(
    (mode: "add" | "erase") => {
      if (draftPoints.length < 3) {
        toast.warning("请至少点 3 个点形成多边形");
        return;
      }
      const op: AnnotationOperation = {
        id: buildOperationId(),
        mode,
        points: draftPoints.map((p) => ({ lat: p.lat, lon: p.lon })),
      };
      setOperations((prev) => [...prev, op]);
      setDraftPoints([]);
      setDrawingEnabled(false);
      toast.success(mode === "add" ? "已加入 caution 多边形" : "已加入擦除多边形");
    },
    [draftPoints]
  );

  const handleSave = useCallback(async () => {
    if (!timestamp) {
      toast.warning("请先选择时间片");
      return;
    }
    setSaving(true);
    try {
      const res = await saveAnnotationPatch({
        timestamp,
        operations,
        note: saveNote,
        author: "web_annotation_workspace",
      });
      setOperations(res.operations ?? operations);
      setPatchStats({
        caution_pixels: Number(res.stats?.caution_pixels ?? 0),
        caution_ratio: Number(res.stats?.caution_ratio ?? 0),
        operations_count: Number(res.stats?.operations_count ?? 0),
      });
      setTileRevision((v) => v + 1);
      toast.success("保存成功，caution_mask/y_class 已更新");
    } catch (error) {
      toast.error(`保存失败: ${getErrorMessage(error)}`);
    } finally {
      setSaving(false);
    }
  }, [operations, saveNote, timestamp]);

  const draftPolyline = useMemo(() => draftPoints.map((p) => [p.lat, p.lon] as [number, number]), [draftPoints]);

  return (
    <div className="bg-slate-100" style={pageStyle}>
      <div style={shellStyle}>
        <div className="border-slate-200 bg-white" style={sideStyle}>
          <div className="p-4 space-y-4">
            <div className="space-y-2">
              <h2 className="text-lg font-semibold">标注工作台</h2>
              <p className="text-xs text-slate-600">
                {t("nav.workspace")}独立标注页：在 Web 里直接画/擦 caution，多边形保存后自动生成训练标签。
              </p>
            </div>

            <div className="space-y-2">
              <Label>布局</Label>
              <div className="flex gap-2">
                <Button size="sm" variant={layoutMode === "desktop" ? "default" : "outline"} onClick={() => setLayoutMode("desktop")}>
                  桌面
                </Button>
                <Button size="sm" variant={layoutMode === "mobile" ? "default" : "outline"} onClick={() => setLayoutMode("mobile")}>
                  移动
                </Button>
              </div>
            </div>

            <div className="space-y-2">
              <Label>时间戳 (UTC)</Label>
              <Select value={timestamp} onValueChange={setTimestamp} disabled={!timestampOptions.length}>
                <SelectTrigger>
                  <SelectValue placeholder={timestampOptions.length ? "" : "加载中..."} />
                </SelectTrigger>
                <SelectContent>
                  {timestampOptions.slice(0, 240).map((ts) => (
                    <SelectItem key={ts} value={ts}>
                      {ts}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Card>
              <CardContent className="p-3 space-y-3">
                <div className="text-sm font-medium">图层叠加</div>
                <div className="space-y-2">
                  {[
                    { id: "bathy", key: "bathy", label: "水深掩膜" },
                    { id: "ais_heatmap", key: "ais", label: "AIS 热力" },
                    { id: "unet_pred", key: "unet", label: "U-Net 分区" },
                    { id: "caution_mask", key: "caution", label: "人工 caution mask" },
                  ].map((cfg) => (
                    <div key={cfg.id} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <label className="flex items-center gap-2">
                          <input
                            type="checkbox"
                            checked={layerState[cfg.key as keyof LayerStates].enabled}
                            onChange={(e) =>
                              setLayerState((prev) => ({
                                ...prev,
                                [cfg.key]: { ...prev[cfg.key as keyof LayerStates], enabled: e.target.checked },
                              }))
                            }
                          />
                          <span>
                            {cfg.label} {availability[cfg.id] ? "" : "（缺失）"}
                          </span>
                        </label>
                        <span>{layerState[cfg.key as keyof LayerStates].opacity}%</span>
                      </div>
                      <Slider
                        value={[layerState[cfg.key as keyof LayerStates].opacity]}
                        min={0}
                        max={100}
                        step={5}
                        onValueChange={(v) =>
                          setLayerState((prev) => ({
                            ...prev,
                            [cfg.key]: { ...prev[cfg.key as keyof LayerStates], opacity: v[0] ?? 0 },
                          }))
                        }
                      />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-3 space-y-2">
                <div className="text-sm font-medium">标注操作（caution）</div>
                <div className="flex gap-2">
                  <Button size="sm" variant={drawTool === "polygon" ? "default" : "outline"} onClick={() => setDrawTool("polygon")}>
                    多边形
                  </Button>
                  <Button size="sm" variant={drawTool === "brush" ? "default" : "outline"} onClick={() => setDrawTool("brush")}>
                    画笔
                  </Button>
                </div>
                <div className="flex gap-2">
                  <Button size="sm" variant={drawMode === "add" ? "default" : "outline"} onClick={() => setDrawMode("add")}>
                    画 caution
                  </Button>
                  <Button size="sm" variant={drawMode === "erase" ? "default" : "outline"} onClick={() => setDrawMode("erase")}>
                    擦除 caution
                  </Button>
                </div>
                {drawTool === "brush" ? (
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <span>画笔粗细（栅格）</span>
                      <span>{brushRadiusCells[0]}</span>
                    </div>
                    <Slider value={brushRadiusCells} min={1} max={10} step={1} onValueChange={setBrushRadiusCells} />
                  </div>
                ) : null}
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" variant={drawingEnabled ? "default" : "outline"} onClick={() => setDrawingEnabled((v) => !v)}>
                    {drawingEnabled ? "停止点选" : "开始点选"}
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setDraftPoints((prev) => prev.slice(0, -1))} disabled={draftPoints.length === 0}>
                    撤销点
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setDraftPoints([])} disabled={draftPoints.length === 0}>
                    清空草稿
                  </Button>
                  <Button size="sm" onClick={() => finalizeDraft(drawMode)} disabled={drawTool !== "polygon" || draftPoints.length < 3}>
                    完成多边形
                  </Button>
                </div>
                <div className="text-xs text-slate-600">
                  草稿点数：{draftPoints.length}，当前工具：{drawTool === "polygon" ? "多边形" : "画笔"}，当前模式：
                  {drawMode === "add" ? "添加 caution" : "擦除 caution"}。
                </div>
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" variant="outline" onClick={() => setOperations((prev) => prev.slice(0, -1))} disabled={operations.length === 0}>
                    撤销上一操作
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setOperations([])} disabled={operations.length === 0}>
                    清空全部操作
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-3 space-y-2">
                <div className="text-sm font-medium">主动学习建议跳转</div>
                <Select value={reviewRunId} onValueChange={setReviewRunId}>
                  <SelectTrigger>
                    <SelectValue placeholder={reviewRuns.length ? "选择建议批次" : "暂无建议批次"} />
                  </SelectTrigger>
                  <SelectContent>
                    {reviewRuns.map((run) => (
                      <SelectItem key={run.run_id} value={run.run_id}>
                        {run.run_id} ({run.accepted_count}/{run.mapping_count})
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {reviewLoading ? <div className="text-xs text-slate-500">加载建议中...</div> : null}
                <div className="max-h-40 overflow-y-auto space-y-1">
                  {reviewItems.slice(0, 10).map((it) => (
                    <button
                      key={`${it.timestamp}-${it.rank}`}
                      className="w-full rounded border border-slate-200 bg-slate-50 px-2 py-1 text-left text-xs hover:bg-slate-100"
                      onClick={() => setTimestamp(it.timestamp.replace("_", "-") + ":00")}
                    >
                      #{it.rank} {it.timestamp} | score {it.score.toFixed(3)}
                    </button>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-3 space-y-2">
                <div className="text-sm font-medium">保存与统计</div>
                <textarea
                  className="w-full min-h-16 rounded border border-slate-300 p-2 text-xs"
                  placeholder="备注（可选）"
                  value={saveNote}
                  onChange={(e) => setSaveNote(e.target.value)}
                />
                <Button className="w-full" onClick={handleSave} disabled={saving || !timestamp}>
                  {saving ? "保存中..." : "保存 patch 并更新训练标签"}
                </Button>
                {loadingPatch ? <div className="text-xs text-slate-500">正在读取当前 patch...</div> : null}
                {patchStats ? (
                  <div className="rounded border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700">
                    <div>caution 像素：{patchStats.caution_pixels}</div>
                    <div>caution 占比：{(patchStats.caution_ratio * 100).toFixed(2)}%</div>
                    <div>操作条数：{patchStats.operations_count}</div>
                    <div>当前会话操作：{operations.length}</div>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          </div>
        </div>

        <div style={mapStyle}>
          <MapContainer key={`annotation-map-${timestamp}`} bounds={INITIAL_BOUNDS} className="h-full w-full" zoomSnap={0.25} minZoom={1} maxZoom={8} noWrap preferCanvas>
            <MapResizeGuard resizeKey={`${timestamp}-${mapResizeKeyRef.current}`} />
            <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" attribution="&copy; OpenStreetMap contributors" noWrap />
            <Pane name="aoi-frame" style={{ zIndex: 390 }}>
              <Rectangle
                bounds={AOI_BOUNDS}
                pathOptions={{
                  color: "#0ea5e9",
                  weight: 2,
                  opacity: 0.95,
                  fill: false,
                  dashArray: "8 6",
                }}
              />
            </Pane>
            <DrawEvents
              enabled={drawingEnabled}
              tool={drawTool}
              onPolygonClick={appendDraftPoint}
              onStrokePreview={setDraftPoints}
              onStrokeEnd={handleStrokeEnd}
            />
            <RasterLayer layerId="bathy" enabled={layerState.bathy.enabled} opacity={layerState.bathy.opacity} timestamp={timestamp} tileRevision={tileRevision} zIndex={300} />
            <RasterLayer layerId="ais_heatmap" enabled={layerState.ais.enabled} opacity={layerState.ais.opacity} timestamp={timestamp} tileRevision={tileRevision} zIndex={340} />
            <RasterLayer layerId="unet_pred" enabled={layerState.unet.enabled} opacity={layerState.unet.opacity} timestamp={timestamp} tileRevision={tileRevision} zIndex={360} />
            <RasterLayer layerId="caution_mask" enabled={layerState.caution.enabled} opacity={layerState.caution.opacity} timestamp={timestamp} tileRevision={tileRevision} zIndex={380} />

            {operations.map((op, idx) =>
              (op.shape ?? "polygon") === "stroke" ? (
                <Polyline
                  key={op.id || `${op.mode}-${idx}`}
                  positions={op.points.map((p) => [p.lat, p.lon] as [number, number])}
                  pathOptions={{
                    color: op.mode === "add" ? "#f59e0b" : "#1d4ed8",
                    weight: Math.max(2, ((op.radius_cells ?? 2) * 2)),
                    opacity: 0.5,
                    dashArray: op.mode === "erase" ? "6 4" : undefined,
                  }}
                />
              ) : (
                <Polygon
                  key={op.id || `${op.mode}-${idx}`}
                  positions={op.points.map((p) => [p.lat, p.lon] as [number, number])}
                  pathOptions={{
                    color: op.mode === "add" ? "#f59e0b" : "#1d4ed8",
                    fillColor: op.mode === "add" ? "#f59e0b" : "#1d4ed8",
                    fillOpacity: 0.2,
                    weight: 2,
                    opacity: 0.9,
                    dashArray: op.mode === "erase" ? "5 5" : undefined,
                  }}
                />
              )
            )}

            {draftPolyline.length >= 2 ? (
              <Polyline positions={draftPolyline} pathOptions={{ color: drawMode === "add" ? "#f59e0b" : "#1d4ed8", weight: 3, dashArray: "6 4" }} />
            ) : null}
            {draftPoints.map((p, idx) => (
              <CircleMarker
                key={`${p.lat}-${p.lon}-${idx}`}
                center={[p.lat, p.lon]}
                radius={4}
                pathOptions={{ color: "#ffffff", weight: 1, fillColor: drawMode === "add" ? "#f59e0b" : "#1d4ed8", fillOpacity: 1 }}
              />
            ))}
          </MapContainer>

          <div className="pointer-events-none absolute right-3 top-3 z-50 rounded bg-white/95 px-3 py-2 text-xs shadow">
            {drawingEnabled
              ? drawTool === "polygon"
                ? "点地图添加多边形顶点"
                : "按住鼠标拖动画笔"
              : "可浏览图层，开启“开始点选”后绘制"}
          </div>
        </div>
      </div>
    </div>
  );
}
