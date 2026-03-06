import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { Download, FileJson, Image, RefreshCw, RotateCcw, Trash2 } from "lucide-react";
import { toast } from "sonner";

import {
  type AisBacktestMetrics,
  type ComplianceNoticesPayload,
  deleteGalleryItem,
  getComplianceNotices,
  getDeletedGalleryList,
  getErrorMessage,
  getGalleryImageUrl,
  getGalleryItem,
  getGalleryList,
  getGalleryReportTemplate,
  getGalleryRiskReport,
  restoreGalleryItem,
  runAisBacktest,
  type GalleryItem,
  type GalleryRiskReport,
} from "../api/client";
import StatCard from "../components/StatCard";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "../components/ui/hover-card";
import { Separator } from "../components/ui/separator";
import { useLanguage } from "../contexts/LanguageContext";

function downloadJsonFile(filename: string, payload: unknown) {
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

async function downloadImageFile(filename: string, imageUrl: string) {
  const res = await fetch(imageUrl);
  if (!res.ok) {
    throw new Error(`Image download failed (HTTP ${res.status})`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadTextFile(filename: string, content: string, mimeType: string) {
  const blob = new Blob([content], { type: `${mimeType};charset=utf-8` });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ExportReport() {
  const { t, language } = useLanguage();
  const [searchParams] = useSearchParams();
  const queryGalleryId = searchParams.get("gallery");

  const [items, setItems] = useState<GalleryItem[]>([]);
  const [deletedItems, setDeletedItems] = useState<GalleryItem[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [selectedItem, setSelectedItem] = useState<GalleryItem | null>(null);
  const [riskReport, setRiskReport] = useState<GalleryRiskReport | null>(null);
  const [complianceNotices, setComplianceNotices] = useState<ComplianceNoticesPayload | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [loadingRiskReport, setLoadingRiskReport] = useState(false);
  const [backtest, setBacktest] = useState<AisBacktestMetrics | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  const loadList = async () => {
    setLoadingList(true);
    try {
      const [activeRes, deletedRes] = await Promise.all([getGalleryList(), getDeletedGalleryList()]);
      const activeItems = activeRes.items ?? [];
      setItems(activeItems);
      setDeletedItems(deletedRes.items ?? []);
      if (!activeItems.length) {
        setSelectedId("");
      } else {
        setSelectedId((prev) => {
          if (queryGalleryId && activeItems.some((it) => it.id === queryGalleryId)) return queryGalleryId;
          if (prev && activeItems.some((it) => it.id === prev)) return prev;
          return activeItems[0].id;
        });
      }
    } catch (error) {
      toast.error(`${t("toast.loadGalleryListFailed")}: ${getErrorMessage(error)}`);
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    void loadList();
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setSelectedItem(null);
      setRiskReport(null);
      setBacktest(null);
      return;
    }
    let active = true;
    async function loadDetail() {
      setLoadingDetail(true);
      try {
        const detail = await getGalleryItem(selectedId);
        if (!active) return;
        setSelectedItem(detail);
        setBacktest(null);
        setRiskReport(null);
      } catch (error) {
        if (!active) return;
        toast.error(`${t("toast.loadGalleryItemFailed")}: ${getErrorMessage(error)}`);
      } finally {
        if (active) setLoadingDetail(false);
      }
    }
    void loadDetail();
    return () => {
      active = false;
    };
  }, [selectedId, t]);

  useEffect(() => {
    if (!selectedId) return;
    let active = true;
    async function loadRiskReport() {
      setLoadingRiskReport(true);
      try {
        const report = await getGalleryRiskReport(selectedId);
        if (!active) return;
        setRiskReport(report);
      } catch {
        if (!active) return;
        setRiskReport(null);
      } finally {
        if (active) setLoadingRiskReport(false);
      }
    }
    void loadRiskReport();
    return () => {
      active = false;
    };
  }, [selectedId]);

  useEffect(() => {
    if (!selectedItem?.timestamp) {
      setComplianceNotices(null);
      return;
    }
    let active = true;
    async function loadCompliance() {
      try {
        const payload = await getComplianceNotices({
          context: "export",
          timestamp: selectedItem.timestamp,
        });
        if (!active) return;
        setComplianceNotices(payload);
      } catch {
        if (!active) return;
        setComplianceNotices(null);
      }
    }
    void loadCompliance();
    return () => {
      active = false;
    };
  }, [selectedItem?.timestamp]);

  const cautionPct = useMemo(() => {
    if (!selectedItem || !selectedItem.distance_km || selectedItem.distance_km <= 0) return 0;
    return (selectedItem.caution_len_km / selectedItem.distance_km) * 100;
  }, [selectedItem]);

  const freshnessStatusKey =
    complianceNotices?.data_freshness?.status && ["fresh", "stale", "outdated", "unknown"].includes(String(complianceNotices.data_freshness.status))
      ? String(complianceNotices.data_freshness.status)
      : "unknown";
  const freshnessHint = complianceNotices?.data_freshness?.hint?.[language] ?? complianceNotices?.data_freshness?.hint?.en ?? "";
  const sourceHealthHint = complianceNotices?.source_credibility?.hint?.[language] ?? complianceNotices?.source_credibility?.hint?.en ?? "";

  const imageUrl = selectedItem ? getGalleryImageUrl(selectedItem.id) : "";

  const getActionSummary = (item: GalleryItem) => {
    const st = item.action?.start_input ?? item.start;
    const gl = item.action?.goal_input ?? item.goal;
    return `${Number(st?.lat ?? 0).toFixed(2)}, ${Number(st?.lon ?? 0).toFixed(2)} -> ${Number(gl?.lat ?? 0).toFixed(2)}, ${Number(gl?.lon ?? 0).toFixed(2)}`;
  };

  const getResultSummary = (item: GalleryItem) => {
    const r = item.result;
    const distance = Number(r?.distance_km ?? item.distance_km ?? 0).toFixed(1);
    const caution = Number(r?.caution_len_km ?? item.caution_len_km ?? 0).toFixed(1);
    const status = String(r?.status ?? "success");
    return `${status} | ${distance} km | ${t("export.resultCaution")} ${caution} km`;
  };

  const handleDelete = async () => {
    if (!selectedItem) return;
    const ok = window.confirm(`${t("export.deleteConfirm.prefix")} ${selectedItem.id}${t("export.deleteConfirm.suffix")}`);
    if (!ok) return;
    try {
      await deleteGalleryItem(selectedItem.id, true);
      toast.success(t("toast.runMovedToRecycle"));
      await loadList();
    } catch (error) {
      toast.error(`${t("toast.deleteFailed")}: ${getErrorMessage(error)}`);
    }
  };

  const handleRestore = async (galleryId: string) => {
    try {
      await restoreGalleryItem(galleryId);
      toast.success(`${t("toast.runRestored")}: ${galleryId}`);
      await loadList();
      setSelectedId(galleryId);
    } catch (error) {
      toast.error(`${t("toast.restoreFailed")}: ${getErrorMessage(error)}`);
    }
  };

  const handleDownloadRoute = () => {
    if (!selectedItem?.route_geojson) {
      toast.error(t("toast.noRouteGeojson"));
      return;
    }
    downloadJsonFile(`route_${selectedItem.id}.geojson`, selectedItem.route_geojson);
    toast.success(t("toast.routeDownloaded"));
  };

  const handleDownloadReport = () => {
    if (!selectedItem) return;
    downloadJsonFile(`report_${selectedItem.id}.json`, {
      ...selectedItem,
      compliance: complianceNotices,
    });
    toast.success(t("toast.reportDownloaded"));
  };

  const handleDownloadRiskReport = () => {
    if (!selectedItem || !riskReport) {
      toast.error(t("export.noRiskReportDownload"));
      return;
    }
    downloadJsonFile(`risk_report_${selectedItem.id}.json`, {
      ...riskReport,
      compliance: riskReport.compliance ?? complianceNotices,
    });
    toast.success(t("toast.riskReportDownloaded"));
  };

  const handleDownloadCandidateComparison = () => {
    if (!selectedItem || !riskReport) {
      toast.error(t("export.noCandidateCompareDownload"));
      return;
    }
    const payload = {
      gallery_id: selectedItem.id,
      timestamp: selectedItem.timestamp,
      candidate_comparison: riskReport.candidate_comparison,
      strategy: riskReport.strategy,
      risk: riskReport.risk,
      compliance: riskReport.compliance ?? complianceNotices,
    };
    downloadJsonFile(`candidate_compare_${selectedItem.id}.json`, payload);
    toast.success(t("toast.candidateCompareDownloaded"));
  };

  const handleDownloadTemplate = async (format: "json" | "csv" | "markdown") => {
    if (!selectedItem) return;
    try {
      const payload = await getGalleryReportTemplate(selectedItem.id, format);
      if (format === "json") {
        downloadJsonFile(`report_template_${selectedItem.id}.json`, payload);
      } else if (format === "csv") {
        downloadTextFile(`report_template_${selectedItem.id}.csv`, String(payload), "text/csv");
      } else {
        downloadTextFile(`report_template_${selectedItem.id}.md`, String(payload), "text/markdown");
      }
      toast.success(t("toast.reportTemplateDownloaded"));
    } catch (error) {
      toast.error(`${t("toast.reportTemplateDownloadFailed")}: ${getErrorMessage(error)}`);
    }
  };

  const handleDownloadImage = async () => {
    if (!selectedItem) return;
    try {
      await downloadImageFile(`gallery_${selectedItem.id}.png`, imageUrl);
      toast.success(t("toast.imageDownloaded"));
    } catch (error) {
      toast.error(getErrorMessage(error));
    }
  };

  const handleRunBacktest = async () => {
    if (!selectedItem) return;
    setEvaluating(true);
    try {
      const res = await runAisBacktest({ gallery_id: selectedItem.id });
      setBacktest(res.metrics);
      toast.success(t("toast.backtestDone"));
    } catch (error) {
      toast.error(`${t("toast.backtestFailed")}: ${getErrorMessage(error)}`);
    } finally {
      setEvaluating(false);
    }
  };

  return (
    <div className="min-h-full bg-gray-50">
      <div className="mx-auto max-w-7xl p-4 sm:p-6 lg:p-8">
        <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="mb-1">{t("export.title")}</h1>
            <p className="text-muted-foreground">{t("export.subtitle")}</p>
          </div>
          <Button variant="outline" className="w-full gap-2 sm:w-auto" onClick={() => void loadList()} disabled={loadingList}>
            <RefreshCw className={`size-4 ${loadingList ? "animate-spin" : ""}`} />
            {t("export.refresh")}
          </Button>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[360px_1fr]">
          <Card>
            <CardHeader>
              <CardTitle>{t("export.savedRuns")}</CardTitle>
              <CardDescription>
                {items.length} {t("export.active")} / {deletedItems.length} {t("export.deleted")}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {items.length === 0 ? (
                <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground">{t("export.noItems")}</div>
              ) : (
                items.map((item) => {
                  const active = item.id === selectedId;
                  return (
                    <HoverCard key={item.id} openDelay={180} closeDelay={80}>
                      <HoverCardTrigger asChild>
                        <button
                          onClick={() => setSelectedId(item.id)}
                          className={`w-full rounded-lg border p-3 text-left transition-colors ${
                            active ? "border-blue-500 bg-blue-50" : "border-border bg-white hover:bg-muted/40"
                          }`}
                        >
                          <div className="text-xs text-muted-foreground">{new Date(item.created_at).toLocaleString()}</div>
                          <div className="font-mono text-sm">{item.id}</div>
                          <div className="text-sm">{item.timestamp}</div>
                          <div className="text-xs text-muted-foreground">{t("export.input")}: {getActionSummary(item)}</div>
                          <div className="text-xs text-muted-foreground">{t("export.result")}: {getResultSummary(item)}</div>
                        </button>
                      </HoverCardTrigger>
                      <HoverCardContent className="w-[min(24rem,80vw)] space-y-2">
                        <div className="text-sm font-semibold">{t("export.runDetails")}</div>
                        <div className="text-xs text-muted-foreground">{t("export.input")}</div>
                        <div className="text-xs font-mono">{getActionSummary(item)}</div>
                        <div className="text-xs text-muted-foreground">{t("export.result")}: {getResultSummary(item)}</div>
                      </HoverCardContent>
                    </HoverCard>
                  );
                })
              )}

              {deletedItems.length > 0 ? (
                <div className="mt-4 rounded-lg border border-amber-200 bg-amber-50 p-3">
                  <div className="mb-2 text-xs font-medium text-amber-900">{t("export.recycleList")}</div>
                  <div className="space-y-2">
                    {deletedItems.slice(0, 5).map((item) => (
                      <div key={`deleted-${item.id}`} className="flex items-center justify-between gap-2 rounded border border-amber-200 bg-white px-2 py-1">
                        <span className="truncate text-xs font-mono">{item.id}</span>
                        <Button size="sm" variant="outline" className="h-7 gap-1" onClick={() => void handleRestore(item.id)}>
                          <RotateCcw className="size-3" />
                          {t("export.restore")}
                        </Button>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>{t("export.runDetail")}</CardTitle>
              <CardDescription>{selectedItem ? selectedItem.id : t("export.noSelection")}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {!selectedItem || loadingDetail ? (
                <div className="rounded-lg border border-dashed p-8 text-center text-sm text-muted-foreground">
                  {loadingDetail ? t("export.loadingDetail") : t("export.noSelection")}
                </div>
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                    <StatCard label={t("summary.distance")} value={Number(selectedItem.distance_km ?? 0).toFixed(1)} unit="km" />
                    <StatCard label={t("summary.caution")} value={cautionPct.toFixed(1)} unit="%" variant="warning" />
                    <StatCard label={t("workspace.corridorBias")} value={Number(selectedItem.corridor_bias ?? 0).toFixed(2)} />
                    <StatCard label={t("export.model")} value={String(selectedItem.model_version ?? "unet_v1")} />
                  </div>

                  <div className="rounded-lg border bg-white p-3 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="text-sm text-muted-foreground">{t("export.riskSummary")}</div>
                      <div className="text-xs text-muted-foreground">
                        {loadingRiskReport ? t("export.status.loading") : riskReport ? t("export.status.ready") : t("export.status.unavailable")}
                      </div>
                    </div>
                    {riskReport ? (
                      <>
                        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                          <StatCard label={t("export.riskExposure")} value={Number((riskReport.risk?.risk_exposure as number) ?? 0).toFixed(3)} />
                          <StatCard
                            label={t("export.highRiskCrossing")}
                            value={(Number((riskReport.risk?.high_risk_crossing_ratio as number) ?? 0) * 100).toFixed(1)}
                            unit="%"
                            variant="warning"
                          />
                          <StatCard label={t("export.avoidanceGain")} value={Number(((riskReport.risk?.avoidance_gain as any)?.risk_reduction ?? 0)).toFixed(3)} variant="success" />
                          <StatCard label={t("export.distanceTradeoff")} value={Number(((riskReport.risk?.avoidance_gain as any)?.distance_tradeoff_km ?? 0)).toFixed(2)} unit="km" />
                        </div>
                        <div className="rounded-md border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700">
                          planner={String((riskReport.strategy?.planner as string) ?? "-")} | risk_mode={String((riskReport.risk?.risk_mode as string) ?? "-")} | caution_mode=
                          {String((riskReport.strategy?.caution_mode as string) ?? "-")} | corridor_bias=
                          {Number((riskReport.strategy?.corridor_bias as number) ?? 0).toFixed(2)}
                        </div>
                      </>
                    ) : (
                      <div className="text-xs text-muted-foreground">{t("export.noRiskReport")}</div>
                    )}
                  </div>

                  {complianceNotices ? (
                    <div className="rounded-lg border border-amber-200 bg-amber-50 p-3 text-xs text-amber-900 space-y-2">
                      <div className="font-medium">{t("compliance.title")}</div>
                      <div>{t("compliance.researchOnly")}</div>
                      <div className="rounded border border-amber-200 bg-white p-2 text-slate-700 space-y-1">
                        <div>
                          {t("compliance.freshness")}: {t(`compliance.status.${freshnessStatusKey}`)}
                          {typeof complianceNotices.data_freshness?.age_hours === "number" ? ` (${complianceNotices.data_freshness.age_hours.toFixed(1)}h)` : ""}
                        </div>
                        {freshnessHint ? <div>{freshnessHint}</div> : null}
                        <div>
                          {t("compliance.sourceHealth")}: {sourceHealthHint || "-"}
                        </div>
                        {complianceNotices.source_credibility?.updated_at ? (
                          <div className="text-[11px] text-slate-500">
                            {t("compliance.updatedAt")}: {String(complianceNotices.source_credibility.updated_at)}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  ) : null}

                  <div className="flex flex-col gap-3 rounded-lg border bg-white p-3 sm:flex-row sm:items-center sm:justify-between">
                    <div className="text-sm text-muted-foreground">{t("export.backtest")}</div>
                    <Button onClick={handleRunBacktest} variant="outline" disabled={evaluating} className="w-full gap-2 sm:w-auto">
                      {evaluating ? t("export.backtest.loading") : t("export.backtest.run")}
                    </Button>
                  </div>

                  {backtest ? (
                    <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
                      <StatCard label="Top10 hit rate" value={(backtest.top10pct_hit_rate * 100).toFixed(1)} unit="%" />
                      <StatCard label="Top25 hit rate" value={(backtest.top25pct_hit_rate * 100).toFixed(1)} unit="%" />
                      <StatCard label="Alignment (0-1)" value={backtest.alignment_norm_0_1.toFixed(3)} />
                      <StatCard label="Z-score" value={backtest.alignment_zscore.toFixed(2)} />
                    </div>
                  ) : null}

                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div className="rounded-lg border bg-white p-3 text-sm">
                      <div className="mb-2 text-muted-foreground">{t("workspace.startPoint")}</div>
                      <div className="font-mono">
                        {Number(selectedItem.start?.lat ?? 0).toFixed(4)}, {Number(selectedItem.start?.lon ?? 0).toFixed(4)}
                      </div>
                    </div>
                    <div className="rounded-lg border bg-white p-3 text-sm">
                      <div className="mb-2 text-muted-foreground">{t("workspace.goalPoint")}</div>
                      <div className="font-mono">
                        {Number(selectedItem.goal?.lat ?? 0).toFixed(4)}, {Number(selectedItem.goal?.lon ?? 0).toFixed(4)}
                      </div>
                    </div>
                  </div>

                  <div className="rounded-lg border bg-white p-3 text-sm space-y-2">
                    <div className="text-muted-foreground">{t("export.inputAndResult")}</div>
                    <div className="font-mono text-xs">{t("export.input")}: {getActionSummary(selectedItem)}</div>
                    <div className="text-xs text-muted-foreground">
                      {t("export.result")}: {getResultSummary(selectedItem)} | points {Number(selectedItem.result?.raw_points ?? selectedItem.explain?.raw_points ?? 0)} {"->"}{" "}
                      {Number(selectedItem.result?.smoothed_points ?? selectedItem.explain?.smoothed_points ?? 0)}
                    </div>
                  </div>

                  <div className="space-y-3">
                    <div className="text-sm text-muted-foreground">{t("export.preview")}</div>
                    <div className="overflow-hidden rounded-lg border bg-white">
                      <img src={imageUrl} alt={`gallery-${selectedItem.id}`} className="h-auto w-full object-contain" />
                    </div>
                  </div>

                  <Separator />

                  <div className="flex flex-wrap gap-2">
                    <Button onClick={handleDownloadRoute} className="w-full gap-2 sm:w-auto">
                      <FileJson className="size-4" />
                      {t("export.downloadRoute")}
                    </Button>
                    <Button onClick={handleDownloadReport} variant="outline" className="w-full gap-2 sm:w-auto">
                      <Download className="size-4" />
                      {t("export.downloadReport")}
                    </Button>
                    <Button onClick={() => void handleDownloadTemplate("json")} variant="outline" className="w-full gap-2 sm:w-auto">
                      <Download className="size-4" />
                      {t("export.downloadTemplate.json")}
                    </Button>
                    <Button onClick={() => void handleDownloadTemplate("csv")} variant="outline" className="w-full gap-2 sm:w-auto">
                      <Download className="size-4" />
                      {t("export.downloadTemplate.csv")}
                    </Button>
                    <Button onClick={() => void handleDownloadTemplate("markdown")} variant="outline" className="w-full gap-2 sm:w-auto">
                      <Download className="size-4" />
                      {t("export.downloadTemplate.md")}
                    </Button>
                    <Button onClick={handleDownloadRiskReport} variant="outline" className="w-full gap-2 sm:w-auto" disabled={!riskReport}>
                      <Download className="size-4" />
                      {t("export.downloadRiskReport")}
                    </Button>
                    <Button
                      onClick={handleDownloadCandidateComparison}
                      variant="outline"
                      className="w-full gap-2 sm:w-auto"
                      disabled={!riskReport || !(riskReport.candidate_comparison?.count > 0)}
                    >
                      <Download className="size-4" />
                      {t("export.downloadCandidateCompare")}
                    </Button>
                    <Button onClick={handleDownloadImage} variant="outline" className="w-full gap-2 sm:w-auto">
                      <Image className="size-4" />
                      {t("export.downloadImage")}
                    </Button>
                    <Button onClick={handleDelete} variant="destructive" className="w-full gap-2 sm:w-auto">
                      <Trash2 className="size-4" />
                      {t("export.delete")}
                    </Button>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
