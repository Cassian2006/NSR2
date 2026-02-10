import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router";
import { Download, FileJson, Image, RefreshCw, Trash2 } from "lucide-react";
import { toast } from "sonner";

import {
  deleteGalleryItem,
  getErrorMessage,
  getGalleryImageUrl,
  getGalleryItem,
  getGalleryList,
  runAisBacktest,
  type AisBacktestMetrics,
  type GalleryItem,
} from "../api/client";
import { useLanguage } from "../contexts/LanguageContext";
import StatCard from "../components/StatCard";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Separator } from "../components/ui/separator";

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
    throw new Error(`Failed to download image: HTTP ${res.status}`);
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function ExportReport() {
  const { t } = useLanguage();
  const [searchParams] = useSearchParams();
  const queryGalleryId = searchParams.get("gallery");

  const [items, setItems] = useState<GalleryItem[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [selectedItem, setSelectedItem] = useState<GalleryItem | null>(null);
  const [loadingList, setLoadingList] = useState(false);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [backtest, setBacktest] = useState<AisBacktestMetrics | null>(null);
  const [evaluating, setEvaluating] = useState(false);

  const loadList = async () => {
    setLoadingList(true);
    try {
      const res = await getGalleryList();
      setItems(res.items ?? []);
      if ((res.items ?? []).length === 0) {
        setSelectedId("");
      } else {
        setSelectedId((prev) => {
          if (queryGalleryId && (res.items ?? []).some((it) => it.id === queryGalleryId)) return queryGalleryId;
          if (prev && (res.items ?? []).some((it) => it.id === prev)) return prev;
          return (res.items ?? [])[0].id;
        });
      }
    } catch (error) {
      toast.error(`${t("toast.loadGalleryListFailed")}: ${getErrorMessage(error)}`);
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    loadList();
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setSelectedItem(null);
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
      } catch (error) {
        if (!active) return;
        toast.error(`${t("toast.loadGalleryItemFailed")}: ${getErrorMessage(error)}`);
      } finally {
        if (active) setLoadingDetail(false);
      }
    }
    loadDetail();
    return () => {
      active = false;
    };
  }, [selectedId]);

  const cautionPct = useMemo(() => {
    if (!selectedItem || !selectedItem.distance_km || selectedItem.distance_km <= 0) return 0;
    return (selectedItem.caution_len_km / selectedItem.distance_km) * 100;
  }, [selectedItem]);

  const imageUrl = selectedItem ? getGalleryImageUrl(selectedItem.id) : "";

  const handleDelete = async () => {
    if (!selectedItem) return;
    const ok = window.confirm(`Delete gallery item ${selectedItem.id}?`);
    if (!ok) return;
    try {
      await deleteGalleryItem(selectedItem.id);
      toast.success(t("toast.galleryDeleted"));
      await loadList();
    } catch (error) {
      toast.error(`${t("toast.deleteFailed")}: ${getErrorMessage(error)}`);
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
    downloadJsonFile(`report_${selectedItem.id}.json`, selectedItem);
    toast.success(t("toast.reportDownloaded"));
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
    <div className="h-full overflow-auto bg-gray-50">
      <div className="max-w-7xl mx-auto p-8">
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="mb-1">{t("export.title")}</h1>
            <p className="text-muted-foreground">{t("export.subtitle")}</p>
          </div>
          <Button variant="outline" className="gap-2" onClick={loadList} disabled={loadingList}>
            <RefreshCw className={`size-4 ${loadingList ? "animate-spin" : ""}`} />
            {t("export.refresh")}
          </Button>
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-[360px_1fr]">
          <Card>
            <CardHeader>
              <CardTitle>{t("export.savedRuns")}</CardTitle>
              <CardDescription>{items.length} item(s)</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {items.length === 0 ? (
                <div className="rounded-lg border border-dashed p-6 text-sm text-muted-foreground">{t("export.noItems")}</div>
              ) : (
                items.map((item) => {
                  const active = item.id === selectedId;
                  return (
                    <button
                      key={item.id}
                      onClick={() => setSelectedId(item.id)}
                      className={`w-full rounded-lg border p-3 text-left transition-colors ${
                        active ? "border-blue-500 bg-blue-50" : "border-border bg-white hover:bg-muted/40"
                      }`}
                    >
                      <div className="text-xs text-muted-foreground">{new Date(item.created_at).toLocaleString()}</div>
                      <div className="font-mono text-sm">{item.id}</div>
                      <div className="text-sm">{item.timestamp}</div>
                      <div className="text-xs text-muted-foreground">distance: {Number(item.distance_km ?? 0).toFixed(1)} km</div>
                    </button>
                  );
                })
              )}
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
                  <div className="grid gap-3 md:grid-cols-4">
                    <StatCard label={t("summary.distance")} value={Number(selectedItem.distance_km ?? 0).toFixed(1)} unit="km" />
                    <StatCard label={t("summary.caution")} value={cautionPct.toFixed(1)} unit="%" variant="warning" />
                    <StatCard label={t("workspace.corridorBias")} value={Number(selectedItem.corridor_bias ?? 0).toFixed(2)} />
                    <StatCard label="Model" value={String(selectedItem.model_version ?? "unet_v1")} />
                  </div>

                  <div className="flex items-center justify-between rounded-lg border bg-white p-3">
                    <div className="text-sm text-muted-foreground">{t("export.backtest")}</div>
                    <Button onClick={handleRunBacktest} variant="outline" disabled={evaluating} className="gap-2">
                      {evaluating ? t("export.backtest.loading") : t("export.backtest.run")}
                    </Button>
                  </div>

                  {backtest ? (
                    <div className="grid gap-3 md:grid-cols-4">
                      <StatCard label="Top10 Hit" value={(backtest.top10pct_hit_rate * 100).toFixed(1)} unit="%" />
                      <StatCard label="Top25 Hit" value={(backtest.top25pct_hit_rate * 100).toFixed(1)} unit="%" />
                      <StatCard label="Align (0-1)" value={backtest.alignment_norm_0_1.toFixed(3)} />
                      <StatCard label="Z-Score" value={backtest.alignment_zscore.toFixed(2)} />
                    </div>
                  ) : null}

                  <div className="grid gap-4 md:grid-cols-2">
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

                  <div className="space-y-3">
                    <div className="text-sm text-muted-foreground">{t("export.preview")}</div>
                    <div className="overflow-hidden rounded-lg border bg-white">
                      <img src={imageUrl} alt={`gallery-${selectedItem.id}`} className="h-auto w-full object-contain" />
                    </div>
                  </div>

                  <Separator />

                  <div className="flex flex-wrap gap-2">
                    <Button onClick={handleDownloadRoute} className="gap-2">
                      <FileJson className="size-4" />
                      {t("export.downloadRoute")}
                    </Button>
                    <Button onClick={handleDownloadReport} variant="outline" className="gap-2">
                      <Download className="size-4" />
                      {t("export.downloadReport")}
                    </Button>
                    <Button onClick={handleDownloadImage} variant="outline" className="gap-2">
                      <Image className="size-4" />
                      {t("export.downloadImage")}
                    </Button>
                    <Button onClick={handleDelete} variant="destructive" className="gap-2">
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
