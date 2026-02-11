import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { Calendar, MapPin } from "lucide-react";

import { getDatasets, getDatasetsQuality, getTimestamps } from "../api/client";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Label } from "../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { useLanguage } from "../contexts/LanguageContext";

type Snapshot = {
  id: string;
  label: string;
  period: string;
};

const RECOMMENDED_SNAPSHOTS: Snapshot[] = [
  { id: "week-28", label: "第28周", period: "2024-07-08 至 2024-07-14" },
  { id: "week-32", label: "第32周", period: "2024-08-05 至 2024-08-11" },
  { id: "week-36", label: "第36周", period: "2024-09-02 至 2024-09-08" },
  { id: "week-40", label: "第40周", period: "2024-09-30 至 2024-10-06" },
  { id: "week-42", label: "第42周", period: "2024-10-14 至 2024-10-20" },
  { id: "week-44", label: "第44周", period: "2024-10-28 至 2024-11-03" },
];

export default function ScenarioSelector() {
  const navigate = useNavigate();
  const { t } = useLanguage();
  const [months, setMonths] = useState<string[]>(["all", "2024-07", "2024-08", "2024-09", "2024-10"]);
  const [selectedMonth, setSelectedMonth] = useState("all");
  const [timestamps, setTimestamps] = useState<string[]>([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState("");
  const [selectedSnapshot, setSelectedSnapshot] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [qualitySummary, setQualitySummary] = useState<{
    status: string;
    timestamp_count: number;
    issues_count?: number;
    first_timestamp?: string;
    last_timestamp?: string;
  } | null>(null);

  useEffect(() => {
    let active = true;
    async function loadDatasetInfo() {
      try {
        const res = await getDatasets();
        const rawMonths = res.dataset.months.length ? res.dataset.months : months.filter((m) => m !== "all");
        const normalizedMonths = Array.from(new Set(rawMonths));
        const nextMonths = ["all", ...normalizedMonths];
        if (active) {
          setMonths(nextMonths);
          setSelectedMonth((prev) => (nextMonths.includes(prev) ? prev : "all"));
        }
      } catch (error) {
        console.warn("datasets api unavailable, using defaults", error);
      }
    }
    void loadDatasetInfo();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    async function loadQuality() {
      try {
        const report = await getDatasetsQuality(40);
        if (!active) return;
        setQualitySummary(report.summary);
      } catch (error) {
        console.warn("dataset quality api unavailable", error);
      }
    }
    void loadQuality();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    let active = true;
    setLoading(true);
    async function loadTimestamps() {
      try {
        const res = await getTimestamps(selectedMonth === "all" ? undefined : selectedMonth);
        if (!active) return;
        setTimestamps(res.timestamps);
        setSelectedTimestamp((prev) => (res.timestamps.includes(prev) ? prev : res.timestamps[0] ?? ""));
      } catch (error) {
        console.warn("timestamps api unavailable", error);
        if (active) {
          setTimestamps([]);
          setSelectedTimestamp("");
        }
      } finally {
        if (active) setLoading(false);
      }
    }
    void loadTimestamps();
    return () => {
      active = false;
    };
  }, [selectedMonth]);

  const timestampOptions = useMemo(() => timestamps.slice(0, 120), [timestamps]);

  const handleLoadLayers = () => {
    if (!selectedTimestamp) return;
    navigate(`/workspace?timestamp=${encodeURIComponent(selectedTimestamp)}`);
  };

  return (
    <div className="h-full overflow-auto bg-gradient-to-br from-blue-50 via-gray-50 to-indigo-50">
      <div className="mx-auto max-w-6xl p-4 sm:p-6 lg:p-12">
        <div className="mb-8">
          <h1 className="mb-2 text-blue-900">{t("scenario.title")}</h1>
          <p className="text-muted-foreground">{t("scenario.subtitle")}</p>
        </div>

        <div className="grid gap-6">
          <Card className="border-blue-200 shadow-lg">
            <CardHeader className="bg-gradient-to-r from-blue-50 to-transparent">
              <CardTitle className="text-blue-900">{t("scenario.config.title")}</CardTitle>
              <CardDescription>{t("scenario.config.desc")}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 md:gap-6">
                <div className="space-y-2">
                  <Label htmlFor="month">{t("scenario.month")}</Label>
                  <Select value={selectedMonth} onValueChange={setSelectedMonth}>
                    <SelectTrigger id="month">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {months.map((month) => (
                        <SelectItem key={month} value={month}>
                          {month === "all" ? "全部（2024-07 至 2024-10）" : month}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="timestamp">{t("scenario.timestamp")}</Label>
                  <Select value={selectedTimestamp} onValueChange={setSelectedTimestamp} disabled={loading || !timestampOptions.length}>
                    <SelectTrigger id="timestamp">
                      <SelectValue placeholder={loading ? "加载中..." : "暂无可用时间片"} />
                    </SelectTrigger>
                    <SelectContent>
                      {timestampOptions.map((ts) => (
                        <SelectItem key={ts} value={ts}>
                          {ts.replace("T", " ")}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <p className="text-xs text-muted-foreground">
                    共 {timestamps.length} 个时间片
                    {timestamps.length > timestampOptions.length ? `，下拉框显示前 ${timestampOptions.length} 个` : ""}
                  </p>
                </div>
              </div>

              <div className="pt-4">
                <Button onClick={handleLoadLayers} size="lg" className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 gap-2" disabled={!selectedTimestamp}>
                  <Calendar className="size-4" />
                  {t("scenario.loadLayers")}
                </Button>
              </div>

              {qualitySummary ? (
                <div
                  className={`rounded-md border px-3 py-2 text-xs ${
                    qualitySummary.status === "pass"
                      ? "border-emerald-300 bg-emerald-50 text-emerald-800"
                      : qualitySummary.status === "warn"
                        ? "border-amber-300 bg-amber-50 text-amber-800"
                        : "border-rose-300 bg-rose-50 text-rose-800"
                  }`}
                >
                  <div className="font-medium">数据质量状态：{qualitySummary.status.toUpperCase()}</div>
                  <div>时间片数量：{qualitySummary.timestamp_count}</div>
                  <div>问题项：{qualitySummary.issues_count ?? 0}</div>
                  {qualitySummary.first_timestamp && qualitySummary.last_timestamp ? (
                    <div>
                      范围：{qualitySummary.first_timestamp} -&gt; {qualitySummary.last_timestamp}
                    </div>
                  ) : null}
                </div>
              ) : null}
            </CardContent>
          </Card>

          <Card className="border-purple-200 shadow-lg">
            <CardHeader className="bg-gradient-to-r from-purple-50 to-transparent">
              <CardTitle className="text-purple-900">{t("scenario.recommended.title")}</CardTitle>
              <CardDescription>{t("scenario.recommended.desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
                {RECOMMENDED_SNAPSHOTS.map((snapshot) => (
                  <button
                    key={snapshot.id}
                    onClick={() => setSelectedSnapshot(snapshot.id)}
                    className={`rounded-lg border-2 p-4 text-left transition-all hover:border-purple-400 hover:shadow-md ${
                      selectedSnapshot === snapshot.id ? "border-purple-600 bg-purple-50 shadow-md" : "border-border bg-white"
                    }`}
                  >
                    <div className="flex min-w-0 items-start gap-3">
                      <div className="mt-0.5">
                        <MapPin className={`size-4 ${selectedSnapshot === snapshot.id ? "text-purple-600" : "text-muted-foreground"}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className={`mb-1 ${selectedSnapshot === snapshot.id ? "text-purple-900" : ""}`}>{snapshot.label}</h4>
                        <p className="font-mono text-xs text-muted-foreground">{snapshot.period}</p>
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
