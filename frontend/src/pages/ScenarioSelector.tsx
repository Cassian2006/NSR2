import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router";
import { Calendar, MapPin } from "lucide-react";

import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Label } from "../components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "../components/ui/select";
import { getDatasets, getTimestamps } from "../api/client";
import { useLanguage } from "../contexts/LanguageContext";

type Snapshot = {
  id: string;
  label: string;
  period: string;
};

const RECOMMENDED_SNAPSHOTS: Snapshot[] = [
  { id: "week-28", label: "Week 28", period: "2024-07-08 to 2024-07-14" },
  { id: "week-32", label: "Week 32", period: "2024-08-05 to 2024-08-11" },
  { id: "week-36", label: "Week 36", period: "2024-09-02 to 2024-09-08" },
  { id: "week-40", label: "Week 40", period: "2024-09-30 to 2024-10-06" },
  { id: "week-42", label: "Week 42", period: "2024-10-14 to 2024-10-20" },
  { id: "week-44", label: "Week 44", period: "2024-10-28 to 2024-11-03" },
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
    loadDatasetInfo();
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
    loadTimestamps();
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
                          {month === "all" ? "All (2024-07 to 2024-10)" : month}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="timestamp">{t("scenario.timestamp")}</Label>
                  <Select value={selectedTimestamp} onValueChange={setSelectedTimestamp} disabled={loading || !timestampOptions.length}>
                    <SelectTrigger id="timestamp">
                      <SelectValue placeholder={loading ? "Loading..." : "No timestamp"} />
                    </SelectTrigger>
                    <SelectContent>
                      {timestampOptions.map((ts) => (
                        <SelectItem key={ts} value={ts}>
                          {ts.replace("T", " ")}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="pt-4">
                <Button onClick={handleLoadLayers} size="lg" className="w-full sm:w-auto bg-blue-600 hover:bg-blue-700 gap-2" disabled={!selectedTimestamp}>
                  <Calendar className="size-4" />
                  {t("scenario.loadLayers")}
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="border-purple-200 shadow-lg">
            <CardHeader className="bg-gradient-to-r from-purple-50 to-transparent">
              <CardTitle className="text-purple-900">{t("scenario.recommended.title")}</CardTitle>
              <CardDescription>{t("scenario.recommended.desc")}</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {RECOMMENDED_SNAPSHOTS.map((snapshot) => (
                  <button
                    key={snapshot.id}
                    onClick={() => setSelectedSnapshot(snapshot.id)}
                    className={`p-4 rounded-lg border-2 transition-all text-left hover:border-purple-400 hover:shadow-md ${
                      selectedSnapshot === snapshot.id ? "border-purple-600 bg-purple-50 shadow-md" : "border-border bg-white"
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <div className="mt-0.5">
                        <MapPin className={`size-4 ${selectedSnapshot === snapshot.id ? "text-purple-600" : "text-muted-foreground"}`} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className={`mb-1 ${selectedSnapshot === snapshot.id ? "text-purple-900" : ""}`}>{snapshot.label}</h4>
                        <p className="text-xs text-muted-foreground font-mono">{snapshot.period}</p>
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
