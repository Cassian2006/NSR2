import { useState } from "react";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Download, FileJson, Image, FileText, CheckCircle2 } from "lucide-react";
import { toast } from "sonner";
import StatCard from "../components/StatCard";
import { Separator } from "../components/ui/separator";

export default function ExportReport() {
  const [downloading, setDownloading] = useState<string | null>(null);

  const handleDownload = (type: string, filename: string) => {
    setDownloading(type);
    toast.loading(`Preparing ${filename}...`, { duration: 1000 });
    
    setTimeout(() => {
      setDownloading(null);
      toast.success(`Downloaded ${filename}`);
    }, 1000);
  };

  return (
    <div className="h-full overflow-auto bg-gray-50">
      <div className="max-w-6xl mx-auto p-12">
        <div className="mb-8">
          <h1 className="mb-2">Export & Report</h1>
          <p className="text-muted-foreground">
            Download route data, visualizations, and comprehensive analysis reports.
          </p>
        </div>

        <div className="grid gap-6">
          {/* Download Options */}
          <Card>
            <CardHeader>
              <CardTitle>Export Options</CardTitle>
              <CardDescription>
                Download route and analysis data in various formats
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid md:grid-cols-3 gap-4">
                <button
                  onClick={() => handleDownload("geojson", "route.geojson")}
                  disabled={downloading === "geojson"}
                  className="p-6 rounded-lg border-2 border-border bg-white hover:border-primary/50 hover:bg-primary/5 transition-all text-left disabled:opacity-50"
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-blue-100 rounded-lg">
                      <FileJson className="size-5 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="mb-1">Route GeoJSON</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Waypoints and geometry
                      </p>
                      <div className="text-xs text-muted-foreground font-mono">
                        route.geojson • ~25 KB
                      </div>
                    </div>
                  </div>
                  <Button 
                    size="sm" 
                    className="w-full mt-4 gap-2"
                    disabled={downloading === "geojson"}
                  >
                    <Download className="size-3" />
                    Download GeoJSON
                  </Button>
                </button>

                <button
                  onClick={() => handleDownload("png", "map_screenshot.png")}
                  disabled={downloading === "png"}
                  className="p-6 rounded-lg border-2 border-border bg-white hover:border-primary/50 hover:bg-primary/5 transition-all text-left disabled:opacity-50"
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-purple-100 rounded-lg">
                      <Image className="size-5 text-purple-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="mb-1">Map Screenshot</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        High-res PNG with legend
                      </p>
                      <div className="text-xs text-muted-foreground font-mono">
                        map_screenshot.png • 1920×1080
                      </div>
                    </div>
                  </div>
                  <Button 
                    size="sm" 
                    className="w-full mt-4 gap-2"
                    disabled={downloading === "png"}
                  >
                    <Download className="size-3" />
                    Download PNG
                  </Button>
                </button>

                <button
                  onClick={() => handleDownload("json", "analysis_report.json")}
                  disabled={downloading === "json"}
                  className="p-6 rounded-lg border-2 border-border bg-white hover:border-primary/50 hover:bg-primary/5 transition-all text-left disabled:opacity-50"
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <FileText className="size-5 text-green-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="mb-1">Analysis Report</h4>
                      <p className="text-sm text-muted-foreground mb-3">
                        Metadata and metrics
                      </p>
                      <div className="text-xs text-muted-foreground font-mono">
                        analysis_report.json • ~8 KB
                      </div>
                    </div>
                  </div>
                  <Button 
                    size="sm" 
                    className="w-full mt-4 gap-2"
                    disabled={downloading === "json"}
                  >
                    <Download className="size-3" />
                    Download JSON
                  </Button>
                </button>
              </div>

              <Separator className="my-6" />

              <div className="flex items-center justify-between">
                <div>
                  <h4 className="mb-1">Download All</h4>
                  <p className="text-sm text-muted-foreground">
                    Export all files as a ZIP archive
                  </p>
                </div>
                <Button
                  onClick={() => handleDownload("zip", "nsr_route_export.zip")}
                  disabled={downloading === "zip"}
                  className="gap-2"
                >
                  <Download className="size-4" />
                  Download All (ZIP)
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Run Summary */}
          <Card>
            <CardHeader>
              <CardTitle>Run Summary</CardTitle>
              <CardDescription>
                Configuration and parameters used for this route calculation
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {/* Settings */}
                <div>
                  <h4 className="mb-3">Settings</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Dataset</div>
                      <div className="font-medium">July–October 2024</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Timestamp</div>
                      <div className="font-medium font-mono">2024-08-15 12:00 UTC</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Safety Policy</div>
                      <div className="font-medium">Blocked = Bathy + U-Net Blocked</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Caution Handling</div>
                      <div className="font-medium">Tie-breaker (Default)</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">AIS Corridor Bias</div>
                      <div className="font-medium">0.2</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg">
                      <div className="text-sm text-muted-foreground mb-1">Model Version</div>
                      <div className="font-medium">U-Net v2.1.3</div>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Route Points */}
                <div>
                  <h4 className="mb-3">Route Endpoints</h4>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
                      <div className="text-sm text-green-700 mb-1">Start Point</div>
                      <div className="font-medium font-mono">78.2467°N, 15.4650°E</div>
                      <div className="text-xs text-muted-foreground mt-1">Longyearbyen, Svalbard</div>
                    </div>
                    <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
                      <div className="text-sm text-red-700 mb-1">Goal Point</div>
                      <div className="font-medium font-mono">81.5074°N, 58.3811°E</div>
                      <div className="text-xs text-muted-foreground mt-1">Franz Josef Land</div>
                    </div>
                  </div>
                </div>

                <Separator />

                {/* Metrics */}
                <div>
                  <h4 className="mb-3">Route Metrics</h4>
                  <div className="grid md:grid-cols-4 gap-3">
                    <StatCard label="Total Distance" value="847" unit="km" />
                    <StatCard label="Safe Zones" value="78.3" unit="%" variant="success" />
                    <StatCard label="Caution Zones" value="21.7" unit="%" variant="warning" />
                    <StatCard label="Corridor Align" value="0.74" variant="success" />
                  </div>
                </div>

                <Separator />

                {/* Validation */}
                <div>
                  <h4 className="mb-3">Validation Metrics</h4>
                  <div className="grid md:grid-cols-3 gap-3">
                    <StatCard label="DTW Distance" value="12.4" unit="km" />
                    <StatCard label="Hausdorff Dist" value="8.7" unit="km" />
                    <StatCard label="AIS Overlap" value="74.2" unit="%" variant="success" />
                  </div>
                </div>

                <Separator />

                {/* Status */}
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                  <div className="flex items-start gap-3">
                    <CheckCircle2 className="size-5 text-green-600 mt-0.5" />
                    <div>
                      <h4 className="text-green-900 mb-1">Route Calculation Successful</h4>
                      <p className="text-sm text-green-800">
                        No safety violations detected. Route successfully avoids all BLOCKED zones 
                        while minimizing distance under the specified constraints.
                      </p>
                      <div className="mt-3 flex gap-4 text-sm text-green-700">
                        <div>
                          <span className="text-muted-foreground">Computed:</span> 2024-02-09 14:32:18 UTC
                        </div>
                        <div>
                          <span className="text-muted-foreground">Duration:</span> 1.47s
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Citation & References */}
          <Card>
            <CardHeader>
              <CardTitle>Citation & References</CardTitle>
              <CardDescription>
                Academic attribution for this route planning system
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="p-4 bg-gray-50 rounded-lg font-mono text-sm">
                <div className="mb-2 text-muted-foreground">BibTeX</div>
                <code className="text-xs leading-relaxed">
                  @article&#123;nsr_planning_2024,<br />
                  &nbsp;&nbsp;title=&#123;Arctic Sea Route Planning with U-Net Safety Predictions&#125;,<br />
                  &nbsp;&nbsp;author=&#123;...&#125;,<br />
                  &nbsp;&nbsp;journal=&#123;...&#125;,<br />
                  &nbsp;&nbsp;year=&#123;2024&#125;<br />
                  &#125;
                </code>
              </div>

              <div>
                <h4 className="mb-2">Data Sources</h4>
                <ul className="text-sm space-y-1 text-muted-foreground">
                  <li>• AIS vessel tracking data (Norwegian Coastal Administration)</li>
                  <li>• Bathymetry: GEBCO 2023 Grid</li>
                  <li>• Ice concentration: EUMETSAT OSI SAF</li>
                  <li>• Wave/Wind: ECMWF ERA5 Reanalysis</li>
                </ul>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
