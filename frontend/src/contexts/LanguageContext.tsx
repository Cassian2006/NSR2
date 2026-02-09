import { createContext, useContext, useState, ReactNode } from "react";

type Language = "en" | "zh";

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [language, setLanguage] = useState<Language>("en");

  const t = (key: string): string => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within LanguageProvider");
  }
  return context;
}

const translations: Record<Language, Record<string, string>> = {
  en: {
    // Navigation
    "app.title": "NSR Route Planning System",
    "nav.scenario": "Scenario",
    "nav.workspace": "Map Workspace",
    "nav.export": "Export & Report",
    "nav.dataset": "Dataset:",
    "nav.export.btn": "Export",
    
    // Scenario Selector
    "scenario.title": "Scenario Selection",
    "scenario.subtitle": "Select a time period and snapshot to load environmental layers and begin route planning.",
    "scenario.config.title": "Dataset Configuration",
    "scenario.config.desc": "Choose the time period and specific timestamp for analysis",
    "scenario.month": "Month / Period",
    "scenario.timestamp": "Timestamp (UTC)",
    "scenario.loadLayers": "Load Layers",
    "scenario.recommended.title": "Recommended Snapshots",
    "scenario.recommended.desc": "Pre-selected 7-day heatmap references for common scenarios",
    "scenario.week28": "Week 28 (Jul 8–14)",
    "scenario.week28.desc": "Early season, high ice concentration",
    "scenario.week32": "Week 32 (Aug 5–11)",
    "scenario.week32.desc": "Peak navigation season",
    "scenario.week36": "Week 36 (Sep 2–8)",
    "scenario.week36.desc": "Optimal conditions",
    "scenario.week40": "Week 40 (Sep 30–Oct 6)",
    "scenario.week40.desc": "Late season, increasing ice",
    "scenario.week42": "Week 42 (Oct 14–20)",
    "scenario.week42.desc": "Season closing",
    "scenario.week44": "Week 44 (Oct 28–Nov 3)",
    "scenario.week44.desc": "End of navigation window",
    "scenario.info.title": "About the Dataset",
    "scenario.info.desc": "This system uses multi-source Arctic environmental data including AIS vessel tracks, bathymetry masks, U-Net predicted safety zones, and meteorological conditions. Recommended snapshots represent typical navigation scenarios during the 2024 season.",
    
    // Map Workspace
    "workspace.scenario": "Scenario",
    "workspace.startPoint": "Start Point",
    "workspace.goalPoint": "Goal Point",
    "workspace.pickOnMap": "Pick on map",
    "workspace.layers": "Layers",
    "workspace.layer.bathymetry": "Bathymetry Mask",
    "workspace.layer.ais": "AIS Corridor Heatmap",
    "workspace.layer.unet": "U-Net Zones",
    "workspace.layer.ice": "Ice Concentration",
    "workspace.layer.wave": "Wave Height",
    "workspace.layer.wind": "Wind Speed",
    "workspace.opacity": "Opacity",
    "workspace.legend": "Map Legend",
    "workspace.legend.safe": "SAFE",
    "workspace.legend.safe.desc": "Navigable",
    "workspace.legend.caution": "CAUTION",
    "workspace.legend.caution.desc": "Proceed with care",
    "workspace.legend.blocked": "BLOCKED",
    "workspace.legend.blocked.desc": "Avoid",
    "workspace.legend.ais": "AIS Corridor",
    "workspace.legend.ais.desc": "Historical traffic",
    "workspace.planning": "Planning",
    "workspace.safetyPolicy": "Safety Policy",
    "workspace.safetyPolicy.opt1": "Blocked = Bathy + U-Net Blocked",
    "workspace.safetyPolicy.opt2": "Blocked = Bathy Only",
    "workspace.safetyPolicy.opt3": "Strict (Include Caution)",
    "workspace.cautionHandling": "Caution Handling",
    "workspace.cautionHandling.opt1": "Tie-breaker (Default)",
    "workspace.cautionHandling.opt2": "Budget Constraint",
    "workspace.cautionHandling.opt3": "Minimize Exposure",
    "workspace.corridorBias": "AIS Corridor Bias",
    "workspace.planRoute": "Plan Route",
    "workspace.mousePosition": "Mouse Position",
    "workspace.grid": "Grid",
    "workspace.activeLayers": "Active Layers",
    
    // Route Summary
    "summary.title": "Route Summary",
    "summary.distance": "Distance",
    "summary.safe": "% in SAFE",
    "summary.caution": "% in CAUTION",
    "summary.alignment": "Corridor Alignment",
    "summary.noViolations": "No Safety Violations",
    "summary.noViolations.desc": "Route avoids all BLOCKED zones",
    "summary.planToSee": "Plan a route to see summary metrics",
    
    // Explainability
    "explain.title": "Explainability",
    "explain.why": "Why This Route?",
    "explain.reason1": "Avoided all BLOCKED zones (bathymetry + U-Net predictions)",
    "explain.reason2": "Minimized total distance under safety constraints",
    "explain.reason3": "Applied mild preference (0.2) toward AIS corridor for validation",
    "explain.reason4": "21.7% of route in CAUTION zones (unavoidable given constraints)",
    "explain.segments.title": "High Caution Segments",
    "explain.segments.desc": "Top 5 segments requiring attention",
    "explain.caution.high": "High caution level",
    "explain.caution.medium": "Medium caution level",
    
    // Validation
    "validation.title": "Validation",
    "validation.compare": "Compare with AIS Track",
    "validation.show": "Show",
    "validation.hide": "Hide",
    "validation.dtw": "DTW Distance",
    "validation.hausdorff": "Hausdorff Distance",
    "validation.overlap": "Corridor Overlap",
    
    // Export & Report
    "export.title": "Export & Report",
    "export.subtitle": "Download route data, visualizations, and comprehensive analysis reports.",
    "export.options.title": "Export Options",
    "export.options.desc": "Download route and analysis data in various formats",
    "export.geojson.title": "Route GeoJSON",
    "export.geojson.desc": "Waypoints and geometry",
    "export.png.title": "Map Screenshot",
    "export.png.desc": "High-res PNG with legend",
    "export.json.title": "Analysis Report",
    "export.json.desc": "Metadata and metrics",
    "export.downloadAll": "Download All",
    "export.downloadAll.desc": "Export all files as a ZIP archive",
    "export.summary.title": "Run Summary",
    "export.summary.desc": "Configuration and parameters used for this route calculation",
    "export.settings": "Settings",
    "export.dataset": "Dataset",
    "export.modelVersion": "Model Version",
    "export.endpoints": "Route Endpoints",
    "export.metrics": "Route Metrics",
    "export.totalDistance": "Total Distance",
    "export.safeZones": "Safe Zones",
    "export.cautionZones": "Caution Zones",
    "export.corridorAlign": "Corridor Align",
    "export.validationMetrics": "Validation Metrics",
    "export.hausdorffDist": "Hausdorff Dist",
    "export.aisOverlap": "AIS Overlap",
    "export.success.title": "Route Calculation Successful",
    "export.success.desc": "No safety violations detected. Route successfully avoids all BLOCKED zones while minimizing distance under the specified constraints.",
    "export.computed": "Computed:",
    "export.duration": "Duration:",
    "export.citation.title": "Citation & References",
    "export.citation.desc": "Academic attribution for this route planning system",
    "export.dataSources": "Data Sources",
    
    // Months
    "month.july": "July 2024",
    "month.august": "August 2024",
    "month.september": "September 2024",
    "month.october": "October 2024",
    "month.july-august": "July–August 2024",
    "month.august-september": "August–September 2024",
    "month.september-october": "September–October 2024",
    "month.july-october": "July–Oct 2024",
    
    // Toast messages
    "toast.planning": "Planning optimal route...",
    "toast.success": "Route calculated successfully",
    "toast.pickStart": "Click on the map to set start point",
    "toast.pickGoal": "Click on the map to set goal point",
    "toast.mapClicked": "Map clicked:",
    "toast.coordsCopied": "Coordinates copied to clipboard",
    "toast.preparing": "Preparing",
    "toast.downloaded": "Downloaded",
  },
  zh: {
    // Navigation
    "app.title": "北极航线规划系统",
    "nav.scenario": "场景选择",
    "nav.workspace": "地图工作区",
    "nav.export": "导出报告",
    "nav.dataset": "数据集：",
    "nav.export.btn": "导出",
    
    // Scenario Selector
    "scenario.title": "场景选择",
    "scenario.subtitle": "选择时间段和快照以加载环境图层并开始航线规划。",
    "scenario.config.title": "数据集配置",
    "scenario.config.desc": "选择分析的时间段和具体时间戳",
    "scenario.month": "月份 / 时段",
    "scenario.timestamp": "时间戳 (UTC)",
    "scenario.loadLayers": "加载图层",
    "scenario.recommended.title": "推荐快照",
    "scenario.recommended.desc": "常见场景的预选7天热图参考",
    "scenario.week28": "第28周（7月8-14日）",
    "scenario.week28.desc": "季节早期，高冰浓度",
    "scenario.week32": "第32周（8月5-11日）",
    "scenario.week32.desc": "航行旺季",
    "scenario.week36": "第36周（9月2-8日）",
    "scenario.week36.desc": "最佳条件",
    "scenario.week40": "第40周（9月30日-10月6日）",
    "scenario.week40.desc": "季节后期，冰层增加",
    "scenario.week42": "第42周（10月14-20日）",
    "scenario.week42.desc": "季节收尾",
    "scenario.week44": "第44周（10月28日-11月3日）",
    "scenario.week44.desc": "航行窗口结束",
    "scenario.info.title": "关于数据集",
    "scenario.info.desc": "本系统使用多源北极环境数据，包括AIS船舶轨迹、水深掩模、U-Net预测安全区域和气象条件。推荐快照代表2024年季节的典型航行场景。",
    
    // Map Workspace
    "workspace.scenario": "场景",
    "workspace.startPoint": "起点",
    "workspace.goalPoint": "终点",
    "workspace.pickOnMap": "从地图选择",
    "workspace.layers": "图层",
    "workspace.layer.bathymetry": "水深掩模",
    "workspace.layer.ais": "AIS航道热图",
    "workspace.layer.unet": "U-Net区域",
    "workspace.layer.ice": "冰浓度",
    "workspace.layer.wave": "波高",
    "workspace.layer.wind": "风速",
    "workspace.opacity": "透明度",
    "workspace.legend": "地图图例",
    "workspace.legend.safe": "安全",
    "workspace.legend.safe.desc": "可通航",
    "workspace.legend.caution": "谨慎",
    "workspace.legend.caution.desc": "谨慎通过",
    "workspace.legend.blocked": "禁止",
    "workspace.legend.blocked.desc": "避开",
    "workspace.legend.ais": "AIS航道",
    "workspace.legend.ais.desc": "历史交通",
    "workspace.planning": "规划",
    "workspace.safetyPolicy": "安全策略",
    "workspace.safetyPolicy.opt1": "禁止区 = 水深 + U-Net禁止",
    "workspace.safetyPolicy.opt2": "禁止区 = 仅水深",
    "workspace.safetyPolicy.opt3": "严格模式（包括谨慎区）",
    "workspace.cautionHandling": "谨慎区处理",
    "workspace.cautionHandling.opt1": "平局决胜（默认）",
    "workspace.cautionHandling.opt2": "预算约束",
    "workspace.cautionHandling.opt3": "最小化暴露",
    "workspace.corridorBias": "AIS航道偏好",
    "workspace.planRoute": "规划航线",
    "workspace.mousePosition": "鼠标位置",
    "workspace.grid": "网格",
    "workspace.activeLayers": "活动图层",
    
    // Route Summary
    "summary.title": "航线摘要",
    "summary.distance": "距离",
    "summary.safe": "安全区占比",
    "summary.caution": "谨慎区占比",
    "summary.alignment": "航道对齐度",
    "summary.noViolations": "无安全违规",
    "summary.noViolations.desc": "航线避开所有禁止区域",
    "summary.planToSee": "规划航线以查看摘要指标",
    
    // Explainability
    "explain.title": "可解释性",
    "explain.why": "为什么选择此航线？",
    "explain.reason1": "避开所有禁止区域（水深 + U-Net预测）",
    "explain.reason2": "在安全约束下最小化总距离",
    "explain.reason3": "对AIS航道应用轻度偏好（0.2）以进行验证",
    "explain.reason4": "21.7%的航线在谨慎区（在约束下不可避免）",
    "explain.segments.title": "高谨慎度航段",
    "explain.segments.desc": "需要注意的前5个航段",
    "explain.caution.high": "高谨慎度",
    "explain.caution.medium": "中等谨慎度",
    
    // Validation
    "validation.title": "验证",
    "validation.compare": "与AIS轨迹比较",
    "validation.show": "显示",
    "validation.hide": "隐藏",
    "validation.dtw": "DTW距离",
    "validation.hausdorff": "豪斯多夫距离",
    "validation.overlap": "航道重叠度",
    
    // Export & Report
    "export.title": "导出报告",
    "export.subtitle": "下载航线数据、可视化图表和综合分析报告。",
    "export.options.title": "导出选项",
    "export.options.desc": "以各种格式下载航线和分析数据",
    "export.geojson.title": "航线 GeoJSON",
    "export.geojson.desc": "航点和几何数据",
    "export.png.title": "地图截图",
    "export.png.desc": "带图例的高分辨率PNG",
    "export.json.title": "分析报告",
    "export.json.desc": "元数据和指标",
    "export.downloadAll": "全部下载",
    "export.downloadAll.desc": "导出所有文件为ZIP压缩包",
    "export.summary.title": "运行摘要",
    "export.summary.desc": "用于此航线计算的配置和参数",
    "export.settings": "设置",
    "export.dataset": "数据集",
    "export.modelVersion": "模型版本",
    "export.endpoints": "航线端点",
    "export.metrics": "航线指标",
    "export.totalDistance": "总距离",
    "export.safeZones": "安全区",
    "export.cautionZones": "谨慎区",
    "export.corridorAlign": "航道对齐",
    "export.validationMetrics": "验证指标",
    "export.hausdorffDist": "豪斯多夫距离",
    "export.aisOverlap": "AIS重叠",
    "export.success.title": "航线计算成功",
    "export.success.desc": "未检测到安全违规。航线成功避开所有禁止区域，同时在指定约束下最小化距离。",
    "export.computed": "计算时间：",
    "export.duration": "耗时：",
    "export.citation.title": "引用与参考",
    "export.citation.desc": "本航线规划系统的学术归属",
    "export.dataSources": "数据来源",
    
    // Months
    "month.july": "2024年7月",
    "month.august": "2024年8月",
    "month.september": "2024年9月",
    "month.october": "2024年10月",
    "month.july-august": "2024年7-8月",
    "month.august-september": "2024年8-9月",
    "month.september-october": "2024年9-10月",
    "month.july-october": "2024年7-10月",
    
    // Toast messages
    "toast.planning": "正在规划最优航线...",
    "toast.success": "航线计算成功",
    "toast.pickStart": "点击地图设置起点",
    "toast.pickGoal": "点击地图设置终点",
    "toast.mapClicked": "地图点击：",
    "toast.coordsCopied": "坐标已复制到剪贴板",
    "toast.preparing": "准备中",
    "toast.downloaded": "已下载",
  },
};
