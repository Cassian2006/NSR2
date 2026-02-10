import { Outlet, Link, useLocation } from "react-router";
import { Button } from "./ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "./ui/select";
import { Download } from "lucide-react";
import { useState } from "react";
import LanguageSwitch from "./LanguageSwitch";
import { useLanguage } from "../contexts/LanguageContext";

export default function Layout() {
  const location = useLocation();
  const [dataset, setDataset] = useState("july-oct-2024");
  const { t } = useLanguage();

  return (
    <div className="h-screen flex flex-col bg-background">
      {/* Top Navigation Bar */}
      <header className="border-b border-border bg-gradient-to-r from-blue-50 via-white to-indigo-50 px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-2">
            <div className="p-1.5 bg-blue-600 rounded-lg">
              <svg className="size-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
              </svg>
            </div>
            <h1 className="text-foreground">{t("app.title")}</h1>
          </Link>
          
          <nav className="flex items-center gap-1">
            <Link to="/">
              <Button
                variant={location.pathname === "/" ? "default" : "ghost"}
                size="sm"
                className={location.pathname === "/" ? "bg-blue-600 hover:bg-blue-700" : ""}
              >
                {t("nav.scenario")}
              </Button>
            </Link>
            <Link to="/workspace">
              <Button
                variant={location.pathname === "/workspace" ? "default" : "ghost"}
                size="sm"
                className={location.pathname === "/workspace" ? "bg-purple-600 hover:bg-purple-700" : ""}
              >
                {t("nav.workspace")}
              </Button>
            </Link>
            <Link to="/export">
              <Button
                variant={location.pathname === "/export" ? "default" : "ghost"}
                size="sm"
                className={location.pathname === "/export" ? "bg-green-600 hover:bg-green-700" : ""}
              >
                {t("nav.export")}
              </Button>
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3">
          <LanguageSwitch />
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">{t("nav.dataset")}</span>
            <Select value={dataset} onValueChange={setDataset}>
              <SelectTrigger className="w-[180px] h-8">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="july-oct-2024">{t("month.july-october")}</SelectItem>
                <SelectItem value="july-sep-2024">{t("month.july-august")}</SelectItem>
                <SelectItem value="aug-oct-2024">{t("month.august-september")}</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Link to="/export">
            <Button size="sm" className="gap-2 bg-green-600 hover:bg-green-700">
              <Download className="size-4" />
              {t("nav.export.btn")}
            </Button>
          </Link>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <Outlet />
      </main>
    </div>
  );
}
