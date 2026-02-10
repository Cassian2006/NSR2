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
import appLogo from "../assets/app-logo.png";

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
            <div className="size-8 rounded-lg border border-blue-200 bg-white p-0.5 shadow-sm">
              <img src={appLogo} alt="NSR logo" className="h-full w-full rounded-md object-cover" />
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
