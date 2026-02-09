import { useLanguage } from "../contexts/LanguageContext";
import { Button } from "./ui/button";
import { Languages } from "lucide-react";

export default function LanguageSwitch() {
  const { language, setLanguage } = useLanguage();

  return (
    <div className="flex items-center gap-1 bg-secondary/50 rounded-lg p-1">
      <Button
        variant={language === "en" ? "default" : "ghost"}
        size="sm"
        onClick={() => setLanguage("en")}
        className="h-7 px-3"
      >
        EN
      </Button>
      <Button
        variant={language === "zh" ? "default" : "ghost"}
        size="sm"
        onClick={() => setLanguage("zh")}
        className="h-7 px-3"
      >
        中文
      </Button>
    </div>
  );
}
