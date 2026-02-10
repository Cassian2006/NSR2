import { Copy, MapPin } from "lucide-react";
import { Input } from "./ui/input";
import { Label } from "./ui/label";
import { Button } from "./ui/button";
import { toast } from "sonner";
import { useLanguage } from "../contexts/LanguageContext";

interface CoordinateInputProps {
  label: string;
  lat: string;
  lon: string;
  onLatChange: (value: string) => void;
  onLonChange: (value: string) => void;
  onPickFromMap: () => void;
}

export default function CoordinateInput({
  label,
  lat,
  lon,
  onLatChange,
  onLonChange,
  onPickFromMap,
}: CoordinateInputProps) {
  const { t } = useLanguage();
  const handleCopy = () => {
    navigator.clipboard.writeText(`${lat}, ${lon}`);
    toast.success(t("toast.coordsCopied"));
  };

  return (
    <div className="space-y-2">
      <Label>{label}</Label>
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Input
            type="text"
            placeholder="Latitude"
            value={lat}
            onChange={(e) => onLatChange(e.target.value)}
            className="font-mono text-sm"
          />
        </div>
        <div className="space-y-1">
          <Input
            type="text"
            placeholder="Longitude"
            value={lon}
            onChange={(e) => onLonChange(e.target.value)}
            className="font-mono text-sm"
          />
        </div>
      </div>
      <div className="flex gap-2">
        <Button
          size="sm"
          variant="outline"
          className="flex-1 gap-2"
          onClick={onPickFromMap}
        >
          <MapPin className="size-3" />
          {t("workspace.pickOnMap")}
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={handleCopy}
        >
          <Copy className="size-3" />
        </Button>
      </div>
    </div>
  );
}
