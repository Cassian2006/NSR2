import { Eye, EyeOff } from "lucide-react";
import { Switch } from "./ui/switch";
import { Slider } from "./ui/slider";
import { Label } from "./ui/label";

interface LayerToggleProps {
  name: string;
  enabled: boolean;
  opacity: number;
  onToggle: (enabled: boolean) => void;
  onOpacityChange: (opacity: number) => void;
}

export default function LayerToggle({
  name,
  enabled,
  opacity,
  onToggle,
  onOpacityChange,
}: LayerToggleProps) {
  return (
    <div className="border-b border-border py-3 last:border-0">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex flex-1 items-center gap-2">
          <button
            onClick={() => onToggle(!enabled)}
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            {enabled ? <Eye className="size-4" /> : <EyeOff className="size-4" />}
          </button>
          <Label className="flex-1 cursor-pointer" onClick={() => onToggle(!enabled)}>
            {name}
          </Label>
        </div>
        <Switch checked={enabled} onCheckedChange={onToggle} />
      </div>
      {enabled && (
        <div className="ml-6 flex items-center gap-3">
          <span className="w-12 text-xs text-muted-foreground">透明度</span>
          <Slider
            value={[opacity]}
            onValueChange={(value) => onOpacityChange(value[0])}
            min={0}
            max={100}
            step={5}
            className="flex-1"
          />
          <span className="w-8 text-right text-xs text-muted-foreground">{opacity}%</span>
        </div>
      )}
    </div>
  );
}
