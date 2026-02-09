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
    <div className="py-3 border-b border-border last:border-0">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 flex-1">
          <button
            onClick={() => onToggle(!enabled)}
            className="text-muted-foreground hover:text-foreground transition-colors"
          >
            {enabled ? <Eye className="size-4" /> : <EyeOff className="size-4" />}
          </button>
          <Label className="cursor-pointer flex-1" onClick={() => onToggle(!enabled)}>
            {name}
          </Label>
        </div>
        <Switch
          checked={enabled}
          onCheckedChange={onToggle}
        />
      </div>
      {enabled && (
        <div className="ml-6 flex items-center gap-3">
          <span className="text-xs text-muted-foreground w-12">Opacity</span>
          <Slider
            value={[opacity]}
            onValueChange={(value) => onOpacityChange(value[0])}
            min={0}
            max={100}
            step={5}
            className="flex-1"
          />
          <span className="text-xs text-muted-foreground w-8 text-right">
            {opacity}%
          </span>
        </div>
      )}
    </div>
  );
}
