import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";

interface LegendItem {
  color: string;
  label: string;
  description?: string;
}

interface LegendCardProps {
  title: string;
  items: LegendItem[];
}

export default function LegendCard({ title, items }: LegendCardProps) {
  return (
    <Card className="bg-white/95 backdrop-blur-sm">
      <CardHeader className="pb-3">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {items.map((item, index) => (
          <div key={index} className="flex items-center gap-2">
            <div
              className="size-4 rounded border border-gray-300 flex-shrink-0"
              style={{ backgroundColor: item.color }}
            />
            <div className="flex-1 min-w-0">
              <div className="text-sm">{item.label}</div>
              {item.description && (
                <div className="text-xs text-muted-foreground">{item.description}</div>
              )}
            </div>
          </div>
        ))}
      </CardContent>
    </Card>
  );
}
