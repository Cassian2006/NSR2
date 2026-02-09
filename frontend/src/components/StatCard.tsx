interface StatCardProps {
  label: string;
  value: string | number;
  unit?: string;
  variant?: "default" | "success" | "warning" | "danger";
}

export default function StatCard({ label, value, unit, variant = "default" }: StatCardProps) {
  const variantStyles = {
    default: "bg-gray-50 border-gray-200",
    success: "bg-green-50 border-green-200",
    warning: "bg-amber-50 border-amber-200",
    danger: "bg-red-50 border-red-200",
  };

  const valueStyles = {
    default: "text-foreground",
    success: "text-green-700",
    warning: "text-amber-700",
    danger: "text-red-700",
  };

  return (
    <div className={`p-3 rounded-lg border ${variantStyles[variant]}`}>
      <div className="text-xs text-muted-foreground mb-1">{label}</div>
      <div className={`text-xl font-medium ${valueStyles[variant]}`}>
        {value}
        {unit && <span className="text-sm ml-1">{unit}</span>}
      </div>
    </div>
  );
}
