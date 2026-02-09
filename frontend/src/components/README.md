# NSR Route Planning UI Components

## Design System

### Color Palette

**Safety Zones:**
- SAFE: `#10b981` (muted green)
- CAUTION: `#f59e0b` (amber/yellow)
- BLOCKED: `#ef4444` (red)

**AIS Heatmap Gradient:**
- Low traffic: `#3b82f6` (blue)
- Medium traffic: `#06b6d4` (cyan)
- High traffic: `#fbbf24` (yellow)
- Peak traffic: `#ef4444` (red)

### Reusable Components

#### LayerToggle
Layer control with visibility toggle and opacity slider.
```tsx
<LayerToggle
  name="Bathymetry Mask"
  enabled={true}
  opacity={80}
  onToggle={(enabled) => {...}}
  onOpacityChange={(opacity) => {...}}
/>
```

#### CoordinateInput
Lat/Lon input with "Pick on map" and copy functionality.
```tsx
<CoordinateInput
  label="Start Point"
  lat="78.2467"
  lon="15.4650"
  onLatChange={(value) => {...}}
  onLonChange={(value) => {...}}
  onPickFromMap={() => {...}}
/>
```

#### LegendCard
Map legend with color swatches and descriptions.
```tsx
<LegendCard
  title="Map Legend"
  items={[
    { color: "#10b981", label: "SAFE", description: "Navigable" },
    { color: "#f59e0b", label: "CAUTION" },
  ]}
/>
```

#### StatCard
Small metric card with optional variant styling.
```tsx
<StatCard
  label="Distance"
  value="847"
  unit="km"
  variant="success" // default | success | warning | danger
/>
```

## Layout Structure

### 3-Column Layout (Map Workspace)
- **Left Panel** (320px): Scenario & layer controls
- **Center**: Map canvas (flex, largest area)
- **Right Panel** (360px): Metrics & explanation

### Typography
Uses Inter font family with clear hierarchy:
- H1: Large titles
- H2: Section headers
- H3: Subsection headers
- H4: Component titles
- Body: Regular text
- Mono: Coordinates and technical values

### Spacing
Generous whitespace for academic aesthetic:
- Section spacing: 24px (`space-y-6`)
- Component spacing: 12px (`space-y-3`)
- Card padding: 16px (`p-4`)
