# NSR 避险航线规划系统（当前实现说明）

最后更新：2026-02-10  
技术栈（当前代码）：React + Vite + FastAPI + Python

---

## 0. 目标与边界

在给定时间片（2024-07 ~ 2024-10）下，基于以下数据进行北极航线规划：

- 环境网格（冰/风/浪）
- 水深与陆地禁行信息（blocked mask）
- AIS 走廊热力图（软偏好）
- U-Net 分割结果（SAFE/CAUTION/BLOCKED）

系统目标：

- 图层浏览（多图层、透明度、取值与可视化）
- 起终点交互选取并规划路线
- 规划结果自动落到 Gallery
- Gallery 支持查看、下载、删除
- 路线可做 AIS 回测评估（第一版）

---

## 1. 当前实现状态

### 已完成

- 后端核心接口：`/v1/datasets`、`/v1/timestamps`、`/v1/layers`
- 图层渲染：`/v1/overlay/{layer}.png`、`/v1/tiles/{layer}/{z}/{x}/{y}.png`
- U-Net 推理接口：`/v1/infer`（支持结果缓存）
- 路线规划接口：`/v1/route/plan`（A* 网格规划）
- Gallery 接口：列表、详情、图片、删除、前端截图上传
- AIS 回测接口：`/v1/eval/ais/backtest`
- 前端页面：场景选择、地图工作台、Gallery/Export

### 进行中/待优化

- 经纬度到网格的映射当前仍是规则网格映射（通过配置边界），尚未接入真实仿射变换参数文件
- 规划平滑策略为简化版，后续可升级 line-of-sight + 曲率约束
- 前端打包体积偏大（Vite 警告 > 500KB），后续可拆包优化

---

## 2. 技术栈（与代码一致）

- 前端：React 18 + Vite + TypeScript + Leaflet
- 后端：FastAPI + Uvicorn
- 算法与模型：Python + NumPy + PyTorch（TinyUNet 推理）
- 数据存储：`npy` + `json`

---

## 3. 数据约定

### 3.1 栅格与坐标

- 当前默认网格边界（可配置）：
  - `grid_lat_min=60.0`
  - `grid_lat_max=86.0`
  - `grid_lon_min=-180.0`
  - `grid_lon_max=180.0`
- 所有核心网格默认按同一 `H x W` 对齐处理

### 3.2 关键文件（annotation pack）

每个时间片目录（如 `data/processed/annotation_pack/2024-07-01_00/`）包含：

- `x_stack.npy`：多通道输入，形状 `(C,H,W)`
- `blocked_mask.npy`：禁行掩膜，形状 `(H,W)`，`0/1`
- `y_class.npy` 或相关标签文件（按训练流程使用）
- `meta.json`：通道名、shape、规则说明

### 3.3 类别编码

- `0 = SAFE`
- `1 = CAUTION`
- `2 = BLOCKED`

---

## 4. 目录结构（当前）

```text
repo/
  backend/
    app/
      api/
        routes_layers.py
        routes_infer.py
        routes_plan.py
        routes_gallery.py
        routes_eval.py
      core/
        config.py
        dataset.py
        gallery.py
        render.py
        schemas.py
      eval/
        compare_ais.py
      model/
        infer.py
        tiny_unet.py
      planning/
        router.py
      main.py
    tests/
  frontend/
    src/
      pages/
        ScenarioSelector.tsx
        MapWorkspace.tsx
        ExportReport.tsx
      components/
        MapCanvas.tsx
      api/
        client.ts
  data/
  outputs/
    pred/
    gallery/
  readme.md
  agent.md
```

---

## 5. 后端 API

### 5.1 数据与图层

- `GET /v1/datasets`
- `GET /v1/timestamps?month=2024-07`
- `GET /v1/layers?timestamp=...`
- `GET /v1/overlay/{layer}.png?timestamp=...&bbox=...&size=...`
- `GET /v1/tiles/{layer}/{z}/{x}/{y}.png?timestamp=...`

支持 layer（当前）：

- `bathy`
- `ais_heatmap`
- `unet_pred`
- `ice`
- `wave`
- `wind`

### 5.2 推理与规划

- `POST /v1/infer`
- `POST /v1/route/plan`

规划策略字段：

- `blocked_sources` 支持 `bathy / unet_blocked / unet_caution`
- `caution_mode` 支持 `tie_breaker / budget / minimize / strict`

### 5.3 Gallery

- `GET /v1/gallery/list`
- `GET /v1/gallery/{id}`
- `GET /v1/gallery/{id}/image.png`
- `POST /v1/gallery/{id}/image`（前端截图上传）
- `DELETE /v1/gallery/{id}`

### 5.4 AIS 回测评估

- `POST /v1/eval/ais/backtest`

可传：

- `gallery_id`（推荐）
- 或 `timestamp + route_geojson`

返回指标（第一版）：

- `top10pct_hit_rate`
- `top25pct_hit_rate`
- `median_or_higher_hit_rate`
- `alignment_norm_0_1`
- `alignment_zscore`
- 以及路线与全局热力图统计

---

## 6. 前端功能（当前）

### 6.1 ScenarioSelector

- 选择月份与时间片
- 进入地图工作台

### 6.2 MapWorkspace

- Leaflet 真地图
- 图层开关 + 透明度
- 地图点选起终点
- 运行 U-Net 推理
- 路径规划并展示 explain
- 规划成功后自动截图并上传到 Gallery

### 6.3 ExportReport（Gallery）

- 浏览历史规划记录
- 查看元数据与预览图
- 下载 GeoJSON/JSON/PNG
- 删除记录
- 一键运行 AIS 回测并展示指标

---

## 7. 运行方式

### 7.1 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### 7.2 前端

```bash
cd frontend
npm install
npm run dev
```

可选环境变量：

- `VITE_API_BASE_URL`（默认 `http://127.0.0.1:8000/v1`）
- `NSR_GRID_LAT_MIN / NSR_GRID_LAT_MAX / NSR_GRID_LON_MIN / NSR_GRID_LON_MAX`

---

## 8. 测试与质量

后端测试：

```bash
cd backend
python -m pytest -q
```

前端构建检查：

```bash
cd frontend
npm run build
```

当前状态：后端测试通过，前端可构建。

---

## 9. 开发约定（执行中）

- 每次推送前执行审查与测试（至少 `pytest`，前端改动时执行 `npm run build`）
- 所有关键操作写入 `agent.md`（带时间戳）
- 阶段性完成即提交并推送
- `readme.md` 可持续更新与修正，以反映真实实现状态

---

## 10. 下一步建议

- 引入真实地理仿射参数，替换规则网格映射
- 增强路径平滑与可航行约束细节
- 增加 AIS 回测批量报告与图表导出
- 优化前端打包体积（按路由/页面拆包）
