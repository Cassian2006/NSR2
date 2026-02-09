NSR 避险航线规划系统 工程手则（Vue + Vite + FastAPI + Python）
0. 项目目标与边界
目标

在给定时间片（2024-07～2024-10）下，基于：

环境网格（Copernicus 冰/风/浪等）

水深网格（陆地/浅水禁行）

AIS 走廊热力图（软偏好）
-（可选）AIS 轨迹（仅回测验证）

实现：

图层浏览（多图层叠加、透明度、取值查看、对比）

交互式路线规划：点选起终点 → 规划 → 地图展示路线0

Gallery：每次规划自动截图（路线+图层+图例）并保存展示，可下载/删除

最佳航线定义（强约束+弱偏好）

硬约束：禁止进入 BLOCKED（水深禁行 + U-Net 预测 BLOCKED）

主目标：在可行域内 距离最短

弱偏好：在距离近似时，轻微偏向 AIS 走廊热力图高的区域（tie-breaker），避免“怪路/无人走廊”

备注：CAUTION 不作为硬禁行，第一版仅作为轻微惩罚或预算（避免大绕路）。

1. 技术栈

前端：Vue 3 + Vite + TypeScript（建议）

地图：MapLibre GL JS（推荐）或 Leaflet

UI 设计：Figma（学术风简洁 dashboard）

后端：FastAPI + Uvicorn

核心：Python（数据对齐/推理/规划/导出）

模型：U-Net（语义分割：SAFE/CAUTION/BLOCKED）

2. 数据约定（强制）
2.1 坐标/栅格对齐

所有网格（env/bathy/ais_heatmap）必须在同一投影/同一分辨率/同一 H×W。

前端与 API 使用 WGS84 经纬度（EPSG:4326）传参；

服务端负责经纬度 → 网格 index（row, col）映射（使用你的对齐网格的仿射变换/经纬度数组）。

2.2 文件命名与层定义

建议统一为：

env/{timestamp}.npy：float32，shape (C,H,W)

bathy/{timestamp}.npy 或 bathy/static.npy：float32，shape (H,W)（水深）

ais_heatmap/{window}/{timestamp}.npy：float32，shape (H,W)（已对齐）

unet_pred/{model_version}/{timestamp}.npy：uint8 或 float16，shape (H,W) 或 (3,H,W)

2.3 类别编码（统一）

0 = SAFE

1 = CAUTION

2 = BLOCKED

水深禁行也最终要输出为 BLOCKED mask（bool 或 0/1）。

3. 目录结构（推荐）
repo/
  backend/
    app/
      main.py
      api/
        routes_layers.py
        routes_infer.py
        routes_plan.py
        routes_gallery.py
      core/
        config.py
        io.py
        geo.py
        tiling.py
        render.py
        gallery.py
      model/
        unet.py
        infer.py
      planning/
        grid_graph.py
        router.py
        smoothing.py
        cost.py
      eval/
        compare_ais.py
    requirements.txt
  frontend/
    src/
      pages/
        Viewer.vue
        Gallery.vue
      components/
        LayerPanel.vue
        InspectorPanel.vue
        RoutePanel.vue
        MapCanvas.vue
        LegendCard.vue
      api/
        client.ts
      store/
        layers.ts
        route.ts
      utils/
        geo.ts
    vite.config.ts
  data/   (不建议直接进git，可放示例/小样)
  outputs/
    gallery/
      runs/
      thumbs/
  README.md

4. 后端 FastAPI 设计（必做接口）
4.1 基础：数据与时间片

GET /v1/datasets

GET /v1/timestamps?month=2024-07

GET /v1/layers?timestamp=...

返回可用 layer 列表、范围、单位、渲染建议

4.2 图层渲染（给前端叠加）

两种方式二选一（推荐 Tiles）：

A. Tiles（推荐）

GET /v1/tiles/{layer}/{z}/{x}/{y}.png?timestamp=...

B. Overlay（简单）

GET /v1/overlay/{layer}.png?timestamp=...&bbox=...&size=...

4.3 U-Net 推理

POST /v1/infer

{ "timestamp": "...", "model_version": "unet_v1" }


返回：

{ "pred_layer": "unet_pred/unet_v1", "stats": {...} }


推理结果应缓存落盘：outputs/pred/{model_version}/{timestamp}.npy

4.4 路线规划（交互核心）

POST /v1/route/plan

{
  "timestamp": "...",
  "start": {"lat": 72.1, "lon": 60.2},
  "goal": {"lat": 70.5, "lon": 150.7},
  "policy": {
    "objective": "shortest_distance_under_safety",
    "blocked_sources": ["bathy", "unet_blocked"],
    "caution_mode": "tie_breaker",
    "corridor_bias": 0.2,
    "smoothing": true
  }
}


返回：

route_geojson（LineString + metrics）

explain（距离、穿越 caution 长度、走廊贴合度等）

gallery_id（本次自动截图保存后的记录 id）

4.5 Gallery（保存规划截图与元信息）

GET /v1/gallery/list

GET /v1/gallery/{id}（元数据）

GET /v1/gallery/{id}/image.png

DELETE /v1/gallery/{id}

5. 核心规划算法规范（避免绕路的关键）
5.1 可行域

blocked = blocked_bathy OR blocked_unet

blocked_bathy：水深阈值 + 陆地

blocked_unet：U-Net 输出 class==2

5.2 代价定义（第一版不调权重）

主代价：几何距离（网格步长）

弱偏好（tie-breaker）：

进入 CAUTION：增加一个封顶的小惩罚（防止大绕路）

偏好走廊：AIS heatmap 高则给小奖励（或减少代价）

关键：惩罚/奖励必须远小于“绕一大圈的距离差”，否则会再次绕路。

5.3 平滑

A* 输出路径做后处理：

去锯齿（line-of-sight 简化）

曲率约束（可选）

输出：

raw_path + smoothed_path

6. U-Net 推理规范
6.1 输入标准化（必须固定尺度）

每个通道使用全季节统计的 mean/std 或固定物理范围 clip

禁止对每张图单独 min-max 拉伸（会学到伪影）

6.2 输出

推荐输出 prob (3,H,W)，前端显示 argmax

同时可以保存 blocked_prob 作为不确定性可视化

7. 前端 Vue 规范
7.1 页面

Viewer：地图 + 图层控制 + 取值检查 + 起终点选择 + 规划按钮

Gallery：规划截图墙（缩略图+元信息），支持打开/下载/删除

7.2 地图交互

点选两次：设 start/goal

Hover：显示 lat/lon + 当前激活层数值

图层叠加：

bathy mask

ais heatmap

unet pred

route polyline（最上层）

7.3 与后端对接

所有请求走 src/api/client.ts

规划后：

渲染 route

弹 toast

自动跳转/刷新 Gallery（或右上角出现“Saved to Gallery”）

8. 自动截图与 Gallery 保存（工程要点）

截图应包含：路线、当前激活图层、图例、时间戳、起终点

推荐实现方式：

前端截图（canvas/map 导出）→ 上传到后端存储

或后端渲染静态图（更一致，但开发稍重）

Gallery 元数据建议存：timestamp、layers、start/goal、distance、caution_len、corridor_bias、model_version

9. 日志、错误处理与性能

后端：

每个 timestamp 的推理与规划耗时记录

缓存：预测结果与常用 overlay/png

前端：

请求中状态（loading）与失败提示

大数组传输避免直接 JSON；图层用 PNG tile/overlay。

10. 测试清单（最低要求）

网格对齐一致性：env/bathy/ais_heatmap H×W 完全一致

经纬度→网格索引正确性：随机点验证

禁行正确性：路线不应进入 blocked

规划稳定性：同起终点重复运行结果一致（或可解释）

Gallery：保存/打开/下载/删除全流程

11. 里程碑（推荐）

M1：Layer Viewer（tiles/overlay + inspector + export）

M2：U-Net infer 接入（unet_pred 显示）

M3：Route planning（start/goal + route + explain）

M4：Gallery（自动截图 + 管理）

M5：AIS 回测评估（可选）