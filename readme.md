# NSR 避险航线规划系统（当前实现说明）

最后更新：2026-02-11  
技术栈：React + Vite + FastAPI + Python

---

## 1. 项目目标
在给定时间片下，基于环境网格（冰/浪/风）、水深禁行、AIS 走廊热力图与 U-Net 分区结果，完成：

- 多图层地图展示与叠加
- 起终点交互式选取与路线规划
- 路线结果可解释（距离/风险/走廊贴合）
- 规划结果落盘到 Gallery（含截图）
- AIS 回测指标输出

---

## 2. 当前功能完成度

### 后端接口（已可用）
- `GET /healthz`
- `GET /v1/datasets`
- `GET /v1/datasets/quality`
- `GET /v1/timestamps`
- `GET /v1/layers`
- `GET /v1/overlay/{layer}.png`
- `GET /v1/tiles/{layer}/{z}/{x}/{y}.png`
- `POST /v1/infer`
- `POST /v1/route/plan`
- `POST /v1/route/plan/dynamic`
- `POST /v1/latest/plan`
- `GET /v1/latest/progress`
- `GET /v1/latest/runtime`
- `GET /v1/latest/sources/health`
- `GET/POST /v1/latest/copernicus/config`
- `GET /v1/latest/status`
- `GET /v1/gallery/list`
- `GET /v1/gallery/{id}`
- `GET /v1/gallery/{id}/image.png`
- `POST /v1/gallery/{id}/image`
- `DELETE /v1/gallery/{id}`
- `POST /v1/eval/ais/backtest`

### 前端页面（已可用）
- `ScenarioSelector`：场景与时间片选择
- `MapWorkspace`：地图工作台、图层控制、规划、推理、latest 进度条
- `ExportReport`：Gallery 浏览、下载、删除、AIS 回测

### 规划算法（已集成）
- A*（基线）
- D* Lite（静态 + 动态增量重规划）
- Any-Angle / Theta*
- Hybrid A*

---

## 3. 本轮关键更新（已落地）

### 地图与图层稳定性
- 修复地图刷新后可能空白的问题：桌面布局高度改为 `100dvh`，地图容器增加最小高度保护。
- 修复图层“看似缺失”问题：新增瓦片版本号 `tileRevision`，避免浏览器缓存旧透明瓦片。
- `MapContainer` 在布局/时间片切换时安全重建，避免 Leaflet 尺寸不同步。

### 部署数据策略
- Docker 默认切为全量数据优先：
  - `NSR_DATA_ROOT=/app/data`
  - `NSR_OUTPUTS_ROOT=/app/outputs`
  - `NSR_ALLOW_DEMO_FALLBACK=0`
- 不再默认静默回退 demo 数据，避免线上误用压缩样本。

### 数据质量报告规则
- 质量规则已调优，避免把“按需推理”场景误判为失败：
  - `auxiliary_layers_coverage`：AIS 覆盖高时，`pred/uncertainty` 低覆盖记为 `WARN` 而非 `FAIL`
  - `numeric_quality_sampled`：增加海域/陆地 NaN 统计，使用分级阈值

---

## 4. 数据质量现状（当前数据集）

- 总体状态：`WARN`（原先 `FAIL` 已修正为更真实评估）
- 时间片数量：`493`
- 时间范围：`2024-07-01_00` -> `2025-02-02_12`
- 当前告警项（2）：
  - `auxiliary_layers_coverage`
  - `numeric_quality_sampled`

说明：这两个告警不阻塞系统运行，属于“生产可运行 + 需要持续优化”状态。

---

## 5. 目录结构（核心）

```text
NSR2/
  backend/
    app/
      api/
      core/
      eval/
      model/
      planning/
      main.py
    tests/
    requirements.txt
  frontend/
    src/
      pages/
      components/
      api/
  data/
  outputs/
  Dockerfile
  DEPLOY_RENDER.md
  readme.md
  agent.md
```

---

## 6. 本地运行

### 后端
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### 前端
```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1 --port 5173
```

默认前端 API 建议：
- `frontend/.env.local`：
  - `VITE_API_BASE_URL=http://127.0.0.1:8000/v1`

---

## 7. 群晖 NAS（DSM）部署建议

推荐目录：

- `/volume1/NSRplanner/app`（仓库代码，含 Dockerfile）
- `/volume1/NSRplanner/data`（真实数据）
- `/volume1/NSRplanner/outputs`（运行输出）

`docker-compose.yml` 建议：

```yaml
services:
  nsr2:
    build:
      context: ./app
      dockerfile: Dockerfile
    container_name: nsr2
    restart: unless-stopped
    ports:
      - "8000:8000"
    environment:
      PORT: "8000"
      NSR_DATA_ROOT: "/app/data"
      NSR_OUTPUTS_ROOT: "/app/outputs"
      NSR_ALLOW_DEMO_FALLBACK: "0"
      NSR_DISABLE_TORCH: "1"
    volumes:
      - /volume1/NSRplanner/data:/app/data:ro
      - /volume1/NSRplanner/outputs:/app/outputs
```

启动后先检查：

- `http://NAS_IP:8000/healthz`
- `http://NAS_IP:8000/v1/datasets`（`sample_count` 必须 > 0）

---

## 8. 提交与质量门禁（执行规则）

- 每次提交 GitHub 前必须执行审查与测试：
  - `python -m pytest -q`（backend）
  - 前端有改动时执行 `npm run build`（frontend）
- 所有关键操作写入 `agent.md`（含时间戳）
- 阶段功能完成后及时提交并推送

---

## 9. 下一步建议

- 补全 `unet_pred/unet_uncertainty` 的历史缓存覆盖率（降低告警）
- 继续优化 latest 拉取链路与错误恢复（Copernicus 不稳定场景）
- 针对 NAS 部署补一份运行巡检脚本（启动后自动检查图层可用性与 sample_count）
