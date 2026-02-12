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

建议再执行一次自动巡检（启动后 20~40 秒）：

```bash
cd backend
python scripts/nas_runtime_healthcheck.py \
  --base-url http://NAS_IP:8000 \
  --min-sample-count 1 \
  --probe-count 12 \
  --out-dir outputs/qa
```

巡检脚本默认会检查：

- `/healthz` 存活
- `/v1/datasets` 的 `sample_count`
- `/v1/timestamps` 是否非空
- `/v1/layers` 的关键图层可用性（`bathy/ais_heatmap/unet_pred/unet_uncertainty`）
- 关键图层 overlay 渲染可访问（返回 `image/png` 且字节数达标）

退出码约定：

- `0`：PASS
- `1`：WARN（仅告警）
- `2`：FAIL（关键项失败）

如需把 WARN 视为通过（例如 Docker/DSM 健康检查），增加：

- `--warn-exit-zero`

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
- 将 NAS 巡检脚本接入 DSM 计划任务（开机后延迟执行并保留最近 7 天报告）

---

## 10. 三个深入升级方向（跨学科 + 创新）

### 方向 A：不确定性驱动的鲁棒航线规划

目标：从“单一最短路”升级为“风险分布最优路径”。

- 创新点：
  - 将 `U-Net uncertainty` 与 `latest` 时序环境预测耦合成概率风险场。
  - 在规划层引入 `CVaR / chance-constrained` 思路，输出保守-平衡-激进三种策略。
- 工程落地：
  - API 增加 `risk_mode`、`risk_budget`、`confidence_level`。
  - 前端展示“风险走廊带 + 多候选路径 + 风险区间条”。
  - 评估新增：`risk_violation_rate`、`expected_delay`、`route_robustness`。

### 方向 B：可解释主动学习的人机协同闭环

目标：降低标注成本，同时提升标签一致性与模型泛化。

- 创新点：
  - 主动学习样本按“高不确定 + 高路线影响”联合排序，而不是只看不确定性。
  - 每个建议区域提供解释信号（冰/风/浪贡献、AIS 偏离度、历史误判类型）。
- 工程落地：
  - 标注端增加“建议原因卡片 + 一键接受/局部修订”。
  - 训练端引入 `focal + dice`、困难样本重采样、类别权重自适应。
  - 质量闭环指标：每轮新增标注对应的 `val_iou`、`route_safety` 增益。

### 方向 C：时序数字孪生与增量重规划基线

目标：发挥 `D* Lite` 优势，从静态规划升级到动态连续决策。

- 创新点：
  - 基于时间步环境更新触发局部增量重规划，构建“动态航程回放”框架。
  - 同场景对比 `A* / D* Lite / Any-angle / Hybrid A*` 的收益边界。
- 工程落地：
  - 扩展 `route/plan/dynamic` 为多时间步批量模式与事件日志输出。
  - 前端增加时间轴回放、重规划触发点、路径演化对比。
  - 基准测试新增：重规划次数、稳定性、风险暴露积分、单位时间算力成本。

---

## 11. 方向 B 执行清单（逐条交付 + 测试验收）

执行顺序建议：`B1 -> B2 -> B3 -> B4 -> B5 -> B6 -> B7 -> B8 -> B9 -> B10`。

### B1. 标注质检规则 v1（数据入口把关）
- [x] 实现内容：
  - 新增标注质检脚本，检查空标注、异常小区域、`caution/blocked` 冲突、标签越界。
  - 生成 `json + md` 质检报告（可追溯到样本 ID）。
- [x] 测试验收：
  - `python -m pytest -q tests/test_train_quality.py`
  - `python scripts/qc_unet_manifest.py` 执行成功并输出报告文件。

### B2. 主动学习排序器 v2（不确定性 + 路线影响）
- [x] 实现内容：
  - 将样本优先级由“仅不确定性”升级为“`uncertainty + route_impact + class_balance`”联合评分。
  - 输出 Top-K 建议样本及评分分解项。
- [x] 测试验收：
  - `python -m pytest -q tests/test_active_learning_suggest.py`
  - `python scripts/active_learning_suggest.py --top-k 20` 可产出排序结果。

### B3. 可解释建议信号（原因分解）
- [x] 实现内容：
  - 为每个建议样本输出解释字段：冰/风/浪贡献、AIS 偏离、历史误判风险。
  - 落盘解释快照（`json` + 可视化图）。
- [x] 测试验收：
  - 新增解释模块单测（贡献值非负、总和一致、字段完整）。
  - 随机抽样 5 个样本，解释文件均可读取。

### B4. 标注任务包自动生成（可直接开工）
- [x] 实现内容：
  - 生成 `riskhint + landmask + blocked overlay` 的标注包目录。
  - 支持按批次导出（例如每批 20 张）与续标。
- [x] 测试验收：
  - `python scripts/prepare_unet_annotation_pack.py` 成功生成批次目录。
  - 批次内文件名、映射清单、图像数量一致。

### B5. 标注端“建议原因卡片 + 一键接受/修订”
- [x] 实现内容：
  - 前端显示建议原因卡片（置信度、贡献来源、推荐动作）。
  - 增加“接受建议 / 局部修订”状态记录。
- [x] 测试验收：
  - `cd frontend && npm run build` 通过。
  - 手工验收：能看到原因卡片，点击操作后状态可持久化。

### B6. 训练损失升级（focal + dice + 类别权重）
- [x] 实现内容：
  - 训练配置支持 `focal + dice` 组合损失与类别权重。
  - 支持配置切换并记录实验参数。
- [x] 测试验收：
  - `python -m pytest -q tests/test_losses.py`
  - 训练脚本 1 个短轮次 smoke run 成功。

### B7. 困难样本重采样（hard example replay）
- [x] 实现内容：
  - 基于误差/不确定性分数建立困难样本池，训练时按比例重放。
  - 支持重采样比例和上限配置。
- [x] 测试验收：
  - 新增采样器单测（抽样分布符合预期）。
  - 训练日志可看到 hard pool 命中率指标。

### B8. 不确定性校准与可视化
- [x] 实现内容：
  - 增加温度缩放或等价校准方法，输出校准前后对比。
  - 前端或报告展示不确定性热图与阈值建议。
- [x] 测试验收：
  - 新增校准单测（ECE/Brier 至少一个指标改善或可追踪）。
  - 生成 1 份校准评估报告。

### B9. 闭环评估报告（标注 -> 训练 -> 规划收益）
- [x] 实现内容：
  - 固化前后对比报告：`val_iou`、`route_safety`、`route_length_delta`、`inference_time`。
  - 输出统一模板报告用于论文/答辩材料。
- [x] 测试验收：
  - `python scripts/report_data_quality.py` 与评估脚本可串联跑通。
  - 生成一份完整对比报告（含结论段）。

### B10. 一键闭环流水线（工程化）
- [x] 实现内容：
  - 提供一条命令串联：质检 -> 建议 -> 标注包 -> 训练 -> 评估 -> 报告。
  - 增加失败重试与断点续跑（最少在建议/训练阶段支持）。
- [x] 测试验收：
  - 全链路脚本在小样本集上可跑通。
  - `python -m pytest -q` 全部通过后再提交 GitHub。
