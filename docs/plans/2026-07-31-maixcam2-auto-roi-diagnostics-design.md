# MaixCAM2 AUTO ROI 诊断日志设计

## 目标

在不改变 AUTO ROI 识别结果和现场阈值的前提下，让电脑端一次日志能够说明黑色A4候选在哪一道门限被拒绝。

## 方案选择

| 方案 | 做法 | 结论 |
|---|---|---|
| 结构化诊断数据 | `paper_locator.py`收集统计并放入`PaperLocation.diagnostics`，`main.py`统一格式化输出 | 采用；算法与输出解耦，测试方便 |
| 定位器直接打印 | 在轮廓循环中直接`print`每个候选 | 不采用；输出过多且定位模块与控制台耦合 |
| 全部显示到小屏幕 | 把候选指标拼到CAL状态栏 | 不采用；640x480界面无法稳定容纳，影响现场观察 |

## 数据流

1. `locate_black_paper()`对本次反向Otsu外轮廓逐个执行面积、四角、矩形度和置信度门。
2. 每道门只累计计数，同时保存最有诊断价值的最大面积、四角顶点分布和最佳候选指标。
3. 成功与失败都把只读语义的诊断字典附加到`PaperLocation`。
4. `log_auto_roi_diagnostics()`仅在`UNKNOWN_SOLVER_DEBUG=True`时生成一条`[ROI] AUTO`日志。

## 日志字段

| 字段 | 含义 |
|---|---|
| `contours` | Otsu暗色掩膜的外轮廓总数 |
| `area_small/area_large` | 面积低于1%或高于50%的拒绝数量 |
| `not_quad` | 凸包在2%周长近似后不是四边形的数量 |
| `rect_low` | 四角候选矩形度低于0.70的数量 |
| `eligible` | 通过面积、四角和矩形度硬门的数量 |
| `largest_area` | 本帧最大暗色外轮廓占画面百分比 |
| `quad_vertices` | 非四角候选的近似顶点数量分布 |
| `best_area/aspect/rect/convex/dark/conf` | 最佳可评分候选的关键指标 |

## 兼容和边界

- 不修改A4四角、H/V、Homography、UART、碎片识别和求解器。
- `diagnostics`缺失或为空时，日志继续兼容现有`PaperLocation`测试替身。
- 不逐轮廓打印，保证每次点击AUTO只增加一条日志。
- 当前只修改A版；B版定位器和发布物保持不变。

## 测试策略

- 先写失败测试，要求面积过大场景返回`area_large`统计。
- 先写失败测试，要求日志包含各拒绝门和最佳候选字段。
- 保留现有成功日志、关闭开关零输出以及全部A/B回归。
