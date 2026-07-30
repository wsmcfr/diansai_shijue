# MaixCAM2 A版噪声角点容错、求解日志与横纸诊断设计

## 目标

在不改变题目目标尺寸和机械可达范围的前提下，让WHITE未知碎片对远距离角点误差、白纸覆盖不完整和T形分段接缝更稳健；同时提供默认开启、可一键关闭的电脑控制台诊断，并让AUTO ROI明确显示横竖方向。

## 已确认根因

| 证据 | 结论 |
|---|---|
| 实机状态为`N=3 EDGE=0 LOCKED PLAN SOLVER...` | 分割、数量和稳定门已经通过，失败位于拼装器 |
| 截图近似5+4+3顶点输入的最佳组合约89.10×56.05mm | 目标尺寸位于题目容差范围 |
| 最佳组合重叠约0.59%，三片均有目标矩形外边 | 组合具备强几何合理性 |
| 最佳填充率约88.78%，生产门为92% | 当前主要拒绝原因是`fill_reject` |
| 填充门试验性设为88%后约40节点得到结果 | 合理的二级容错门可以同时缩短求解 |
| CAL截图显示`X 33.5mm` | 当前截图已被AUTO识别为横纸，缺少的是明确H/V反馈 |

## 现场可调常量

三个用户需要直接修改的值统一放在`maixcam2_app_A_quad/assembly_planner.py`导入区后的“现场调试开关”块，不散落在算法内部：

```python
UNKNOWN_STRICT_MIN_FILL_RATIO = 0.92
UNKNOWN_RELAXED_MIN_FILL_RATIO = 0.86
UNKNOWN_SOLVER_DEBUG = True
```

- `UNKNOWN_STRICT_MIN_FILL_RATIO`是第一轮严格填充门。
- `UNKNOWN_RELAXED_MIN_FILL_RATIO`只用于WHITE严格轮失败后的容错验收。
- `UNKNOWN_SOLVER_DEBUG`控制电脑串口/控制台日志；设为`False`时不进入日志格式化分支。

## 双层验收

严格层保持现有规则：目标长边88～122mm、短边48～92mm、填充率至少92%、重叠率不超过3%。任何严格候选成功都优先返回。

WHITE严格层无解时启用容错层：尺寸和重叠规则不变，仅把填充率降到86%，并新增“每片至少一条边位于目标矩形外框”的硬约束。外边判断使用约2～4mm的尺度相关余量，抵消Homography和角点误差，但不允许整片悬在矩形内部。

容错候选按缺口、重叠、接缝长度误差和闭环误差综合评分。GRAPH在最多90组连接中保留最佳容错结果；分段边兜底在首次容错解后再比较固定数量节点，随后返回最佳结果，避免重新跑满搜索预算。CARD继续只使用严格验收，不因白片容错改变花纹拼接规则。

## 调试日志

开启开关后只在关键状态输出一组结构化ASCII日志：

```text
[ROI] AUTO orientation=H edges_px=[...] confidence=0.82
[SOLVER] SNAPSHOT profile=WHITE count=3
[SOLVER] PIECE id=U1 vertices_mm=[...] edges_mm=[...]
[SOLVER] GRAPH candidates=... sets=... checked=... best_fill=...
[SOLVER] FALLBACK complete=... size=... fill=... overlap=... outer=...
[SOLVER] RESULT success=... reason=... nodes=... active_ms=...
```

日志不逐节点打印，避免串口输出反过来拖慢求解。关闭开关后不计算边长字符串和顶点字符串。

## 横纸AUTO

现有对边平均长度判向、297×210mm Homography、左右各裁33.5mm和105mm红线逻辑继续保留。AUTO成功状态改为`AUTO ROI OK H 82%`或`AUTO ROI OK V 82%`，控制台同时打印四边像素长度与最终方向。新增接近用户截图角度和透视的横纸测试，验证方向、机械区和红线同步为H。

## 测试边界

1. 截图近似5+4+3轮廓必须走容错层成功，并报告`relaxed_accept=1`。
2. 严格干净矩形继续报告严格成功，不得降级到容错层。
3. 尺寸超界、重叠过大、缺少目标外边和随机三角形负例继续失败。
4. CARD不使用86%容错门。
5. 调试开关关闭时控制台无求解日志；开启时包含毫米顶点、拒绝计数和结果。
6. 横纸AUTO状态包含H，强透视合成场景仍得到297×210mm映射。

## 版本边界

版本提升到A版v1.7.0。只修改A版、A版测试、文档和打包清单；稳定版与B版业务源码及B版发布包保持不变。
