# MaixCAM2 A版CAL接口修复设计

- **trace_id**：`maixcam2-a-cal-interface-fix-20260729`
- **故障设备**：MaixCAM2 / MaixPy
- **故障范围**：仅A版四边形掩膜应用
- **用户确认**：2026-07-29确认采用方案1

## 故障与根因

按下 `CAL` 后，A版 `main.py` 调用 `draw_calibration_frame()` 时传入
`paper_quad` 和 `active_quad` 两个关键字参数，而A版 `calibration_ui.py`
中的函数签名不接收它们，因此在进入调参分支的第一帧抛出 `TypeError`。

该问题已在源码、A版发布ZIP和PC端 `inspect.signature().bind()` 最小复现中
得到相同结果。B版没有传入这两个关键字参数，所以不受影响。

## 选定方案

删除A版主循环中的两个冗余关键字参数，不扩展绘制函数接口。ROI页面已经从
`CalibrationSession.settings` 读取 `paper_quad` 和 `inset_mm` 并计算有效四边形，
因此删除参数不会丢失青色完整A4和黄色机械有效区。

这能保持纸张参数只有一个数据来源，避免运行时设置与调参会话设置不一致。

## 发布策略

- A版应用版本从 `1.1.0` 升级为 `1.1.1`。
- 新包命名为 `diansai_quad-v1.1.1.zip`。
- 保留旧 `diansai_quad-v1.1.0.zip`，便于识别故障包，不覆盖历史证据。
- B版继续使用 `diansai_warp-v1.1.0.zip`，业务代码和版本均不修改。

## 测试设计

1. 新增A版主循环到调参绘制函数的接口契约测试，自动解析真实 `main.py` 调用，
   确认所有关键字参数都被目标函数签名支持。
2. 先运行契约测试并观察其因 `paper_quad/active_quad` 失败，再实施最小修复。
3. 发布测试分别断言A版 `1.1.1` 和B版 `1.1.0`，并验证两个ZIP平铺导入。
4. 最终执行全部测试、compileall、A/B源码导入和稳定版9项SHA256核对。

## 验证边界

PC端测试可证明接口、发布结构和导入路径成立。MaixCAM2上再次按下 `CAL` 并完成
页面切换，属于本次修复完成后的实机验收项。
