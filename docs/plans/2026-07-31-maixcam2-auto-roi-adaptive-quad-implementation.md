# MaixCAM2 AUTO ROI Adaptive Quad Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task.

**Goal:** 在保留严格AUTO ROI优先级和全部旧验收门的前提下，让轻微边缘凸起形成的5/6角黑纸恢复为可靠四角。

**Architecture:** `config.py`集中定义递增epsilon序列；`paper_locator.py`只在严格2%失败时逐级重试并返回所用比例；`main.py`把严格顶点数和最终比例追加到现有单行诊断。

**Tech Stack:** Python 3、OpenCV、NumPy、pytest、MaixPy控制台

---

### Task 1: 定义自适应四角行为

**Files:**
- Modify: `tests_ab/test_paper_locator.py`
- Modify: `tests_ab/test_quad_main.py`

1. 增加5角和6角合成黑纸，断言恢复为基础A4四角且采用2.5%。
2. 断言干净A4继续采用2%。
3. 断言空、非递增、非正数和过大的epsilon序列被拒绝。
4. 断言AUTO日志输出`strict_vertices`和`quad_eps`。
5. 运行新增用例，确认因当前固定2%实现而失败。

### Task 2: 实现严格优先的多epsilon拟合

**Files:**
- Modify: `maixcam2_app_A_quad/config.py`
- Modify: `maixcam2_app_A_quad/paper_locator.py`

1. 在配置顶部增加`PAPER_QUAD_EPSILON_RATIOS`并接入`DEFAULT_CONFIG`。
2. 校验epsilon序列后，按顺序执行多边形近似；第一个有效四角立即返回。
3. 把严格顶点数和最终epsilon写入最佳候选诊断，不改变评分和硬门。
4. 运行定位器新增与原有回归，确认RED转GREEN。

### Task 3: 完成日志、发布和验证

**Files:**
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip`

1. 扩展最佳候选日志字段，保持每次AUTO一行和关闭调试零输出。
2. 运行`python -m pytest tests_ab -q`和A版compileall。
3. 重建A版v2.1.0正式ZIP并运行发布一致性测试。
4. 检查差异，确认未修改原有AUTO门限、B版源码或用户未跟踪压缩包。
