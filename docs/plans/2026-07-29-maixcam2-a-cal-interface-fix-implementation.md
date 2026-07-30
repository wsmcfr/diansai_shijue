# MaixCAM2 A版CAL接口修复 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复A版按下CAL时因调参绘制函数关键字参数不匹配而退出的问题，并发布可明确区分的A版v1.1.1。

**Architecture:** 保持 `CalibrationSession.settings` 为ROI绘制的唯一数据来源，只删除A版主循环的冗余参数。发布层仅提升A版补丁版本，B版源码、版本和ZIP保持不变。

**Tech Stack:** Python 3、AST、inspect、pytest、MaixPy应用清单、ZIP平铺发布

---

## 执行约束

| 项目 | 要求 |
|---|---|
| trace_id | `maixcam2-a-cal-interface-fix-20260729` |
| 执行方式 | 当前会话单代理串行执行 |
| TDD | 每个生产改动前必须先看到对应测试按预期失败 |
| 稳定版 | `maixcam2_app/` 的9项SHA256必须保持一致 |
| B版 | 不修改 `maixcam2_app_B_warp/` 业务源码和版本 |
| Git | 当前目录不是有效Git仓库，以测试、SHA256和编辑清单替代提交证据 |

### Task 1：建立A版CAL接口契约回归并修复调用

**Files:**
- Modify: `tests_ab/test_quad_main.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1：写失败测试**

在 `tests_ab/test_quad_main.py` 增加AST接口契约测试。测试解析真实A版入口中的
`draw_calibration_frame()` 调用，并与导入函数的 `inspect.signature()` 比较：

```python
def test_quad_calibration_draw_call_only_uses_supported_keywords():
    """验证A版进入CAL时不会向绘制函数传入其不支持的关键字参数。"""
    unsupported = _unsupported_draw_calibration_keywords()
    assert unsupported == set()
```

**Step 2：确认红灯**

Run: `python -m pytest tests_ab/test_quad_main.py::test_quad_calibration_draw_call_only_uses_supported_keywords -v`

Expected: FAIL，差异为 `{'paper_quad', 'active_quad'}`。

**Step 3：最小修复**

从 `maixcam2_app_A_quad/main.py` 的调参绘制调用中删除：

```python
paper_quad=runtime_settings.get("paper_quad"),
active_quad=build_runtime_active_quad(runtime_settings),
```

保留其他位置的 `build_runtime_active_quad()`，因为识别流程仍需使用有效四边形。

**Step 4：确认绿灯与A版定向回归**

Run: `python -m pytest tests_ab/test_quad_main.py tests_ab/test_variant_calibration_ui.py -v`

Expected: 全部PASS，ROI页颜色测试继续证明完整A4和有效区仍会绘制。

### Task 2：发布A版v1.1.1

**Files:**
- Modify: `tests_ab/test_variant_packages.py`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `tools/package_variants.py`
- Modify: `README.md`
- Modify: `docs/maixcam2-auto-roi-ab-guide.md`
- Create: `maixcam2_app_A_quad/dist/diansai_quad-v1.1.1.zip`

**Step 1：先修改发布预期测试**

让发布参数表分别携带版本：A版期望 `1.1.1` 和
`diansai_quad-v1.1.1.zip`，B版仍期望 `1.1.0`。

**Step 2：确认发布红灯**

Run: `python -m pytest tests_ab/test_variant_packages.py -v`

Expected: FAIL，A版清单仍为1.1.0且新ZIP不存在。

**Step 3：更新A版发布元数据和文档**

- `maixcam2_app_A_quad/app.yaml` 改为 `version: 1.1.1`；
- `tools/package_variants.py` 的A版归档名改为 `diansai_quad-v1.1.1.zip`；
- README与现场手册把A版部署路径改为新包，并注明v1.1.1修复CAL崩溃；
- 不删除旧A版v1.1.0，不修改B版版本。

**Step 4：重打包并确认绿灯**

Run: `python tools/package_variants.py`

Run: `python -m pytest tests_ab/test_variant_packages.py -v`

Expected: 新A包和原B包均通过清单、平铺文件与解压导入测试。

### Task 3：全量验证与故障记录

**Files:**
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `研究发现.md`

**Step 1：执行全量完成门**

Run: `python -m pytest -v`

Run: `python -m compileall -q maixcam2_app maixcam2_app_A_quad maixcam2_app_B_warp tools`

Run: `python -c "import maixcam2_app_A_quad.main, maixcam2_app_B_warp.main; print('AB_IMPORT_OK')"`

Expected: 全部退出码0，测试数比修复前增加1项。

**Step 2：核对稳定版与B版边界**

- 稳定版9个Python/YAML文件与既有SHA256基线逐项一致；
- B版业务源码与修复前保持一致；
- A版新ZIP平铺解压导入成功。

**Step 3：更新三份项目记录**

记录实机日志、根因、TDD红绿证据、新ZIP路径与SHA256，并把“MaixCAM2重新按CAL”
保留为设备端待验证项。硬件分配没有变化，不修改 `硬件资源表.md`。
