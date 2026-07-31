# MaixCAM2 A版蓝框手动标定 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在A版CAL的ROI页加入AUTO/MANUAL双模式，使用户可用`X/Y/W/H`和`1/5/10px`步进修正蓝色A4四边形，同时保持黄色工作区默认与蓝框重合。

**Architecture:** 仅在`CalibrationSession`中保存未持久化的ROI模式、蓝框参数项和步进；`paper_quad`仍是唯一持久化蓝框。手动几何使用纯函数生成候选四边形，再统一经过边界、最小尺寸和现有凸四边形校验；入口只负责把六槽动作路由到会话，绘制层按页面职责显示蓝框或黄色工作区参数。

**Tech Stack:** MaixPy/Python 3、NumPy、OpenCV、pytest、现有JSON设置与A版发布脚本。

---

### Task 1: ROI会话模式、参数循环与初始框

**Files:**
- Modify: `tests_ab/test_variant_calibration_ui.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`

**Step 1: Write the failing tests**

新增A版专属测试，先声明期望API：

```python
def test_a_roi_page_starts_in_auto_and_cycles_manual_items():
    session = _make_a_session()
    assert session.roi_mode == "AUTO"
    assert session.current_item == "MODE"
    assert [session.cycle_roi_item() for _ in range(6)] == [
        "X", "Y", "W", "H", "STEP", "MODE"
    ]


def test_a_switching_to_manual_builds_centered_a4_when_quad_is_missing():
    session = _make_a_session(frame_size=(1280, 960))
    assert session.set_roi_mode("MANUAL") is True
    quad = np.asarray(session.settings["paper_quad"])
    assert quad.shape == (4, 2)
    assert np.allclose(quad.mean(axis=0), (640.0, 480.0))
    assert session.settings["work_x_mm"] == 0.0
    assert session.settings["work_y_mm"] == 0.0


def test_a_switching_to_manual_preserves_existing_quad_and_resets_full_work_area():
    session = _make_a_session_with_quad()
    old_quad = np.asarray(session.settings["paper_quad"]).copy()
    session.settings["work_x_mm"] = 10.0
    assert session.set_roi_mode("MANUAL") is True
    np.testing.assert_allclose(session.settings["paper_quad"], old_quad)
    assert _work_region(session) == default_work_region_mm(
        session.settings["paper_orientation"]
    )
```

同时把现有A/B共用断言拆成变体分支：A版ROI页期望蓝框动作，B版保持原动作不变。

**Step 2: Run tests to verify RED**

Run:

```powershell
pytest -q tests_ab/test_variant_calibration_ui.py -k "roi_page_starts or switching_to_manual"
```

Expected: FAIL，原因是`roi_mode`、`cycle_roi_item()`和`set_roi_mode()`尚不存在，而不是导入或夹具错误。

**Step 3: Write minimal implementation**

在`calibration_ui.py`增加会话常量：

```python
ROI_MODE_AUTO = "AUTO"
ROI_MODE_MANUAL = "MANUAL"
PAPER_ROI_ITEMS = ("MODE", "X", "Y", "W", "H", "STEP")
MANUAL_ROI_STEPS_PX = (1, 5, 10)
MIN_MANUAL_PAPER_EDGE_PX = 80.0
```

在`CalibrationSession.__init__()`中初始化`roi_mode`、`_paper_roi_item_index`和默认5px步进。新增：

- `current_roi_item`：返回ROI页蓝框参数。
- `cycle_roi_item()`：循环`MODE/X/Y/W/H/STEP`。
- `_build_centered_paper_quad()`：按当前横竖方向和80%可用画面生成居中A4框。
- `_reset_full_work_region()`：把黄色毫米工作区和分界线恢复为完整纸面默认值。
- `set_roi_mode(mode)`：切到MANUAL时保留现有蓝框；没有蓝框时创建初始框；随后重置黄色区和测量窗口。

所有新增函数必须包含中文函数注释，说明流程、参数、返回值和为什么状态不持久化。

**Step 4: Run tests to verify GREEN**

Run:

```powershell
pytest -q tests_ab/test_variant_calibration_ui.py -k "roi_page_starts or switching_to_manual or auto_roi_failure or auto_roi_success"
```

Expected: PASS；AUTO失败仍保留旧蓝框，AUTO成功仍重置完整黄色工作区。

**Step 5: Commit checkpoint**

```powershell
git add maixcam2_app_A_quad/calibration_ui.py tests_ab/test_variant_calibration_ui.py
git commit -m "feat: add manual paper roi session state"
git push origin feature/four-piece-dedicated
```

---

### Task 2: 透视保持的X/Y/W/H几何调整与安全门

**Files:**
- Modify: `tests_ab/test_variant_calibration_ui.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`

**Step 1: Write the failing geometry tests**

```python
def test_a_manual_xy_translates_every_corner_by_selected_step():
    session = _make_a_manual_session(step=5)
    before = np.asarray(session.settings["paper_quad"])
    assert session.adjust_paper_quad("X", 1) is True
    np.testing.assert_allclose(session.settings["paper_quad"], before + (5.0, 0.0))


def test_a_manual_width_keeps_top_and_bottom_edge_directions():
    session = _make_a_perspective_manual_session()
    before = np.asarray(session.settings["paper_quad"])
    before_directions = _row_unit_vectors(before)
    assert session.adjust_paper_quad("W", 1) is True
    after = np.asarray(session.settings["paper_quad"])
    np.testing.assert_allclose(_row_unit_vectors(after), before_directions, atol=1e-6)
    assert _mean_width(after) > _mean_width(before)


def test_a_manual_height_keeps_left_and_right_edge_directions():
    session = _make_a_perspective_manual_session()
    before = np.asarray(session.settings["paper_quad"])
    assert session.adjust_paper_quad("H", -1) is True
    after = np.asarray(session.settings["paper_quad"])
    np.testing.assert_allclose(_column_unit_vectors(after), _column_unit_vectors(before), atol=1e-6)
    assert _mean_height(after) < _mean_height(before)


@pytest.mark.parametrize("item,direction", [("X", -1), ("Y", -1), ("W", -1), ("H", -1)])
def test_a_manual_adjustment_rejects_out_of_frame_or_too_small_quad(item, direction):
    session = _make_limit_manual_session(item)
    before = session.snapshot()
    assert session.adjust_paper_quad(item, direction) is False
    assert session.settings["paper_quad"] == before["paper_quad"]
    assert session.status_text == "ROI LIMIT"
```

**Step 2: Run tests to verify RED**

Run:

```powershell
pytest -q tests_ab/test_variant_calibration_ui.py -k "manual_xy or manual_width or manual_height or manual_adjustment_rejects"
```

Expected: FAIL，原因是`adjust_paper_quad()`尚不存在。

**Step 3: Implement pure geometry helpers and session adjustment**

新增纯函数：

- `_paper_quad_mean_size(quad)`：返回上下边平均宽、左右边平均高。
- `_scale_quad_width(quad, delta_px)`：上边和下边分别绕各自中点使用同一比例缩放。
- `_scale_quad_height(quad, delta_px)`：左边和右边分别绕各自中点使用同一比例缩放。
- `_validate_manual_paper_quad(quad, frame_size)`：检查有限数、80px最小平均边长、画面边界，并通过`validate_runtime_settings()`复核连续凸性。

新增`CalibrationSession.adjust_paper_quad(item, direction)`：

1. 非MANUAL模式显示`SWITCH MANUAL`并返回False。
2. `X/Y`平移所有角点；`W/H`调用纯几何函数。
3. 候选不合法时不修改会话并显示`ROI LIMIT`。
4. 成功时写回`paper_quad`、重置测量窗口并显示`MAN X/Y/W/H`状态。
5. `STEP`由独立`cycle_manual_step(direction)`在1/5/10px中循环。

**Step 4: Run geometry tests and refactor**

Run:

```powershell
pytest -q tests_ab/test_variant_calibration_ui.py -k "manual_ or auto_roi"
```

Expected: PASS。重构只允许抽取重复的四边形数组校验，不改变AUTO定位逻辑。

**Step 5: Commit checkpoint**

```powershell
git add maixcam2_app_A_quad/calibration_ui.py tests_ab/test_variant_calibration_ui.py
git commit -m "feat: adjust paper quad with perspective preservation"
git push origin feature/four-piece-dedicated
```

---

### Task 3: 六槽入口路由、页面职责和绘制标签

**Files:**
- Modify: `tests_ab/test_variant_calibration_ui.py`
- Modify: `tests_ab/test_a_work_region.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_A_quad/main.py`

**Step 1: Write failing action and rendering tests**

覆盖以下行为：

```python
def test_a_roi_controls_map_to_auto_minus_manual_value_plus_lock_send():
    session = _make_a_session()
    assert session.bottom_actions() == (
        "auto_roi", "paper_dec", "paper_value", "paper_inc", "lock_roi", "send_a4"
    )


def test_a_roi_mode_and_geometry_are_driven_by_minus_plus_actions():
    runtime, interface, detection = _make_calibration_runtime()
    _tap_value_until(interface.calibration_session, "MODE")
    runtime, message = handle_calibration_action("control_4", ...)
    assert message == "ROI MANUAL"
    _tap_value_until(interface.calibration_session, "X")
    before = np.asarray(interface.calibration_session.settings["paper_quad"])
    runtime, message = handle_calibration_action("control_4", ...)
    assert message.startswith("MAN X")
    assert np.all(np.asarray(interface.calibration_session.settings["paper_quad"])[:, 0] > before[:, 0])


def test_a_roi_draws_manual_field_but_mask_keeps_work_mm_field(monkeypatch):
    # ROI页应捕获到"MODE AUTO"或"X ...px"；MASK页仍捕获到"X 0.0mm"。
```

补充回归：AUTO成功切回AUTO模式；AUTO失败保持MANUAL模式和手动蓝框；LOCK保存当前手动`paper_quad`；不按LOCK退出CAL后运行参数不变；B版动作和标签不变。

**Step 2: Run tests to verify RED**

Run:

```powershell
pytest -q tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py -k "paper_dec or roi_mode_and_geometry or roi_draws_manual or lock"
```

Expected: FAIL，原因是ROI页仍返回`work_dec/work_value/work_inc`。

**Step 3: Implement routing and labels**

在`CalibrationSession`中按页面返回动作：

- ROI：`auto_roi/paper_dec/paper_value/paper_inc/lock_roi/send_a4`
- MASK、RESULT、MEASURE：保留`auto_roi/work_dec/work_value/work_inc/lock_roi/send_a4`
- ADV：完全保持现有动作

在`main.handle_calibration_action()`中：

- `paper_value`循环ROI蓝框参数。
- `paper_dec/paper_inc`按当前项执行模式切换、步进切换或几何调整。
- AUTO成功后设置AUTO模式；AUTO失败不改变当前模式。

在`draw_calibration_frame()`中：

- ROI页第三槽显示`MODE AUTO/MAN`、`X/Y/W/H <值>px`或`STEP <值>px`。
- MASK/RESULT/MEASURE继续显示黄色工作区毫米参数。
- 蓝框仍用青色粗线，黄色完整重合时用黄色细线，保证两者同时可见。
- LOCK和SEND A4的启用条件保持依赖`paper_quad`。

**Step 4: Run focused and variant regression tests**

Run:

```powershell
pytest -q tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py tests_ab/test_calibration_quality.py
```

Expected: PASS；A版新交互通过，B版旧交互通过。

**Step 5: Commit checkpoint**

```powershell
git add maixcam2_app_A_quad/calibration_ui.py maixcam2_app_A_quad/main.py tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py
git commit -m "feat: expose manual paper roi controls"
git push origin feature/four-piece-dedicated
```

---

### Task 4: 文档、全量验证与正式发布

**Files:**
- Modify: `maixcam2_app_A_quad/调试说明.md`
- Modify: `项目规划清单.md`
- Modify: `研究发现.md`
- Modify: `编辑清单.md`
- Regenerate: `maixcam2_app_A_quad/dist/diansai_quad-v2.1.0.zip`

**Step 1: Add documentation assertions before editing docs**

在现有发布/文档测试中增加字符串断言，要求正式说明包含：

- `MODE AUTO`与`MODE MANUAL`
- `X/Y/W/H`
- `STEP 1/5/10px`
- AUTO失败的历史框/居中框回退
- 蓝框与黄色框默认重合
- 必须按`LOCK ROI`保存

**Step 2: Run documentation test to verify RED**

Run:

```powershell
pytest -q tests_ab/test_variant_packages.py -k "documentation or manifest"
```

Expected: FAIL，指出调试说明缺少新手动流程。

**Step 3: Update the field guide and project records**

在`调试说明.md`增加现场步骤和故障表；同步四文件记录每轮RED/GREEN证据、硬件资源无变化、实机待验证项。不得修改用户的`maixcam2_app_A_quad.7z`和`maix-diansai_quad-v2.1.0.zip`。

**Step 4: Run the complete verification gate**

依次执行：

```powershell
pytest -q tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py tests_ab/test_calibration_quality.py
pytest -q tests tests_ab
python -m compileall -q maixcam2_app_A_quad
python tools/package_variants.py
pytest -q tests_ab/test_variant_packages.py
git diff --check
```

Expected: 所有命令退出码0；记录新的总测试数、ZIP字节数和SHA256。若全量测试或发布测试失败，不得更新“已完成”状态。

**Step 5: Commit and push the release snapshot**

只暂存本任务文件和正式`diansai_quad-v2.1.0.zip`，明确排除两个用户文件：

```powershell
git add maixcam2_app_A_quad tests_ab 项目规划清单.md 研究发现.md 编辑清单.md
git commit -m "release: add manual paper roi calibration"
git push origin feature/four-piece-dedicated
```

完成后再次运行`git status --short`，确认只剩预期的用户未跟踪压缩包。
