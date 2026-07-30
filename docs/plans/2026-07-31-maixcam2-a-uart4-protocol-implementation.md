# MaixCAM2 A版 UART4 二进制通信 Implementation Plan

> **For Codex:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `maixcam2_app_A_quad` 中增加可靠的UART4二进制通信、ROI页A4发送按钮、求解结果自动发送和完整A4黄色区域，并提供STM32F4解析文档。

**Architecture:** 新建不依赖OpenCV和Maix硬件的 `serial_protocol.py`，把帧编解码、流式解析、可靠发送和链路状态集中管理；`main.py` 只注入Maix UART4工厂并转交界面动作和成功规划。调参布局保持固定六槽，纸面区域校验以完整A4边界为上限。

**Tech Stack:** Python 3、MaixPy `maix.uart`/`maix.pinmap`、`struct`、pytest、CRC16-CCITT-FALSE、UART 115200 8N1。

---

### Task 1: 协议编解码与流式解析

**Files:**
- Create: `maixcam2_app_A_quad/serial_protocol.py`
- Create: `tests_ab/test_a_uart_protocol.py`

**Step 1: Write the failing tests**

在 `tests_ab/test_a_uart_protocol.py` 添加以下独立行为测试：

```python
def test_crc16_matches_ccitt_false_standard_vector():
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_frame_encoding_uses_little_endian_header_length_and_crc():
    frame = encode_frame(MSG_PAPER_FRAME, b"\x01\x02", sequence=0x1234, flags=FLAG_ACK_REQUIRED)
    assert frame[:9] == b"\xAA\x55\x01\x10\x01\x34\x12\x02\x00"
    assert int.from_bytes(frame[-2:], "little") == crc16_ccitt_false(frame[2:-2])


def test_stream_parser_recovers_from_noise_split_frames_and_bad_crc():
    parser = FrameStreamParser()
    good = encode_frame(MSG_ACK, encode_ack_payload(MSG_HEARTBEAT, 7, 0), sequence=9)
    bad = bytearray(good)
    bad[-1] ^= 0xFF
    assert parser.feed(b"noise" + bytes(bad) + good[:4]) == []
    frames = parser.feed(good[4:] + good)
    assert [(item.message_type, item.sequence) for item in frames] == [(MSG_ACK, 9), (MSG_ACK, 9)]


def test_paper_payload_uses_full_a4_millimetres_for_both_orientations():
    assert decode_paper_payload(encode_paper_payload("portrait"))["corners_mm"] == (
        (0.0, 0.0), (210.0, 0.0), (210.0, 297.0), (0.0, 297.0)
    )


@pytest.mark.parametrize("piece_count", (1, 2, 3, 4))
def test_puzzle_payload_keeps_all_piece_records_in_one_payload(piece_count):
    payload = encode_puzzle_result_payload("unknown", "portrait", make_placements(piece_count))
    decoded = decode_puzzle_result_payload(payload)
    assert decoded["piece_count"] == piece_count
    assert len(decoded["pieces"]) == piece_count
```

补充定点数四舍五入、负旋转角、非法碎片数、非有限数、溢出、版本错误、64字节长度上限和序号边界测试。

**Step 2: Run tests to verify RED**

Run: `pytest -q tests_ab/test_a_uart_protocol.py`

Expected: FAIL，原因是 `maixcam2_app_A_quad.serial_protocol` 尚不存在。

**Step 3: Write minimal implementation**

在 `serial_protocol.py` 实现以下公开接口，并为每个函数写中文函数注释，说明参数、流程、返回值和异常边界：

```python
FRAME_SOF = b"\xAA\x55"
PROTOCOL_VERSION = 1
MAX_PAYLOAD_LENGTH = 64
MSG_HEARTBEAT = 0x01
MSG_PAPER_FRAME = 0x10
MSG_PUZZLE_RESULT = 0x20
MSG_ACK = 0x80
FLAG_ACK_REQUIRED = 0x01
FLAG_RETRY = 0x02

def crc16_ccitt_false(data): ...
def encode_frame(message_type, payload=b"", sequence=0, flags=0): ...
def encode_ack_payload(acked_type, acked_sequence, status=0): ...
def decode_ack_payload(payload): ...
def encode_heartbeat_payload(uptime_ms, app_state, last_error=0): ...
def encode_paper_payload(paper_orientation): ...
def decode_paper_payload(payload): ...
def encode_puzzle_result_payload(mode, paper_orientation, placements): ...
def decode_puzzle_result_payload(payload): ...

class ProtocolFrame:
    # 保存一个已经通过版本、长度和CRC校验的帧。
    ...

class FrameStreamParser:
    def feed(self, data): ...
```

解析器以 `bytearray` 保存残留数据；每次先搜索帧头，再读取固定头和载荷长度，CRC失败时仅丢弃候选帧头的第一个字节并继续同步。缓存设置硬上限，末尾单个 `0xAA` 必须保留以接续下一批数据。

**Step 4: Run tests to verify GREEN**

Run: `pytest -q tests_ab/test_a_uart_protocol.py`

Expected: PASS，协议黄金字节、CRC和流式恢复全部通过。

**Step 5: Commit**

```bash
git add maixcam2_app_A_quad/serial_protocol.py tests_ab/test_a_uart_protocol.py
git commit -m "feat: add MaixCAM2 UART4 binary codec"
git push origin main
```

### Task 2: 非阻塞UART运行器、心跳、ACK与可靠重发

**Files:**
- Modify: `maixcam2_app_A_quad/serial_protocol.py`
- Modify: `tests_ab/test_a_uart_protocol.py`

**Step 1: Write the failing tests**

使用只实现 `read/write/close` 的 `FakeUart` 和可控毫秒时钟，添加：

```python
def test_runtime_sends_heartbeat_every_500ms_without_blocking_reads(): ...
def test_runtime_becomes_online_only_after_matching_ack_and_offline_after_1500ms(): ...
def test_reliable_message_retries_same_type_and_sequence_with_retry_flag(): ...
def test_puzzle_result_is_queued_once_per_capture_context(): ...
def test_new_capture_context_cancels_old_pending_result(): ...
def test_short_write_and_uart_factory_failure_keep_visual_runtime_alive(): ...
def test_sequence_wraps_from_65535_to_zero(): ...
```

**Step 2: Run tests to verify RED**

Run: `pytest -q tests_ab/test_a_uart_protocol.py -k runtime`

Expected: FAIL，原因是 `VisionSerialRuntime` 和运行器状态接口不存在。

**Step 3: Write minimal implementation**

实现：

```python
class VisionSerialRuntime:
    def __init__(self, uart_factory, clock_ms=None): ...
    def poll(self, app_state=0): ...
    def queue_paper_frame(self, paper_orientation): ...
    def queue_puzzle_result_once(self, mode, paper_orientation, placements): ...
    def reset_result_context(self): ...
    def close(self): ...

    @property
    def link_text(self): ...

    @property
    def last_event_text(self): ...

def create_maix_uart4(): ...
```

`create_maix_uart4()` 必须延迟导入 `maix.err/pinmap/uart`，依次设置 `A21=UART4_TX`、`A22=UART4_RX`，再打开 `/dev/ttyS4`。运行器所有读取均调用无参数 `read()` 非阻塞接口；可靠A4/结果帧前三次按250ms重发，之后按1000ms重发；心跳每500ms产生新序号，最近心跳序号保留到ACK窗口结束。

**Step 4: Run tests to verify GREEN**

Run: `pytest -q tests_ab/test_a_uart_protocol.py`

Expected: PASS，且测试不导入Maix硬件模块。

**Step 5: Commit**

```bash
git add maixcam2_app_A_quad/serial_protocol.py tests_ab/test_a_uart_protocol.py
git commit -m "feat: add reliable UART4 heartbeat runtime"
git push origin main
```

### Task 3: 六槽调参界面与完整A4黄色区域

**Files:**
- Modify: `maixcam2_app_A_quad/touch_ui.py`
- Modify: `maixcam2_app_A_quad/calibration_ui.py`
- Modify: `maixcam2_app_A_quad/paper_locator.py`
- Modify: `tests_ab/test_variant_calibration_ui.py`
- Modify: `tests_ab/test_a_work_region.py`

**Step 1: Write the failing tests**

添加以下行为：

```python
def test_a_calibration_layout_has_six_non_overlapping_control_slots(): ...
def test_roi_sixth_control_resolves_to_send_a4_and_adv_sixth_is_disabled(): ...
def test_send_a4_button_is_enabled_only_after_paper_quad_exists(): ...
def test_default_portrait_work_region_is_full_a4():
    assert default_work_region_mm("portrait") == (0.0, 0.0, 210.0, 297.0)

def test_default_landscape_work_region_is_full_a4():
    assert default_work_region_mm("landscape") == (0.0, 0.0, 297.0, 210.0)

def test_auto_roi_resets_work_region_split_and_inset_to_full_paper(): ...
def test_work_region_rejects_only_values_outside_the_current_paper(): ...
```

**Step 2: Run tests to verify RED**

Run: `pytest -q tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py`

Expected: FAIL，现有布局只有五槽且工作区最大值仍为230mm。

**Step 3: Write minimal implementation**

- `build_calibration_layout()` 固定创建 `control_1` 至 `control_6`，顶部五页签不变。
- `CalibrationSession.bottom_actions()` 在普通页返回六个动作，最后一个为 `send_a4`；ADV页第六项返回 `disabled`。
- `draw_calibration_frame()` 普通页绘制 `SEND A4`，仅在 `paper_quad` 存在时启用；ADV页第六槽禁用并留空。
- `default_work_region_mm()` 返回当前方向的完整纸面。
- `validate_work_region_mm()` 只以当前纸宽、纸高为最大值。
- `apply_auto_roi()` 成功时设置完整纸面区域、纸面中线和 `inset_mm=0`。
- 更新相关中文注释，删除“固定230mm上限”的过期说明，但保留旧版本设置迁移兼容。

**Step 4: Run tests to verify GREEN**

Run: `pytest -q tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py tests_ab/test_variant_settings.py`

Expected: PASS，横竖方向均可黄色框等于蓝框，旧设置文件仍可读取。

**Step 5: Commit**

```bash
git add maixcam2_app_A_quad/touch_ui.py maixcam2_app_A_quad/calibration_ui.py maixcam2_app_A_quad/paper_locator.py tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py
git commit -m "feat: allow full A4 calibration work area"
git push origin main
```

### Task 4: 主循环接入按钮、链路状态和成功规划发送

**Files:**
- Modify: `maixcam2_app_A_quad/main.py`
- Modify: `tests_ab/test_quad_main.py`
- Modify: `tests_ab/test_a_start_gate.py`
- Modify: `tests_ab/test_calibration_quality.py`

**Step 1: Write the failing tests**

通过注入假的 `VisionSerialRuntime` 添加：

```python
def test_send_a4_calibration_action_queues_current_orientation_without_saving_settings(): ...
def test_send_a4_without_paper_quad_returns_roi_not_set(): ...
def test_successful_plan_queues_all_placements_once(): ...
def test_failed_or_incomplete_plan_is_never_sent(): ...
def test_start_mode_and_cal_changes_reset_result_communication_context(): ...
def test_uart_status_is_added_only_to_display_text_without_repeated_suffixes(): ...
def test_flat_main_import_includes_serial_protocol_fallback(): ...
```

**Step 2: Run tests to verify RED**

Run: `pytest -q tests_ab/test_quad_main.py tests_ab/test_a_start_gate.py tests_ab/test_calibration_quality.py -k "uart or send_a4 or communication"`

Expected: FAIL，主循环尚未创建通信运行器，也未处理 `send_a4`。

**Step 3: Write minimal implementation**

- 包导入和MaixVision平铺导入两条路径都导入 `VisionSerialRuntime/create_maix_uart4`。
- `handle_calibration_action()` 增加可选 `serial_runtime` 参数，处理 `send_a4` 时只排队发送，不调用设置保存。
- 新增纯函数把成功 `AssemblyPlan.placements` 提交给通信运行器，便于PC单测，不在主循环拼字段。
- `run_app()` 创建通信运行器，每帧非阻塞 `poll()`；CAL、模式、START重置旧结果上下文。
- 新成功规划调用 `queue_puzzle_result_once()`，同一规划后续绘制不重复创建帧。
- 正常页和CAL页只在最终 `display_status` 拼接一个 `UART:OK/OFFLINE/ERROR`，不污染核心视觉状态。
- 应用正常退出时关闭UART对象；UART异常只改变通信状态，不退出相机循环。

**Step 4: Run tests to verify GREEN**

Run: `pytest -q tests_ab/test_a_uart_protocol.py tests_ab/test_quad_main.py tests_ab/test_a_start_gate.py tests_ab/test_calibration_quality.py`

Expected: PASS，通信触发与视觉状态隔离。

**Step 5: Commit**

```bash
git add maixcam2_app_A_quad/main.py tests_ab/test_quad_main.py tests_ab/test_a_start_gate.py tests_ab/test_calibration_quality.py
git commit -m "feat: send puzzle plans over UART4"
git push origin main
```

### Task 5: F4解析文档、发布清单与完整回归

**Files:**
- Create: `maixcam2_app_A_quad/MaixCAM2与STM32F4串口协议说明.md`
- Modify: `maixcam2_app_A_quad/A版实机调试手册.md`
- Modify: `maixcam2_app_A_quad/app.yaml`
- Modify: `tools/package_variants.py`
- Modify: `tests_ab/test_variant_packages.py`
- Modify: `项目规划清单.md`
- Modify: `编辑清单.md`
- Modify: `硬件资源表.md`
- Modify: `研究发现.md`

**Step 1: Write the failing release test**

把A版目标版本更新为 `2.0.0`，发布白名单加入 `serial_protocol.py` 和协议说明文档，并断言B版文件及哈希边界不变。

Run: `pytest -q tests_ab/test_variant_packages.py`

Expected: FAIL，清单版本、文件列表和ZIP尚未更新。

**Step 2: Write documentation and release metadata**

协议说明文档必须包含：

- UART4接线和115200 8N1。
- 通用帧逐字节偏移、四类消息载荷和毫米/角度定点换算。
- CRC16-CCITT-FALSE的可直接移植C函数。
- F4 DMA或中断环形缓冲接收状态机伪代码。
- 粘包、断包、CRC错误、长度错误的处理。
- ACK构造、`TYPE+SEQ`去重和不得重复执行电机动作。
- 竖放/横放PAPER_FRAME与三片PUZZLE_RESULT完整十六进制示例。
- 现场逻辑分析仪和串口联调步骤。

更新 `app.yaml` 为 `2.0.0` 并加入新运行模块/文档；更新打包规格并生成 `maixcam2_app_A_quad/dist/diansai_quad-v2.0.0.zip`。

**Step 3: Run focused and full verification**

Run: `pytest -q tests_ab/test_a_uart_protocol.py tests_ab/test_variant_calibration_ui.py tests_ab/test_a_work_region.py tests_ab/test_quad_main.py tests_ab/test_a_start_gate.py tests_ab/test_variant_packages.py`

Expected: PASS。

Run: `pytest -q tests tests_ab`

Expected: 全部PASS，B版隔离测试不变。

Run: `python -m compileall -q maixcam2_app_A_quad`

Expected: 退出码0。

Run: `git diff --check`

Expected: 无空白错误。

**Step 4: Update project records**

在四文件中记录 `trace_id=maixcam2-a-uart4-protocol-20260731`、设计决策、逐轮RED/GREEN证据、UART4引脚资源、发布ZIP哈希和实机未验证边界。

**Step 5: Commit and push**

```bash
git add maixcam2_app_A_quad tools/package_variants.py tests_ab/test_variant_packages.py 项目规划清单.md 编辑清单.md 硬件资源表.md 研究发现.md
git commit -m "release: add UART4 protocol to A variant v2.0.0"
git push origin main
```

实施清单：

1. [创建协议编解码和流式解析，verify:`pytest -q tests_ab/test_a_uart_protocol.py`，review:false]
2. [实现非阻塞心跳、ACK和可靠重发，verify:可控时钟运行器测试全部通过，review:false]
3. [增加六槽ROI界面并放开完整A4区域，verify:横竖工作区与调参专项通过，review:false]
4. [把A4按钮和成功规划接入主循环，verify:按钮、结果一次发送和状态隔离专项通过，review:true]
5. [完成F4文档、v2.0.0发布和全量回归，verify:`pytest -q tests tests_ab`、compileall、ZIP逐字节一致，review:true]
