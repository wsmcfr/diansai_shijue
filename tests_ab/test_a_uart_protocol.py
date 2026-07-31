"""验证A版UART4二进制协议、可靠发送和Maix硬件适配边界。"""

import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from maixcam2_app_A_quad.serial_protocol import (
    FLAG_ACK_REQUIRED,
    FLAG_RETRY,
    FRAME_SOF,
    MAX_PAYLOAD_LENGTH,
    MSG_ACK,
    MSG_HEARTBEAT,
    MSG_PAPER_FRAME,
    MSG_PUZZLE_RESULT,
    PROTOCOL_VERSION,
    RESULT_FLAG_BEST_EFFORT,
    FrameStreamParser,
    VisionSerialRuntime,
    create_maix_uart4,
    crc16_ccitt_false,
    decode_ack_payload,
    decode_heartbeat_payload,
    decode_paper_payload,
    decode_puzzle_result_payload,
    encode_ack_payload,
    encode_frame,
    encode_heartbeat_payload,
    encode_paper_payload,
    encode_puzzle_result_payload,
)


def _make_placements(piece_count):
    """构造包含正负角度的规划记录，供1～4片参数化协议测试复用。"""
    return [
        SimpleNamespace(
            piece_id=f"U{index}",
            source_center_mm=(10.0 * index + 0.04, 20.0 * index + 0.05),
            target_center_mm=(100.0 + index, 160.0 + index),
            rotation_delta_deg=(-1 if index % 2 else 1) * (12.3 + index),
        )
        for index in range(1, piece_count + 1)
    ]


class _FakeClock:
    """提供可手动推进的单调毫秒时钟，避免可靠性测试依赖真实等待。"""

    def __init__(self, now_ms=0):
        self.now_ms = int(now_ms)

    def __call__(self):
        """返回当前测试毫秒值。"""
        return self.now_ms

    def advance(self, delta_ms):
        """把测试时间向前推进指定毫秒数。"""
        self.now_ms += int(delta_ms)


class _FakeUart:
    """模拟Maix UART非阻塞read、完整write和close接口。"""

    def __init__(self, short_write=False):
        self.writes = []
        self.reads = []
        self.short_write = bool(short_write)
        self.closed = False

    def write(self, data):
        """记录发送帧；short_write模式故意少报一个字节。"""
        raw = bytes(data)
        self.writes.append(raw)
        return max(0, len(raw) - 1) if self.short_write else len(raw)

    def read(self):
        """没有测试输入时立即返回空bytes，模拟官方非阻塞read()。"""
        return self.reads.pop(0) if self.reads else b""

    def close(self):
        """记录资源释放状态。"""
        self.closed = True


def _decode_writes(fake_uart):
    """把FakeUart记录的每次完整write解析成协议帧列表。"""
    parser = FrameStreamParser()
    frames = []
    for raw in fake_uart.writes:
        frames.extend(parser.feed(raw))
    return frames


def _ack_frame(acked_frame, status=0):
    """为指定发送帧构造F4返回的合法ACK通用帧。"""
    return encode_frame(
        MSG_ACK,
        encode_ack_payload(acked_frame.message_type, acked_frame.sequence, status),
        sequence=0x9000,
    )


def test_crc16_matches_ccitt_false_standard_vector():
    """标准向量必须得到0x29B1，防止两端采用不同CRC变体。"""
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_frame_encoding_uses_little_endian_header_length_and_crc():
    """通用帧头、序号、长度和CRC必须完全符合协议文档的字节顺序。"""
    assert PROTOCOL_VERSION == 3
    frame = encode_frame(
        MSG_PAPER_FRAME,
        b"\x01\x02",
        sequence=0x1234,
        flags=FLAG_ACK_REQUIRED,
    )

    assert frame[:9] == b"\xAA\x55\x03\x10\x01\x34\x12\x02\x00"
    assert int.from_bytes(frame[-2:], "little") == crc16_ccitt_false(frame[2:-2])


def test_heartbeat_payload_carries_session_id_before_uptime_and_state():
    """心跳必须先发送非零启动会话ID，供F4在Maix重启后清空旧去重缓存。"""
    payload = encode_heartbeat_payload(
        session_id=0x12345678,
        uptime_ms=0x89ABCDEF,
        app_state=3,
        last_error=4,
    )

    assert payload == b"\x78\x56\x34\x12\xEF\xCD\xAB\x89\x03\x04"
    assert decode_heartbeat_payload(payload) == {
        "session_id": 0x12345678,
        "uptime_ms": 0x89ABCDEF,
        "app_state": 3,
        "last_error": 4,
    }

    # 完整黄金帧同时锁定协议版本3、10字节长度和CRC，供F4文档逐字节核对。
    frame = encode_frame(
        MSG_HEARTBEAT,
        encode_heartbeat_payload(0x12345678, 1000, 3, 0),
        sequence=0x0001,
        flags=FLAG_ACK_REQUIRED,
    )
    assert frame == bytes.fromhex(
        "AA 55 03 01 01 01 00 0A 00 "
        "78 56 34 12 E8 03 00 00 03 00 86 76"
    )


def test_stream_parser_recovers_from_noise_split_frames_and_bad_crc():
    """坏帧不能吞掉后续合法断包，多帧粘包也必须一次全部返回。"""
    parser = FrameStreamParser()
    good = encode_frame(
        MSG_ACK,
        encode_ack_payload(MSG_HEARTBEAT, 7, 0),
        sequence=9,
    )
    bad = bytearray(good)
    bad[-1] ^= 0xFF

    assert parser.feed(b"noise" + bytes(bad) + good[:4]) == []
    frames = parser.feed(good[4:] + good)

    assert [(item.message_type, item.sequence) for item in frames] == [
        (MSG_ACK, 9),
        (MSG_ACK, 9),
    ]
    assert decode_ack_payload(frames[0].payload) == {
        "acked_type": MSG_HEARTBEAT,
        "acked_sequence": 7,
        "status": 0,
    }


def test_stream_parser_preserves_partial_sof_after_large_noise():
    """缓存裁剪时必须保留末尾0xAA，使下一批0x55仍能组成帧头。"""
    parser = FrameStreamParser()
    frame = encode_frame(MSG_ACK, encode_ack_payload(MSG_HEARTBEAT, 1, 0), sequence=2)

    assert parser.feed(b"x" * 300 + FRAME_SOF[:1]) == []
    parsed = parser.feed(frame[1:])

    assert len(parsed) == 1
    assert parsed[0].sequence == 2


@pytest.mark.parametrize(
    ("orientation", "width", "height", "corners"),
    (
        (
            "portrait",
            210.0,
            297.0,
            ((0.0, 0.0), (210.0, 0.0), (210.0, 297.0), (0.0, 297.0)),
        ),
        (
            "landscape",
            297.0,
            210.0,
            ((0.0, 0.0), (297.0, 0.0), (297.0, 210.0), (0.0, 210.0)),
        ),
    ),
)
def test_paper_payload_uses_full_a4_millimetres(orientation, width, height, corners):
    """调试帧只发送完整A4毫米边界，不发送黄色区域或相机像素。"""
    decoded = decode_paper_payload(encode_paper_payload(orientation))

    assert decoded["orientation"] == orientation
    assert decoded["paper_size_mm"] == (width, height)
    assert decoded["corners_mm"] == corners


@pytest.mark.parametrize("piece_count", (1, 2, 3, 4))
def test_puzzle_payload_keeps_all_piece_records_in_one_payload(piece_count):
    """1～4片记录必须连续编码在同一个PUZZLE_RESULT载荷中。"""
    payload = encode_puzzle_result_payload(
        "unknown",
        "portrait",
        _make_placements(piece_count),
    )
    decoded = decode_puzzle_result_payload(payload)

    assert decoded["mode"] == "unknown"
    assert decoded["orientation"] == "portrait"
    assert decoded["piece_count"] == piece_count
    assert len(decoded["pieces"]) == piece_count
    assert len(payload) == 4 + 11 * piece_count
    assert [piece["piece_index"] for piece in decoded["pieces"]] == list(
        range(1, piece_count + 1)
    )


def test_puzzle_payload_rounds_millimetres_and_signed_angle_to_tenths():
    """定点转换采用对称四舍五入，并保留负角度符号。"""
    placement = SimpleNamespace(
        piece_id="U1",
        source_center_mm=(12.34, 56.75),
        target_center_mm=(100.06, 200.04),
        rotation_delta_deg=-36.55,
    )

    piece = decode_puzzle_result_payload(
        encode_puzzle_result_payload("unknown", "landscape", [placement])
    )["pieces"][0]

    assert piece["source_center_mm"] == (12.3, 56.8)
    assert piece["target_center_mm"] == (100.1, 200.0)
    assert piece["rotation_deg"] == -36.6


def test_puzzle_payload_marks_best_effort_and_rejects_unknown_result_flags():
    """结果头bit0必须标记可能不准确，F4解析模型不得接受任何未知高位。"""
    reliable_payload = encode_puzzle_result_payload(
        "unknown",
        "portrait",
        _make_placements(2),
    )
    best_effort_payload = encode_puzzle_result_payload(
        "unknown",
        "portrait",
        _make_placements(2),
        best_effort=True,
    )

    assert reliable_payload[3] == 0
    assert best_effort_payload[3] == RESULT_FLAG_BEST_EFFORT
    assert decode_puzzle_result_payload(reliable_payload)["reliable"] is True
    decoded_best = decode_puzzle_result_payload(best_effort_payload)
    assert decoded_best["best_effort"] is True
    assert decoded_best["reliable"] is False

    malformed = bytearray(best_effort_payload)
    malformed[3] |= 0x80
    with pytest.raises(ValueError, match="未知标志"):
        decode_puzzle_result_payload(malformed)


@pytest.mark.parametrize("piece_count", (0, 5))
def test_puzzle_payload_rejects_piece_counts_outside_one_to_four(piece_count):
    """题目范围外的记录数不得生成可执行机械结果。"""
    with pytest.raises(ValueError, match="1到4"):
        encode_puzzle_result_payload("unknown", "portrait", _make_placements(piece_count))


def test_puzzle_payload_rejects_non_finite_or_overflowing_values():
    """NaN和超出int16的坐标必须在发送前失败，不能发生静默截断。"""
    invalid = SimpleNamespace(
        piece_id="U1",
        source_center_mm=(float("nan"), 10.0),
        target_center_mm=(20.0, 30.0),
        rotation_delta_deg=0.0,
    )
    overflow = SimpleNamespace(
        piece_id="U1",
        source_center_mm=(4000.0, 10.0),
        target_center_mm=(20.0, 30.0),
        rotation_delta_deg=0.0,
    )

    with pytest.raises(ValueError, match="有限"):
        encode_puzzle_result_payload("unknown", "portrait", [invalid])
    with pytest.raises(ValueError, match="int16"):
        encode_puzzle_result_payload("unknown", "portrait", [overflow])


def test_frame_rejects_invalid_fields_and_oversized_payload():
    """协议版本字段范围和64字节载荷上限必须在编码入口统一保护。"""
    with pytest.raises(ValueError, match="消息类型"):
        encode_frame(256, b"")
    with pytest.raises(ValueError, match="序号"):
        encode_frame(MSG_HEARTBEAT, b"", sequence=65536)
    with pytest.raises(ValueError, match="载荷"):
        encode_frame(MSG_HEARTBEAT, b"x" * (MAX_PAYLOAD_LENGTH + 1))


def test_parser_rejects_wrong_version_and_continues_with_next_frame():
    """不同协议版本不能被误解析，但后续当前版本帧仍应正常恢复。"""
    parser = FrameStreamParser()
    wrong = bytearray(encode_frame(MSG_ACK, encode_ack_payload(MSG_HEARTBEAT, 1, 0)))
    wrong[2] = PROTOCOL_VERSION + 1
    wrong[-2:] = crc16_ccitt_false(wrong[2:-2]).to_bytes(2, "little")
    good = encode_frame(MSG_ACK, encode_ack_payload(MSG_HEARTBEAT, 2, 0), sequence=3)

    parsed = parser.feed(bytes(wrong) + good)

    assert len(parsed) == 1
    assert parsed[0].sequence == 3


def test_runtime_sends_heartbeat_every_500ms_without_blocking_reads():
    """轮询可立即返回，且心跳在0、500、1000ms各发送一次。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    runtime = VisionSerialRuntime(lambda: fake_uart, clock_ms=clock)

    runtime.poll(app_state=2)
    assert [frame.message_type for frame in _decode_writes(fake_uart)] == [MSG_HEARTBEAT]

    fake_uart.writes.clear()
    clock.advance(499)
    runtime.poll(app_state=2)
    assert fake_uart.writes == []

    clock.advance(1)
    runtime.poll(app_state=3)
    heartbeat = _decode_writes(fake_uart)
    assert len(heartbeat) == 1
    assert heartbeat[0].message_type == MSG_HEARTBEAT
    assert heartbeat[0].flags == FLAG_ACK_REQUIRED


def test_runtime_sends_session_heartbeat_before_queued_business_frame():
    """首次轮询必须先声明当前会话，再发送同序号空间中的A4或机械业务帧。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    runtime = VisionSerialRuntime(
        lambda: fake_uart,
        clock_ms=clock,
        session_id=0x10203040,
    )
    runtime.queue_paper_frame("portrait")

    runtime.poll(app_state=2)

    frames = _decode_writes(fake_uart)
    assert [frame.message_type for frame in frames] == [MSG_HEARTBEAT, MSG_PAPER_FRAME]
    assert decode_heartbeat_payload(frames[0].payload)["session_id"] == 0x10203040
    assert runtime.session_id == 0x10203040


def test_zero_session_id_is_replaced_with_nonzero_value():
    """零值不能作为有效会话标识，测试注入零时必须规范为1。"""
    runtime = VisionSerialRuntime(
        lambda: _FakeUart(),
        clock_ms=_FakeClock(),
        session_id=0,
    )

    assert runtime.session_id == 1


def test_runtime_becomes_online_after_matching_ack_and_offline_after_1500ms():
    """只有匹配已发送帧的ACK能置在线，ACK超时后自动恢复离线。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    runtime = VisionSerialRuntime(lambda: fake_uart, clock_ms=clock)

    runtime.poll()
    heartbeat = _decode_writes(fake_uart)[0]
    assert runtime.link_text == "UART:OFFLINE"

    fake_uart.reads.append(_ack_frame(heartbeat))
    runtime.poll()
    assert runtime.link_text == "UART:OK"

    clock.advance(1501)
    runtime.poll()
    assert runtime.link_text == "UART:OFFLINE"


def test_unmatched_ack_does_not_create_false_online_state():
    """随机序号ACK不能伪造链路在线状态。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    runtime = VisionSerialRuntime(lambda: fake_uart, clock_ms=clock)

    runtime.poll()
    fake_uart.reads.append(
        encode_frame(MSG_ACK, encode_ack_payload(MSG_HEARTBEAT, 0x4321, 0), sequence=2)
    )
    runtime.poll()

    assert runtime.link_text == "UART:OFFLINE"


def test_reliable_paper_message_retries_same_sequence_and_stops_after_ack():
    """A4可靠帧250ms后用同序号和RETRY标志重发，ACK后立即移出队列。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    runtime = VisionSerialRuntime(lambda: fake_uart, clock_ms=clock)
    runtime.poll()
    fake_uart.writes.clear()

    queued_sequence = runtime.queue_paper_frame("portrait")
    runtime.poll()
    first = _decode_writes(fake_uart)[0]
    assert first.message_type == MSG_PAPER_FRAME
    assert first.sequence == queued_sequence
    assert first.flags == FLAG_ACK_REQUIRED

    fake_uart.writes.clear()
    clock.advance(249)
    runtime.poll()
    assert fake_uart.writes == []

    clock.advance(1)
    runtime.poll()
    retry = _decode_writes(fake_uart)[0]
    assert retry.sequence == first.sequence
    assert retry.flags == FLAG_ACK_REQUIRED | FLAG_RETRY

    fake_uart.reads.append(_ack_frame(retry))
    runtime.poll()
    assert runtime.pending_count == 0
    assert runtime.last_event_text == "A4 ACK"


def test_nack_keeps_reliable_message_pending_but_proves_link_is_alive():
    """F4返回非零状态时保留待发消息，同时确认物理链路确实连通。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    runtime = VisionSerialRuntime(lambda: fake_uart, clock_ms=clock)
    runtime.poll()
    fake_uart.writes.clear()
    runtime.queue_paper_frame("landscape")
    runtime.poll()
    paper_frame = _decode_writes(fake_uart)[0]

    fake_uart.reads.append(_ack_frame(paper_frame, status=2))
    runtime.poll()

    assert runtime.pending_count == 1
    assert runtime.link_text == "UART:OK"
    assert runtime.last_event_text == "A4 NACK 2"


def test_puzzle_result_is_queued_only_once_per_capture_context():
    """同一次START即使每帧看到相同规划，也只能创建一个结果序号。"""
    clock = _FakeClock()
    runtime = VisionSerialRuntime(lambda: _FakeUart(), clock_ms=clock)
    placements = _make_placements(3)

    assert runtime.queue_puzzle_result_once("unknown", "portrait", placements) is True
    assert runtime.queue_puzzle_result_once("unknown", "portrait", placements) is False
    assert runtime.pending_count == 1

    runtime.reset_result_context()
    assert runtime.pending_count == 0
    assert runtime.queue_puzzle_result_once("unknown", "portrait", placements) is True


def test_runtime_queues_best_effort_flag_without_changing_retry_semantics():
    """不可靠结果仍走原单次可靠队列，但载荷必须把风险标志发送给F4。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    runtime = VisionSerialRuntime(lambda: fake_uart, clock_ms=clock)

    assert runtime.queue_puzzle_result_once(
        "unknown",
        "landscape",
        _make_placements(3),
        best_effort=True,
    ) is True
    runtime.poll()
    result_frame = next(
        frame for frame in _decode_writes(fake_uart)
        if frame.message_type == MSG_PUZZLE_RESULT
    )
    decoded = decode_puzzle_result_payload(result_frame.payload)

    assert result_frame.flags == FLAG_ACK_REQUIRED
    assert decoded["best_effort"] is True
    assert runtime.pending_count == 1  # 心跳单独跟踪，可靠结果仍只占一个业务队列项。


def test_runtime_rejects_invalid_plan_without_raising_or_consuming_context():
    """NaN等结果编码错误必须留在通信层，并允许修正后的同次结果再次排队。"""
    clock = _FakeClock()
    runtime = VisionSerialRuntime(lambda: _FakeUart(), clock_ms=clock)
    invalid = SimpleNamespace(
        piece_id="U1",
        source_center_mm=(float("nan"), 20.0),
        target_center_mm=(100.0, 160.0),
        rotation_delta_deg=30.0,
    )

    assert runtime.queue_puzzle_result_once("unknown", "portrait", [invalid]) is False
    assert runtime.last_event_text == "RESULT ERROR"
    assert runtime.pending_count == 0

    assert (
        runtime.queue_puzzle_result_once(
            "unknown",
            "portrait",
            _make_placements(1),
        )
        is True
    )


def test_reset_result_context_preserves_pending_manual_a4_frame():
    """重新START只取消旧机械结果，用户手动发送的A4调试帧仍等待ACK。"""
    clock = _FakeClock()
    runtime = VisionSerialRuntime(lambda: _FakeUart(), clock_ms=clock)
    runtime.queue_paper_frame("portrait")
    runtime.queue_puzzle_result_once("unknown", "portrait", _make_placements(2))

    runtime.reset_result_context()

    assert runtime.pending_message_types == (MSG_PAPER_FRAME,)


def test_factory_failure_retries_later_without_raising_into_visual_loop():
    """UART打开失败后1000ms重试，poll本身不得把异常抛给相机主循环。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    attempts = []

    def factory():
        attempts.append(clock())
        if len(attempts) == 1:
            raise RuntimeError("busy")
        return fake_uart

    runtime = VisionSerialRuntime(factory, clock_ms=clock)
    runtime.poll()
    assert runtime.link_text == "UART:ERROR"
    assert attempts == [0]

    clock.advance(999)
    runtime.poll()
    assert attempts == [0]

    clock.advance(1)
    runtime.poll()
    assert attempts == [0, 1000]
    assert len(fake_uart.writes) == 1


def test_short_write_keeps_message_pending_and_reopens_uart_later():
    """短写不能被当成成功发送，可靠帧必须保留并等待重新打开串口。"""
    clock = _FakeClock()
    short_uart = _FakeUart(short_write=True)
    good_uart = _FakeUart()
    devices = [short_uart, good_uart]
    runtime = VisionSerialRuntime(lambda: devices.pop(0), clock_ms=clock)
    runtime.queue_paper_frame("portrait")

    runtime.poll()
    assert runtime.pending_count == 1
    assert runtime.link_text == "UART:ERROR"
    assert short_uart.closed is True

    clock.advance(1000)
    runtime.poll()
    assert any(frame.message_type == MSG_PAPER_FRAME for frame in _decode_writes(good_uart))


def test_sequence_wraps_from_65535_to_zero_without_reusing_pending_frame():
    """uint16序号达到65535后自然回到0，既有待确认帧仍保留原序号。"""
    clock = _FakeClock()
    runtime = VisionSerialRuntime(lambda: _FakeUart(), clock_ms=clock)
    runtime._next_sequence_value = 0xFFFF

    first = runtime.queue_paper_frame("portrait")
    second = runtime.queue_paper_frame("landscape")

    assert first == 0xFFFF
    assert second == 0


def test_close_releases_uart_and_stops_future_polling():
    """应用退出后必须释放设备，后续误调用poll也不能重新打开串口。"""
    clock = _FakeClock()
    fake_uart = _FakeUart()
    calls = []

    def factory():
        calls.append(1)
        return fake_uart

    runtime = VisionSerialRuntime(factory, clock_ms=clock)
    runtime.poll()
    runtime.close()
    clock.advance(1000)
    runtime.poll()

    assert fake_uart.closed is True
    assert len(calls) == 1


def _install_fake_maix_uart_modules(monkeypatch, uart_factory):
    """安装只覆盖UART4工厂所需接口的假maix模块，并返回调用顺序列表。"""
    events = []
    fake_maix = ModuleType("maix")
    fake_maix.comm = SimpleNamespace(
        rm_default_comm_listener=lambda: events.append("comm_remove"),
        add_default_comm_listener=lambda: events.append("comm_restore"),
    )
    fake_maix.err = SimpleNamespace(
        check_raise=lambda result, _message: events.append(("check", result)),
    )
    fake_maix.pinmap = SimpleNamespace(
        set_pin_function=lambda pin, function: events.append(("pin", pin, function)) or 0,
    )

    def open_uart(device, baudrate):
        """记录UART设备与波特率，然后调用测试指定工厂。"""
        events.append(("uart", device, baudrate))
        return uart_factory(events)

    fake_maix.uart = SimpleNamespace(UART=open_uart)
    monkeypatch.setitem(sys.modules, "maix", fake_maix)
    return events


def test_maix_uart4_factory_takes_and_returns_default_comm_listener(monkeypatch):
    """应用占用UART4前必须停默认监听，关闭UART后必须把监听归还系统。"""
    raw_uart = _FakeUart()
    events = _install_fake_maix_uart_modules(
        monkeypatch,
        lambda _events: raw_uart,
    )

    uart_lease = create_maix_uart4()
    uart_lease.write(b"abc")
    uart_lease.close()

    assert events[0] == "comm_remove"
    assert events[1:5] == [
        ("pin", "A21", "UART4_TX"),
        ("check", 0),
        ("pin", "A22", "UART4_RX"),
        ("check", 0),
    ]
    assert events[5] == ("uart", "/dev/ttyS4", 115200)
    assert events[-1] == "comm_restore"
    assert raw_uart.closed is True


def test_maix_uart4_factory_restores_listener_when_open_fails(monkeypatch):
    """UART构造失败时也必须恢复默认监听，不能让Maix通信口永久失联。"""

    def fail_open(_events):
        """模拟设备被占用或驱动打开失败。"""
        raise RuntimeError("uart busy")

    events = _install_fake_maix_uart_modules(monkeypatch, fail_open)

    with pytest.raises(RuntimeError, match="uart busy"):
        create_maix_uart4()

    assert events[0] == "comm_remove"
    assert events[-1] == "comm_restore"
