"""MaixCAM2与STM32F4之间的UART4二进制协议和可靠发送状态机。"""

import math
import os
import struct
import time as _python_time


# 通用帧常量集中定义，F4端必须使用完全相同的数值和小端字节序。
FRAME_SOF = b"\xAA\x55"
# 版本3把结果载荷第4字节定义为可靠性标志；F4必须按版本拒绝旧保留字段语义。
PROTOCOL_VERSION = 3
MAX_PAYLOAD_LENGTH = 64
MAX_RX_BUFFER_LENGTH = 256

# 消息类型按功能分段，0x80以上保留给F4返回的响应帧。
MSG_HEARTBEAT = 0x01
MSG_PAPER_FRAME = 0x10
MSG_PUZZLE_RESULT = 0x20
MSG_ACK = 0x80

# FLAGS位定义；重发时沿用原TYPE和SEQ，只增加FLAG_RETRY供F4诊断。
FLAG_ACK_REQUIRED = 0x01
FLAG_RETRY = 0x02

# PUZZLE_RESULT载荷内部标志与通用帧FLAGS无关；bit0表示结果可能不准确。
RESULT_FLAG_BEST_EFFORT = 0x01
RESULT_FLAG_KNOWN_MASK = RESULT_FLAG_BEST_EFFORT

# 协议层只传数字方向和模式；公开编码函数同时接受业务层使用的字符串。
ORIENTATION_PORTRAIT = 0
ORIENTATION_LANDSCAPE = 1
MODE_KNOWN = 0
MODE_UNKNOWN = 1


def _validate_uint(value, maximum, field_name):
    """校验无符号整数字段并返回int。

    主要流程：拒绝布尔值以外的隐式字符串和越界数字，避免struct.pack在设备端抛出
    含义不清的异常。关键参数maximum为字段最大值，field_name用于形成现场可读错误。
    返回值为规范int；非法输入抛出ValueError。
    """
    try:
        normalized = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是整数") from error
    if normalized != value or not 0 <= normalized <= maximum:
        raise ValueError(f"{field_name}必须位于0到{maximum}之间")
    return normalized


def _orientation_code(paper_orientation):
    """把portrait/landscape或数字方向转换为协议方向码。"""
    if isinstance(paper_orientation, str):
        normalized = paper_orientation.strip().lower()
        if normalized == "portrait":
            return ORIENTATION_PORTRAIT
        if normalized == "landscape":
            return ORIENTATION_LANDSCAPE
    elif paper_orientation in (ORIENTATION_PORTRAIT, ORIENTATION_LANDSCAPE):
        return int(paper_orientation)
    raise ValueError("纸张方向必须是portrait或landscape")


def _orientation_name(orientation_code):
    """把协议方向码转换为业务层使用的方向字符串。"""
    if orientation_code == ORIENTATION_PORTRAIT:
        return "portrait"
    if orientation_code == ORIENTATION_LANDSCAPE:
        return "landscape"
    raise ValueError("协议纸张方向无效")


def _mode_code(mode):
    """把known/unknown或数字模式转换为协议模式码。"""
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized == "known":
            return MODE_KNOWN
        if normalized == "unknown":
            return MODE_UNKNOWN
    elif mode in (MODE_KNOWN, MODE_UNKNOWN):
        return int(mode)
    raise ValueError("拼图模式必须是known或unknown")


def _mode_name(mode_code):
    """把协议模式码转换为业务层使用的模式字符串。"""
    if mode_code == MODE_KNOWN:
        return "known"
    if mode_code == MODE_UNKNOWN:
        return "unknown"
    raise ValueError("协议拼图模式无效")


def _fixed_int16(value, field_name):
    """把毫米或角度浮点值转换为0.1单位的有符号16位整数。

    主要流程：先检查有限数，再执行远离零方向的0.5四舍五入，最后检查int16范围。
    这样正负角度在临界小数处使用对称规则，不依赖Python的银行家舍入。
    返回值为可直接按小端编码的int；NaN、无穷和溢出抛出ValueError。
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name}必须是有限数字") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name}必须是有限数字")
    scaled = numeric * 10.0
    rounded = math.floor(scaled + 0.5) if scaled >= 0.0 else math.ceil(scaled - 0.5)
    if not -32768 <= rounded <= 32767:
        raise ValueError(f"{field_name}超出int16定点范围")
    return int(rounded)


def crc16_ccitt_false(data):
    """计算CRC16-CCITT-FALSE校验值。

    关键参数data必须可转换为bytes。算法使用多项式0x1021、初值0xFFFF、不反射、
    xorout=0，返回0到65535的整数。通用帧只对VERSION至PAYLOAD末尾调用本函数。
    """
    try:
        raw = bytes(data)
    except (TypeError, ValueError) as error:
        raise ValueError("CRC输入必须是字节数据") from error

    crc = 0xFFFF
    for byte in raw:
        crc ^= byte << 8
        for _bit in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def encode_frame(message_type, payload=b"", sequence=0, flags=0):
    """编码一个完整UART二进制帧。

    关键参数message_type和flags为uint8，sequence为uint16，payload最大64字节。
    返回值为可直接交给UART.write的bytes；字段非法时在写串口前抛出ValueError。
    """
    normalized_type = _validate_uint(message_type, 0xFF, "消息类型")
    normalized_sequence = _validate_uint(sequence, 0xFFFF, "序号")
    normalized_flags = _validate_uint(flags, 0xFF, "标志位")
    try:
        normalized_payload = bytes(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("载荷必须是字节数据") from error
    if len(normalized_payload) > MAX_PAYLOAD_LENGTH:
        raise ValueError(f"载荷长度不能超过{MAX_PAYLOAD_LENGTH}字节")

    body = struct.pack(
        "<BBBHH",
        PROTOCOL_VERSION,
        normalized_type,
        normalized_flags,
        normalized_sequence,
        len(normalized_payload),
    ) + normalized_payload
    return FRAME_SOF + body + struct.pack("<H", crc16_ccitt_false(body))


def encode_ack_payload(acked_type, acked_sequence, status=0):
    """编码F4 ACK载荷，主要供PC测试和F4文档黄金帧使用。"""
    return struct.pack(
        "<BHB",
        _validate_uint(acked_type, 0xFF, "ACK消息类型"),
        _validate_uint(acked_sequence, 0xFFFF, "ACK序号"),
        _validate_uint(status, 0xFF, "ACK状态"),
    )


def decode_ack_payload(payload):
    """解析固定4字节ACK载荷并返回字段字典。"""
    raw = bytes(payload)
    if len(raw) != 4:
        raise ValueError("ACK载荷长度必须是4字节")
    acked_type, acked_sequence, status = struct.unpack("<BHB", raw)
    return {
        "acked_type": acked_type,
        "acked_sequence": acked_sequence,
        "status": status,
    }


def encode_heartbeat_payload(session_id, uptime_ms, app_state, last_error=0):
    """编码10字节心跳载荷，会话ID与运行毫秒数均按uint32发送。"""
    return struct.pack(
        "<IIBB",
        _validate_uint(session_id, 0xFFFFFFFF, "启动会话ID"),
        _validate_uint(uptime_ms, 0xFFFFFFFF, "运行毫秒数"),
        _validate_uint(app_state, 0xFF, "应用状态"),
        _validate_uint(last_error, 0xFF, "通信错误码"),
    )


def decode_heartbeat_payload(payload):
    """解析固定10字节心跳，并返回F4去重和状态判断所需字段。"""
    raw = bytes(payload)
    if len(raw) != 10:
        raise ValueError("心跳载荷长度必须是10字节")
    session_id, uptime_ms, app_state, last_error = struct.unpack("<IIBB", raw)
    if session_id == 0:
        raise ValueError("启动会话ID不能为0")
    return {
        "session_id": session_id,
        "uptime_ms": uptime_ms,
        "app_state": app_state,
        "last_error": last_error,
    }


def _normalize_session_id(session_id):
    """生成或校验非零uint32启动会话ID。

    测试可传固定值；设备默认读取4字节系统随机数。随机结果或显式输入为0时统一改为1，
    确保F4可以把0保留为“尚未收到合法心跳”的哨兵值。返回1至0xFFFFFFFF整数。
    """
    if session_id is None:
        normalized = int.from_bytes(os.urandom(4), "little")
    else:
        normalized = _validate_uint(session_id, 0xFFFFFFFF, "启动会话ID")
    return 1 if normalized == 0 else normalized


def encode_paper_payload(paper_orientation):
    """按当前横竖方向编码完整A4四角毫米载荷。

    返回载荷固定22字节，角点顺序为左上、右上、右下、左下。这里不接收相机四角，
    从接口层确保调试帧不会误发黄色工作区或像素坐标。
    """
    orientation = _orientation_code(paper_orientation)
    if orientation == ORIENTATION_PORTRAIT:
        width_x10, height_x10 = 2100, 2970
    else:
        width_x10, height_x10 = 2970, 2100
    corners = (
        (0, 0),
        (width_x10, 0),
        (width_x10, height_x10),
        (0, height_x10),
    )
    payload = bytearray(struct.pack("<BBHH", orientation, 4, width_x10, height_x10))
    for x_x10, y_x10 in corners:
        payload.extend(struct.pack("<HH", x_x10, y_x10))
    return bytes(payload)


def decode_paper_payload(payload):
    """解析PAPER_FRAME载荷，返回便于测试和F4文档核对的毫米字段。"""
    raw = bytes(payload)
    if len(raw) != 22:
        raise ValueError("A4载荷长度必须是22字节")
    orientation, corner_count, width_x10, height_x10 = struct.unpack_from("<BBHH", raw, 0)
    if corner_count != 4:
        raise ValueError("A4角点数量必须是4")
    corners = []
    for index in range(corner_count):
        x_x10, y_x10 = struct.unpack_from("<HH", raw, 6 + index * 4)
        corners.append((x_x10 / 10.0, y_x10 / 10.0))
    return {
        "orientation": _orientation_name(orientation),
        "paper_size_mm": (width_x10 / 10.0, height_x10 / 10.0),
        "corners_mm": tuple(corners),
    }


def encode_puzzle_result_payload(
    mode,
    paper_orientation,
    placements,
    best_effort=False,
):
    """把一次成功规划的全部碎片编码到同一个结果载荷。

    关键参数placements中的每项需提供piece_id、source_center_mm、target_center_mm和
    rotation_delta_deg。记录先按piece_id排序，再生成从1开始的传输序号；碎片数必须
    为1至4。best_effort=True时设置结果头bit0，提醒F4该拼法可能不准确。返回
    4+11*N字节载荷，任何非有限或溢出坐标都会在发送前抛出ValueError。
    """
    normalized_placements = sorted(
        list(placements),
        key=lambda placement: str(getattr(placement, "piece_id", "")),
    )
    piece_count = len(normalized_placements)
    if not 1 <= piece_count <= 4:
        raise ValueError("拼图结果碎片数必须位于1到4之间")
    result_flags = RESULT_FLAG_BEST_EFFORT if bool(best_effort) else 0

    payload = bytearray(
        struct.pack(
            "<BBBB",
            _mode_code(mode),
            _orientation_code(paper_orientation),
            piece_count,
            result_flags,
        )
    )
    for piece_index, placement in enumerate(normalized_placements, start=1):
        try:
            source_x, source_y = placement.source_center_mm
            target_x, target_y = placement.target_center_mm
            rotation_deg = placement.rotation_delta_deg
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("碎片规划缺少中心或旋转字段") from error
        payload.extend(
            struct.pack(
                "<Bhhhhh",
                piece_index,
                _fixed_int16(source_x, "源中心X"),
                _fixed_int16(source_y, "源中心Y"),
                _fixed_int16(target_x, "目标中心X"),
                _fixed_int16(target_y, "目标中心Y"),
                _fixed_int16(rotation_deg, "旋转角"),
            )
        )
    if len(payload) > MAX_PAYLOAD_LENGTH:
        raise ValueError("拼图结果载荷超出协议上限")
    return bytes(payload)


def decode_puzzle_result_payload(payload):
    """解析PUZZLE_RESULT载荷并恢复为毫米和角度字典。

    本函数用于PC回归和协议示例，不参与MaixCAM2发送热路径。载荷长度、结果未知标志、
    模式、方向和碎片数量不一致时抛出ValueError，防止F4示例接受畸形机械命令。
    """
    raw = bytes(payload)
    if len(raw) < 4:
        raise ValueError("拼图结果载荷过短")
    mode, orientation, piece_count, result_flags = struct.unpack_from("<BBBB", raw, 0)
    if not 1 <= piece_count <= 4:
        raise ValueError("拼图结果碎片数必须位于1到4之间")
    if result_flags & ~RESULT_FLAG_KNOWN_MASK:
        raise ValueError("拼图结果包含未知标志位")
    expected_length = 4 + piece_count * 11
    if len(raw) != expected_length:
        raise ValueError("拼图结果载荷长度与碎片数不一致")

    pieces = []
    for index in range(piece_count):
        values = struct.unpack_from("<Bhhhhh", raw, 4 + index * 11)
        piece_index, source_x, source_y, target_x, target_y, rotation = values
        if piece_index != index + 1:
            raise ValueError("碎片序号必须从1连续递增")
        pieces.append(
            {
                "piece_index": piece_index,
                "source_center_mm": (source_x / 10.0, source_y / 10.0),
                "target_center_mm": (target_x / 10.0, target_y / 10.0),
                "rotation_deg": rotation / 10.0,
            }
        )
    return {
        "mode": _mode_name(mode),
        "orientation": _orientation_name(orientation),
        "piece_count": piece_count,
        "result_flags": result_flags,
        "best_effort": bool(result_flags & RESULT_FLAG_BEST_EFFORT),
        "reliable": not bool(result_flags & RESULT_FLAG_BEST_EFFORT),
        "pieces": tuple(pieces),
    }


class ProtocolFrame:
    """保存一个已经通过版本、长度和CRC校验的协议帧。"""

    def __init__(self, message_type, flags, sequence, payload):
        """初始化不可变约定字段；调用者只能从FrameStreamParser获得本对象。"""
        self.version = PROTOCOL_VERSION
        self.message_type = int(message_type)
        self.flags = int(flags)
        self.sequence = int(sequence)
        self.payload = bytes(payload)


class FrameStreamParser:
    """把任意断包、粘包和带噪声的UART字节流恢复为完整协议帧。"""

    def __init__(self):
        """创建空接收缓存；解析器不持有UART对象，也不会执行阻塞读取。"""
        self._buffer = bytearray()

    def _trim_noise_without_header(self):
        """没有完整帧头时清理噪声，并保留可能跨批次的末尾0xAA。"""
        if self._buffer and self._buffer[-1] == FRAME_SOF[0]:
            self._buffer[:] = FRAME_SOF[:1]
        else:
            self._buffer.clear()

    def feed(self, data):
        """追加一批UART数据并返回其中所有CRC正确的完整帧。

        关键参数data为空时只返回空列表。解析过程中遇到错误版本、超长载荷或坏CRC，
        只丢弃当前候选帧头第一个字节后重新搜索，保证紧随其后的合法帧不被吞掉。
        返回值为按接收顺序排列的ProtocolFrame列表。
        """
        if data is None:
            return []
        try:
            raw = bytes(data)
        except (TypeError, ValueError) as error:
            raise ValueError("串口接收数据必须是字节流") from error
        if not raw:
            return []
        self._buffer.extend(raw)
        if len(self._buffer) > MAX_RX_BUFFER_LENGTH:
            del self._buffer[: len(self._buffer) - MAX_RX_BUFFER_LENGTH]

        frames = []
        while True:
            start_index = self._buffer.find(FRAME_SOF)
            if start_index < 0:
                self._trim_noise_without_header()
                break
            if start_index > 0:
                del self._buffer[:start_index]
            if len(self._buffer) < 9:
                break

            version, message_type, flags, sequence, payload_length = struct.unpack_from(
                "<BBBHH",
                self._buffer,
                2,
            )
            if version != PROTOCOL_VERSION or payload_length > MAX_PAYLOAD_LENGTH:
                del self._buffer[0]
                continue

            frame_length = 11 + payload_length
            if len(self._buffer) < frame_length:
                break
            candidate = bytes(self._buffer[:frame_length])
            expected_crc = struct.unpack_from("<H", candidate, frame_length - 2)[0]
            actual_crc = crc16_ccitt_false(candidate[2:-2])
            if expected_crc != actual_crc:
                del self._buffer[0]
                continue

            payload = candidate[9:-2]
            frames.append(ProtocolFrame(message_type, flags, sequence, payload))
            del self._buffer[:frame_length]
        return frames


class _PendingMessage:
    """保存一条等待ACK的可靠逻辑消息及其下一次发送时刻。"""

    def __init__(self, message_type, sequence, payload, next_send_ms):
        """初始化待发消息；send_count只在UART完整写入后增加。"""
        self.message_type = int(message_type)
        self.sequence = int(sequence)
        self.payload = bytes(payload)
        self.send_count = 0
        self.next_send_ms = int(next_send_ms)


class VisionSerialRuntime:
    """在相机主循环中非阻塞维护UART4心跳、ACK和可靠结果重发。"""

    HEARTBEAT_INTERVAL_MS = 500
    LINK_TIMEOUT_MS = 1500
    FAST_RETRY_INTERVAL_MS = 250
    SLOW_RETRY_INTERVAL_MS = 1000
    REOPEN_INTERVAL_MS = 1000

    def __init__(self, uart_factory, clock_ms=None, session_id=None):
        """创建尚未打开串口的通信运行器。

        关键参数uart_factory为无参工厂，成功返回具有read/write/close的UART对象；
        clock_ms为空时使用Python单调时钟，测试可注入可控时钟；session_id为空时生成
        随机非零uint32，测试可注入固定值。构造过程不打开硬件，第一次poll才初始化。
        """
        if not callable(uart_factory):
            raise ValueError("uart_factory必须可调用")
        if clock_ms is not None and not callable(clock_ms):
            raise ValueError("clock_ms必须可调用")
        self._uart_factory = uart_factory
        self._clock_ms = clock_ms or (
            lambda: int(round(_python_time.monotonic() * 1000.0))
        )
        started_ms = int(self._clock_ms())
        self._started_ms = started_ms
        self._session_id = _normalize_session_id(session_id)
        self._uart = None
        self._parser = FrameStreamParser()
        self._next_open_ms = started_ms
        self._next_heartbeat_ms = started_ms
        self._next_sequence_value = 0
        self._pending = {}
        self._recent_heartbeats = {}
        self._last_ack_ms = None
        self._last_event_text = "UART INIT"
        self._result_queued = False
        self._closed = False

    def _now_ms(self):
        """读取并规范化单调毫秒时钟，异常值直接暴露为ValueError。"""
        try:
            return int(self._clock_ms())
        except (TypeError, ValueError) as error:
            raise ValueError("通信时钟必须返回整数毫秒") from error

    def _next_sequence(self):
        """分配uint16序号并在65535后自然回到0。"""
        sequence = self._next_sequence_value
        self._next_sequence_value = (self._next_sequence_value + 1) & 0xFFFF
        return sequence

    def _close_uart_after_error(self, now_ms, event_text="UART ERROR"):
        """发送或读取异常后安全关闭UART，并安排稍后重开。

        待确认消息不会在这里清除，因为重新打开设备后仍需继续发送同一逻辑结果。
        close本身的异常被忽略，避免资源清理错误覆盖最初通信故障。
        """
        uart_object = self._uart
        self._uart = None
        if uart_object is not None:
            try:
                uart_object.close()
            except Exception:
                pass
        self._next_open_ms = int(now_ms) + self.REOPEN_INTERVAL_MS
        self._last_event_text = str(event_text)

    def _ensure_uart(self, now_ms):
        """到达重试时刻时调用硬件工厂；失败只更新状态并返回False。"""
        if self._uart is not None:
            return True
        if now_ms < self._next_open_ms:
            return False
        try:
            uart_object = self._uart_factory()
            if uart_object is None:
                raise RuntimeError("UART工厂返回None")
        except Exception:
            self._next_open_ms = now_ms + self.REOPEN_INTERVAL_MS
            self._last_event_text = "UART ERROR"
            return False
        self._uart = uart_object
        self._last_event_text = "UART READY"
        # 重连后立即发心跳，让F4无需等待旧周期即可确认链路。
        self._next_heartbeat_ms = now_ms
        return True

    def _write_complete(self, frame_bytes, now_ms):
        """执行一次UART写入，只有返回完整长度才视为成功。

        官方write返回实际发送字节数；负值、短写和异常都可能造成F4收到残帧，因此统一
        关闭设备并稍后重开。返回True表示整帧已经交给驱动，False表示调用方不得推进
        发送计数或重试阶段。
        """
        if self._uart is None:
            return False
        try:
            written = int(self._uart.write(frame_bytes))
        except Exception:
            self._close_uart_after_error(now_ms)
            return False
        if written != len(frame_bytes):
            self._close_uart_after_error(now_ms, "UART WRITE ERROR")
            return False
        return True

    def _send_heartbeat_if_due(self, now_ms, app_state):
        """到期时发送一个新心跳，不追赶主循环阻塞期间错过的多个周期。"""
        if self._uart is None or now_ms < self._next_heartbeat_ms:
            return
        payload = encode_heartbeat_payload(
            self._session_id,
            (now_ms - self._started_ms) & 0xFFFFFFFF,
            app_state,
            0,
        )
        sequence = self._next_sequence()
        frame = encode_frame(
            MSG_HEARTBEAT,
            payload,
            sequence=sequence,
            flags=FLAG_ACK_REQUIRED,
        )
        if self._write_complete(frame, now_ms):
            self._recent_heartbeats[sequence] = now_ms + self.LINK_TIMEOUT_MS
            self._next_heartbeat_ms = now_ms + self.HEARTBEAT_INTERVAL_MS

    def _send_pending_if_due(self, now_ms):
        """按稳定副本遍历所有到期可靠消息，避免ACK处理期间修改字典。"""
        if self._uart is None:
            return
        for pending in tuple(self._pending.values()):
            if self._uart is None:
                break
            if now_ms < pending.next_send_ms:
                continue
            flags = FLAG_ACK_REQUIRED
            if pending.send_count > 0:
                flags |= FLAG_RETRY
            frame = encode_frame(
                pending.message_type,
                pending.payload,
                sequence=pending.sequence,
                flags=flags,
            )
            if not self._write_complete(frame, now_ms):
                break
            pending.send_count += 1
            interval_ms = (
                self.FAST_RETRY_INTERVAL_MS
                if pending.send_count <= 3
                else self.SLOW_RETRY_INTERVAL_MS
            )
            pending.next_send_ms = now_ms + interval_ms

    def _event_prefix(self, message_type):
        """把可靠消息类型转换为屏幕使用的短事件名称。"""
        if message_type == MSG_PAPER_FRAME:
            return "A4"
        if message_type == MSG_PUZZLE_RESULT:
            return "RESULT"
        if message_type == MSG_HEARTBEAT:
            return "HEARTBEAT"
        return f"TYPE {message_type:02X}"

    def _handle_ack(self, frame, now_ms):
        """验证ACK是否匹配近期心跳或待确认消息，并更新链路与队列。

        随机TYPE/SEQ不会置在线。非零status证明链路存在但表示F4拒绝业务载荷，因此
        保留可靠消息继续重发；status=0才从队列移除。返回值表示ACK是否匹配。
        """
        try:
            ack = decode_ack_payload(frame.payload)
        except ValueError:
            return False
        acked_type = ack["acked_type"]
        acked_sequence = ack["acked_sequence"]
        status = ack["status"]
        pending_key = (acked_type, acked_sequence)

        matched_heartbeat = (
            acked_type == MSG_HEARTBEAT
            and acked_sequence in self._recent_heartbeats
        )
        pending = self._pending.get(pending_key)
        if not matched_heartbeat and pending is None:
            return False

        self._last_ack_ms = now_ms
        if matched_heartbeat:
            self._recent_heartbeats.pop(acked_sequence, None)
        if pending is not None:
            prefix = self._event_prefix(acked_type)
            if status == 0:
                self._pending.pop(pending_key, None)
                self._last_event_text = f"{prefix} ACK"
            else:
                self._last_event_text = f"{prefix} NACK {status}"
        return True

    def _read_available(self, now_ms):
        """执行一次官方非阻塞read，并解析本批次全部ACK帧。"""
        if self._uart is None:
            return
        try:
            data = self._uart.read()
        except Exception:
            self._close_uart_after_error(now_ms)
            return
        if not data:
            return
        try:
            frames = self._parser.feed(data)
        except ValueError:
            self._last_event_text = "UART RX ERROR"
            return
        for frame in frames:
            if frame.message_type == MSG_ACK:
                self._handle_ack(frame, now_ms)

    def _prune_heartbeat_sequences(self, now_ms):
        """删除超过链路窗口的旧心跳序号，限制长期运行内存。"""
        expired = [
            sequence
            for sequence, expires_ms in self._recent_heartbeats.items()
            if now_ms > expires_ms
        ]
        for sequence in expired:
            self._recent_heartbeats.pop(sequence, None)

    def _queue_reliable(self, message_type, payload, event_text):
        """替换同类型旧消息并创建一个立即到期的新可靠消息。"""
        now_ms = self._now_ms()
        for key in tuple(self._pending):
            if key[0] == message_type:
                self._pending.pop(key, None)
        sequence = self._next_sequence()
        pending = _PendingMessage(message_type, sequence, payload, now_ms)
        self._pending[(message_type, sequence)] = pending
        self._last_event_text = str(event_text)
        return sequence

    def queue_paper_frame(self, paper_orientation):
        """排队发送当前方向的完整A4毫米边界，返回分配的uint16序号。"""
        payload = encode_paper_payload(paper_orientation)
        return self._queue_reliable(MSG_PAPER_FRAME, payload, "A4 QUEUED")

    def queue_puzzle_result_once(
        self,
        mode,
        paper_orientation,
        placements,
        best_effort=False,
    ):
        """在当前采集上下文中最多排队一次完整拼图结果。

        best_effort=True时把“可能不准确”写入结果载荷bit0，但仍使用相同ACK、重发和
        单次上下文规则。返回True表示新建结果，False表示已经排过或输入无法编码。
        NaN、字段缺失和定点溢出会转换为RESULT ERROR，且不会消耗本次结果上下文。
        """
        if self._result_queued:
            return False
        try:
            payload = encode_puzzle_result_payload(
                mode,
                paper_orientation,
                placements,
                best_effort=best_effort,
            )
        except Exception:
            # 协议边界必须吸收任何规划对象结构异常；机械端不会收到半帧或错误目标。
            self._last_event_text = "RESULT ERROR"
            return False
        self._queue_reliable(MSG_PUZZLE_RESULT, payload, "RESULT QUEUED")
        self._result_queued = True
        return True

    def reset_result_context(self):
        """开始新采集上下文时取消旧结果，但保留用户手动A4帧和心跳状态。"""
        for key in tuple(self._pending):
            if key[0] == MSG_PUZZLE_RESULT:
                self._pending.pop(key, None)
        self._result_queued = False

    def poll(self, app_state=0):
        """推进一次非阻塞接收、心跳和可靠重发状态机。

        关键参数app_state为心跳携带的uint8界面状态。函数最多执行一次非阻塞read和
        有限次短帧write，不等待ACK、不sleep；UART打开、读写异常均转为内部状态，
        不抛入视觉主循环。已调用close后本函数直接返回。
        """
        if self._closed:
            return
        normalized_state = _validate_uint(app_state, 0xFF, "应用状态")
        now_ms = self._now_ms()
        if not self._ensure_uart(now_ms):
            return
        self._read_available(now_ms)
        if self._uart is None:
            return
        self._send_heartbeat_if_due(now_ms, normalized_state)
        self._send_pending_if_due(now_ms)
        self._prune_heartbeat_sequences(now_ms)

    @property
    def session_id(self):
        """返回本次视觉应用启动期间固定不变的非零uint32会话ID。"""
        return self._session_id

    @property
    def link_text(self):
        """返回屏幕使用的UART:OK、UART:OFFLINE或UART:ERROR短状态。"""
        if self._uart is None:
            if "ERROR" in self._last_event_text:
                return "UART:ERROR"
            return "UART:OFFLINE"
        now_ms = self._now_ms()
        if self._last_ack_ms is not None and now_ms - self._last_ack_ms <= self.LINK_TIMEOUT_MS:
            return "UART:OK"
        return "UART:OFFLINE"

    @property
    def last_event_text(self):
        """返回最近一次通信动作的短文本，供调参按钮反馈使用。"""
        return self._last_event_text

    @property
    def pending_count(self):
        """返回当前等待ACK的A4和结果逻辑消息总数。"""
        return len(self._pending)

    @property
    def pending_message_types(self):
        """返回按数值排序的待确认消息类型，主要用于状态和单元测试。"""
        return tuple(sorted(key[0] for key in self._pending))

    def close(self):
        """释放UART并永久停止本运行器后续重开尝试。"""
        if self._closed:
            return
        self._closed = True
        uart_object = self._uart
        self._uart = None
        if uart_object is not None:
            try:
                uart_object.close()
            except Exception:
                pass


class _MaixUart4Lease:
    """包装应用独占的UART4对象，并在关闭时恢复Maix默认通信监听。"""

    def __init__(self, uart_object, restore_listener):
        """保存底层UART和无参恢复回调；两者分别负责数据收发与系统资源归还。"""
        self._uart_object = uart_object
        self._restore_listener = restore_listener
        self._closed = False

    def read(self):
        """转发官方无参数非阻塞read调用，返回当前可用字节或空值。"""
        return self._uart_object.read()

    def write(self, data):
        """转发UART写入并返回底层驱动报告的实际字节数。"""
        return self._uart_object.write(data)

    def close(self):
        """只执行一次底层关闭，并在任何关闭结果下尝试恢复默认监听。"""
        if self._closed:
            return
        self._closed = True
        try:
            self._uart_object.close()
        finally:
            try:
                self._restore_listener()
            except Exception:
                # 应用退出或错误重开时，恢复监听失败不能覆盖最初的UART异常。
                pass


def create_maix_uart4():
    """按Sipeed官方MaixCAM2示例映射并打开UART4。

    主要流程：延迟导入maix模块，先停止占用UART4的默认Maix Comm监听，再映射
    A21/A22并以115200打开/dev/ttyS4。打开失败立即恢复监听；成功返回租约包装对象，
    其close会关闭UART并归还默认监听。默认构造参数即8N1、无流控。
    """
    from maix import comm, err, pinmap, uart

    comm.rm_default_comm_listener()
    try:
        err.check_raise(
            pinmap.set_pin_function("A21", "UART4_TX"),
            "Failed set pin A21 function to UART4_TX",
        )
        err.check_raise(
            pinmap.set_pin_function("A22", "UART4_RX"),
            "Failed set pin A22 function to UART4_RX",
        )
        uart_object = uart.UART("/dev/ttyS4", 115200)
    except Exception:
        try:
            comm.add_default_comm_listener()
        except Exception:
            # 原始打开异常决定现场诊断；恢复失败不能把它替换成次生异常。
            pass
        raise
    return _MaixUart4Lease(uart_object, comm.add_default_comm_listener)
