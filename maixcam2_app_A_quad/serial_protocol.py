"""MaixCAM2与STM32F4之间的UART4二进制协议和可靠发送状态机。"""

import math
import struct


# 通用帧常量集中定义，F4端必须使用完全相同的数值和小端字节序。
FRAME_SOF = b"\xAA\x55"
PROTOCOL_VERSION = 1
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


def encode_heartbeat_payload(uptime_ms, app_state, last_error=0):
    """编码6字节心跳载荷，运行毫秒数按uint32自然回绕。"""
    return struct.pack(
        "<IBB",
        _validate_uint(uptime_ms, 0xFFFFFFFF, "运行毫秒数"),
        _validate_uint(app_state, 0xFF, "应用状态"),
        _validate_uint(last_error, 0xFF, "通信错误码"),
    )


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


def encode_puzzle_result_payload(mode, paper_orientation, placements):
    """把一次成功规划的全部碎片编码到同一个结果载荷。

    关键参数placements中的每项需提供piece_id、source_center_mm、target_center_mm和
    rotation_delta_deg。记录先按piece_id排序，再生成从1开始的传输序号；碎片数必须
    为1至4。返回4+11*N字节载荷，任何非有限或溢出坐标都会在发送前抛出ValueError。
    """
    normalized_placements = sorted(
        list(placements),
        key=lambda placement: str(getattr(placement, "piece_id", "")),
    )
    piece_count = len(normalized_placements)
    if not 1 <= piece_count <= 4:
        raise ValueError("拼图结果碎片数必须位于1到4之间")

    payload = bytearray(
        struct.pack(
            "<BBBB",
            _mode_code(mode),
            _orientation_code(paper_orientation),
            piece_count,
            0,
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

    本函数用于PC回归和协议示例，不参与MaixCAM2发送热路径。载荷长度、保留字节、
    模式、方向和碎片数量不一致时抛出ValueError，防止F4示例接受畸形机械命令。
    """
    raw = bytes(payload)
    if len(raw) < 4:
        raise ValueError("拼图结果载荷过短")
    mode, orientation, piece_count, reserved = struct.unpack_from("<BBBB", raw, 0)
    if not 1 <= piece_count <= 4:
        raise ValueError("拼图结果碎片数必须位于1到4之间")
    if reserved != 0:
        raise ValueError("拼图结果保留字段必须为0")
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
