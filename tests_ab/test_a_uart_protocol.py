"""验证A版UART4二进制协议、可靠发送和Maix硬件适配边界。"""

from types import SimpleNamespace

import pytest

from maixcam2_app_A_quad.serial_protocol import (
    FLAG_ACK_REQUIRED,
    FRAME_SOF,
    MAX_PAYLOAD_LENGTH,
    MSG_ACK,
    MSG_HEARTBEAT,
    MSG_PAPER_FRAME,
    MSG_PUZZLE_RESULT,
    PROTOCOL_VERSION,
    FrameStreamParser,
    crc16_ccitt_false,
    decode_ack_payload,
    decode_paper_payload,
    decode_puzzle_result_payload,
    encode_ack_payload,
    encode_frame,
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


def test_crc16_matches_ccitt_false_standard_vector():
    """标准向量必须得到0x29B1，防止两端采用不同CRC变体。"""
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_frame_encoding_uses_little_endian_header_length_and_crc():
    """通用帧头、序号、长度和CRC必须完全符合协议文档的字节顺序。"""
    frame = encode_frame(
        MSG_PAPER_FRAME,
        b"\x01\x02",
        sequence=0x1234,
        flags=FLAG_ACK_REQUIRED,
    )

    assert frame[:9] == b"\xAA\x55\x01\x10\x01\x34\x12\x02\x00"
    assert int.from_bytes(frame[-2:], "little") == crc16_ccitt_false(frame[2:-2])


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
