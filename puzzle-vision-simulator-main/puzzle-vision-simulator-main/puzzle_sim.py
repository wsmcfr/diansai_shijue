#!/usr/bin/env python3
"""E题拼图装置：随机切割、随机摆放、视觉识别和矩形还原仿真。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import cv2
import numpy as np


CANVAS_W, CANVAS_H = 900, 1200
DIVIDER_Y = 575
PIXELS_PER_CM = 40.0
CARD_WIDTH_CM, CARD_HEIGHT_CM = 10.0, 6.0
CARD_W = CARD_WIDTH_CM * PIXELS_PER_CM
CARD_H = CARD_HEIGHT_CM * PIXELS_PER_CM
TARGET_Y = 790
PIECE_BGR = (245, 245, 245)
PAPER_BGR = (18, 18, 18)
CARD_ASSET = Path(__file__).resolve().parent / "mujoco_sim/assets/cards/big_joker.png"


def _clip_half_plane(poly: np.ndarray, point: np.ndarray,
                     normal: np.ndarray, keep_positive: bool) -> np.ndarray:
    """Clip a convex polygon by one side of an infinite straight cut."""
    result = []
    signed = (poly - point) @ normal
    if not keep_positive:
        signed = -signed
    for i, current in enumerate(poly):
        previous = poly[i - 1]
        current_d, previous_d = signed[i], signed[i - 1]
        current_in, previous_in = current_d >= -1e-7, previous_d >= -1e-7
        if current_in != previous_in:
            t = previous_d / (previous_d - current_d)
            result.append(previous + t * (current - previous))
        if current_in:
            result.append(current)
    return np.asarray(result, dtype=float)


def _valid_piece(poly: np.ndarray) -> bool:
    if not 3 <= len(poly) <= 5:
        return False
    lengths = np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1)
    return (lengths.min() >= 2.0 * PIXELS_PER_CM and
            abs(cv2.contourArea(poly.astype(np.float32))) >= 6500)


def _edge_on_target_boundary(a: np.ndarray, b: np.ndarray,
                             tolerance: float = 1e-6) -> bool:
    return (
        (abs(a[0]) <= tolerance and abs(b[0]) <= tolerance)
        or (abs(a[0] - CARD_W) <= tolerance
            and abs(b[0] - CARD_W) <= tolerance)
        or (abs(a[1]) <= tolerance and abs(b[1]) <= tolerance)
        or (abs(a[1] - CARD_H) <= tolerance
            and abs(b[1] - CARD_H) <= tolerance)
    )


def validate_cut_layout(pieces: list[np.ndarray]) -> None:
    """Enforce all three field-piece geometry requirements."""
    for index, piece in enumerate(pieces):
        if not 3 <= len(piece) <= 5:
            raise RuntimeError(f"P{index} 边数为 {len(piece)}，要求3～5边")
        edge_list = edges(piece)
        lengths = [np.linalg.norm(b - a) for a, b in edge_list]
        if min(lengths) < 2.0 * PIXELS_PER_CM - 1e-6:
            raise RuntimeError(
                f"P{index} 最短边 {min(lengths) / PIXELS_PER_CM:.3f}cm，小于2cm")
        if not any(_edge_on_target_boundary(a, b) for a, b in edge_list):
            raise RuntimeError(f"P{index} 没有位于目标矩形外边上的边")


def _common_vertex_cut(rng: np.random.Generator,
                       piece_count: int) -> list[np.ndarray]:
    """Original proven radial/common-vertex layouts."""
    tl, tr = np.array([0., 0.]), np.array([CARD_W, 0.])
    br, bl = np.array([CARD_W, CARD_H]), np.array([0., CARD_H])
    if piece_count == 1:
        return [np.array([tl, tr, br, bl])]
    if piece_count == 2:
        t = np.array([rng.uniform(.25, .75) * CARD_W, 0.0])
        b = np.array([rng.uniform(.25, .75) * CARD_W, CARD_H])
        return [np.array([tl, t, b, bl]), np.array([t, tr, br, b])]
    if piece_count == 3:
        c = np.array([rng.uniform(.43, .57) * CARD_W,
                      rng.uniform(.34, .48) * CARD_H])
        t = np.array([rng.uniform(.32, .68) * CARD_W, 0.0])
        # Keep both resulting outer vertical segments at least 2 cm.
        r = np.array([CARD_W, rng.uniform(.50, .65) * CARD_H])
        l = np.array([0.0, rng.uniform(.50, .65) * CARD_H])
        return [
            np.array([t, tr, r, c]),
            np.array([r, br, bl, l, c]),
            np.array([l, tl, t, c]),
        ]
    c = np.array([rng.uniform(.38, .62) * CARD_W,
                  rng.uniform(.35, .65) * CARD_H])
    t = np.array([rng.uniform(.20, .80) * CARD_W, 0.0])
    r = np.array([CARD_W, rng.uniform(1/3, 2/3) * CARD_H])
    b = np.array([rng.uniform(.20, .80) * CARD_W, CARD_H])
    l = np.array([0.0, rng.uniform(1/3, 2/3) * CARD_H])
    return [
        np.array([tl, t, c, l]),
        np.array([t, tr, r, c]),
        np.array([c, r, br, b]),
        np.array([l, c, b, bl]),
    ]


def _parallel_non_common_cut(rng: np.random.Generator,
                             piece_count: int) -> list[np.ndarray]:
    """Parallel full-edge cuts with no shared global vertex."""
    if piece_count == 1:
        return [np.array([
            [0., 0.], [CARD_W, 0.], [CARD_W, CARD_H], [0., CARD_H]])]
    # Four horizontal strips cannot satisfy the official 2 cm minimum edge
    # inside a 6 cm height, so four-piece layouts use vertical/oblique strips.
    vertical = piece_count == 4 or bool(rng.integers(0, 2))
    span = CARD_W if vertical else CARD_H
    minimum = 2.0 * PIXELS_PER_CM
    remainder = span - piece_count * minimum

    def boundaries():
        if remainder <= 1e-8:
            widths = np.full(piece_count, minimum)
        else:
            widths = minimum + rng.dirichlet(
                np.full(piece_count, 2.5)) * remainder
        return np.r_[0.0, np.cumsum(widths)]

    first, second = boundaries(), boundaries()
    pieces = []
    for i in range(piece_count):
        if vertical:
            pieces.append(np.array([
                [first[i], 0.0], [first[i + 1], 0.0],
                [second[i + 1], CARD_H], [second[i], CARD_H]]))
        else:
            pieces.append(np.array([
                [0.0, first[i]], [CARD_W, second[i]],
                [CARD_W, second[i + 1]], [0.0, first[i + 1]]]))
    return pieces


def _equal_rectangle_cut(piece_count: int) -> list[np.ndarray]:
    """Equal rectangular tiles, including the ambiguous four-piece 2×2 case."""
    if piece_count == 4:
        mid_x, mid_y = CARD_W / 2, CARD_H / 2
        return [
            np.array([[0., 0.], [mid_x, 0.], [mid_x, mid_y], [0., mid_y]]),
            np.array([[mid_x, 0.], [CARD_W, 0.],
                      [CARD_W, mid_y], [mid_x, mid_y]]),
            np.array([[0., mid_y], [mid_x, mid_y],
                      [mid_x, CARD_H], [0., CARD_H]]),
            np.array([[mid_x, mid_y], [CARD_W, mid_y],
                      [CARD_W, CARD_H], [mid_x, CARD_H]]),
        ]
    # For 1–3 pieces, equal vertical rectangles keep every edge >=2 cm.
    bounds = np.linspace(0., CARD_W, piece_count + 1)
    return [
        np.array([[bounds[i], 0.], [bounds[i + 1], 0.],
                  [bounds[i + 1], CARD_H], [bounds[i], CARD_H]])
        for i in range(piece_count)
    ]


def _boundary_fan_cut(rng: np.random.Generator,
                      piece_count: int) -> list[np.ndarray]:
    """Rays share one point on the top boundary rather than an interior point."""
    if piece_count <= 2:
        return _common_vertex_cut(rng, piece_count)
    center = np.array([rng.uniform(.42, .58) * CARD_W, 0.])
    minimum = 2.0 * PIXELS_PER_CM
    remainder = CARD_W - piece_count * minimum
    widths = (np.full(piece_count, minimum) if remainder <= 1e-8 else
              minimum + rng.dirichlet(np.full(piece_count, 2.5)) * remainder)
    bottom = np.r_[0.0, np.cumsum(widths)]
    cuts = [np.array([x, CARD_H]) for x in bottom[1:-1]]
    pieces = [
        np.array([[0., 0.], center, cuts[0], [0., CARD_H]])
    ]
    for left, right in zip(cuts, cuts[1:]):
        pieces.append(np.array([center, right, left]))
    pieces.append(np.array([
        center, [CARD_W, 0.], [CARD_W, CARD_H], cuts[-1]]))
    return pieces


def _t_junction_cut(rng: np.random.Generator,
                    piece_count: int) -> list[np.ndarray]:
    """Hierarchical T cuts, including long-edge to short-edge adjacency."""
    if piece_count <= 2:
        return _common_vertex_cut(rng, piece_count)
    top = rng.uniform(.42, .58) * CARD_W
    bottom = rng.uniform(.42, .58) * CARD_W
    p0, p1 = np.array([top, 0.]), np.array([bottom, CARD_H])
    if piece_count == 3:
        t = rng.uniform(.34, .66)
        junction = p0 * (1 - t) + p1 * t
        right = np.array([CARD_W, rng.uniform(.34, .66) * CARD_H])
        return [
            np.array([[0., 0.], p0, p1, [0., CARD_H]]),
            np.array([p0, [CARD_W, 0.], right, junction]),
            np.array([junction, right, [CARD_W, CARD_H], p1]),
        ]
    left_t = rng.uniform(.34, .54)
    right_t = rng.uniform(.46, .66)
    left_junction = p0 * (1 - left_t) + p1 * left_t
    right_junction = p0 * (1 - right_t) + p1 * right_t
    left = np.array([0., rng.uniform(.34, .60) * CARD_H])
    right = np.array([CARD_W, rng.uniform(.40, .66) * CARD_H])
    return [
        np.array([[0., 0.], p0, left_junction, left]),
        np.array([left, left_junction, p1, [0., CARD_H]]),
        np.array([p0, [CARD_W, 0.], right, right_junction]),
        np.array([right_junction, right, [CARD_W, CARD_H], p1]),
    ]


def _corner_polygon_cut(rng: np.random.Generator,
                        piece_count: int) -> list[np.ndarray]:
    """Opposite corner triangles plus central quadrilateral/pentagon pieces."""
    if piece_count <= 2:
        return _common_vertex_cut(rng, piece_count)
    a = np.array([rng.uniform(.20, .28) * CARD_W, 0.])
    b = np.array([0., rng.uniform(.35, .52) * CARD_H])
    e = np.array([rng.uniform(.52, .60) * CARD_W, 0.])
    f = np.array([rng.uniform(.40, .48) * CARD_W, CARD_H])
    if piece_count == 3:
        return [
            np.array([[0., 0.], a, b]),
            np.array([a, e, f, [0., CARD_H], b]),
            np.array([e, [CARD_W, 0.], [CARD_W, CARD_H], f]),
        ]
    c = np.array([CARD_W, rng.uniform(.48, .65) * CARD_H])
    d = np.array([rng.uniform(.72, .80) * CARD_W, CARD_H])
    return [
        np.array([[0., 0.], a, b]),
        np.array([a, e, f, [0., CARD_H], b]),
        np.array([e, [CARD_W, 0.], c, d, f]),
        np.array([c, [CARD_W, CARD_H], d]),
    ]


def _concave_polyline_cut(rng: np.random.Generator,
                          piece_count: int) -> list[np.ndarray]:
    """A bent cut producing one legal concave pentagon.

    The internal boundary R→J→BL is a two-segment polyline.  Additional
    pieces fan from J to independently spaced points on the bottom edge.
    Every result has 3–5 sides, a >=2 cm outside edge and >=2 cm edges.
    """
    if piece_count == 1:
        return _common_vertex_cut(rng, 1)
    tl, tr = np.array([0., 0.]), np.array([CARD_W, 0.])
    br, bl = np.array([CARD_W, CARD_H]), np.array([0., CARD_H])
    # Starting the polyline at the top-right corner yields a concave
    # quadrilateral; starting lower on the right edge yields a pentagon.
    right = (
        np.array([CARD_W, 0.])
        if bool(rng.integers(0, 2))
        else np.array([CARD_W, rng.uniform(.35, .47) * CARD_H])
    )
    junction = np.array([
        rng.uniform(.43, .60) * CARD_W,
        # A deep, camera-resolvable re-entrant corner.  Keeping it above the
        # right boundary endpoint also avoids a near-collinear "fake" notch.
        rng.uniform(.08, .14) * CARD_H,
    ])
    concave = (
        np.array([tl, tr, junction, bl])
        if np.allclose(right, tr)
        else np.array([tl, tr, right, junction, bl])
    )
    pieces = [concave]

    lower_count = piece_count - 1
    minimum = 2.0 * PIXELS_PER_CM
    remainder = CARD_W - lower_count * minimum
    widths = (np.full(lower_count, minimum) if remainder <= 1e-8 else
              minimum + rng.dirichlet(np.full(lower_count, 3.0)) * remainder)
    bounds = np.r_[0., np.cumsum(widths)]
    # Walk intervals from right to left to preserve the boundary winding.
    for interval in range(lower_count - 1, -1, -1):
        lo, hi = bounds[interval], bounds[interval + 1]
        if interval == lower_count - 1:
            poly = [right, br, np.array([lo, CARD_H]), junction]
        elif interval == 0:
            poly = [np.array([hi, CARD_H]), bl, junction]
        else:
            poly = [
                np.array([hi, CARD_H]),
                np.array([lo, CARD_H]),
                junction,
            ]
        pieces.append(np.asarray(poly, dtype=float))
    return pieces


def random_cut(rng: np.random.Generator, piece_count: int = 4,
               cut_mode: str = "sequential") -> list[np.ndarray]:
    """Make 1-4 pieces by independent sequential cuts.

    Unlike a radial-only generator, this produces parallel cuts, T junctions,
    branch cuts and mixed layouts. No global common vertex is assumed.
    """
    if not 2 <= piece_count <= 4:
        raise ValueError("切割后的碎片数量必须在 2～4 之间")
    aliases = {"sequential": "strips"}
    cut_mode = aliases.get(cut_mode, cut_mode)
    allowed = {
        "common", "boundary_fan", "strips", "equal_rectangles",
        "t_junction", "corner", "concave", "mixed"}
    if cut_mode not in allowed:
        raise ValueError(f"未知切割方式：{cut_mode}")
    if cut_mode == "mixed":
        choices = ["common", "boundary_fan", "strips", "equal_rectangles"]
        if piece_count >= 3:
            choices += ["t_junction", "corner"]
        if piece_count >= 2:
            choices += ["concave"]
        cut_mode = str(rng.choice(choices))
    if cut_mode == "common":
        pieces = _common_vertex_cut(rng, piece_count)
    elif cut_mode == "boundary_fan":
        pieces = _boundary_fan_cut(rng, piece_count)
    elif cut_mode == "strips":
        pieces = _parallel_non_common_cut(rng, piece_count)
    elif cut_mode == "equal_rectangles":
        pieces = _equal_rectangle_cut(piece_count)
    elif cut_mode == "t_junction":
        pieces = _t_junction_cut(rng, piece_count)
    elif cut_mode == "concave":
        pieces = _concave_polyline_cut(rng, piece_count)
    else:
        pieces = _corner_polygon_cut(rng, piece_count)
    validate_cut_layout(pieces)
    return pieces


def resolve_cut_mode(seed: int, piece_count: int, cut_mode: str) -> str:
    """Resolve the deterministic category used by a mixed-mode scene."""
    cut_mode = {"sequential": "strips"}.get(cut_mode, cut_mode)
    if cut_mode != "mixed":
        return cut_mode
    choices = ["common", "boundary_fan", "strips", "equal_rectangles"]
    if piece_count >= 3:
        choices += ["t_junction", "corner"]
    if piece_count >= 2:
        choices += ["concave"]
    return str(np.random.default_rng(seed).choice(choices))


def rigid(angle: float, tx: float, ty: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def apply_h(points: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = np.c_[points, np.ones(len(points))] @ h.T
    return q[:, :2] / q[:, 2, None]


def place_randomly(polys: list[np.ndarray], rng: np.random.Generator):
    placed = [None] * len(polys)
    occupancy = np.zeros((DIVIDER_Y - 25, CANVAS_W), np.uint8)
    # Place large pieces first so that 3-piece layouts do not trap the largest
    # sector in the remaining narrow gaps.
    ordered = sorted(range(len(polys)),
                     key=lambda i: abs(cv2.contourArea(polys[i].astype(np.float32))),
                     reverse=True)
    for index in ordered:
        poly = polys[index]
        centroid = poly.mean(axis=0)
        local = poly - centroid
        accepted = False
        for _ in range(2000):
            angle = rng.uniform(-math.pi, math.pi)
            rot = rigid(angle, 0, 0)
            rotated = apply_h(local, rot)
            mn, mx = rotated.min(0), rotated.max(0)
            tx = rng.uniform(25 - mn[0], CANVAS_W - 25 - mx[0])
            ty = rng.uniform(25 - mn[1], DIVIDER_Y - 35 - mx[1])
            scene_poly = rotated + [tx, ty]
            mask = np.zeros_like(occupancy)
            cv2.fillPoly(mask, [np.round(scene_poly).astype(np.int32)], 255)
            dilated = cv2.dilate(mask, np.ones((19, 19), np.uint8))
            if not np.any((dilated > 0) & (occupancy > 0)):
                cv2.fillPoly(occupancy, [np.round(scene_poly).astype(np.int32)], 255)
                placed[index] = scene_poly
                accepted = True
                break
        if not accepted:
            raise RuntimeError("无法在上半区无重叠摆放碎片，请更换随机种子")
    return placed


def _joker_card() -> np.ndarray:
    card = cv2.imread(str(CARD_ASSET), cv2.IMREAD_COLOR)
    if card is None:
        raise RuntimeError(f"无法读取大鬼扑克牌素材：{CARD_ASSET}")
    return cv2.resize(card, (int(CARD_W), int(CARD_H)),
                      interpolation=cv2.INTER_AREA)


def render_scene(placed: list[np.ndarray], source: list[np.ndarray] | None = None,
                 material_mode: str = "color") -> np.ndarray:
    image = np.full((CANVAS_H, CANVAS_W, 3), PAPER_BGR, np.uint8)
    cv2.line(image, (0, DIVIDER_Y), (CANVAS_W, DIVIDER_Y), (210, 210, 210), 4)
    target_x = int((CANVAS_W - CARD_W) / 2)
    target_y = TARGET_Y
    cv2.rectangle(image, (target_x, target_y),
                  (int(target_x + CARD_W), int(target_y + CARD_H)), (190, 190, 190), 2)
    cv2.putText(image, "Target: 10 cm x 6 cm", (target_x, target_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (220, 220, 220), 2, cv2.LINE_AA)
    joker = _joker_card() if material_mode == "joker" else None
    for index, poly in enumerate(placed):
        pts = np.round(poly).astype(np.int32)
        if joker is None or source is None:
            cv2.fillPoly(image, [pts], PIECE_BGR, lineType=cv2.LINE_8)
        else:
            src = source[index].astype(np.float32)
            dst = poly.astype(np.float32)
            transform = cv2.getAffineTransform(src[:3], dst[:3])
            warped = cv2.warpAffine(
                joker, transform, (CANVAS_W, CANVAS_H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            mask = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
            cv2.fillPoly(mask, [pts], 255)
            image[mask > 0] = warped[mask > 0]
        # Use a darker color of the same hue so the visual border remains part
        # of the segmented foreground instead of clipping polygon corners.
        cv2.polylines(image, [pts], True, (120, 120, 120), 2, cv2.LINE_AA)
    return image


def generate_camera_frame(seed: int, piece_count: int,
                          material_mode: str = "color",
                          cut_mode: str = "sequential") -> np.ndarray:
    """Generation boundary: return pixels only, never geometry or true poses."""
    rng = np.random.default_rng(seed)
    source_polygons = random_cut(rng, piece_count, cut_mode)
    placed_polygons = place_randomly(source_polygons, rng)
    rendered = render_scene(
        placed_polygons, source_polygons, material_mode=material_mode)

    # Simulate a camera/file boundary. Geometry variables stay local to this
    # function and the vision stage receives only decoded image pixels.
    ok, encoded = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("仿真相机图像编码失败")
    camera_frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if camera_frame is None:
        raise RuntimeError("仿真相机图像解码失败")
    return camera_frame


def order_clockwise(vertices: np.ndarray) -> np.ndarray:
    c = vertices.mean(axis=0)
    a = np.arctan2(vertices[:, 1] - c[1], vertices[:, 0] - c[0])
    return vertices[np.argsort(a)]


def detect_pieces(image: np.ndarray, expected_count: int | None = None,
                  preserve_concavity: bool = False) -> list[np.ndarray]:
    roi = image[:DIVIDER_Y]
    # The official setup uses white/printed pieces on black A4 paper. Estimate
    # the dominant work-plane brightness on every frame so MuJoCo lighting
    # gradients and real camera exposure changes do not require a fixed value.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    paper_level = float(np.median(gray))
    threshold = max(70.0, paper_level + 38.0)
    mask = np.where(gray > threshold, 255, 0).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pieces = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 3000:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if x <= 1 or y <= 1 or x + w >= roi.shape[1] - 1 or y + h >= roi.shape[0] - 1:
            # Ignore the visible table/camera-rig strips outside the A4 sheet.
            continue
        # For concave mode the external contour order must be retained: sorting
        # vertices around the centroid or taking a convex hull fills the notch.
        # The supplied Joker has a white perimeter, so texture holes remain
        # interior contours and do not alter this outer boundary.
        boundary = cnt if preserve_concavity else cv2.convexHull(cnt)
        peri = cv2.arcLength(boundary, True)
        approx = cv2.approxPolyDP(
            boundary, 0.025 * peri, True).reshape(-1, 2).astype(float)
        if 3 <= len(approx) <= 5:
            pieces.append(approx if preserve_concavity
                          else order_clockwise(approx))
    pieces.sort(key=lambda p: (p.mean(0)[1], p.mean(0)[0]))
    if not 2 <= len(pieces) <= 4:
        raise RuntimeError(f"视觉检测到 {len(pieces)} 块碎片，要求为 2～4 块")
    if expected_count is not None and len(pieces) != expected_count:
        raise RuntimeError(f"视觉检测到 {len(pieces)} 块碎片，设置数量为 {expected_count}")
    return pieces


def edges(poly: np.ndarray):
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]


def align_edge(src_a, src_b, dst_a, dst_b) -> np.ndarray:
    """Rigid transform mapping src_a->dst_a and src_b->dst_b."""
    u, v = src_b - src_a, dst_b - dst_a
    angle = math.atan2(v[1], v[0]) - math.atan2(u[1], u[0])
    r = rigid(angle, 0, 0)
    mapped = apply_h(np.array([src_a]), r)[0]
    r[:2, 2] = dst_a - mapped
    return r


def optimize_pose_graph(pieces, matches, initial):
    """Globally distribute closed-loop edge error over all movable pieces."""
    if len(pieces) < 3:
        return initial

    def pack(poses):
        values = []
        for h in poses[1:]:
            values.extend([math.atan2(h[1, 0], h[0, 0]), h[0, 2], h[1, 2]])
        return np.asarray(values, dtype=float)

    def unpack(x):
        poses = [initial[0]]
        for k in range(len(pieces) - 1):
            theta, tx, ty = x[3 * k:3 * k + 3]
            poses.append(rigid(theta, tx, ty))
        return poses

    def residual(x):
        poses = unpack(x)
        values = []
        for match in matches:
            _, i, _ei, j, _ej = match[:5]
            ia, ib, ja, jb = match_segments(pieces, match)
            wi = apply_h(np.array([ia, ib]), poses[i])
            wj = apply_h(np.array([jb, ja]), poses[j])
            values.extend((wi - wj).ravel())
        return np.asarray(values)

    x = pack(initial)
    for _ in range(20):
        r0 = residual(x)
        jac = np.empty((len(r0), len(x)))
        for k in range(len(x)):
            step = 1e-5 if k % 3 == 0 else 1e-3
            shifted = x.copy()
            shifted[k] += step
            jac[:, k] = (residual(shifted) - r0) / step
        delta, *_ = np.linalg.lstsq(jac, -r0, rcond=None)
        x += delta
        if np.linalg.norm(delta) < 1e-7:
            break
    return unpack(x)


def candidate_matchings(pieces: list[np.ndarray]):
    all_edges = {(i, e): edge for i, p in enumerate(pieces) for e, edge in enumerate(edges(p))}
    candidates = []
    for (i, ei), (j, ej) in itertools.combinations(all_edges, 2):
        if i == j:
            continue
        a, b = all_edges[(i, ei)]
        c, d = all_edges[(j, ej)]
        la, lb = np.linalg.norm(b - a), np.linalg.norm(d - c)
        rel = abs(la - lb) / max(la, lb)
        # Rasterized shallow vertices (especially on legal five-sided pieces)
        # can shift an endpoint by several pixels, so use a tolerant shortlist;
        # final selection is decided by closure, overlap and rectangle fill.
        if rel < 0.12:
            candidates.append((rel, i, ei, j, ej, 0., 1., 0., 1.))
        ratio = min(la, lb) / max(la, lb)
        if 0.22 <= ratio <= 0.88:
            # T junction: the shorter complete edge can occupy either end of
            # a longer collinear edge. The global rectangle score resolves
            # which endpoint is the physical junction.
            # Full-edge evidence is more reliable. Keep partial candidates
            # behind all plausible full matches, then let rectangle coverage
            # select them only when a T junction genuinely requires one.
            penalty = 0.15
            if la > lb:
                candidates.extend([
                    (penalty, i, ei, j, ej, 0., ratio, 0., 1.),
                    (penalty, i, ei, j, ej, 1. - ratio, 1., 0., 1.),
                ])
            else:
                candidates.extend([
                    (penalty, i, ei, j, ej, 0., 1., 0., ratio),
                    (penalty, i, ei, j, ej, 0., 1., 1. - ratio, 1.),
                ])
    candidates.sort()
    # Keep ambiguity bounded while preserving all near-exact cut-edge matches.
    return candidates[:80]


def match_segments(pieces, match):
    """Return the two full or partial edge segments encoded by a candidate."""
    _, i, ei, j, ej, ia0, ia1, ja0, ja1 = match
    a, b = edges(pieces[i])[ei]
    c, d = edges(pieces[j])[ej]
    return (a + (b - a) * ia0, a + (b - a) * ia1,
            c + (d - c) * ja0, c + (d - c) * ja1)


def matching_sets(pieces: list[np.ndarray], cut_mode: str = "auto"):
    count = len(pieces)
    if count == 1:
        yield ()
        return
    cand = candidate_matchings(pieces)
    # Every sequential straight cut adds one adjacency edge, so a connected
    # spanning tree (N-1 matches) is sufficient. Additional junction contacts
    # are evaluated implicitly by overlap and rectangular fill quality.
    pair_count = (
        count if ((cut_mode == "common" and count >= 3)
                  or (cut_mode == "concave" and count >= 2))
        else count - 1)
    full = [m for m in cand if tuple(m[5:]) == (0., 1., 0., 1.)]
    partial = [m for m in cand if tuple(m[5:]) != (0., 1., 0., 1.)]
    if cut_mode == "t_junction" and count >= 3:
        combos = (
            tuple(base) + (part,)
            for base in itertools.combinations(full, pair_count - 1)
            for part in partial
        )
    elif cut_mode in {
            "common", "boundary_fan", "strips", "corner",
            "concave", "equal_rectangles", "sequential"}:
        combos = itertools.combinations(full, pair_count)
    else:
        combos = itertools.chain(
            itertools.combinations(full, pair_count),
            (tuple(base) + (part,)
             for base in itertools.combinations(full, pair_count - 1)
             for part in partial))
    for combo in combos:
        used, degree = set(), [0] * count
        ok = True
        graph = [set() for _ in range(count)]
        for match in combo:
            _, i, ei, j, ej = match[:5]
            if (i, ei) in used or (j, ej) in used:
                ok = False
                break
            used |= {(i, ei), (j, ej)}
            degree[i] += 1
            degree[j] += 1
            graph[i].add(j)
            graph[j].add(i)
        if not ok or any(d == 0 for d in degree):
            continue
        if cut_mode == "common" and count >= 3 and any(d != 2 for d in degree):
            continue
        seen, stack = {0}, [0]
        while stack:
            for j in graph[stack.pop()]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(seen) == count:
            yield combo


def assemble_from_matches(pieces, matches):
    adjacency = [[] for _ in pieces]
    for match in matches:
        _, i, _ei, j, _ej = match[:5]
        adjacency[i].append((j, match, False))
        adjacency[j].append((i, match, True))
    transforms = [None] * len(pieces)
    transforms[0] = np.eye(3)
    stack = [0]
    closure_error = 0.0
    while stack:
        i = stack.pop()
        for j, match, reversed_sides in adjacency[i]:
            ia, ib, ja, jb = match_segments(pieces, match)
            if reversed_sides:
                ia, ib, ja, jb = ja, jb, ia, ib
            wa, wb = apply_h(np.array([ia, ib]), transforms[i])
            proposed = align_edge(ja, jb, wb, wa)  # cutting edges meet in reverse order
            if transforms[j] is None:
                transforms[j] = proposed
                stack.append(j)
            else:
                closure_error += np.linalg.norm(apply_h(pieces[j], proposed) -
                                                apply_h(pieces[j], transforms[j]), axis=1).mean()
    assembled = [apply_h(p, h) for p, h in zip(pieces, transforms)]
    # Quality: low overlap, compact rectangular union, and graph closure.
    allp = np.vstack(assembled)
    mn, mx = allp.min(0), allp.max(0)
    scale = 1.0
    shift = -mn + 10
    w, h = np.ceil(mx - mn + 20).astype(int)
    masks = []
    for p in assembled:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.round(p * scale + shift).astype(np.int32)], 1)
        masks.append(m)
    total = sum(masks)
    overlap = float(np.count_nonzero(total > 1))
    union = (total > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(union, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(cnt)
    rw, rh = rect[1]
    rect_area = rw * rh
    union_area = float(np.count_nonzero(union))
    expected_area = CARD_W * CARD_H
    fill_error = max(0.0, rect_area - union_area)
    aspect = max(rw, rh) / max(1.0, min(rw, rh))
    expected_aspect = CARD_W / CARD_H
    aspect_error = abs(math.log(max(aspect, 1e-6) / expected_aspect))
    disconnected_area = sum(cv2.contourArea(c) for c in cnts) - cv2.contourArea(cnt)
    perimeter_error = abs(cv2.arcLength(cnt, True) -
                          2.0 * (CARD_W + CARD_H))
    match_error = sum(x[0] for x in matches) * 5000
    # A correct solution must cover one 10x6 rectangle. These global terms
    # reject locally plausible equal-length outer-edge matches that otherwise
    # fold pieces over each other or create an L-shaped/oversized assembly.
    score = (
        closure_error * 8
        + overlap * 12
        + fill_error * 8
        + abs(union_area - expected_area) * 4
        + abs(rect_area - expected_area) * 3
        + aspect_error * 80000
        + disconnected_area * 20
        + perimeter_error * 25
        + match_error
    )
    return score, transforms, assembled


def solve(pieces: list[np.ndarray], cut_mode: str = "auto"):
    if cut_mode == "equal_rectangles":
        count = len(pieces)
        if count == 4:
            cell_w, cell_h = CARD_W / 2, CARD_H / 2
            slots = [(0., 0.), (cell_w, 0.),
                     (0., cell_h), (cell_w, cell_h)]
        else:
            cell_w, cell_h = CARD_W / count, CARD_H
            slots = [(i * cell_w, 0.) for i in range(count)]
        target_origin = np.array([
            (CANVAS_W - CARD_W) / 2, float(TARGET_Y)])
        transforms = []
        for piece, slot in zip(pieces, slots):
            best = None
            for a, b in edges(piece):
                vector = b - a
                angle = -math.atan2(vector[1], vector[0])
                rotation = rigid(angle, 0., 0.)
                rotated = apply_h(piece, rotation)
                low, high = rotated.min(axis=0), rotated.max(axis=0)
                size = high - low
                cost = abs(size[0] - cell_w) + abs(size[1] - cell_h)
                if best is None or cost < best[0]:
                    best = (cost, rotation, low)
            _, rotation, low = best
            destination = target_origin + np.asarray(slot)
            translation = rigid(0., *(destination - low))
            transforms.append(translation @ rotation)
        # Identical blank rectangles have no observable identity. Any bijection
        # between detected pieces and equal target cells is a correct solution.
        return transforms, ()

    best = None
    for matches in matching_sets(pieces, cut_mode):
        result = assemble_from_matches(pieces, matches)
        if best is None or result[0] < best[0]:
            best = (*result, matches)
    if best is None:
        raise RuntimeError("未找到满足边长配对与碎片邻接关系的拼接")
    _, transforms, assembled, matches = best
    # Candidate search uses the inexpensive propagated poses. Only the winning
    # topology receives global nonlinear refinement.
    transforms = optimize_pose_graph(pieces, matches, transforms)
    assembled = [apply_h(p, h) for p, h in zip(pieces, transforms)]

    # Normalize recovered rectangle to the requested lower-half target.
    allp = np.vstack(assembled).astype(np.float32)
    center, size, angle = cv2.minAreaRect(allp)
    if size[0] < size[1]:
        angle += 90.0
    normalize = rigid(math.radians(-angle), 0, 0)
    rotated = apply_h(allp, normalize)
    mn, mx = rotated.min(0), rotated.max(0)
    if (mx - mn)[0] < (mx - mn)[1]:
        normalize = rigid(math.radians(90) - math.radians(angle), 0, 0)
        rotated = apply_h(allp, normalize)
        mn, mx = rotated.min(0), rotated.max(0)
    target_origin = np.array([(CANVAS_W - CARD_W) / 2, float(TARGET_Y)])
    translate = rigid(0, *(target_origin - mn))
    final = [translate @ normalize @ h for h in transforms]
    return final, matches


def analyze_camera_frame(image: np.ndarray, cut_mode: str = "auto"):
    """Pure vision/planning boundary: input is an image, output is detected poses."""
    detected = detect_pieces(
        image, preserve_concavity=(cut_mode == "concave"))
    transforms, matches = solve(detected, cut_mode)
    return detected, transforms, matches


def annotate_detection(image, pieces):
    out = image.copy()
    for i, p in enumerate(pieces):
        pts = np.round(p).astype(np.int32)
        cv2.polylines(out, [pts], True, (0, 220, 255), 3)
        for k, pt in enumerate(pts):
            cv2.circle(out, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(out, str(k), tuple(pt + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 160), 1, cv2.LINE_AA)
        c = np.round(p.mean(0)).astype(int)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .8,
                    (0, 0, 0), 1, cv2.LINE_AA)
    return out


def render_solution(image, pieces, transforms, preserve_texture=False):
    out = image.copy()
    colors = [PIECE_BGR] * 4
    for i, (p, h) in enumerate(zip(pieces, transforms)):
        q = np.round(apply_h(p, h)).astype(np.int32)
        if preserve_texture:
            source_mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(source_mask, [np.round(p).astype(np.int32)], 255)
            warped_image = cv2.warpPerspective(
                image, h, (image.shape[1], image.shape[0]))
            warped_mask = cv2.warpPerspective(
                source_mask, h, (image.shape[1], image.shape[0]))
            out[warped_mask > 127] = warped_image[warped_mask > 127]
        else:
            cv2.fillPoly(out, [q], colors[i])
        cv2.polylines(out, [q], True, (20, 20, 20), 2, cv2.LINE_AA)
        c = np.round(q.mean(0)).astype(int)
        cv2.putText(out, f"P{i}", tuple(c), cv2.FONT_HERSHEY_SIMPLEX, .7,
                    (20, 20, 20), 2, cv2.LINE_AA)
    return out


def make_summary(scene, detected, solution):
    scale = 0.48
    panels = []
    for title, im in [("1 INPUT", scene), ("2 DETECT", detected), ("3 RESTORE", solution)]:
        x = cv2.resize(im, None, fx=scale, fy=scale)
        cv2.rectangle(x, (0, 0), (x.shape[1], 42), (255, 255, 255), -1)
        cv2.putText(x, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .75, (20, 20, 20), 2)
        panels.append(x)
    return np.hstack(panels)


def run_once(seed: int, output: Path, save=True, piece_count: int = 4):
    scene = generate_camera_frame(seed, piece_count)
    detected_pieces, transforms, matches = analyze_camera_frame(scene)
    # This assertion belongs to simulation evaluation only. It is not supplied
    # to detection or planning and cannot influence their result.
    if len(detected_pieces) != piece_count:
        raise RuntimeError(
            f"独立视觉检测到 {len(detected_pieces)} 块，仿真评测设置为 {piece_count} 块")

    # Pixel-domain reconstruction metrics.
    restored = [apply_h(p, h) for p, h in zip(detected_pieces, transforms)]
    allp = np.vstack(restored).astype(np.float32)
    rect = cv2.minAreaRect(allp)
    rw, rh = sorted(rect[1], reverse=True)
    dimension_error = max(abs(rw - CARD_W), abs(rh - CARD_H))

    if save:
        output.mkdir(parents=True, exist_ok=True)
        detected = annotate_detection(scene, detected_pieces)
        solution = render_solution(scene, detected_pieces, transforms)
        cv2.imwrite(str(output / "scene.png"), scene)
        cv2.imwrite(str(output / "detected.png"), detected)
        cv2.imwrite(str(output / "solution.png"), solution)
        cv2.imwrite(str(output / "summary.png"), make_summary(scene, detected, solution))
        records = []
        for i, (p, h) in enumerate(zip(detected_pieces, transforms)):
            angle = math.degrees(math.atan2(h[1, 0], h[0, 0]))
            records.append({
                "piece_id": i,
                "detected_center_px": p.mean(0).round(3).tolist(),
                "rotation_deg": round(angle, 6),
                "translation_px": h[:2, 2].round(6).tolist(),
                "matrix_2x3": h[:2].round(9).tolist(),
                "matrix_3x3": h.round(9).tolist(),
            })
        data = {
            "seed": seed,
            "piece_count": piece_count,
            "coordinate_system": "image pixels: x right, y down",
            "scale_px_per_cm": PIXELS_PER_CM,
            "target_rectangle_cm": {
                "width": CARD_WIDTH_CM,
                "height": CARD_HEIGHT_CM,
            },
            "target_rectangle_px": {
                "x": (CANVAS_W - CARD_W) / 2,
                "y": TARGET_Y,
                "width": CARD_W,
                "height": CARD_H,
            },
            "matched_cut_edges": [
                [int(match[1]), int(match[2]),
                 int(match[3]), int(match[4]),
                 bool(tuple(match[5:]) != (0., 1., 0., 1.))]
                for match in matches],
            "dimension_error_px": round(float(dimension_error), 4),
            "pieces": records,
        }
        (output / "transforms.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return dimension_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7, help="随机种子")
    parser.add_argument("--output", type=Path, default=Path("output/demo"), help="输出目录")
    parser.add_argument("--pieces", type=int, choices=range(2, 5), default=4,
                        help="碎片数量，范围 2～4")
    parser.add_argument("--batch", type=int, default=0, help="批量验证次数（不保存图片）")
    args = parser.parse_args()
    if args.batch:
        errors, failures = [], []
        for seed in range(args.seed, args.seed + args.batch):
            try:
                errors.append(run_once(seed, args.output, save=False, piece_count=args.pieces))
            except Exception as exc:
                failures.append((seed, str(exc)))
        print(f"批量测试: {args.batch} 次，成功 {len(errors)}，失败 {len(failures)}")
        if errors:
            print(f"矩形尺寸最大误差: {max(errors):.3f} px，平均: {np.mean(errors):.3f} px")
        if failures:
            print("失败样例:", failures[:10])
            raise SystemExit(1)
    else:
        err = run_once(args.seed, args.output, save=True, piece_count=args.pieces)
        print(f"完成。输出目录: {args.output.resolve()}")
        print(f"还原矩形尺寸误差: {err:.3f} px")
        print(f"位姿矩阵: {(args.output / 'transforms.json').resolve()}")


if __name__ == "__main__":
    main()
