"""
Puzzle piece detector for CanMV K230.

Scene:
- Black A4 paper as background.
- White sticker / white poker face on each metal puzzle piece.
- A red horizontal line separates the upper random area from the lower target area.

Output:
- LCD/IDE overlay: blue ROI, yellow divider line, red piece boxes, green centers.
- REPL print: id, center, size, angle estimate, pixels, area.
"""

import math
import time

from media.sensor import *
from media.display import *
from media.media import *


FRAME_W = 640
FRAME_H = 480

# Approximate area containing the black A4 paper. Tune this once after fixing camera.
PAPER_ROI = (150, 0, 365, 430)

# Used only if the red divider line is not found.
FALLBACK_UPPER_ROI = (150, 0, 365, 250)

# Keep several pixels above the red line so the red line itself is not detected as a piece.
DIVIDER_MARGIN = 8

# LAB thresholds.
# Bright threshold: white pieces may look pink/blue under camera auto white balance,
# so detect bright objects on the black paper instead of strict white.
BRIGHT_LAB_THRESHOLD = (45, 100, -128, 127, -128, 127)

# Red divider line threshold. If the line is not found, reduce A_min from 25 to 15.
RED_LAB_THRESHOLD = (20, 100, 25, 127, -20, 80)

# Piece filtering.
MIN_PIXELS = 100
MIN_AREA = 140
MAX_BLOBS = 4
MAX_ASPECT_RATIO = 5
MAX_THIN_WIDTH = 28
MIN_THIN_HEIGHT = 80

# Short labels for the four self-made pieces.
# These are heuristic labels based on size and bounding-box shape.
LABEL_UNKNOWN = "UNK"
LABEL_LARGE_TRI = "TRI"
LABEL_LONG_PIECE = "LONG"
LABEL_SIDE_PIECE = "SIDE"
LABEL_SMALL_PIECE = "SMALL"

# Red line filtering.
MIN_LINE_WIDTH = 180
MAX_LINE_HEIGHT = 25

# Current white pieces are separated objects. Do not merge them.
MERGE_BLOBS = False
MERGE_MARGIN = 0

# Reject bright areas touching the left ROI border.
# Top/right/bottom borders are allowed because pieces may be close to the paper edge/divider.
EDGE_GUARD = 4

PRINT_INTERVAL_MS = 300
DIVIDER_SCAN_INTERVAL_FRAMES = 5

# Vertex extraction. Use our own contour scan, not blob API corners.
ENABLE_PIXEL_VERTEX_SCAN = True
PIXEL_SCAN_STEP = 2
PIXEL_LUMA_THRESHOLD = 85
POLY_EPSILON = 5
MAX_VERTICES = 10
MIN_CONTOUR_POINTS = 12
CONTOUR_MAX_STEPS = 600
ENABLE_MAIN_EDGE_REFINE = True
MIN_MAIN_VERTICES = 3
MAX_MAIN_VERTICES = 5
MIN_EDGE_RATIO = 0.12
COLLINEAR_DOT_LIMIT = 0.94
COLLINEAR_DOT_LIMIT_SQ = COLLINEAR_DOT_LIMIT * COLLINEAR_DOT_LIMIT

# Plan overlay. The plan is drawn in the lower target area and printed in pixels.
ENABLE_PLAN_OVERLAY = True
PLAN_TARGET_W = 310
PLAN_TARGET_H = 170
PLAN_MARGIN_X = 24
PLAN_MARGIN_Y = 18
PLAN_MIN_H = 80
PLAN_FIT_MARGIN = 8
AUTO_SOLVE_PLAN = True
FAST_SOLVE_PLAN = True
USE_FIG2_TEMPLATE_FOR_KNOWN_SET = True
EDGE_MATCH_RATIO = 0.28
GRAPH_MATCH_RATIO = 0.16
MAX_GRAPH_CANDIDATES = 32
MAX_GRAPH_MATCH_SETS = 90
MAX_ACCEPTED_GRAPH_SCORE = 1.05
MAX_ACCEPTED_SOLVER_SCORE = 0.22
MAX_SOLVER_STATES = 80
MAX_SOLVER_BRANCHES = 24
MAX_FAST_ANCHORS = 12
MAX_FAST_STATES = 36
CORNER_DOT_LIMIT = 0.35
CORNER_DOT_LIMIT_SQ = CORNER_DOT_LIMIT * CORNER_DOT_LIMIT
OUTER_EDGE_MARGIN_RATIO = 0.045
MISSING_OUTER_EDGE_PENALTY = 0.45
MIN_POLYGON_AREA = 80
OVERLAP_EPSILON = 1.5
OVERLAP_EPSILON_SQ = OVERLAP_EPSILON * OVERLAP_EPSILON
FREEZE_PLAN_AFTER_STABLE = True
PLAN_STABLE_MS = 3000
PLAN_SIGNATURE_POS_TOL = 20
PLAN_UNLOCK_MOVE_TOL = 45
PLAN_CACHE_DIVIDER_TOL = 6
MIN_REASONABLE_ASPECT = 12 / 9
MAX_REASONABLE_ASPECT = 9 / 5
PLAN_MOVE_ORDER = (
    LABEL_LARGE_TRI,
    LABEL_LONG_PIECE,
    LABEL_SIDE_PIECE,
    LABEL_SMALL_PIECE,
)

# Normalized target layout from problem Figure 2.
# Target rectangle: 10cm x 6cm.
# Left edge segments: 2cm, 1cm, 3cm. Top edge segments: 2cm, 8cm.
# Diagonal B(2,0)->D(10,6): B->P is 2cm and H->D is 3cm.
PLAN_TEMPLATES = (
    (LABEL_SMALL_PIECE, ((0.00, 0.00), (0.20, 0.00), (0.36, 0.20), (0.00, 0.333))),
    (LABEL_LARGE_TRI, ((0.20, 0.00), (1.00, 0.00), (1.00, 1.00))),
    (LABEL_SIDE_PIECE, ((0.00, 0.333), (0.36, 0.20), (0.76, 0.70), (0.00, 0.50))),
    (LABEL_LONG_PIECE, ((0.00, 0.50), (0.76, 0.70), (1.00, 1.00), (0.00, 1.00))),
)


def safe_call(obj, name, default=None):
    try:
        fn = getattr(obj, name)
        return fn()
    except Exception:
        return default


def blob_rect(blob):
    rect = safe_call(blob, "rect", None)
    if rect:
        return rect

    x = safe_call(blob, "x", 0)
    y = safe_call(blob, "y", 0)
    w = safe_call(blob, "w", 0)
    h = safe_call(blob, "h", 0)
    return (x, y, w, h)


def blob_info(blob, index):
    x, y, w, h = blob_rect(blob)
    cx = safe_call(blob, "cx", x + w // 2)
    cy = safe_call(blob, "cy", y + h // 2)
    pixels = safe_call(blob, "pixels", 0)
    area = safe_call(blob, "area", w * h)
    theta = safe_call(blob, "rotation", 0)
    aspect = max(w, h) / max(1, min(w, h))
    fill = pixels / max(1, area)
    return {
        "id": index,
        "label": LABEL_UNKNOWN,
        "vertices": [],
        "edge_lengths": [],
        "main_edge_count": 0,
        "vertex_source": "none",
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "cx": cx,
        "cy": cy,
        "pixels": pixels,
        "area": area,
        "theta": theta,
        "aspect": aspect,
        "fill": fill,
    }


def dist(a, b):
    return math.sqrt(dist_sq(a, b))


def dist_sq(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def point_line_distance(p, a, b):
    return math.sqrt(point_line_distance_sq(p, a, b))


def point_line_distance_sq(p, a, b):
    ax, ay = a
    bx, by = b
    px, py = p
    dx = bx - ax
    dy = by - ay
    if dx == 0 and dy == 0:
        return dist_sq(p, a)
    numerator = dy * px - dx * py + bx * ay - by * ax
    return (numerator * numerator) / (dx * dx + dy * dy)


def rdp(points, epsilon):
    if len(points) < 3:
        return points

    first = points[0]
    last = points[-1]
    index = 0
    max_dist_sq = 0
    for i in range(1, len(points) - 1):
        d = point_line_distance_sq(points[i], first, last)
        if d > max_dist_sq:
            index = i
            max_dist_sq = d

    if max_dist_sq > epsilon * epsilon:
        left = rdp(points[: index + 1], epsilon)
        right = rdp(points[index:], epsilon)
        return left[:-1] + right
    return [first, last]


def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def normalize_pixel(pixel):
    if isinstance(pixel, tuple) or isinstance(pixel, list):
        if len(pixel) >= 3:
            return pixel[0], pixel[1], pixel[2]
        if len(pixel) == 1:
            return pixel[0], pixel[0], pixel[0]

    # RGB565 fallback.
    value = int(pixel)
    r = ((value >> 11) & 0x1F) * 255 // 31
    g = ((value >> 5) & 0x3F) * 255 // 63
    b = (value & 0x1F) * 255 // 31
    return r, g, b


def is_bright_pixel(img, x, y):
    try:
        pixel = img.get_pixel(x, y)
    except Exception:
        return False

    r, g, b = normalize_pixel(pixel)
    luma = (30 * r + 59 * g + 11 * b) // 100
    return luma >= PIXEL_LUMA_THRESHOLD


def box_vertices(info):
    x = info["x"]
    y = info["y"]
    w = info["w"]
    h = info["h"]
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def edge_lengths(vertices):
    lengths = []
    count = len(vertices)
    for i in range(count):
        lengths.append(dist(vertices[i], vertices[(i + 1) % count]))
    return lengths


def polygon_edges(vertices):
    items = []
    count = len(vertices)
    for i in range(count):
        items.append((vertices[i], vertices[(i + 1) % count]))
    return items


def mask_at(mask, gx, gy):
    if gy < 0 or gy >= len(mask):
        return 0
    row = mask[gy]
    if gx < 0 or gx >= len(row):
        return 0
    return row[gx]


def is_mask_boundary(mask, gx, gy):
    if not mask_at(mask, gx, gy):
        return False
    return (
        not mask_at(mask, gx - 1, gy)
        or not mask_at(mask, gx + 1, gy)
        or not mask_at(mask, gx, gy - 1)
        or not mask_at(mask, gx, gy + 1)
    )


def remove_isolated_mask_pixels(mask):
    cleaned = []
    height = len(mask)
    for gy in range(height):
        row = mask[gy]
        clean_row = bytearray(len(row))
        for gx in range(len(row)):
            if not row[gx]:
                continue

            neighbors = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    if mask_at(mask, gx + dx, gy + dy):
                        neighbors += 1

            if neighbors >= 2:
                clean_row[gx] = 1
        cleaned.append(clean_row)
    return cleaned


def build_piece_mask(img, x0, y0, x1, y1):
    mask = []
    for y in range(y0, y1 + 1, PIXEL_SCAN_STEP):
        row = bytearray()
        for x in range(x0, x1 + 1, PIXEL_SCAN_STEP):
            row.append(1 if is_bright_pixel(img, x, y) else 0)
        mask.append(row)
    return remove_isolated_mask_pixels(mask)


def trace_mask_contour(mask):
    start = None
    for gy in range(len(mask)):
        for gx in range(len(mask[gy])):
            if is_mask_boundary(mask, gx, gy):
                start = (gx, gy)
                break
        if start is not None:
            break

    if start is None:
        return []

    dirs = (
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    )

    current = start
    previous = (start[0] - 1, start[1])
    contour = []

    for _ in range(CONTOUR_MAX_STEPS):
        contour.append(current)

        back_dx = previous[0] - current[0]
        back_dy = previous[1] - current[1]
        back_index = 4
        for i, direction in enumerate(dirs):
            if direction[0] == back_dx and direction[1] == back_dy:
                back_index = i
                break

        found = False
        for offset in range(1, 9):
            direction_index = (back_index + offset) % 8
            dx, dy = dirs[direction_index]
            nx = current[0] + dx
            ny = current[1] + dy
            if not mask_at(mask, nx, ny):
                continue

            previous_direction = dirs[(direction_index - 1) % 8]
            previous = (
                current[0] + previous_direction[0],
                current[1] + previous_direction[1],
            )
            current = (nx, ny)
            found = True
            break

        if not found:
            break
        if current == start and len(contour) > 1:
            break

    return contour


def grid_to_pixels(contour, x0, y0):
    points = []
    for gx, gy in contour:
        points.append((x0 + gx * PIXEL_SCAN_STEP, y0 + gy * PIXEL_SCAN_STEP))
    return points


def remove_near_duplicate_points(points, min_distance):
    if not points:
        return points

    min_distance_sq = min_distance * min_distance
    filtered = []
    for point in points:
        if not filtered or dist_sq(point, filtered[-1]) >= min_distance_sq:
            filtered.append(point)

    if len(filtered) > 1 and dist_sq(filtered[0], filtered[-1]) < min_distance_sq:
        filtered.pop()
    return filtered


def max_edge_length(points):
    if len(points) < 2:
        return 0

    longest_sq = 0
    for i in range(len(points)):
        length_sq = dist_sq(points[i], points[(i + 1) % len(points)])
        if length_sq > longest_sq:
            longest_sq = length_sq
    return math.sqrt(longest_sq)


def remove_short_polygon_edges(points):
    if len(points) <= MIN_MAIN_VERTICES:
        return points

    longest = max_edge_length(points)
    min_len = longest * MIN_EDGE_RATIO
    min_len_sq = min_len * min_len
    refined = points[:]
    changed = True

    while changed and len(refined) > MIN_MAIN_VERTICES:
        changed = False
        shortest_index = -1
        shortest_len_sq = min_len_sq
        for i in range(len(refined)):
            length_sq = dist_sq(refined[i], refined[(i + 1) % len(refined)])
            if length_sq < shortest_len_sq:
                shortest_len_sq = length_sq
                shortest_index = i

        if shortest_index >= 0:
            remove_index = (shortest_index + 1) % len(refined)
            refined.pop(remove_index)
            changed = True

    return refined


def is_near_collinear(prev_point, point, next_point):
    v1 = (point[0] - prev_point[0], point[1] - prev_point[1])
    v2 = (next_point[0] - point[0], next_point[1] - point[1])
    return vector_dot_abs_sq(v1, v2, 1) >= COLLINEAR_DOT_LIMIT_SQ


def vector_dot_abs_sq(v1, v2, zero_default):
    len1_sq = v1[0] * v1[0] + v1[1] * v1[1]
    len2_sq = v2[0] * v2[0] + v2[1] * v2[1]
    if len1_sq <= 0 or len2_sq <= 0:
        return zero_default
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    return (dot * dot) / (len1_sq * len2_sq)


def merge_collinear_polygon_edges(points):
    if len(points) <= MIN_MAIN_VERTICES:
        return points

    refined = points[:]
    changed = True
    while changed and len(refined) > MIN_MAIN_VERTICES:
        changed = False
        for i in range(len(refined)):
            prev_point = refined[(i - 1) % len(refined)]
            point = refined[i]
            next_point = refined[(i + 1) % len(refined)]
            if is_near_collinear(prev_point, point, next_point):
                refined.pop(i)
                changed = True
                break

    return refined


def corner_strength(prev_point, point, next_point):
    v1 = (prev_point[0] - point[0], prev_point[1] - point[1])
    v2 = (next_point[0] - point[0], next_point[1] - point[1])
    return 1 - math.sqrt(vector_dot_abs_sq(v1, v2, 1))


def remove_weakest_corners(points):
    refined = points[:]
    while len(refined) > MAX_MAIN_VERTICES:
        weakest_index = 0
        weakest_score = -1
        for i in range(len(refined)):
            point = refined[i]
            prev_point = refined[(i - 1) % len(refined)]
            next_point = refined[(i + 1) % len(refined)]
            v1 = (prev_point[0] - point[0], prev_point[1] - point[1])
            v2 = (next_point[0] - point[0], next_point[1] - point[1])
            score = vector_dot_abs_sq(v1, v2, 1)
            if score > weakest_score:
                weakest_score = score
                weakest_index = i
        refined.pop(weakest_index)
    return refined


def refine_main_edges(points):
    if not ENABLE_MAIN_EDGE_REFINE or len(points) <= MAX_MAIN_VERTICES:
        return points

    refined = remove_near_duplicate_points(points, PIXEL_SCAN_STEP * 2)
    refined = remove_short_polygon_edges(refined)
    refined = merge_collinear_polygon_edges(refined)
    refined = remove_weakest_corners(refined)
    refined = remove_short_polygon_edges(refined)
    refined = merge_collinear_polygon_edges(refined)

    if len(refined) < MIN_MAIN_VERTICES:
        return points
    return refined


def simplify_closed_contour(contour):
    if len(contour) < MIN_CONTOUR_POINTS:
        return None

    first = contour[0]
    split_index = 0
    split_distance_sq = 0
    for i, point in enumerate(contour):
        d = dist_sq(first, point)
        if d > split_distance_sq:
            split_distance_sq = d
            split_index = i

    if split_index <= 1 or split_index >= len(contour) - 1:
        return None

    epsilon = POLY_EPSILON
    path_a = contour[: split_index + 1]
    path_b = contour[split_index:] + [contour[0]]
    simplified = []
    while epsilon <= 24:
        simplified = rdp(path_a, epsilon)[:-1] + rdp(path_b, epsilon)[:-1]
        simplified = remove_near_duplicate_points(simplified, PIXEL_SCAN_STEP * 2)

        if 3 <= len(simplified) <= MAX_VERTICES:
            return refine_main_edges(simplified)
        epsilon += 2

    if len(simplified) >= 3:
        return refine_main_edges(simplified[:MAX_VERTICES])
    return None


def extract_pixel_vertices(img, info):
    if not ENABLE_PIXEL_VERTEX_SCAN:
        return None

    x0 = max(0, info["x"] - PIXEL_SCAN_STEP)
    y0 = max(0, info["y"] - PIXEL_SCAN_STEP)
    x1 = min(FRAME_W - 1, info["x"] + info["w"] + PIXEL_SCAN_STEP)
    y1 = min(FRAME_H - 1, info["y"] + info["h"] + PIXEL_SCAN_STEP)

    mask = build_piece_mask(img, x0, y0, x1, y1)
    contour = trace_mask_contour(mask)
    if len(contour) < MIN_CONTOUR_POINTS:
        return None

    return simplify_closed_contour(grid_to_pixels(contour, x0, y0))


def vertex_cache_key(info):
    return (info["x"], info["y"], info["w"], info["h"])


def attach_vertices(img, infos, vertex_cache=None):
    next_cache = {}
    for info in infos:
        key = vertex_cache_key(info)
        cached = None
        if vertex_cache is not None:
            cached = vertex_cache.get(key)

        if cached is not None:
            vertices, lengths, source = cached
            info["vertices"] = vertices
            info["edge_lengths"] = lengths
            info["main_edge_count"] = len(vertices)
            info["vertex_source"] = source
            next_cache[key] = cached
            continue

        vertices = extract_pixel_vertices(img, info)

        if vertices is None:
            vertices = box_vertices(info)
            info["vertex_source"] = "box"
        else:
            info["vertex_source"] = "contour"

        info["vertices"] = vertices
        info["edge_lengths"] = edge_lengths(vertices)
        info["main_edge_count"] = len(vertices)
        next_cache[key] = (info["vertices"], info["edge_lengths"], info["vertex_source"])

    return next_cache


def touches_roi_edge(info, roi, guard):
    rx, ry, rw, rh = roi
    if info["x"] <= rx + guard:
        return True
    return False


def find_divider_y(img):
    blobs = img.find_blobs(
        [RED_LAB_THRESHOLD],
        roi=PAPER_ROI,
        pixels_threshold=80,
        area_threshold=80,
        merge=True,
        margin=5,
    )

    best = None
    best_score = 0
    for blob in blobs:
        info = blob_info(blob, 0)
        if info["w"] < MIN_LINE_WIDTH:
            continue
        if info["h"] > MAX_LINE_HEIGHT:
            continue

        score = info["w"] * info["pixels"]
        if score > best_score:
            best = info
            best_score = score

    if best is None:
        return None
    return best["cy"]


def make_upper_roi(divider_y):
    if divider_y is None:
        return FALLBACK_UPPER_ROI

    x, y, w, h = PAPER_ROI
    upper_h = divider_y - y - DIVIDER_MARGIN
    if upper_h < 80:
        return FALLBACK_UPPER_ROI
    return (x, y, w, upper_h)


def is_valid_piece(info, roi):
    if info["w"] < 10 or info["h"] < 10:
        return False
    if info["area"] < MIN_AREA:
        return False

    long_side = max(info["w"], info["h"])
    short_side = max(1, min(info["w"], info["h"]))
    if long_side / short_side > MAX_ASPECT_RATIO:
        return False
    if info["w"] <= MAX_THIN_WIDTH and info["h"] >= MIN_THIN_HEIGHT:
        return False

    if touches_roi_edge(info, roi, EDGE_GUARD):
        return False

    return True


def classify_pieces(infos):
    """Assign stable piece labels from simple geometry.

    This is intended for the fixed 4-piece training set:
    - SMALL: smallest white area.
    - LONG: most elongated remaining piece.
    - TRI: largest remaining piece.
    - SIDE: the remaining piece.

    It does not depend on the random position of each piece.
    """
    for info in infos:
        info["label"] = LABEL_UNKNOWN

    if not infos:
        return

    remaining = infos[:]

    if len(remaining) >= 3:
        small = min(remaining, key=lambda item: item["pixels"])
        small["label"] = LABEL_SMALL_PIECE
        remaining.remove(small)

    if remaining:
        long_piece = max(remaining, key=lambda item: item["aspect"])
        if long_piece["aspect"] >= 1.6:
            long_piece["label"] = LABEL_LONG_PIECE
            remaining.remove(long_piece)

    if remaining:
        tri = max(remaining, key=lambda item: item["pixels"])
        tri["label"] = LABEL_LARGE_TRI
        remaining.remove(tri)

    for info in remaining:
        info["label"] = LABEL_SIDE_PIECE


def make_plan_rect(divider_y, aspect=None):
    rx, ry, rw, rh = PAPER_ROI
    if divider_y is None:
        target_top = FALLBACK_UPPER_ROI[1] + FALLBACK_UPPER_ROI[3] + PLAN_MARGIN_Y
    else:
        target_top = divider_y + PLAN_MARGIN_Y

    target_bottom = ry + rh - PLAN_MARGIN_Y
    available_h = target_bottom - target_top
    if available_h < PLAN_MIN_H:
        target_top = ry + rh - PLAN_MARGIN_Y - PLAN_MIN_H
        available_h = PLAN_MIN_H

    max_w = min(PLAN_TARGET_W, rw - PLAN_MARGIN_X * 2)
    max_h = min(PLAN_TARGET_H, available_h)
    if aspect is None or aspect <= 0:
        target_w = max_w
        target_h = max_h
    else:
        target_w = max_w
        target_h = int(target_w / aspect)
        if target_h > max_h:
            target_h = max_h
            target_w = int(target_h * aspect)

    target_x = rx + (rw - target_w) // 2
    target_y = target_top + (available_h - target_h) // 2
    return (target_x, target_y, target_w, target_h)


def scale_plan_points(points, rect):
    x, y, w, h = rect
    scaled = []
    for px, py in points:
        sx = x + int(px * w + 0.5)
        sy = y + int(py * h + 0.5)
        scaled.append((sx, sy))
    return scaled


def polygon_center(points):
    if not points:
        return (0, 0)

    sx = 0
    sy = 0
    for point in points:
        sx += point[0]
        sy += point[1]
    return (sx // len(points), sy // len(points))


def polygon_angle(points):
    if len(points) < 2:
        return 0

    longest = 0
    best_angle = 0
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        length = dist(p1, p2)
        if length > longest:
            longest = length
            best_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    return int(best_angle * 180 / math.pi)


def polygon_signed_area(points):
    area = 0
    count = len(points)
    for i in range(count):
        p1 = points[i]
        p2 = points[(i + 1) % count]
        area += p1[0] * p2[1] - p2[0] * p1[1]
    return area / 2


def polygon_abs_area(points):
    area = polygon_signed_area(points)
    if area < 0:
        return -area
    return area


def ensure_ccw_polygon(points):
    if polygon_signed_area(points) < 0:
        reversed_points = points[:]
        reversed_points.reverse()
        return reversed_points
    return points


def rotate_point(point, angle):
    ca = math.cos(angle)
    sa = math.sin(angle)
    return (point[0] * ca - point[1] * sa, point[0] * sa + point[1] * ca)


def rotate_polygon(points, angle):
    ca = math.cos(angle)
    sa = math.sin(angle)
    rotated = []
    for point in points:
        rotated.append((point[0] * ca - point[1] * sa, point[0] * sa + point[1] * ca))
    return rotated


def transform_point(point, angle, tx, ty):
    rotated = rotate_point(point, angle)
    return (rotated[0] + tx, rotated[1] + ty)


def transform_polygon(points, angle, tx, ty):
    ca = math.cos(angle)
    sa = math.sin(angle)
    transformed = []
    for point in points:
        transformed.append((point[0] * ca - point[1] * sa + tx, point[0] * sa + point[1] * ca + ty))
    return transformed


def edge_len(p1, p2):
    return dist(p1, p2)


def edge_is_match(len_a, len_b):
    long_len = max(len_a, len_b)
    if long_len <= 0:
        return False
    return abs(len_a - len_b) / long_len <= EDGE_MATCH_RATIO


def edge_endpoint_error_with_lengths(src_a, src_b, src_len, dst_a, dst_b, dst_len):
    long_len = max(src_len, dst_len)
    if long_len <= 0:
        return 999999

    pose = align_edge_pose(src_a, src_b, dst_b, dst_a)
    mapped_a = apply_pose_point(src_a, pose)
    mapped_b = apply_pose_point(src_b, pose)
    return (dist(mapped_a, dst_b) + dist(mapped_b, dst_a)) / long_len


def edge_endpoint_error(src_a, src_b, dst_a, dst_b):
    src_len = edge_len(src_a, src_b)
    dst_len = edge_len(dst_a, dst_b)
    return edge_endpoint_error_with_lengths(src_a, src_b, src_len, dst_a, dst_b, dst_len)


def edge_match_error_with_lengths(src_a, src_b, src_len, dst_a, dst_b, dst_len):
    long_len = max(src_len, dst_len)
    if long_len <= 0:
        return 999999

    length_error = abs(src_len - dst_len) / long_len
    endpoint_error = edge_endpoint_error_with_lengths(src_a, src_b, src_len, dst_a, dst_b, dst_len)
    return max(length_error, endpoint_error)


def edge_match_error(src_a, src_b, dst_a, dst_b):
    src_len = edge_len(src_a, src_b)
    dst_len = edge_len(dst_a, dst_b)
    return edge_match_error_with_lengths(src_a, src_b, src_len, dst_a, dst_b, dst_len)


def align_polygon_edge_to_edge(poly, edge_index, target_p1, target_p2):
    source_p1 = poly[edge_index]
    source_p2 = poly[(edge_index + 1) % len(poly)]
    source_angle = math.atan2(source_p2[1] - source_p1[1], source_p2[0] - source_p1[0])
    target_angle = math.atan2(target_p1[1] - target_p2[1], target_p1[0] - target_p2[0])
    rotate_angle = target_angle - source_angle
    rotated_p1 = rotate_point(source_p1, rotate_angle)
    tx = target_p2[0] - rotated_p1[0]
    ty = target_p2[1] - rotated_p1[1]
    return transform_polygon(poly, rotate_angle, tx, ty), rotate_angle


def point_on_segment(point, a, b):
    line_len_sq = dist_sq(a, b)
    if line_len_sq <= 0:
        return dist_sq(point, a) <= OVERLAP_EPSILON_SQ
    if point_line_distance_sq(point, a, b) > OVERLAP_EPSILON_SQ:
        return False
    dot = (point[0] - a[0]) * (point[0] - b[0]) + (point[1] - a[1]) * (point[1] - b[1])
    return dot <= OVERLAP_EPSILON


def point_inside_polygon(point, poly):
    for i in range(len(poly)):
        if point_on_segment(point, poly[i], poly[(i + 1) % len(poly)]):
            return False

    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        pi = poly[i]
        pj = poly[j]
        if (pi[1] > point[1]) != (pj[1] > point[1]):
            x_cross = (pj[0] - pi[0]) * (point[1] - pi[1]) / (pj[1] - pi[1]) + pi[0]
            if point[0] < x_cross:
                inside = not inside
        j = i
    return inside


def segment_crosses_strict(a1, a2, b1, b2):
    if point_on_segment(a1, b1, b2) or point_on_segment(a2, b1, b2):
        return False
    if point_on_segment(b1, a1, a2) or point_on_segment(b2, a1, a2):
        return False

    o1 = cross(a1, a2, b1)
    o2 = cross(a1, a2, b2)
    o3 = cross(b1, b2, a1)
    o4 = cross(b1, b2, a2)
    return o1 * o2 < 0 and o3 * o4 < 0


def polygons_overlap(poly_a, poly_b):
    for point in poly_a:
        if point_inside_polygon(point, poly_b):
            return True
    for point in poly_b:
        if point_inside_polygon(point, poly_a):
            return True

    for i in range(len(poly_a)):
        a1 = poly_a[i]
        a2 = poly_a[(i + 1) % len(poly_a)]
        for j in range(len(poly_b)):
            b1 = poly_b[j]
            b2 = poly_b[(j + 1) % len(poly_b)]
            if segment_crosses_strict(a1, a2, b1, b2):
                return True
    return False


def bbox_of_points(points):
    min_x = points[0][0]
    max_x = points[0][0]
    min_y = points[0][1]
    max_y = points[0][1]
    for point in points:
        if point[0] < min_x:
            min_x = point[0]
        if point[0] > max_x:
            max_x = point[0]
        if point[1] < min_y:
            min_y = point[1]
        if point[1] > max_y:
            max_y = point[1]
    return (min_x, min_y, max_x, max_y)


def bbox_of_polygons(polys):
    first = True
    min_x = max_x = min_y = max_y = 0
    for poly in polys:
        for point in poly:
            if first:
                min_x = max_x = point[0]
                min_y = max_y = point[1]
                first = False
                continue
            if point[0] < min_x:
                min_x = point[0]
            if point[0] > max_x:
                max_x = point[0]
            if point[1] < min_y:
                min_y = point[1]
            if point[1] > max_y:
                max_y = point[1]
    return (min_x, min_y, max_x, max_y)


def all_polygon_points(polys):
    points = []
    for poly in polys:
        points += poly
    return points


def make_piece_records(infos):
    records = []
    for info in infos:
        vertices = refine_main_edges(info["vertices"])
        if len(vertices) < 3:
            continue
        if len(vertices) > MAX_MAIN_VERTICES:
            continue
        area = polygon_abs_area(vertices)
        if area < MIN_POLYGON_AREA:
            continue

        local = []
        for point in vertices:
            local.append((point[0] - info["cx"], point[1] - info["cy"]))
        local = ensure_ccw_polygon(local)
        local_edges = polygon_edges(local)
        local_edge_lengths = edge_lengths(local)
        records.append(
            {
                "id": info["id"],
                "label": info["label"],
                "from_x": info["cx"],
                "from_y": info["cy"],
                "local_vertices": local,
                "local_edges": local_edges,
                "edge_lengths": local_edge_lengths,
                "area": area,
            }
        )
    return records


def state_overlap(candidate_poly, placed):
    for item in placed:
        if polygons_overlap(candidate_poly, item["target_vertices"]):
            return True
    return False


def make_pose(angle, tx, ty):
    return {"angle": angle, "tx": tx, "ty": ty}


def identity_pose():
    return make_pose(0, 0, 0)


def apply_pose_point(point, pose):
    return transform_point(point, pose["angle"], pose["tx"], pose["ty"])


def apply_pose_polygon(points, pose):
    return transform_polygon(points, pose["angle"], pose["tx"], pose["ty"])


def align_edge_pose(src_a, src_b, dst_a, dst_b):
    src_angle = math.atan2(src_b[1] - src_a[1], src_b[0] - src_a[0])
    dst_angle = math.atan2(dst_b[1] - dst_a[1], dst_b[0] - dst_a[0])
    angle = dst_angle - src_angle
    mapped = rotate_point(src_a, angle)
    return make_pose(angle, dst_a[0] - mapped[0], dst_a[1] - mapped[1])


def pose_delta_error(poly, pose_a, pose_b):
    if pose_a is None or pose_b is None:
        return 999999
    total = 0
    for point in poly:
        pa = apply_pose_point(point, pose_a)
        pb = apply_pose_point(point, pose_b)
        total += dist(pa, pb)
    return total / max(1, len(poly))


def graph_candidate_matchings(records):
    candidates = []
    for i in range(len(records)):
        edges_i = records[i]["local_edges"]
        lengths_i = records[i]["edge_lengths"]
        for j in range(i + 1, len(records)):
            edges_j = records[j]["local_edges"]
            lengths_j = records[j]["edge_lengths"]
            for ei in range(len(edges_i)):
                ia, ib = edges_i[ei]
                len_i = lengths_i[ei]
                for ej in range(len(edges_j)):
                    ja, jb = edges_j[ej]
                    match_error = edge_match_error_with_lengths(ia, ib, len_i, ja, jb, lengths_j[ej])
                    if match_error <= GRAPH_MATCH_RATIO:
                        candidates.append((match_error, i, ei, j, ej))
    candidates.sort(key=lambda item: item[0])
    return candidates[:MAX_GRAPH_CANDIDATES]


def graph_is_connected(graph):
    if not graph:
        return False

    seen = [False] * len(graph)
    stack = [0]
    seen[0] = True
    while stack:
        node = stack.pop()
        for other in graph[node]:
            if not seen[other]:
                seen[other] = True
                stack.append(other)
    for item in seen:
        if not item:
            return False
    return True


def collect_graph_matching_sets(records):
    count = len(records)
    if count == 1:
        return [[]]

    candidates = graph_candidate_matchings(records)
    if not candidates:
        return []

    pair_counts = [1] if count == 2 else [count - 1, count]
    results = []

    def dfs(start, pair_count, selected, used_edges, degree, graph):
        if len(results) >= MAX_GRAPH_MATCH_SETS:
            return

        if len(selected) == pair_count:
            for value in degree:
                if value < 1:
                    return
            if not graph_is_connected(graph):
                return
            results.append(selected[:])
            return

        remaining = pair_count - len(selected)
        if len(candidates) - start < remaining:
            return

        for index in range(start, len(candidates)):
            rel, i, ei, j, ej = candidates[index]
            edge_a = (i, ei)
            edge_b = (j, ej)
            if edge_a in used_edges or edge_b in used_edges:
                continue
            if degree[i] >= 3 or degree[j] >= 3:
                continue

            used_edges.add(edge_a)
            used_edges.add(edge_b)
            degree[i] += 1
            degree[j] += 1
            graph[i].append(j)
            graph[j].append(i)
            selected.append((rel, i, ei, j, ej))

            dfs(index + 1, pair_count, selected, used_edges, degree, graph)

            selected.pop()
            graph[i].pop()
            graph[j].pop()
            degree[i] -= 1
            degree[j] -= 1
            used_edges.remove(edge_a)
            used_edges.remove(edge_b)

    for pair_count in pair_counts:
        dfs(0, pair_count, [], set(), [0] * count, [[] for _ in range(count)])
        if results:
            break

    return results


def assemble_graph_matches(records, matches):
    adjacency = [[] for _ in records]
    for rel, i, ei, j, ej in matches:
        adjacency[i].append((j, ei, ej))
        adjacency[j].append((i, ej, ei))

    poses = [None] * len(records)
    poses[0] = identity_pose()
    stack = [0]
    closure_error = 0

    while stack:
        i = stack.pop()
        poly_i = records[i]["local_vertices"]
        pose_i = poses[i]
        for j, ei, ej in adjacency[i]:
            poly_j = records[j]["local_vertices"]
            ia, ib = records[i]["local_edges"][ei]
            ja, jb = records[j]["local_edges"][ej]
            wa = apply_pose_point(ia, pose_i)
            wb = apply_pose_point(ib, pose_i)
            proposed = align_edge_pose(jb, ja, wa, wb)
            if poses[j] is None:
                poses[j] = proposed
                stack.append(j)
            else:
                closure_error += pose_delta_error(poly_j, proposed, poses[j])

    for pose in poses:
        if pose is None:
            return None

    placed = []
    for record, pose in zip(records, poses):
        placed.append(
            {
                "id": record["id"],
                "label": record["label"],
                "from_x": record["from_x"],
                "from_y": record["from_y"],
                "target_vertices": apply_pose_polygon(record["local_vertices"], pose),
                "area": record["area"],
            }
        )
    return {"placed": placed, "closure_error": closure_error, "match_error": sum(item[0] for item in matches)}


def graph_layout_score(layout):
    state = {"placed": layout["placed"]}
    base_score = score_complete_state(state)
    if base_score is None:
        return None

    overlap_penalty = 0
    placed = layout["placed"]
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            if polygons_overlap(placed[i]["target_vertices"], placed[j]["target_vertices"]):
                overlap_penalty += 1

    total_score = (
        base_score["score"]
        + layout["closure_error"] * 0.02
        + layout["match_error"] * 4
        + overlap_penalty * 2
    )
    base_score["score"] = total_score
    base_score["match_error"] = layout["match_error"]
    return base_score


def solve_piece_layout_by_graph(records):
    if len(records) < 2:
        return None

    best_layout = None
    best_score = None
    for matches in collect_graph_matching_sets(records):
        layout = assemble_graph_matches(records, matches)
        if layout is None:
            continue
        score = graph_layout_score(layout)
        if score is None:
            continue
        if best_score is None or score["score"] < best_score["score"]:
            best_layout = layout
            best_score = score

    if best_layout is None:
        return None
    if best_score["score"] > MAX_ACCEPTED_GRAPH_SCORE:
        return None
    return {"state": {"placed": best_layout["placed"]}, "score": best_score, "mode": "GRAPH_AUTO"}


def make_initial_states(records):
    states = []
    for i, record in enumerate(records):
        placed_item = {
            "id": record["id"],
            "label": record["label"],
            "from_x": record["from_x"],
            "from_y": record["from_y"],
            "target_vertices": record["local_vertices"],
            "target_edges": record["local_edges"],
            "target_edge_lengths": record["edge_lengths"],
            "area": record["area"],
        }
        unused = []
        for j, other in enumerate(records):
            if i != j:
                unused.append(other)
        states.append({"placed": [placed_item], "unused": unused, "match_error": 0})
    return states


def local_edge_count(record):
    return len(record["local_edges"])


def local_edge(record, edge_index):
    return record["local_edges"][edge_index]


def make_edge_match_counts(records):
    counts = {}
    for ri, record in enumerate(records):
        for edge_index in range(local_edge_count(record)):
            counts[(record["id"], edge_index)] = 0

    for ai, record_a in enumerate(records):
        for bi, record_b in enumerate(records):
            if bi <= ai:
                continue
            for edge_a in range(local_edge_count(record_a)):
                len_a = record_a["edge_lengths"][edge_a]
                for edge_b in range(local_edge_count(record_b)):
                    if edge_is_match(len_a, record_b["edge_lengths"][edge_b]):
                        counts[(record_a["id"], edge_a)] += 1
                        counts[(record_b["id"], edge_b)] += 1
    return counts


def corner_perpendicular_score(poly, vertex_index):
    point = poly[vertex_index]
    prev_point = poly[(vertex_index - 1) % len(poly)]
    next_point = poly[(vertex_index + 1) % len(poly)]
    v1 = (prev_point[0] - point[0], prev_point[1] - point[1])
    v2 = (next_point[0] - point[0], next_point[1] - point[1])
    dot_sq = vector_dot_abs_sq(v1, v2, 1)
    if dot_sq > CORNER_DOT_LIMIT_SQ:
        return CORNER_DOT_LIMIT + 1
    return math.sqrt(dot_sq)


def anchor_polygon_at_corner(poly, vertex_index, use_next_as_x):
    corner = poly[vertex_index]
    prev_point = poly[(vertex_index - 1) % len(poly)]
    next_point = poly[(vertex_index + 1) % len(poly)]

    if use_next_as_x:
        axis_point = next_point
    else:
        axis_point = prev_point

    angle = -math.atan2(axis_point[1] - corner[1], axis_point[0] - corner[0])
    anchored = []
    for point in poly:
        shifted = (point[0] - corner[0], point[1] - corner[1])
        anchored.append(rotate_point(shifted, angle))

    bbox = bbox_of_points(anchored)
    if bbox[0] < -OVERLAP_EPSILON or bbox[1] < -OVERLAP_EPSILON:
        shifted = []
        for point in anchored:
            shifted.append((point[0] - bbox[0], point[1] - bbox[1]))
        anchored = shifted
    return anchored


def make_fast_anchor_states(records, match_counts):
    anchors = []
    for ri, record in enumerate(records):
        poly = record["local_vertices"]
        for vertex_index in range(len(poly)):
            corner_score = corner_perpendicular_score(poly, vertex_index)
            if corner_score > CORNER_DOT_LIMIT:
                continue

            prev_edge = (vertex_index - 1) % len(poly)
            next_edge = vertex_index
            outer_score = match_counts[(record["id"], prev_edge)] + match_counts[(record["id"], next_edge)]
            anchors.append((outer_score + corner_score, ri, vertex_index))

    if not anchors:
        return make_initial_states(records)

    anchors.sort(key=lambda item: item[0])
    states = []
    for _, ri, vertex_index in anchors[:MAX_FAST_ANCHORS]:
        record = records[ri]
        for use_next_as_x in (True, False):
            anchored_poly = anchor_polygon_at_corner(record["local_vertices"], vertex_index, use_next_as_x)
            placed_item = {
                "id": record["id"],
                "label": record["label"],
                "from_x": record["from_x"],
                "from_y": record["from_y"],
                "target_vertices": anchored_poly,
                "target_edges": polygon_edges(anchored_poly),
                "target_edge_lengths": record["edge_lengths"],
                "area": record["area"],
            }
            unused = []
            for i, other in enumerate(records):
                if i != ri:
                    unused.append(other)
            states.append({"placed": [placed_item], "unused": unused, "match_error": 0})
    return states


def partial_state_score(state):
    polys = []
    total_area = 0
    for item in state["placed"]:
        polys.append(item["target_vertices"])
        total_area += item["area"]
    bbox = bbox_of_polygons(polys)
    bbox_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return bbox_area / max(1, total_area) + state.get("match_error", 0) * 0.25


def expand_solver_state(state):
    candidates = []
    placed = state["placed"]

    for unused_index, record in enumerate(state["unused"]):
        source_poly = record["local_vertices"]
        source_edges = record["local_edges"]
        source_lengths = record["edge_lengths"]
        for source_edge in range(len(source_edges)):
            sp1, sp2 = source_edges[source_edge]
            source_len = source_lengths[source_edge]

            for base_item in placed:
                base_poly = base_item["target_vertices"]
                base_edges = base_item.get("target_edges")
                if base_edges is None:
                    base_edges = polygon_edges(base_poly)
                base_lengths = base_item.get("target_edge_lengths")
                if base_lengths is None:
                    base_lengths = edge_lengths(base_poly)
                for base_edge in range(len(base_edges)):
                    bp1, bp2 = base_edges[base_edge]
                    match_error = edge_match_error_with_lengths(
                        sp1, sp2, source_len, bp1, bp2, base_lengths[base_edge])
                    if match_error > EDGE_MATCH_RATIO:
                        continue

                    target_poly, _ = align_polygon_edge_to_edge(source_poly, source_edge, bp1, bp2)
                    if state_overlap(target_poly, placed):
                        continue

                    new_item = {
                        "id": record["id"],
                        "label": record["label"],
                        "from_x": record["from_x"],
                        "from_y": record["from_y"],
                        "target_vertices": target_poly,
                        "target_edges": polygon_edges(target_poly),
                        "target_edge_lengths": source_lengths,
                        "area": record["area"],
                    }
                    new_unused = []
                    for i, other in enumerate(state["unused"]):
                        if i != unused_index:
                            new_unused.append(other)
                    new_placed = placed[:] + [new_item]
                    candidates.append(
                        {
                            "placed": new_placed,
                            "unused": new_unused,
                            "match_error": state.get("match_error", 0) + match_error,
                        }
                    )

    candidates.sort(key=partial_state_score)
    return candidates[:MAX_SOLVER_BRANCHES]


def candidate_rect_angles(polys):
    angles = []
    for poly in polys:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            angle = -math.atan2(p2[1] - p1[1], p2[0] - p1[0])
            angles.append(angle)
    return angles


def point_near_value(value, target, margin):
    return abs(value - target) <= margin


def edge_on_rotated_bbox(p1, p2, bbox, margin):
    min_x, min_y, max_x, max_y = bbox
    if point_near_value(p1[0], min_x, margin) and point_near_value(p2[0], min_x, margin):
        return True
    if point_near_value(p1[0], max_x, margin) and point_near_value(p2[0], max_x, margin):
        return True
    if point_near_value(p1[1], min_y, margin) and point_near_value(p2[1], min_y, margin):
        return True
    if point_near_value(p1[1], max_y, margin) and point_near_value(p2[1], max_y, margin):
        return True
    return False


def count_pieces_with_outer_edges(rotated_polys, bbox, margin):
    count = 0
    for poly in rotated_polys:
        has_outer = False
        for i in range(len(poly)):
            if edge_on_rotated_bbox(poly[i], poly[(i + 1) % len(poly)], bbox, margin):
                has_outer = True
                break
        if has_outer:
            count += 1
    return count


def score_complete_state(state):
    polys = []
    total_area = 0
    for item in state["placed"]:
        polys.append(item["target_vertices"])
        total_area += item["area"]

    best = None
    for angle in candidate_rect_angles(polys):
        rotated_polys = []
        for poly in polys:
            rotated_polys.append(rotate_polygon(poly, angle))

        bbox = bbox_of_polygons(rotated_polys)
        width = max(1, bbox[2] - bbox[0])
        height = max(1, bbox[3] - bbox[1])
        bbox_area = width * height
        fill_error = abs(bbox_area - total_area) / max(1, total_area)
        aspect = width / height
        if aspect < 1:
            aspect = 1 / aspect
        aspect_penalty = 0
        if aspect < MIN_REASONABLE_ASPECT:
            aspect_penalty = MIN_REASONABLE_ASPECT - aspect
        elif aspect > MAX_REASONABLE_ASPECT:
            aspect_penalty = aspect - MAX_REASONABLE_ASPECT

        outer_margin = min(width, height) * OUTER_EDGE_MARGIN_RATIO
        outer_pieces = count_pieces_with_outer_edges(rotated_polys, bbox, outer_margin)
        missing_outer = len(rotated_polys) - outer_pieces
        outer_penalty = missing_outer * MISSING_OUTER_EDGE_PENALTY

        score = fill_error + aspect_penalty + outer_penalty

        if best is None or score < best["score"]:
            best = {
                "score": score,
                "angle": angle,
                "bbox": bbox,
                "width": width,
                "height": height,
                "aspect": aspect,
                "fill_error": fill_error,
                "outer_pieces": outer_pieces,
            }
    return best


def search_solver_states(states, target_count, state_limit):
    complete = []

    while states:
        next_states = []
        for state in states:
            if len(state["placed"]) >= target_count:
                complete.append(state)
                continue
            next_states += expand_solver_state(state)

        if not next_states:
            break
        next_states.sort(key=partial_state_score)
        states = next_states[:state_limit]

    return complete


def best_solution_from_complete(complete, mode):
    best_state = None
    best_score = None
    for state in complete:
        score = score_complete_state(state)
        if score is None:
            continue
        if best_score is None or score["score"] < best_score["score"]:
            best_state = state
            best_score = score

    if best_state is None:
        return None
    match_error = best_state.get("match_error", 0)
    best_score["score"] += match_error * 0.25
    best_score["match_error"] = match_error
    if best_score["score"] > MAX_ACCEPTED_SOLVER_SCORE:
        return None
    return {"state": best_state, "score": best_score, "mode": mode}


def solve_piece_layout(infos):
    records = make_piece_records(infos)
    if len(records) < 2:
        return None

    graph_solution = solve_piece_layout_by_graph(records)
    if graph_solution is not None:
        return graph_solution

    if FAST_SOLVE_PLAN:
        match_counts = make_edge_match_counts(records)
        fast_states = make_fast_anchor_states(records, match_counts)
        complete = search_solver_states(fast_states, len(records), MAX_FAST_STATES)
        solution = best_solution_from_complete(complete, "FAST_AUTO")
        if solution is not None:
            return solution

    states = make_initial_states(records)
    complete = search_solver_states(states, len(records), MAX_SOLVER_STATES)
    return best_solution_from_complete(complete, "AUTO")


def fit_solution_to_rect(solution, divider_y):
    score = solution["score"]
    plan_rect = make_plan_rect(divider_y, None)

    x, y, w, h = plan_rect
    bbox = score["bbox"]
    bbox_w = max(1, bbox[2] - bbox[0])
    bbox_h = max(1, bbox[3] - bbox[1])
    scale_x = (w - PLAN_FIT_MARGIN * 2) / bbox_w
    scale_y = (h - PLAN_FIT_MARGIN * 2) / bbox_h
    scale = min(scale_x, scale_y)
    draw_w = bbox_w * scale
    draw_h = bbox_h * scale
    offset_x = x + (w - draw_w) / 2
    offset_y = y + (h - draw_h) / 2

    steps = []
    for item in solution["state"]["placed"]:
        display_vertices = []
        for point in item["target_vertices"]:
            rotated = rotate_point(point, score["angle"])
            px = int(offset_x + (rotated[0] - bbox[0]) * scale + 0.5)
            py = int(offset_y + (rotated[1] - bbox[1]) * scale + 0.5)
            display_vertices.append((px, py))

        target_center = polygon_center(display_vertices)
        steps.append(
            {
                "id": item["id"],
                "label": item["label"],
                "from_x": item["from_x"],
                "from_y": item["from_y"],
                "to_x": target_center[0],
                "to_y": target_center[1],
                "to_theta": polygon_angle(display_vertices),
                "target_vertices": display_vertices,
                "mode": solution["mode"],
                "score": score["score"],
                "fill_error": score.get("fill_error", 0),
                "match_error": score.get("match_error", 0),
                "outer_pieces": score.get("outer_pieces", 0),
            }
        )

    return plan_rect, steps


def find_piece_for_label(infos, label, used):
    for info in infos:
        if info["label"] == label and info["id"] not in used:
            return info
    return None


def find_next_unused_piece(infos, used):
    for info in infos:
        if info["id"] not in used:
            return info
    return None


def has_complete_fig2_labels(infos):
    found = {}
    for info in infos:
        found[info["label"]] = True
    return (
        len(infos) == 4
        and found.get(LABEL_LARGE_TRI, False)
        and found.get(LABEL_LONG_PIECE, False)
        and found.get(LABEL_SIDE_PIECE, False)
        and found.get(LABEL_SMALL_PIECE, False)
    )


def make_fig2_plan_steps(infos, divider_y, mode):
    rect = make_plan_rect(divider_y, 10 / 6)
    used = []
    steps = []

    for label, template_points in PLAN_TEMPLATES:
        info = find_piece_for_label(infos, label, used)
        if info is None:
            info = find_next_unused_piece(infos, used)
        if info is None:
            continue

        target_vertices = scale_plan_points(template_points, rect)
        target_center = polygon_center(target_vertices)
        used.append(info["id"])
        steps.append(
            {
                "id": info["id"],
                "label": info["label"],
                "from_x": info["cx"],
                "from_y": info["cy"],
                "to_x": target_center[0],
                "to_y": target_center[1],
                "to_theta": polygon_angle(target_vertices),
                "target_vertices": target_vertices,
                "mode": mode,
                "score": 0,
                "fill_error": 0,
                "match_error": 0,
                "outer_pieces": len(infos),
            }
        )

    ordered = []
    for label in PLAN_MOVE_ORDER:
        for step in steps:
            if step["label"] == label and step not in ordered:
                ordered.append(step)
    for step in steps:
        if step not in ordered:
            ordered.append(step)

    return rect, ordered


def make_plan_steps(infos, divider_y):
    if USE_FIG2_TEMPLATE_FOR_KNOWN_SET and has_complete_fig2_labels(infos):
        return make_fig2_plan_steps(infos, divider_y, "FIG2_KNOWN")

    if AUTO_SOLVE_PLAN:
        solution = solve_piece_layout(infos)
        if solution is not None:
            rect, auto_steps = fit_solution_to_rect(solution, divider_y)
            auto_steps.sort(key=lambda item: -polygon_abs_area(item["target_vertices"]))
            return rect, auto_steps

    return make_fig2_plan_steps(infos, divider_y, "FIG2_FALLBACK")


def plan_signature(infos):
    signature = []
    for info in infos:
        qx = info["cx"] // PLAN_SIGNATURE_POS_TOL
        qy = info["cy"] // PLAN_SIGNATURE_POS_TOL
        signature.append((info["label"], qx, qy, info["main_edge_count"]))
    signature.sort()
    return signature


def divider_y_close(a, b):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= PLAN_CACHE_DIVIDER_TOL


def plan_steps_match_infos(infos, plan_steps):
    if not plan_steps:
        return False
    if len(plan_steps) != len(infos):
        return False
    if plan_steps[0]["mode"] == "FIG2_FALLBACK":
        return False

    ids = []
    for info in infos:
        ids.append(info["id"])
    for step in plan_steps:
        if step["id"] not in ids:
            return False
    return True


def plan_signature_close(a, b):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False

    for item_a, item_b in zip(a, b):
        if item_a[0] != item_b[0]:
            return False
        if abs(item_a[1] - item_b[1]) > 1:
            return False
        if abs(item_a[2] - item_b[2]) > 1:
            return False
        if abs(item_a[3] - item_b[3]) > 1:
            return False
    return True


def plan_scene_changed_after_lock(current_signature, locked_signature):
    if current_signature is None or locked_signature is None:
        return True
    if len(current_signature) != len(locked_signature):
        return True

    move_limit = max(1, PLAN_UNLOCK_MOVE_TOL // PLAN_SIGNATURE_POS_TOL)
    for item_a, item_b in zip(current_signature, locked_signature):
        if item_a[0] != item_b[0]:
            return True
        if abs(item_a[1] - item_b[1]) > move_limit:
            return True
        if abs(item_a[2] - item_b[2]) > move_limit:
            return True
        if abs(item_a[3] - item_b[3]) > 1:
            return True
    return False


def draw_polygon(img, points, color, thickness):
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        img.draw_line(p1[0], p1[1], p2[0], p2[1], color=color, thickness=thickness)


def draw_plan(img, plan_rect, plan_steps):
    if not ENABLE_PLAN_OVERLAY:
        return

    img.draw_rectangle(plan_rect, color=(255, 0, 255), thickness=2)
    img.draw_string(plan_rect[0], max(0, plan_rect[1] - 16), "PLAN", color=(255, 0, 255), scale=2)

    for step in plan_steps:
        points = step["target_vertices"]
        draw_polygon(img, points, color=(255, 255, 255), thickness=2)

        cx, cy = polygon_center(points)
        text = str(step["id"])
        img.draw_string(cx - 6, cy - 8, text, color=(0, 255, 0), scale=2)
        img.draw_line(step["from_x"], step["from_y"], step["to_x"], step["to_y"], color=(255, 0, 255), thickness=1)


def draw_blob(img, info):
    img.draw_cross(info["cx"], info["cy"], color=(0, 255, 0), size=10, thickness=2)

    vertices = info["vertices"]
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        img.draw_line(p1[0], p1[1], p2[0], p2[1], color=(0, 255, 255), thickness=2)
        img.draw_cross(p1[0], p1[1], color=(255, 255, 0), size=5, thickness=1)

    text = "%d:%s" % (info["id"], info["label"])
    img.draw_string(info["x"], max(0, info["y"] - 16), text, color=(255, 255, 0), scale=2)


def draw_status(img, fps, piece_count):
    text = "FPS:%.1f P:%d" % (fps, piece_count)
    img.draw_string(4, 4, text, color=(0, 255, 0), scale=2)


def draw_recognition_result(img, infos, plan_steps, plan_locked, stable_elapsed_ms):
    valid_plan = (
        len(infos) > 0
        and len(plan_steps) == len(infos)
        and plan_steps
        and plan_steps[0]["mode"] != "FIG2_FALLBACK"
    )

    if valid_plan:
        if plan_locked:
            text = "RECOG:OK LOCK"
            color = (0, 255, 0)
        else:
            stable_s = stable_elapsed_ms / 1000
            if stable_s > PLAN_STABLE_MS / 1000:
                stable_s = PLAN_STABLE_MS / 1000
            text = "RECOG:OK WAIT %.1fs" % stable_s
            color = (255, 255, 0)
    else:
        text = "RECOG:FAIL"
        color = (255, 0, 0)

    img.draw_string(4, 28, text, color=color, scale=2)


def print_infos(infos, fps, divider_y, roi):
    print("fps=%.1f pieces=%d divider=%s roi=%s" % (fps, len(infos), str(divider_y), str(roi)))
    for info in infos:
        print(
            "id=%d cx=%d cy=%d w=%d h=%d theta=%s pixels=%d area=%d"
            " aspect=%.2f fill=%.2f label=%s vsrc=%s edges_n=%d vertices=%s edges=%s"
            % (
                info["id"],
                info["cx"],
                info["cy"],
                info["w"],
                info["h"],
                str(info["theta"]),
                info["pixels"],
                info["area"],
                info["aspect"],
                info["fill"],
                info["label"],
                info["vertex_source"],
                info["main_edge_count"],
                str(info["vertices"]),
                str([int(v) for v in info["edge_lengths"]]),
            )
        )


def print_plan(plan_rect, plan_steps, locked=False):
    mode = "NONE"
    score = 0
    fill_error = 0
    match_error = 0
    outer_pieces = 0
    if plan_steps:
        mode = plan_steps[0]["mode"]
        score = plan_steps[0].get("score", 0)
        fill_error = plan_steps[0].get("fill_error", 0)
        match_error = plan_steps[0].get("match_error", 0)
        outer_pieces = plan_steps[0].get("outer_pieces", 0)
    print(
        "PLAN_MOVE mode=%s locked=%s count=%d score=%.3f fill=%.3f match=%.3f outer=%d target_rect=%s"
        % (mode, str(locked), len(plan_steps), score, fill_error, match_error, outer_pieces, str(plan_rect))
    )
    for index, step in enumerate(plan_steps):
        print(
            "move=%d id=%d label=%s from_x=%d from_y=%d to_x=%d to_y=%d to_theta=%d target_vertices=%s"
            % (
                index + 1,
                step["id"],
                step["label"],
                step["from_x"],
                step["from_y"],
                step["to_x"],
                step["to_y"],
                step["to_theta"],
                str(step["target_vertices"]),
            )
        )


def init_camera_display():
    sensor = Sensor(width=1280, height=960)
    sensor.reset()
    sensor.set_framesize(width=FRAME_W, height=FRAME_H)
    sensor.set_pixformat(Sensor.RGB565)

    Display.init(Display.ST7701, width=FRAME_W, height=FRAME_H, to_ide=True)
    MediaManager.init()

    sensor.run()
    return sensor


def main():
    sensor = init_camera_display()
    clock = time.clock()
    last_print = time.ticks_ms()
    frame_index = 0
    cached_divider_y = None
    last_signature = None
    stable_since_ms = last_print
    plan_locked = False
    locked_signature = None
    cached_plan_rect = None
    cached_plan_steps = []
    cached_plan_signature = None
    cached_plan_divider_y = None
    vertex_cache = {}

    while True:
        frame_index += 1
        clock.tick()
        img = sensor.snapshot()

        if (
            cached_divider_y is None
            or frame_index % DIVIDER_SCAN_INTERVAL_FRAMES == 0
        ):
            found_divider_y = find_divider_y(img)
            if found_divider_y is not None:
                cached_divider_y = found_divider_y
        divider_y = cached_divider_y
        upper_roi = make_upper_roi(divider_y)

        img.draw_rectangle(upper_roi, color=(0, 0, 255), thickness=2)
        if divider_y is not None:
            img.draw_line(0, divider_y, FRAME_W - 1, divider_y, color=(255, 255, 0), thickness=2)

        blobs = img.find_blobs(
            [BRIGHT_LAB_THRESHOLD],
            roi=upper_roi,
            pixels_threshold=MIN_PIXELS,
            area_threshold=MIN_AREA,
            merge=MERGE_BLOBS,
            margin=MERGE_MARGIN,
        )

        infos = []
        for blob in blobs:
            info = blob_info(blob, len(infos) + 1)
            if is_valid_piece(info, upper_roi):
                infos.append(info)

        infos.sort(key=lambda item: item["pixels"], reverse=True)
        infos = infos[:MAX_BLOBS]
        vertex_cache = attach_vertices(img, infos, vertex_cache)
        classify_pieces(infos)

        for i, info in enumerate(infos):
            info["id"] = i + 1

        now = time.ticks_ms()
        signature = plan_signature(infos)
        if plan_locked:
            if plan_scene_changed_after_lock(signature, locked_signature):
                plan_locked = False
                locked_signature = None
                last_signature = signature
                stable_since_ms = now
        elif plan_signature_close(signature, last_signature):
            pass
        else:
            last_signature = signature
            stable_since_ms = now

        stable_elapsed_ms = time.ticks_diff(now, stable_since_ms)

        if plan_locked and cached_plan_rect is not None:
            plan_rect = cached_plan_rect
            plan_steps = cached_plan_steps
        elif (
            cached_plan_rect is not None
            and plan_signature_close(signature, cached_plan_signature)
            and divider_y_close(divider_y, cached_plan_divider_y)
            and plan_steps_match_infos(infos, cached_plan_steps)
        ):
            plan_rect = cached_plan_rect
            plan_steps = cached_plan_steps
        else:
            plan_rect, plan_steps = make_plan_steps(infos, divider_y)
            cached_plan_rect = plan_rect
            cached_plan_steps = plan_steps
            cached_plan_signature = signature
            cached_plan_divider_y = divider_y
            if (
                FREEZE_PLAN_AFTER_STABLE
                and len(plan_steps) == len(infos)
                and stable_elapsed_ms >= PLAN_STABLE_MS
                and plan_steps
                and plan_steps[0]["mode"] != "FIG2_FALLBACK"
            ):
                plan_locked = True
                locked_signature = signature

        for info in infos:
            draw_blob(img, info)

        draw_plan(img, plan_rect, plan_steps)
        draw_status(img, clock.fps(), len(infos))
        draw_recognition_result(img, infos, plan_steps, plan_locked, stable_elapsed_ms)

        if time.ticks_diff(now, last_print) >= PRINT_INTERVAL_MS:
            print_infos(infos, clock.fps(), divider_y, upper_roi)
            print_plan(plan_rect, plan_steps, plan_locked)
            last_print = now

        Display.show_image(img)


main()
