"""Regression test: planning must work from saved pixels without generator state."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from puzzle_sim import (CARD_H, CARD_W, PIXELS_PER_CM, analyze_camera_frame,
                        generate_camera_frame, random_cut,
                        validate_cut_layout)


class ImageIsolationTest(unittest.TestCase):
    def test_saved_image_is_sufficient_for_planning(self):
        for piece_count in range(2, 5):
            with self.subTest(piece_count=piece_count):
                image = generate_camera_frame(seed=100 + piece_count,
                                              piece_count=piece_count)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "camera.png"
                    self.assertTrue(cv2.imwrite(str(path), image))
                    reloaded = cv2.imread(str(path), cv2.IMREAD_COLOR)

                pieces, transforms, _matches = analyze_camera_frame(reloaded)
                self.assertEqual(len(pieces), piece_count)
                self.assertEqual(len(transforms), piece_count)

    def test_joker_texture_is_detected_from_pixels_only(self):
        for piece_count in range(2, 5):
            with self.subTest(piece_count=piece_count):
                image = generate_camera_frame(
                    seed=200 + piece_count,
                    piece_count=piece_count,
                    material_mode="joker")
                reloaded = cv2.imdecode(
                    cv2.imencode(".png", image)[1], cv2.IMREAD_COLOR)
                pieces, transforms, _matches = analyze_camera_frame(reloaded)
                self.assertEqual(len(pieces), piece_count)
                self.assertEqual(len(transforms), piece_count)
                for transform in transforms:
                    # Each planned pose remains a proper rigid transform.
                    self.assertAlmostEqual(
                        float(np.linalg.det(transform[:2, :2])), 1.0, places=5)

    def test_sequential_cuts_cover_non_common_vertex_layouts(self):
        found_without_global_vertex = False
        for seed in range(20):
            pieces = random_cut(np.random.default_rng(seed), 4)
            for piece in pieces:
                self.assertLessEqual(len(piece), 5)
                lengths = np.linalg.norm(
                    np.roll(piece, -1, axis=0) - piece, axis=1)
                self.assertGreaterEqual(lengths.min(), 2 * PIXELS_PER_CM - 1e-6)
            candidates = pieces[0]
            common = [
                point for point in candidates
                if all(any(np.linalg.norm(point - other) < 1e-5
                           for other in piece)
                       for piece in pieces[1:])
            ]
            if not common:
                found_without_global_vertex = True
                break
        self.assertTrue(found_without_global_vertex)

    def test_all_cut_categories_are_visually_solvable(self):
        for mode in (
                "common", "boundary_fan", "strips", "equal_rectangles",
                "t_junction", "corner", "concave"):
            with self.subTest(mode=mode):
                image = generate_camera_frame(
                    seed=7, piece_count=4, material_mode="color",
                    cut_mode=mode)
                pieces, transforms, matches = analyze_camera_frame(image, mode)
                self.assertEqual(len(pieces), 4)
                self.assertEqual(len(transforms), 4)
                partial_count = sum(
                    tuple(match[5:]) != (0., 1., 0., 1.)
                    for match in matches)
                self.assertEqual(partial_count, 1 if mode == "t_junction" else 0)
                if mode == "concave":
                    self.assertTrue(any(
                        not cv2.isContourConvex(
                            p.astype(np.float32).reshape(-1, 1, 2))
                        for p in pieces))

    def test_field_geometry_requirements_for_every_category(self):
        modes = (
            "common", "boundary_fan", "strips", "equal_rectangles",
            "t_junction", "corner", "concave")
        for mode in modes:
            for piece_count in range(2, 5):
                for seed in range(30):
                    with self.subTest(
                            mode=mode, pieces=piece_count, seed=seed):
                        validate_cut_layout(random_cut(
                            np.random.default_rng(seed), piece_count, mode))

    def test_concave_outline_survives_pixel_only_detection(self):
        seen_vertex_counts = set()
        for piece_count in range(2, 5):
            for seed in range(30):
                with self.subTest(pieces=piece_count, seed=seed):
                    image = generate_camera_frame(
                        seed, piece_count, "color", "concave")
                    pieces, transforms, _ = analyze_camera_frame(
                        image, "concave")
                    self.assertEqual(len(pieces), piece_count)
                    self.assertEqual(len(transforms), piece_count)
                    self.assertTrue(any(
                        not cv2.isContourConvex(
                            p.astype(np.float32).reshape(-1, 1, 2))
                        for p in pieces))
                    seen_vertex_counts.update(
                        len(p) for p in pieces
                        if not cv2.isContourConvex(
                            p.astype(np.float32).reshape(-1, 1, 2)))
        self.assertTrue({4, 5}.issubset(seen_vertex_counts))

    def test_four_identical_rectangles_are_solved_without_texture(self):
        source = random_cut(np.random.default_rng(0), 4, "equal_rectangles")
        widths = []
        heights = []
        for piece in source:
            low, high = piece.min(axis=0), piece.max(axis=0)
            widths.append(high[0] - low[0])
            heights.append(high[1] - low[1])
        self.assertTrue(np.allclose(widths, CARD_W / 2))
        self.assertTrue(np.allclose(heights, CARD_H / 2))
        for seed in range(20):
            image = generate_camera_frame(
                seed, 4, "color", "equal_rectangles")
            pieces, transforms, _ = analyze_camera_frame(
                image, "equal_rectangles")
            self.assertEqual(len(pieces), 4)
            self.assertEqual(len(transforms), 4)


if __name__ == "__main__":
    unittest.main()
