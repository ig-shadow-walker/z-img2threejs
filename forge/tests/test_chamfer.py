#!/usr/bin/env python3
"""Tests for directional chamfer distance (`_shared/chamfer.py`).

`divine_eye.py` listed "Directional Chamfer Distance" as a deferred signal. The reason it
is worth having is `test_catches_boundary_error_iou_is_insensitive_to`: a serrated contour
and a smooth one of the same radius share a high IoU, because the mismatched pixels sit in
a thin band along the edge and a thin band is a small fraction of a filled disc. Chamfer
measures the contour itself and separates them.

Pure stdlib. Run: python3 forge/tests/test_chamfer.py
"""
from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

SKILL = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SKILL / "forge" / "_shared"))

from chamfer import (  # noqa: E402
    DEFAULT_ORIENTATION_BINS,
    MAX_BOUNDARY_POINTS,
    boundary_points,
    directional_chamfer,
)

SIZE = 224


def disc(radius: float = 70.0, *, serration: float = 0.0, teeth: int = 36, size: int = SIZE) -> list[bool]:
    """A filled disc, optionally with a sawtooth radius.

    Serration changes the CONTOUR while leaving the enclosed area almost identical, which
    is exactly the configuration IoU handles badly.
    """
    mask = []
    for y in range(size):
        for x in range(size):
            dx, dy = x - size // 2, y - size // 2
            effective = radius
            if serration:
                angle = math.atan2(dy, dx)
                sector = int((angle + math.pi) / (2 * math.pi / teeth))
                effective = radius + (serration if sector % 2 else -serration)
            mask.append(dx * dx + dy * dy < effective * effective)
    return mask


def rectangle(width: int, height: int, size: int = SIZE) -> list[bool]:
    mask = []
    for y in range(size):
        for x in range(size):
            mask.append(
                abs(x - size // 2) < width // 2 and abs(y - size // 2) < height // 2
            )
    return mask


def iou(a: list[bool], b: list[bool]) -> float:
    intersection = sum(1 for x, y in zip(a, b, strict=True) if x and y)
    union = sum(1 for x, y in zip(a, b, strict=True) if x or y)
    return intersection / union if union else 0.0


class BoundaryExtractionTest(unittest.TestCase):
    def test_boundary_is_the_shell_not_the_interior(self):
        mask = disc(radius=60.0)
        points = boundary_points(mask, SIZE)
        filled = sum(1 for value in mask if value)
        # A disc's perimeter is ~2*pi*r, far smaller than its area.
        self.assertLess(len(points), filled // 4)
        self.assertGreater(len(points), 0)

    def test_empty_mask_has_no_boundary(self):
        self.assertEqual(boundary_points([False] * (SIZE * SIZE), SIZE), [])

    def test_orientation_bins_are_in_range(self):
        points = boundary_points(disc(), SIZE)
        for _x, _y, orientation in points:
            self.assertIn(orientation, range(DEFAULT_ORIENTATION_BINS))

    def test_axis_aligned_edges_get_distinct_orientations(self):
        # A wide, short rectangle's boundary is dominated by horizontal runs; a tall, narrow
        # one by vertical runs. The dominant orientation bin must differ between them.
        wide = boundary_points(rectangle(160, 24), SIZE)
        tall = boundary_points(rectangle(24, 160), SIZE)

        def dominant(points):
            counts: dict[int, int] = {}
            for _x, _y, orientation in points:
                counts[orientation] = counts.get(orientation, 0) + 1
            return max(counts, key=lambda key: counts[key])

        self.assertNotEqual(dominant(wide), dominant(tall))


class DirectionalChamferTest(unittest.TestCase):
    def test_identical_contours_score_perfectly(self):
        mask = disc()
        result = directional_chamfer(mask, mask, SIZE)
        self.assertEqual(result["distance"], 0.0)
        self.assertEqual(result["agreement"], 1.0)

    def test_catches_boundary_error_iou_is_insensitive_to(self):
        smooth = disc(radius=70.0)
        serrated = disc(radius=70.0, serration=5.0)
        area_iou = iou(smooth, serrated)
        result = directional_chamfer(smooth, serrated, SIZE)

        # IoU stays high because the disagreement is a thin band along the contour.
        self.assertGreater(area_iou, 0.80)
        # Chamfer must be materially more pessimistic about the same pair.
        self.assertLess(result["agreement"], area_iou)
        self.assertGreater(result["distance"], 1.0)

    def test_agreement_degrades_monotonically_with_serration(self):
        smooth = disc(radius=70.0)
        scores = [
            directional_chamfer(smooth, disc(radius=70.0, serration=depth), SIZE)["agreement"]
            for depth in (0.0, 3.0, 6.0, 10.0)
        ]
        self.assertEqual(scores, sorted(scores, reverse=True), scores)

    def test_bidirectional_max_catches_one_sided_missing_contour(self):
        # A small disc against a large one: the forward and backward distances differ, and
        # the reported distance must be the worse of the two.
        small = disc(radius=30.0)
        large = disc(radius=90.0)
        result = directional_chamfer(small, large, SIZE)
        self.assertAlmostEqual(
            result["distance"], max(result["forward"], result["backward"]), places=3
        )

    def test_empty_reference_is_absent_not_zero(self):
        # No contour means no comparison. Reporting 0.0 would read as a perfect match, and
        # reporting 1.0 would read as a pass; both are wrong.
        result = directional_chamfer([False] * (SIZE * SIZE), disc(), SIZE)
        self.assertIsNone(result["distance"])
        self.assertIsNone(result["agreement"])

    def test_empty_render_is_absent_not_zero(self):
        result = directional_chamfer(disc(), [False] * (SIZE * SIZE), SIZE)
        self.assertIsNone(result["agreement"])

    def test_distant_contours_floor_at_zero(self):
        # Two small discs far apart, beyond the full-mismatch distance.
        left = [False] * (SIZE * SIZE)
        right = [False] * (SIZE * SIZE)
        for y in range(SIZE):
            for x in range(SIZE):
                if (x - 30) ** 2 + (y - 30) ** 2 < 100:
                    left[y * SIZE + x] = True
                if (x - 190) ** 2 + (y - 190) ** 2 < 100:
                    right[y * SIZE + x] = True
        self.assertEqual(directional_chamfer(left, right, SIZE)["agreement"], 0.0)

    def test_orientation_awareness_separates_transposed_shapes(self):
        # A wide bar vs a tall bar overlap at the centre, so a plain nearest-boundary match
        # would look decent. Orientation matching must punish it.
        wide = rectangle(160, 24)
        tall = rectangle(24, 160)
        result = directional_chamfer(wide, tall, SIZE)
        self.assertLess(result["agreement"], 0.5)

    def test_boundary_points_are_strided_and_the_stride_is_reported(self):
        # Coverage honesty: a pathological mask must not silently stall or silently
        # under-sample. Fill a checkerboard so nearly every pixel is a boundary pixel.
        noisy = [((x + y) % 2 == 0) for y in range(SIZE) for x in range(SIZE)]
        result = directional_chamfer(noisy, disc(), SIZE)
        self.assertLessEqual(result["referenceBoundaryPoints"], MAX_BOUNDARY_POINTS)
        self.assertGreater(result["strides"]["reference"], 1)

    def test_result_is_deterministic(self):
        smooth, serrated = disc(), disc(serration=4.0)
        first = directional_chamfer(smooth, serrated, SIZE)
        second = directional_chamfer(smooth, serrated, SIZE)
        self.assertEqual(first, second)


def write_disc_png(path: Path, radius: float, *, serration: float = 0.0, size: int = SIZE) -> None:
    """Render one of the test discs as a real PNG so divine_eye.evaluate can read it."""
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            dx, dy = x - size // 2, y - size // 2
            effective = radius
            if serration:
                angle = math.atan2(dy, dx)
                sector = int((angle + math.pi) / (2 * math.pi / 36))
                effective = radius + (serration if sector % 2 else -serration)
            inside = dx * dx + dy * dy < effective * effective
            row += bytes((40, 40, 40) if inside else (250, 250, 250))
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


class DivineEyeIntegrationTest(unittest.TestCase):
    """The signal must be visible in the Eye's output without moving its verdict."""

    def setUp(self):
        sys.path.insert(0, str(SKILL / "forge" / "stage1_intake"))
        sys.path.insert(0, str(SKILL / "forge" / "stage4_review"))
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.smooth = self.tmp / "smooth.png"
        self.serrated = self.tmp / "serrated.png"
        write_disc_png(self.smooth, 70.0)
        write_disc_png(self.serrated, 70.0, serration=5.0)
        self.addCleanup(self._tmp.cleanup)

    def test_chamfer_appears_in_the_evaluation_output(self):
        import divine_eye

        result = divine_eye.evaluate(self.smooth, self.serrated)
        self.assertIn("directionalChamfer", result)
        self.assertIsNotNone(result["directionalChamfer"]["agreement"])
        self.assertIsNotNone(result["signals"]["directionalChamferAgreement"])

    def test_chamfer_is_declared_report_only(self):
        import divine_eye

        result = divine_eye.evaluate(self.smooth, self.serrated)
        self.assertIn("directionalChamfer", result["reportOnlySignals"])

    def test_chamfer_does_not_change_the_fidelity_score(self):
        # The decisive property of report-only: an uncalibrated signal must not move a
        # verdict. Fidelity is the weighted mean of the soft ensemble, so if chamfer had
        # been added to it, the score would differ from the ensemble's own mean.
        import divine_eye

        result = divine_eye.evaluate(self.smooth, self.serrated)
        weights = result["weights"]
        signals = result["signals"]
        self.assertNotIn("directionalChamfer", weights)
        self.assertNotIn("directionalChamferAgreement", weights)
        # Recompute fidelity from the declared weights and confirm it reproduces exactly.
        name_by_weight_key = {
            "proportion": None,  # derived, not echoed as a raw signal
            "symmetry": "symmetryParity",
            "phash": "phashSimilarity",
            "ssim": "ssim",
            "edgeOverlap": "edgeOverlap",
            "blowoutParity": "blowoutParity",
            "flatRegion": "flatRegionScore",
            "tonalParity": "tonalParity",
            "objectness": "objectness",
        }
        total = 0.0
        total_weight = 0.0
        for key, weight in weights.items():
            signal_key = name_by_weight_key.get(key, key)
            if signal_key is None:
                # proportion is computed from aspectRatioDelta; reconstruct it.
                value = max(0.0, 1.0 - signals["aspectRatioDelta"] / 0.05)
            else:
                value = signals[signal_key]
            total += value * weight
            total_weight += weight
        self.assertAlmostEqual(result["fidelity"], round(total / total_weight, 4), places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
