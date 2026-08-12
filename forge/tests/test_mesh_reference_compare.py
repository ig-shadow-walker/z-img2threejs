#!/usr/bin/env python3
"""Tests for `stage4_review/mesh_reference_compare.py` (362 LOC, previously untested).

The module's docstring records three specific comparisons that measured the wrong thing,
and each of its design decisions exists to prevent one of them. Those decisions are what
these tests pin, because a silent regression in any of them reproduces a failure the
project has already paid for once:

* FEET, NOT BBOX TOP -- bands measured down from the bbox top were offset by whatever
  poked up highest (one ear tip vs two ears), so every band below described the
  misalignment rather than the shapes.
* PERCENTILES, NOT EXTREMES -- a long thin staff contributes few vertices but dominates
  any min/max width.
* CENTROID OFFSET, NOT ONLY WIDTH -- two bands can share a width and sit in different
  places, which a width-only comparison passes silently.

Also covered: the node-transform composition, which the docstring says was wrong twice --
once ignoring rotation entirely, once refusing the `matrix` form that three.js's own
GLTFExporter emits.

Pure stdlib; builds real uncompressed GLBs with struct. Run:
python3 forge/tests/test_mesh_reference_compare.py
"""
from __future__ import annotations

import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SKILL = Path(__file__).resolve().parents[2]
FORGE = SKILL / "forge"

sys.path.insert(0, str(FORGE / "stage4_review"))

import mesh_reference_compare as mrc  # noqa: E402

Point = tuple[float, float, float]


def write_glb(path: Path, points: list[Point], *, node: dict | None = None) -> None:
    """Write a minimal uncompressed GLB holding one POSITION accessor.

    `node` overrides the single node's transform record, which is how the transform tests
    exercise translation / quaternion rotation / the `matrix` form.
    """
    payload = b"".join(struct.pack("<3f", *point) for point in points)
    # The BIN chunk must be 4-byte aligned; 3 floats is already 12 bytes per vertex.
    gltf = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [node if node is not None else {"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,  # FLOAT
                "count": len(points),
                "type": "VEC3",
            }
        ],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(payload)}],
        "buffers": [{"byteLength": len(payload)}],
    }
    json_bytes = json.dumps(gltf).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)
    bin_bytes = payload + b"\x00" * (-len(payload) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    path.write_bytes(
        struct.pack("<III", 0x46546C67, 2, total)
        + struct.pack("<II", len(json_bytes), 0x4E4F534A)
        + json_bytes
        + struct.pack("<II", len(bin_bytes), 0x004E4942)
        + bin_bytes
    )


def column(
    *,
    half_width: float = 0.2,
    height: float = 2.0,
    steps: int = 40,
    ring: int = 12,
) -> list[Point]:
    """A vertical cylindrical column of vertices: enough points to clear the 20/band floor."""
    points: list[Point] = []
    for i in range(steps):
        y = height * i / (steps - 1)
        for j in range(ring):
            angle = 2 * math.pi * j / ring
            points.append((half_width * math.cos(angle), y, half_width * math.sin(angle)))
    return points


def figure() -> list[Point]:
    """A crude standing figure: wide hips, narrow waist, wide shoulders, narrow neck, head.

    find_landmarks() reads the waist and neck off the width curve's two minima, so the
    fixture needs those minima to actually exist.
    """
    points: list[Point] = []
    profile = [
        (0.00, 0.34), (0.10, 0.34), (0.20, 0.32), (0.30, 0.26),
        (0.40, 0.20), (0.50, 0.22), (0.60, 0.30), (0.70, 0.34),
        (0.78, 0.30), (0.84, 0.14), (0.88, 0.16), (0.94, 0.20), (1.00, 0.14),
    ]
    for fraction, half_width in profile:
        y = fraction * 2.0
        for j in range(24):
            angle = 2 * math.pi * j / 24
            points.append((half_width * math.cos(angle), y, half_width * 0.6 * math.sin(angle)))
    return points


def run(*args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FORGE / "stage4_review" / "mesh_reference_compare.py"), *map(str, args)],
        capture_output=True,
        text=True,
    )


class GlbReadingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_positions(self):
        path = self.tmp / "column.glb"
        points = column()
        write_glb(path, points)
        read = mrc.read_glb_positions(path)
        self.assertEqual(len(read), len(points))
        for original, loaded in zip(points, read, strict=True):
            for a, b in zip(original, loaded, strict=True):
                self.assertAlmostEqual(a, b, places=5)

    def test_node_translation_is_applied(self):
        # A mesh parked under a translated node must be measured where it actually is;
        # ignoring the transform makes a placement error look like a proportion error.
        path = self.tmp / "translated.glb"
        write_glb(path, [(0.0, 0.0, 0.0)], node={"mesh": 0, "translation": [1.0, 2.0, 3.0]})
        self.assertEqual(
            [tuple(round(value, 5) for value in point) for point in mrc.read_glb_positions(path)],
            [(1.0, 2.0, 3.0)],
        )

    def test_node_rotation_is_applied(self):
        # Regression on the first wrong version, which ignored rotation entirely.
        path = self.tmp / "rotated.glb"
        half = math.sqrt(0.5)  # 90 degrees about Z
        write_glb(path, [(1.0, 0.0, 0.0)], node={"mesh": 0, "rotation": [0.0, 0.0, half, half]})
        x, y, z = mrc.read_glb_positions(path)[0]
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)
        self.assertAlmostEqual(z, 0.0, places=5)

    def test_node_scale_is_applied(self):
        path = self.tmp / "scaled.glb"
        write_glb(path, [(1.0, 1.0, 1.0)], node={"mesh": 0, "scale": [2.0, 3.0, 4.0]})
        self.assertEqual(
            [tuple(round(value, 5) for value in point) for point in mrc.read_glb_positions(path)],
            [(2.0, 3.0, 4.0)],
        )

    def test_matrix_form_node_is_accepted(self):
        # Regression on the second wrong version, which REFUSED any node carrying a
        # `matrix` -- exactly how three.js's GLTFExporter writes this project's own model,
        # so the candidate could not be read at all.
        path = self.tmp / "matrix.glb"
        column_major = [
            2.0, 0.0, 0.0, 0.0,
            0.0, 2.0, 0.0, 0.0,
            0.0, 0.0, 2.0, 0.0,
            5.0, 6.0, 7.0, 1.0,
        ]
        write_glb(path, [(1.0, 1.0, 1.0)], node={"mesh": 0, "matrix": column_major})
        self.assertEqual(
            [tuple(round(value, 5) for value in point) for point in mrc.read_glb_positions(path)],
            [(7.0, 8.0, 9.0)],
        )


class NormalisationTest(unittest.TestCase):
    def test_feet_go_to_zero_and_height_to_one(self):
        points = [(0.0, 5.0, 0.0), (0.0, 9.0, 0.0), (1.0, 7.0, 0.0)]
        normalised = mrc.normalise(points)
        ys = [point[1] for point in normalised]
        self.assertAlmostEqual(min(ys), 0.0, places=6)
        self.assertAlmostEqual(max(ys), 1.0, places=6)

    def test_a_spike_on_top_does_not_shift_the_body_datum(self):
        # FEET, NOT BBOX TOP. Adding one tall spike must not move where the body sits,
        # which is the failure the docstring records: bands measured from the top were
        # offset by whichever single vertex poked up highest.
        body = column(height=2.0)
        with_spike = body + [(0.0, 2.6, 0.0)]

        base = mrc.normalise(body)
        spiked = mrc.normalise(with_spike)
        # The body's own lowest point stays pinned at 0 in both.
        self.assertAlmostEqual(min(point[1] for point in base), 0.0, places=6)
        self.assertAlmostEqual(min(point[1] for point in spiked), 0.0, places=6)
        # And the spike changes the scale only, not the datum: the body's normalised
        # heights stay in proportion to each other.
        base_mid = sorted(point[1] for point in base)[len(base) // 2]
        spiked_mid = sorted(point[1] for point in spiked)[: len(base)][len(base) // 2]
        self.assertLess(spiked_mid, base_mid)

    def test_lateral_centring_uses_the_median_not_the_bbox_centre(self):
        # A single long horizontal prop drags a bbox centre sideways; the median x does not.
        body = column(half_width=0.2, height=2.0)
        prop = [(4.0, 1.0, 0.0)]  # one far-out vertex, like a staff tip
        normalised = mrc.normalise(body + prop)
        body_xs = [point[0] for point in normalised[: len(body)]]
        median_x = mrc.percentile(body_xs, 0.5)
        # The body remains centred on ~0 despite the outlier.
        self.assertLess(abs(median_x), 0.05)

    def test_zero_height_mesh_exits_rather_than_dividing_by_zero(self):
        with self.assertRaises(SystemExit):
            mrc.normalise([(0.0, 1.0, 0.0), (1.0, 1.0, 0.0)])


class PercentileWidthTest(unittest.TestCase):
    def test_percentile_width_ignores_a_thin_prop_that_full_width_reports(self):
        # PERCENTILES, NOT EXTREMES: the documented reason the module does not use min/max.
        body = column(half_width=0.2, height=2.0, steps=40, ring=24)
        # One thin prop sticking far out at a single height band.
        prop = [(3.0, 1.0 + 0.001 * i, 0.0) for i in range(5)]
        profile = mrc.band_profile(mrc.normalise(body + prop), 10)
        affected = [band for band in profile if "width" in band and band["widthFull"] > band["width"] * 2]
        self.assertTrue(
            affected,
            "expected at least one band where the raw extent is dominated by the prop while "
            "the percentile width is not",
        )

    def test_percentile_interpolates(self):
        self.assertAlmostEqual(mrc.percentile([0.0, 1.0], 0.5), 0.5, places=6)
        self.assertAlmostEqual(mrc.percentile([0.0, 10.0], 0.9), 9.0, places=6)

    def test_percentile_of_empty_is_zero(self):
        self.assertEqual(mrc.percentile([], 0.5), 0.0)

    def test_sparse_bands_are_reported_without_metrics(self):
        # A band with too few vertices must not emit a width; a number from 3 samples would
        # read as measured when it is noise.
        profile = mrc.band_profile(mrc.normalise(column(steps=6, ring=3)), 40)
        sparse = [band for band in profile if "width" not in band]
        self.assertTrue(sparse)
        for band in sparse:
            self.assertLess(band["points"], 20)


class CentroidTest(unittest.TestCase):
    def test_same_width_in_a_different_place_is_caught(self):
        # CENTROID OFFSET, NOT ONLY WIDTH: a limb the right size on the wrong side.
        centred = column(half_width=0.2, height=2.0)
        shifted = [(x + 0.5, y, z) for x, y, z in centred]

        # normalise() re-centres on the median x, so compare the profiles of a mesh that
        # contains BOTH a centred trunk and a displaced limb.
        trunk = column(half_width=0.15, height=2.0, steps=40, ring=20)
        limb_centred = [(x, y * 0.25 + 0.8, z) for x, y, z in column(half_width=0.08, steps=30, ring=12)]
        limb_shifted = [(x + 0.45, y, z) for x, y, z in limb_centred]

        reference = mrc.band_profile(mrc.normalise(trunk + limb_centred), 10)
        candidate = mrc.band_profile(mrc.normalise(trunk + limb_shifted), 10)

        deltas = [
            abs(c["centroidX"] - r["centroidX"])
            for r, c in zip(reference, candidate, strict=True)
            if "centroidX" in r and "centroidX" in c
        ]
        self.assertTrue(deltas)
        self.assertGreater(max(deltas), 0.02, "a displaced limb must move some band's centroid")
        # Sanity: the plain shift fixture is unused for profiling but proves the helper works.
        self.assertEqual(len(shifted), len(centred))


class LandmarkBandingTest(unittest.TestCase):
    def test_landmarks_are_found_and_ordered(self):
        marks = mrc.find_landmarks(mrc.normalise(figure()))
        self.assertEqual(marks["ground"], 0.0)
        self.assertEqual(marks["top"], 1.0)
        self.assertLess(marks["ground"], marks["waist"])
        self.assertLess(marks["waist"], marks["neck"])
        self.assertLess(marks["neck"], marks["top"])

    def test_landmark_bands_return_profile_and_marks(self):
        profile, marks = mrc.landmark_bands(mrc.normalise(figure()), 3)
        self.assertIsInstance(profile, list)
        self.assertIn("waist", marks)
        self.assertTrue(profile)

    def test_landmark_bands_span_the_whole_height(self):
        profile, _marks = mrc.landmark_bands(mrc.normalise(figure()), 3)
        self.assertAlmostEqual(profile[0]["lo"], 0.0, places=6)
        self.assertAlmostEqual(profile[-1]["hi"], 1.0, places=6)


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "reference.glb"
        self.candidate = self.tmp / "candidate.glb"
        write_glb(self.reference, figure())
        self.addCleanup(self._tmp.cleanup)

    def test_identical_meshes_score_zero_deviation(self):
        write_glb(self.candidate, figure())
        result = run(self.reference, self.candidate, "--json", "--align", "height")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        # `bands` is the band COUNT; `rows` holds the per-band comparison.
        self.assertTrue(report["rows"])
        for row in report["rows"]:
            self.assertAlmostEqual(row["widthDelta"], 0.0, places=4)
            self.assertAlmostEqual(row["centroidXDelta"], 0.0, places=4)
        self.assertAlmostEqual(report["meanBandError"], 0.0, places=5)

    def test_a_wider_candidate_reports_positive_width_deltas(self):
        write_glb(self.candidate, [(x * 1.5, y, z) for x, y, z in figure()])
        result = run(self.reference, self.candidate, "--json", "--align", "height")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        deltas = [row["widthDelta"] for row in report["rows"]]
        self.assertTrue(deltas)
        self.assertGreater(max(deltas), 0.0)
        self.assertGreater(report["meanBandError"], 0.0)

    def test_both_alignment_modes_run(self):
        write_glb(self.candidate, figure())
        for mode in ("height", "landmarks"):
            result = run(self.reference, self.candidate, "--json", "--align", mode)
            self.assertEqual(result.returncode, 0, f"{mode}: {result.stderr}")
            self.assertIn("rows", json.loads(result.stdout))

    def test_text_output_names_the_worst_bands(self):
        # The module's stated purpose is to say WHERE the candidate is wrong, so the
        # human-readable path must surface that rather than only a score.
        write_glb(self.candidate, [(x * 1.5, y, z) for x, y, z in figure()])
        result = run(self.reference, self.candidate, "--align", "height")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("worst bands", result.stdout)
        self.assertIn("mean band error", result.stdout)

    def test_non_glb_input_is_refused(self):
        bogus = self.tmp / "not.glb"
        bogus.write_bytes(b"this is not a GLB at all")
        result = run(self.reference, bogus, "--json")
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
