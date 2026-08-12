#!/usr/bin/env python3
"""Tests for `stage4_review/compare_region_passes.py`.

SKILL.md:102 relies on this module to block unsupported per-region claims: "missing
semantic-ID data blocks per-region confidence instead of falling back to whole-image color
or silhouette scores." docs/ARCHITECTURE.md lists it as a gate. It had no test coverage at
all, so nothing prevented that fallback from being reintroduced.

The load-bearing test here is `test_missing_semantic_id_blocks_per_region_claims`: without
a semantic-ID pass the module must report `regionEvidence: "unavailable"`, populate
`blockedClaims`, and exit non-zero -- never quietly substitute a global score.

Pure stdlib. Run: python3 forge/tests/test_compare_region_passes.py
"""
from __future__ import annotations

import json
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

SKILL = Path(__file__).resolve().parents[2]
FORGE = SKILL / "forge"

sys.path.insert(0, str(FORGE / "stage1_intake"))
sys.path.insert(0, str(FORGE / "stage4_review"))

import compare_region_passes as crp  # noqa: E402

SIZE = 32
REGION_A_COLOR = (255, 0, 0)
REGION_B_COLOR = (0, 255, 0)


def write_rgba_png(path: Path, pixel_fn, size: int = SIZE) -> None:
    """Write an RGBA PNG whose pixels come from `pixel_fn(x, y) -> (r,g,b,a)`."""
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            row += bytes(pixel_fn(x, y))
        rows.append(bytes(row))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        # colour type 6 = RGBA
        + chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + chunk(b"IEND", b"")
    )


def semantic_pixels(x: int, y: int) -> tuple[int, int, int, int]:
    """Left half region-a, right half region-b — a flat unique colour per region."""
    return (*(REGION_A_COLOR if x < SIZE // 2 else REGION_B_COLOR), 255)


def beauty_pixels(shift: int = 0):
    def inner(x: int, y: int) -> tuple[int, int, int, int]:
        value = max(0, min(255, 120 + shift if x < SIZE // 2 else 200))
        return (value, value, value, 255)

    return inner


def silhouette_pixels(radius: int):
    def inner(x: int, y: int) -> tuple[int, int, int, int]:
        inside = (x - SIZE // 2) ** 2 + (y - SIZE // 2) ** 2 < radius * radius
        return (255, 255, 255, 255) if inside else (0, 0, 0, 0)

    return inner


def run(script: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FORGE / script), *map(str, args)],
        capture_output=True,
        text=True,
    )


class RegionPassComparisonTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write_pass_images(self, *, with_semantic: bool, render_shift: int = 0) -> dict:
        """Build reference and render captures and return the manifest dict."""
        reference_passes: dict = {}
        render_passes: dict = {}

        def record(name: str, ref_fn, render_fn) -> None:
            ref_path = self.tmp / f"ref-{name}.png"
            render_path = self.tmp / f"out-{name}.png"
            write_rgba_png(ref_path, ref_fn)
            write_rgba_png(render_path, render_fn)
            reference_passes[name] = {"status": "recorded", "path": str(ref_path)}
            render_passes[name] = {"status": "recorded", "path": str(render_path)}

        record("beauty", beauty_pixels(0), beauty_pixels(render_shift))
        record("alpha-silhouette", silhouette_pixels(12), silhouette_pixels(12))
        if with_semantic:
            record("semantic-id", semantic_pixels, semantic_pixels)

        profile_path = self.tmp / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "regions": [
                        {"id": "region-a", "idColor": list(REGION_A_COLOR)},
                        {"id": "region-b", "idColor": list(REGION_B_COLOR)},
                    ]
                }
            ),
            encoding="utf-8",
        )
        return {
            "renderProfile": {"path": str(profile_path)},
            "captures": [
                {
                    "id": "hero",
                    "passes": render_passes,
                    "reference": {"passes": reference_passes},
                }
            ],
        }

    def _manifest(self, payload: dict) -> Path:
        path = self.tmp / "manifest.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_missing_semantic_id_blocks_per_region_claims(self):
        # The contract SKILL.md depends on. Without semantic-ID evidence there must be no
        # per-region numbers at all, rather than a global score wearing a region label.
        manifest = self._manifest(self._write_pass_images(with_semantic=False))
        result = crp.compare_capture(manifest, "hero")
        self.assertEqual(result["regionEvidence"], "unavailable")
        self.assertTrue(result["blockedClaims"])
        for pass_id, payload in result["results"].items():
            self.assertEqual(payload["perRegion"], {}, pass_id)

    def test_semantic_id_present_enables_per_region_scores(self):
        manifest = self._manifest(self._write_pass_images(with_semantic=True))
        result = crp.compare_capture(manifest, "hero")
        self.assertEqual(result["regionEvidence"], "semantic-id-reference-pass")
        self.assertEqual(result["blockedClaims"], [])
        self.assertIn("region-a", result["results"]["beauty"]["perRegion"])
        self.assertIn("region-b", result["results"]["beauty"]["perRegion"])

    def test_per_region_scores_localize_the_error(self):
        # The render differs only in the left half, which is region-a. Region-b must score
        # better than region-a, or per-region scoring is not actually per-region.
        manifest = self._manifest(self._write_pass_images(with_semantic=True, render_shift=-60))
        result = crp.compare_capture(manifest, "hero")
        per_region = result["results"]["beauty"]["perRegion"]
        self.assertLess(per_region["region-a"]["similarity"], per_region["region-b"]["similarity"])
        self.assertEqual(per_region["region-b"]["similarity"], 1.0)

    def test_identical_passes_score_perfectly(self):
        manifest = self._manifest(self._write_pass_images(with_semantic=True))
        result = crp.compare_capture(manifest, "hero")
        self.assertEqual(result["results"]["beauty"]["global"]["similarity"], 1.0)
        self.assertEqual(result["results"]["alpha-silhouette"]["global"]["iou"], 1.0)
        self.assertEqual(result["results"]["semantic-id"]["global"]["exactMatch"], 1.0)

    def test_unrecorded_passes_are_skipped_not_scored(self):
        payload = self._write_pass_images(with_semantic=True)
        payload["captures"][0]["passes"]["beauty"]["status"] = "pending"
        manifest = self._manifest(payload)
        result = crp.compare_capture(manifest, "hero")
        self.assertNotIn("beauty", result["passesCompared"])

    def test_mismatched_dimensions_are_refused(self):
        payload = self._write_pass_images(with_semantic=True)
        small = self.tmp / "small.png"
        write_rgba_png(small, beauty_pixels(0), size=16)
        payload["captures"][0]["passes"]["beauty"]["path"] = str(small)
        manifest = self._manifest(payload)
        with self.assertRaises(ValueError) as caught:
            crp.compare_capture(manifest, "hero")
        self.assertIn("dimensions differ", str(caught.exception))

    def test_unknown_capture_id_names_the_available_ids(self):
        # Regression: a bare next() raised StopIteration, whose str() is empty, so the CLI
        # reported {"error": ""} and said nothing about the cause.
        manifest = self._manifest(self._write_pass_images(with_semantic=True))
        with self.assertRaises(ValueError) as caught:
            crp.compare_capture(manifest, "no-such-capture")
        message = str(caught.exception)
        self.assertIn("no-such-capture", message)
        self.assertIn("hero", message)

    def test_capture_without_reference_record_is_refused(self):
        payload = self._write_pass_images(with_semantic=True)
        del payload["captures"][0]["reference"]
        manifest = self._manifest(payload)
        with self.assertRaises(ValueError):
            crp.compare_capture(manifest, "hero")


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.helper = RegionPassComparisonTest("run")
        self.helper.tmp = self.tmp

    def test_cli_exits_nonzero_when_region_claims_are_blocked(self):
        payload = self.helper._write_pass_images(with_semantic=False)
        manifest = self.tmp / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        out = self.tmp / "comparison.json"
        result = run(
            "stage4_review/compare_region_passes.py",
            "--manifest", manifest, "--capture-id", "hero", "--out", out,
        )
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertTrue(out.exists())
        self.assertTrue(json.loads(out.read_text(encoding="utf-8"))["blockedClaims"])

    def test_cli_exits_zero_with_semantic_evidence(self):
        payload = self.helper._write_pass_images(with_semantic=True)
        manifest = self.tmp / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        out = self.tmp / "comparison.json"
        result = run(
            "stage4_review/compare_region_passes.py",
            "--manifest", manifest, "--capture-id", "hero", "--out", out,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cli_reports_a_real_message_for_a_bad_capture_id(self):
        payload = self.helper._write_pass_images(with_semantic=True)
        manifest = self.tmp / "manifest.json"
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        result = run(
            "stage4_review/compare_region_passes.py",
            "--manifest", manifest, "--capture-id", "ghost", "--out", self.tmp / "x.json",
        )
        self.assertEqual(result.returncode, 2)
        payload = json.loads(result.stdout)
        self.assertTrue(payload["error"], "error message must not be empty")


if __name__ == "__main__":
    unittest.main(verbosity=2)
