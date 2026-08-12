#!/usr/bin/env python3
"""Tests for the Divine Eye calibration corpus (`stage4_review/build_corpus.py`).

`calibrate_eye.py` has always required a labeled corpus before the Eye may hard-gate
anything -- "Only then may report-only be flipped to hard-gate" -- and no corpus existed,
while `divine_eye.py` hard-gated on hand-picked constants regardless.

The property these tests protect is the one that keeps the harness honest: a corpus of
synthetic perturbations is a valid REGRESSION fixture and must never authorize a threshold
change. Its images carry the reference's own shading and noise, so their absolute fidelity
numbers are not the numbers a Three.js render produces.

Pure stdlib. Run: python3 forge/tests/test_build_corpus.py
"""
from __future__ import annotations

import json
import math
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
sys.path.insert(0, str(FORGE / "_shared"))
sys.path.insert(0, str(FORGE / "stage4_review"))

import build_corpus  # noqa: E402
import calibrate_eye  # noqa: E402

SIZE = 160


def write_reference(path: Path, size: int = SIZE) -> None:
    """A shaded, coloured blob on white: enough structure for every perturbation to bite."""
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            dx, dy = x - size // 2, y - size // 2
            if dx * dx / (48**2) + dy * dy / (60**2) < 1:
                # A real gradient, so the flatten perturbation is not a no-op.
                shade = 0.55 + 0.45 * (1.0 - math.hypot(dx, dy) / 70.0)
                pixel = (
                    max(0, min(255, int(200 * shade))),
                    max(0, min(255, int(120 * shade))),
                    max(0, min(255, int(70 * shade))),
                )
            else:
                pixel = (250, 250, 250)
            row += bytes(pixel)
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


def run(script: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FORGE / script), *map(str, args)],
        capture_output=True,
        text=True,
    )


class CorpusStructureTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        write_reference(self.reference)
        self.addCleanup(self._tmp.cleanup)

    def test_builds_a_render_per_perturbation(self):
        manifest = build_corpus.build_corpus(self.reference, self.tmp / "out")
        self.assertEqual(len(manifest["pairs"]), len(build_corpus.PERTURBATIONS))
        for pair in manifest["pairs"]:
            self.assertTrue(Path(pair["render"]).exists(), pair["perturbation"])

    def test_every_pair_is_stamped_synthetic(self):
        manifest = build_corpus.build_corpus(self.reference, self.tmp / "out")
        for pair in manifest["pairs"]:
            self.assertEqual(pair["provenance"], build_corpus.SYNTHETIC_PROVENANCE)
        self.assertFalse(manifest["thresholdFlipAuthorized"])

    def test_corpus_without_identity_is_refused(self):
        # A corpus with only bad pairs measures false rejects and nothing else.
        with self.assertRaises(ValueError):
            build_corpus.build_corpus(self.reference, self.tmp / "out", names=["scale"])

    def test_unknown_perturbation_is_refused(self):
        with self.assertRaises(ValueError):
            build_corpus.build_corpus(
                self.reference, self.tmp / "out", names=["identity", "not-a-thing"]
            )

    def test_identity_render_is_pixel_identical_to_the_reference(self):
        manifest = build_corpus.build_corpus(self.reference, self.tmp / "out")
        identity = next(p for p in manifest["pairs"] if p["perturbation"] == "identity")
        from extract_pbr_evidence import load_image

        _w1, _h1, original, _ = load_image(self.reference)
        _w2, _h2, produced, _ = load_image(Path(identity["render"]))
        self.assertEqual(
            [pixel[:3] for pixel in original], [pixel[:3] for pixel in produced]
        )

    def test_parameter_overrides_are_recorded(self):
        manifest = build_corpus.build_corpus(
            self.reference, self.tmp / "out", overrides={"scale": 0.5}
        )
        scale = next(p for p in manifest["pairs"] if p["perturbation"] == "scale")
        self.assertEqual(scale["parameter"], 0.5)

    def test_hue_shift_preserves_luma(self):
        # The perturbation's whole purpose is to be invisible to luma signals, so that it
        # isolates whether the Eye can gate on colour at all.
        from extract_pbr_evidence import load_image

        _w, _h, pixels, _ = load_image(self.reference)
        shifted = build_corpus.perturb_hue_shift(pixels, SIZE, SIZE, 60.0)

        def luma(r, g, b):
            return 0.299 * r + 0.587 * g + 0.114 * b

        for index in range(0, len(shifted), 3 * 997):  # sparse sample, deterministic
            original = pixels[index // 3]
            before = luma(*original[:3])
            after = luma(shifted[index], shifted[index + 1], shifted[index + 2])
            self.assertAlmostEqual(before, after, delta=3.0)

    def test_known_gaps_are_declared_not_hidden(self):
        manifest = build_corpus.build_corpus(self.reference, self.tmp / "out")
        gaps = {pair["perturbation"] for pair in manifest["pairs"] if pair["knownGap"]}
        # Recorded from measurement: the Eye passes both of these today.
        self.assertIn("hue-shift", gaps)
        self.assertIn("blur", gaps)

    def test_negative_control_is_marked(self):
        manifest = build_corpus.build_corpus(self.reference, self.tmp / "out")
        control = next(p for p in manifest["pairs"] if p["perturbation"] == "remove-feature")
        self.assertTrue(control["negativeControl"])
        self.assertFalse(control["desiredDetection"])


class ProvenanceAuditTest(unittest.TestCase):
    """The load-bearing honesty rule."""

    def test_synthetic_corpus_never_authorizes_a_threshold_flip(self):
        audit = calibrate_eye.provenance_audit(
            [{"provenance": build_corpus.SYNTHETIC_PROVENANCE}] * 4
        )
        self.assertFalse(audit["thresholdFlipAuthorized"])
        self.assertTrue(audit["blockers"])

    def test_unlabelled_pairs_block_the_flip(self):
        audit = calibrate_eye.provenance_audit([{"reference": "a", "render": "b"}])
        self.assertFalse(audit["thresholdFlipAuthorized"])
        self.assertIn("no provenance", audit["blockers"][0])

    def test_real_renders_are_authorized(self):
        audit = calibrate_eye.provenance_audit(
            [{"provenance": build_corpus.REAL_RENDER_PROVENANCE}] * 3
        )
        self.assertTrue(audit["thresholdFlipAuthorized"])
        self.assertEqual(audit["blockers"], [])

    def test_mixed_provenance_blocks_the_flip(self):
        audit = calibrate_eye.provenance_audit(
            [
                {"provenance": build_corpus.REAL_RENDER_PROVENANCE},
                {"provenance": build_corpus.SYNTHETIC_PROVENANCE},
            ]
        )
        self.assertFalse(audit["thresholdFlipAuthorized"])


class ExpectationAuditTest(unittest.TestCase):
    def _rows(self, verdict: str) -> list[dict]:
        return [{"render": "r.png", "verdict": verdict, "fidelity": 0.9}]

    def test_recorded_behaviour_matching_reality_is_clean(self):
        pairs = [{"render": "r.png", "perturbation": "p", "expectedDetection": False,
                  "desiredDetection": False}]
        audit = calibrate_eye.expectation_audit(self._rows("pass"), pairs)
        self.assertTrue(audit["regressionClean"])
        self.assertEqual(audit["knownGaps"], [])

    def test_desensitized_signal_is_a_regression(self):
        pairs = [{"render": "r.png", "perturbation": "p", "expectedDetection": True,
                  "desiredDetection": True}]
        audit = calibrate_eye.expectation_audit(self._rows("pass"), pairs)
        self.assertFalse(audit["regressionClean"])

    def test_newly_sharpened_signal_is_reported_as_an_improvement_not_a_failure(self):
        pairs = [{"render": "r.png", "perturbation": "p", "expectedDetection": False,
                  "desiredDetection": True}]
        audit = calibrate_eye.expectation_audit(self._rows("reject"), pairs)
        self.assertTrue(audit["improvements"])
        self.assertTrue(audit["regressionClean"])

    def test_known_gap_is_reported_while_staying_clean(self):
        # The point: a passing run must still show the Eye's sensitivity holes.
        pairs = [{"render": "r.png", "perturbation": "hue-shift", "expectedDetection": False,
                  "desiredDetection": True, "defectClass": "wrong-colour"}]
        audit = calibrate_eye.expectation_audit(self._rows("pass"), pairs)
        self.assertTrue(audit["regressionClean"])
        self.assertEqual(len(audit["knownGaps"]), 1)
        self.assertEqual(audit["knownGaps"][0]["defectClass"], "wrong-colour")

    def test_not_asserted_pairs_are_skipped_with_a_reason(self):
        pairs = [{"render": "r.png", "perturbation": "flatten", "expectedDetection": None,
                  "desiredDetection": True, "rationale": "fixture-dependent"}]
        audit = calibrate_eye.expectation_audit(self._rows("pass"), pairs)
        self.assertTrue(audit["regressionClean"])
        self.assertIn("fixture-dependent", audit["notAsserted"][0])


class EndToEndTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        write_reference(self.reference)
        self.manifest = self.tmp / "corpus.json"
        self.addCleanup(self._tmp.cleanup)

    def test_cli_builds_then_calibrates(self):
        build = run(
            "stage4_review/build_corpus.py", self.reference,
            "--out-dir", self.tmp / "out", "--manifest", self.manifest,
        )
        self.assertEqual(build.returncode, 0, build.stderr)
        self.assertIn("thresholdFlipAuthorized: False", build.stdout)

        calibrate = run("stage4_review/calibrate_eye.py", "--corpus", self.manifest, "--json")
        payload = json.loads(calibrate.stdout)
        self.assertFalse(payload["thresholdFlipAuthorized"])
        self.assertTrue(payload["reportOnly"])
        # The harness must still be usable as a regression gate despite refusing the flip.
        self.assertIn("regressionClean", payload["expectations"])

    def test_calibrate_accepts_a_bare_pair_list_too(self):
        # Backward compatibility with the original corpus format.
        bare = self.tmp / "bare.json"
        renders = self.tmp / "out"
        build_corpus.build_corpus(self.reference, renders)
        bare.write_text(
            json.dumps(
                [
                    {"reference": str(self.reference), "render": str(renders / "identity.png"),
                     "label": "good", "provenance": build_corpus.REAL_RENDER_PROVENANCE},
                    {"reference": str(self.reference), "render": str(renders / "scale.png"),
                     "label": "bad", "provenance": build_corpus.REAL_RENDER_PROVENANCE},
                ]
            ),
            encoding="utf-8",
        )
        result = run("stage4_review/calibrate_eye.py", "--corpus", bare, "--json")
        payload = json.loads(result.stdout)
        # Real provenance + clean separation on this trivial pair => flip authorized.
        self.assertTrue(payload["provenance"]["thresholdFlipAuthorized"])

    def test_malformed_corpus_exits_two(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps({"nope": 1}), encoding="utf-8")
        result = run("stage4_review/calibrate_eye.py", "--corpus", bad)
        self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
