#!/usr/bin/env python3
"""Tests for feature-scale fidelity measurement (`stage4_review/microscope.py`).

The headline test is `test_divine_eye_passes_what_the_microscope_catches`: it builds a
512x512 pair differing only by one 14x14px feature (0.075% of the frame), shows that
`divine_eye.py` scores it a comfortable pass, and shows that the microscope fails it.
That is the documented blind spot -- "a feature a few pixels wide is not scored badly, it
is ABSENT before any comparison happens" -- demonstrated rather than asserted.

Also covered are the three rules the microscope spec insists on, each of which had
already produced a wrong answer in a real build before it was understood:

* the visible-footprint rule (never measure on an isolation render),
* the concave-feature rule (never colour-gate cavity shading),
* the hard limit (a feature the reference cannot resolve is excluded, not scored).

Pure stdlib. Run: python3 forge/tests/test_microscope.py
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
sys.path.insert(0, str(FORGE / "_shared"))
sys.path.insert(0, str(FORGE / "stage4_review"))

import microscope  # noqa: E402

SIZE = 512
BODY_RGB = (170, 140, 110)
BACKGROUND_RGB = (250, 250, 250)
NOTCH_RGB = (25, 25, 28)
# A 14x14 notch: 196px, which is 0.075% of a 512x512 frame and sub-pixel on a 64x64 grid.
NOTCH_BOX = (300, 314, 130, 144)
NOTCH_ROI = [0.566, 0.234, 0.633, 0.301]
BODY_ROI = [0.30, 0.40, 0.70, 0.70]


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(
        ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
    )


def write_scene(
    path: Path,
    *,
    notch: bool = True,
    occluder: bool = False,
    size: int = SIZE,
) -> None:
    """An elliptical body on white, optionally with the small notch and an occluder.

    `occluder` draws an opaque bar across the notch, which is how the visible-footprint
    fixtures model "this part is hidden in the real frame".
    """
    x0, x1, y0, y1 = NOTCH_BOX
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            inside = (x - size // 2) ** 2 / (150**2) + (y - size // 2) ** 2 / (200**2) < 1
            pixel = BODY_RGB if inside else BACKGROUND_RGB
            if notch and x0 <= x < x1 and y0 <= y < y1:
                pixel = NOTCH_RGB
            if occluder and y0 - 4 <= y < y1 + 4 and x0 - 4 <= x < x1 + 4:
                pixel = (90, 90, 95)
            row += bytes(pixel)
        rows.append(bytes(row))
    payload = b"".join(rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
        + _chunk(b"IDAT", zlib.compress(payload))
        + _chunk(b"IEND", b"")
    )


def run(script: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FORGE / script), *map(str, args)],
        capture_output=True,
        text=True,
    )


def critical_notch(**overrides) -> dict:
    descriptor = {
        "featureId": "body-notch-01",
        "componentId": "body",
        "criticality": "critical",
        "referenceRoi": NOTCH_ROI,
        "minimumProjectedPixels": 128,
    }
    descriptor.update(overrides)
    return descriptor


class MicroscopeCoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        self.with_notch = self.tmp / "with.png"
        self.without_notch = self.tmp / "without.png"
        write_scene(self.reference, notch=True)
        write_scene(self.with_notch, notch=True)
        write_scene(self.without_notch, notch=False)
        self.addCleanup(self._tmp.cleanup)

    def _run(self, render: Path, descriptors: list[dict], **kwargs) -> dict:
        return microscope.run_microscope(descriptors, self.reference, render, {}, **kwargs)

    def test_missing_small_feature_fails(self):
        # The spec's acceptance criterion: "A fixture identical everywhere but missing one
        # small tear FAILS."
        result = self._run(self.without_notch, [critical_notch()])
        self.assertFalse(result["passed"])
        self.assertIn("body-notch-01", result["gatingFailures"])

    def test_identical_render_passes(self):
        result = self._run(self.with_notch, [critical_notch()])
        self.assertTrue(result["passed"], result["features"])
        self.assertEqual(result["featureScores"]["body-notch-01"], 1.0)

    def test_divine_eye_passes_what_the_microscope_catches(self):
        # The whole reason this module exists, as an executable comparison.
        eye = run(
            "stage4_review/divine_eye.py",
            "--reference", self.reference, "--render", self.without_notch, "--json",
        )
        eye_result = json.loads(eye.stdout)
        self.assertGreater(
            eye_result["fidelity"], 0.9,
            "fixture no longer reproduces the blind spot: divine_eye should score this high",
        )
        self.assertEqual(eye_result["hardGateFailures"], [])

        scope = self._run(self.without_notch, [critical_notch()])
        self.assertFalse(
            scope["passed"],
            "the microscope must catch the feature divine_eye's 64x64 grid cannot see",
        )

    def test_score_is_worst_metric_not_average(self):
        # A correct colour must not be able to mask wrong linework.
        result = self._run(self.without_notch, [critical_notch()])
        record = result["features"][0]
        metrics = record["metrics"]
        gating_metrics = [
            metrics[name]
            for name in ("patchSsim", "edgeAgreement", "colourAgreement")
            if name in metrics and (name != "colourAgreement" or metrics.get("colourGates"))
        ]
        self.assertAlmostEqual(record["score"], min(gating_metrics), places=4)

    def test_worst_first_aggregation_reports_the_worst_feature(self):
        result = self._run(
            self.without_notch,
            [critical_notch(), {
                "featureId": "body-mass",
                "componentId": "body",
                "criticality": "important",
                "referenceRoi": BODY_ROI,
            }],
        )
        self.assertEqual(result["worstFeature"]["featureId"], "body-notch-01")
        # A perfectly-scoring large feature must not lift the verdict.
        self.assertEqual(result["featureScores"]["body-mass"], 1.0)
        self.assertFalse(result["passed"])

    def test_per_feature_action_distinguishes_absent_from_wrong(self):
        # Absent -> refine-spec (never built); present but wrong -> refine-code.
        wrong = self._run(self.without_notch, [critical_notch()])
        self.assertEqual(wrong["action"], "refine-code")

    def test_non_gating_feature_below_threshold_does_not_fail_the_pass(self):
        result = self._run(self.without_notch, [critical_notch(criticality="detail")])
        self.assertTrue(result["passed"])
        self.assertEqual(result["gatingFailures"], [])


class HardLimitTest(unittest.TestCase):
    """Zoom cannot create information the reference does not contain."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        self.render = self.tmp / "render.png"
        write_scene(self.reference, notch=True)
        write_scene(self.render, notch=False)
        self.addCleanup(self._tmp.cleanup)

    def test_tiny_roi_is_excluded_not_scored(self):
        tiny = critical_notch(
            referenceRoi=[0.60, 0.26, 0.607, 0.267],  # ~3.5px square
            minimumProjectedPixels=128,
        )
        result = microscope.run_microscope([tiny], self.reference, self.render, {})
        record = result["features"][0]
        self.assertEqual(record["status"], "insufficient-reference-resolution")
        self.assertIsNone(record["score"])
        self.assertIn("body-notch-01", result["insufficientReferenceResolution"])
        # Crucially it must NOT appear in the gating scores: failing the model for a
        # feature the reference cannot resolve would gate against an invention.
        self.assertNotIn("body-notch-01", result["featureScores"])
        self.assertTrue(result["passed"])


class VisibleFootprintTest(unittest.TestCase):
    """Measure the visible footprint, never an isolation render."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        self.render = self.tmp / "render.png"
        self.hidden = self.tmp / "hidden.png"
        write_scene(self.reference, notch=True)
        write_scene(self.render, notch=True)
        # The component-hidden frame: same scene with the notch component absent.
        write_scene(self.hidden, notch=False)
        self.addCleanup(self._tmp.cleanup)

    def test_footprint_restricts_measurement_and_is_reported(self):
        result = microscope.run_microscope(
            [critical_notch(componentId="notch")],
            self.reference,
            self.render,
            {"notch": self.hidden},
        )
        record = result["features"][0]
        self.assertEqual(record["measurementBasis"], "visible-footprint")
        # Exactly the 14x14 notch pixels differ between full and component-hidden frames.
        self.assertEqual(record["visibleFootprintPixels"], 196)

    def test_absent_footprint_is_not_a_silent_pass(self):
        # If the component-hidden frame is identical to the full frame, the component is
        # nowhere frontmost in this ROI: there is nothing to measure, and saying so beats
        # inventing a score.
        result = microscope.run_microscope(
            [critical_notch(componentId="notch")],
            self.reference,
            self.render,
            {"notch": self.render},
        )
        record = result["features"][0]
        self.assertEqual(record["status"], "component-not-visible")
        self.assertIsNone(record["score"])

    def test_missing_footprint_evidence_is_flagged_in_notes(self):
        result = microscope.run_microscope(
            [critical_notch(componentId="notch")], self.reference, self.render, {}
        )
        record = result["features"][0]
        self.assertEqual(record["measurementBasis"], "roi-foreground")
        self.assertTrue(
            any("visible footprint" in note for note in record["notes"]), record["notes"]
        )


class ConcaveColourRuleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        self.render = self.tmp / "render.png"
        write_scene(self.reference, notch=True)
        write_scene(self.render, notch=True)
        self.addCleanup(self._tmp.cleanup)

    def test_concave_feature_colour_is_report_only(self):
        result = microscope.run_microscope(
            [critical_notch(concave=True)], self.reference, self.render, {}
        )
        record = result["features"][0]
        self.assertFalse(record["metrics"]["colourGates"])
        self.assertIn("body-notch-01", result["reportOnly"])
        self.assertTrue(any("concave" in note for note in record["notes"]))

    def test_convex_dark_material_still_colour_gates(self):
        # The counterpart: a genuinely dark MATERIAL clusters near-black, so it must keep
        # its colour metric. Otherwise the concave rule would silence colour everywhere.
        result = microscope.run_microscope(
            [critical_notch()], self.reference, self.render, {}
        )
        record = result["features"][0]
        self.assertTrue(record["metrics"]["colourGates"])
        self.assertEqual(
            record["metrics"]["referenceDarkProfile"]["classification"], "material-like"
        )


class FailClosedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        self.render = self.tmp / "render.png"
        write_scene(self.reference)
        write_scene(self.render)
        self.addCleanup(self._tmp.cleanup)

    def test_empty_manifest_is_an_error_not_a_vacuous_pass(self):
        with self.assertRaises(microscope.MicroscopeError):
            microscope.run_microscope([], self.reference, self.render, {})

    def test_mismatched_resolution_is_refused(self):
        small = self.tmp / "small.png"
        write_scene(small, size=256)
        with self.assertRaises(microscope.MicroscopeError):
            microscope.run_microscope([critical_notch()], self.reference, small, {})

    def test_missing_roi_is_refused(self):
        with self.assertRaises(microscope.MicroscopeError):
            microscope.run_microscope(
                [{"featureId": "no-roi", "criticality": "critical"}],
                self.reference,
                self.render,
                {},
            )

    def test_inverted_roi_is_refused(self):
        with self.assertRaises(microscope.MicroscopeError):
            microscope.run_microscope(
                [critical_notch(referenceRoi=[0.7, 0.7, 0.2, 0.2])],
                self.reference,
                self.render,
                {},
            )

    def test_descriptor_without_feature_id_is_refused(self):
        with self.assertRaises(microscope.MicroscopeError):
            microscope.run_microscope(
                [{"referenceRoi": NOTCH_ROI}], self.reference, self.render, {}
            )


class CliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.reference = self.tmp / "ref.png"
        self.render = self.tmp / "render.png"
        write_scene(self.reference, notch=True)
        write_scene(self.render, notch=False)
        self.manifest = self.tmp / "features.json"
        self.manifest.write_text(json.dumps([critical_notch()]), encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_cli_exits_nonzero_on_gate_failure(self):
        result = run(
            "stage4_review/microscope.py",
            "--reference", self.reference, "--render", self.render,
            "--manifest", self.manifest,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("GATE FAILURE", result.stdout)

    def test_cli_writes_per_feature_ready_scores(self):
        scores = self.tmp / "scores.json"
        run(
            "stage4_review/microscope.py",
            "--reference", self.reference, "--render", self.render,
            "--manifest", self.manifest, "--scores-out", scores,
        )
        payload = json.loads(scores.read_text(encoding="utf-8"))
        self.assertIn("body-notch-01", payload)

        # The real integration: hand those scores straight to per_feature.py.
        targets = self.tmp / "targets.json"
        targets.write_text(
            json.dumps([{"id": "body-notch-01", "tier": "critical"}]), encoding="utf-8"
        )
        gate = run("stage4_review/per_feature.py", "--targets", targets, "--scores", scores)
        self.assertNotEqual(gate.returncode, 0, gate.stdout)

    def test_cli_writes_native_resolution_patches(self):
        patch_dir = self.tmp / "patches"
        run(
            "stage4_review/microscope.py",
            "--reference", self.reference, "--render", self.render,
            "--manifest", self.manifest, "--patch-dir", patch_dir,
        )
        written = sorted(path.name for path in patch_dir.glob("*.png"))
        self.assertEqual(written, ["body-notch-01-reference.png", "body-notch-01-render.png"])

    def test_cli_requires_a_descriptor_source(self):
        result = run(
            "stage4_review/microscope.py",
            "--reference", self.reference, "--render", self.render,
        )
        self.assertNotEqual(result.returncode, 0)

    def test_cli_errors_exit_two(self):
        # Fail closed: an unmeasurable request is an error, not a printed complaint.
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps([{"featureId": "x"}]), encoding="utf-8")
        result = run(
            "stage4_review/microscope.py",
            "--reference", self.reference, "--render", self.render, "--manifest", bad,
        )
        self.assertEqual(result.returncode, 2)

    def test_from_spec_derives_descriptors_from_detail_inventory(self):
        spec = self.tmp / "spec.json"
        spec.write_text(
            json.dumps(
                {
                    "targetName": "Fixture",
                    "schemaVersion": "2.2",
                    "detailInventory": {
                        "details": [
                            {
                                "id": "body-notch-01",
                                "componentId": "body",
                                "criticality": "critical",
                                "referenceRoi": NOTCH_ROI,
                            }
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )
        result = run(
            "stage4_review/microscope.py",
            "--reference", self.reference, "--render", self.render, "--from-spec", spec,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("body-notch-01", result.stdout)

    def test_from_spec_without_rois_is_an_error(self):
        spec = self.tmp / "empty.json"
        spec.write_text(
            json.dumps({"detailInventory": {"details": [{"id": "a", "note": "no roi"}]}}),
            encoding="utf-8",
        )
        result = run(
            "stage4_review/microscope.py",
            "--reference", self.reference, "--render", self.render, "--from-spec", spec,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("vacuous pass", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
