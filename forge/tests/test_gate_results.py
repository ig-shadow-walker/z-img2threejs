#!/usr/bin/env python3
"""Tests for the recorded off-axis gate contract (`_shared/gate_results.py`).

The behavior under test is the one that was previously impossible: an off-axis
gate verdict reaching the deterministic layer at all. Before this contract,
`turntable_gate.py` / `self_intersection.py` / `attachment_anchor.py` /
`diagnose_render_multi_angle.py` printed a verdict to stdout and nothing read it,
so SKILL.md could call them blocking while `orchestrate_passes.py check` passed a
front-only render by construction.

The second thing under test is that coverage is part of the verdict. A gate that
ran clean over zero sampled vertices, or over zero orbit views, must NOT unlock a
pass — both gates already report their own coverage and both already carry source
comments warning that "0 violations" from a check that never ran is not a pass.

Pure stdlib. Run: python3 forge/tests/test_gate_results.py
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

sys.path.insert(0, str(FORGE / "_shared"))
sys.path.insert(0, str(FORGE / "stage3_build"))

from gate_results import (  # noqa: E402
    GATE_ATTACHMENT_ANCHOR,
    GATE_MULTI_ANGLE,
    GATE_RESULTS_KEY,
    GATE_SELF_INTERSECTION,
    GATE_TURNTABLE,
    GateResultError,
    build_entry,
    coverage_for,
    missing_gate_reasons,
    record_gate_result,
    satisfies_gate,
)
import orchestrate_passes  # noqa: E402


def run(script: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FORGE / script), *map(str, args)],
        capture_output=True,
        text=True,
    )


def write_png(path: Path, w: int = 48, h: int = 48, solid: bool = True) -> None:
    """A minimal valid RGB PNG. `solid` draws a centred opaque blob on white."""
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            inside = solid and abs(x - w // 2) < w // 4 and abs(y - h // 2) < h // 4
            raw += bytes((40, 40, 40) if inside else (250, 250, 250))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


class CoverageSemanticsTest(unittest.TestCase):
    def test_self_intersection_clean_run_over_zero_vertices_is_not_complete(self):
        # The exact vacuous pass this contract exists to refuse.
        coverage = coverage_for(
            GATE_SELF_INTERSECTION,
            {"selfIntersecting": False, "sampledVertexCount": 0, "totalVertexCount": 0},
        )
        self.assertFalse(coverage["complete"])

    def test_self_intersection_with_samples_is_complete(self):
        coverage = coverage_for(
            GATE_SELF_INTERSECTION,
            {"selfIntersecting": False, "sampledVertexCount": 512, "totalVertexCount": 900},
        )
        self.assertTrue(coverage["complete"])

    def test_turntable_missing_azimuth_is_not_complete(self):
        coverage = coverage_for(
            GATE_TURNTABLE,
            {"covered": False, "missingAzimuths": [180.0], "angles": [{}, {}, {}]},
        )
        self.assertFalse(coverage["complete"])
        self.assertEqual(coverage["missingAzimuths"], [180.0])

    def test_multi_angle_needs_at_least_two_orbit_views(self):
        # SKILL.md: a non-planar form must hold from >= 2 angles.
        self.assertFalse(coverage_for(GATE_MULTI_ANGLE, {"angles": [{}]})["complete"])
        self.assertTrue(coverage_for(GATE_MULTI_ANGLE, {"angles": [{}, {}]})["complete"])

    def test_attachment_unmeasured_is_not_complete(self):
        incomplete = coverage_for(
            GATE_ATTACHMENT_ANCHOR,
            {"attachmentCount": 2, "unmeasuredAttachments": ["branch-a"]},
        )
        self.assertFalse(incomplete["complete"])
        # A spec with nothing to measure is a real answer, not a missing one.
        vacuous = coverage_for(
            GATE_ATTACHMENT_ANCHOR, {"attachmentCount": 0, "unmeasuredAttachments": []}
        )
        self.assertTrue(vacuous["complete"])

    def test_unknown_gate_is_rejected(self):
        with self.assertRaises(GateResultError):
            coverage_for("not-a-gate", {})


class SatisfiesGateTest(unittest.TestCase):
    def test_passed_but_incomplete_coverage_does_not_satisfy(self):
        spec: dict = {}
        record_gate_result(
            spec,
            build_entry(
                GATE_SELF_INTERSECTION,
                {"selfIntersecting": False, "sampledVertexCount": 0, "totalVertexCount": 0},
                passed=True,
                pass_id="blockout",
            ),
        )
        self.assertFalse(satisfies_gate(spec, GATE_SELF_INTERSECTION, "blockout"))
        self.assertIn("measured nothing usable", missing_gate_reasons(spec, GATE_SELF_INTERSECTION, "blockout"))

    def test_failed_result_reports_as_failed_not_missing(self):
        spec: dict = {}
        record_gate_result(
            spec,
            build_entry(
                GATE_MULTI_ANGLE,
                {"angles": [{}, {}], "degenerate": True},
                passed=False,
                pass_id="blockout",
            ),
        )
        self.assertEqual(missing_gate_reasons(spec, GATE_MULTI_ANGLE, "blockout"), "multi-angle: recorded result failed")

    def test_no_result_reports_as_never_ran(self):
        self.assertIn("no recorded result", missing_gate_reasons({}, GATE_TURNTABLE, "blockout"))

    def test_pass_scoping_is_enforced(self):
        spec: dict = {}
        record_gate_result(
            spec,
            build_entry(GATE_MULTI_ANGLE, {"angles": [{}, {}]}, passed=True, pass_id="blockout"),
        )
        self.assertTrue(satisfies_gate(spec, GATE_MULTI_ANGLE, "blockout"))
        # Evidence from an earlier pass must not satisfy a later one.
        self.assertFalse(satisfies_gate(spec, GATE_MULTI_ANGLE, "material-pass"))


class RequirementDetectionTest(unittest.TestCase):
    def test_pre_2_2_specs_are_exempt(self):
        # Backward compatibility: a spec authored before the requirement existed is
        # not retroactively blocked, mirroring requires_topology_classification.
        spec = {"schemaVersion": "2.1", "componentTree": [{"primitive": "box"}]}
        self.assertFalse(orchestrate_passes.requires_off_axis_evidence(spec))

    def test_2_2_volumetric_spec_requires_off_axis(self):
        spec = {"schemaVersion": "2.2", "componentTree": [{"primitive": "box"}]}
        self.assertTrue(orchestrate_passes.requires_off_axis_evidence(spec))

    def test_all_plane_card_spec_is_exempt(self):
        # A plane card's silhouette is SUPPOSED to collapse when orbited.
        spec = {"schemaVersion": "2.2", "componentTree": [{"primitive": "plane-card"}]}
        self.assertFalse(orchestrate_passes.requires_off_axis_evidence(spec))

    def test_attachment_requirement_follows_component_roles(self):
        without = {"schemaVersion": "2.2", "componentTree": [{"id": "body", "primitive": "box"}]}
        self.assertFalse(orchestrate_passes.requires_attachment_evidence(without))


class GateCliRecordingTest(unittest.TestCase):
    """The CLIs must write their verdict where the deterministic layer can read it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.spec_path = self.tmp / "spec.json"
        self.png = self.tmp / "view.png"
        write_png(self.png)
        self.addCleanup(self._tmp.cleanup)

    def _write_spec(self) -> None:
        self.spec_path.write_text(
            json.dumps(
                {
                    "targetName": "Gate Fixture",
                    "schemaVersion": "2.2",
                    "suitability": "pass",
                    "componentTree": [{"id": "body", "primitive": "box"}],
                }
            ),
            encoding="utf-8",
        )

    def test_multi_angle_cli_records_into_spec(self):
        self._write_spec()
        result = run(
            "stage4_review/diagnose_render_multi_angle.py",
            "--reference", self.png,
            "--orbit", self.png,
            "--orbit", self.png,
            "--spec", self.spec_path,
            "--pass-id", "blockout",
            "--in-place",
            "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        entries = spec[GATE_RESULTS_KEY]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["gate"], GATE_MULTI_ANGLE)
        self.assertTrue(entries[0]["passed"])
        self.assertTrue(entries[0]["coverageComplete"])
        self.assertEqual(entries[0]["passId"], "blockout")

    def test_multi_angle_cli_without_recording_flags_leaves_spec_untouched(self):
        self._write_spec()
        before = self.spec_path.read_text(encoding="utf-8")
        result = run(
            "stage4_review/diagnose_render_multi_angle.py",
            "--reference", self.png, "--orbit", self.png, "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.spec_path.read_text(encoding="utf-8"), before)

    def test_attachment_anchor_records_without_colliding_with_positional_spec(self):
        # attachment_anchor.py already takes the spec positionally, so the recording
        # helper must not also add a --spec flag.
        self._write_spec()
        result = run(
            "stage4_review/attachment_anchor.py",
            self.spec_path, "--pass-id", "blockout", "--in-place", "--json",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        self.assertEqual(spec[GATE_RESULTS_KEY][0]["gate"], GATE_ATTACHMENT_ANCHOR)

    def test_self_intersection_records_load_error_as_failure(self):
        # A mesh that cannot be read is not a clean bill of health.
        self._write_spec()
        meshes = self.tmp / "meshes.json"
        meshes.write_text(json.dumps({"meshes": [{"name": "broken"}]}), encoding="utf-8")
        run(
            "stage4_review/self_intersection.py",
            meshes, "--spec", self.spec_path, "--pass-id", "blockout", "--in-place", "--json",
        )
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        entry = spec[GATE_RESULTS_KEY][0]
        self.assertEqual(entry["gate"], GATE_SELF_INTERSECTION)
        self.assertFalse(entry["passed"])

    def test_recorded_gate_results_validate(self):
        self._write_spec()
        run(
            "stage4_review/diagnose_render_multi_angle.py",
            "--reference", self.png, "--orbit", self.png, "--orbit", self.png,
            "--spec", self.spec_path, "--pass-id", "blockout", "--in-place", "--json",
        )
        validated = run("stage2_spec/validate_sculpt_spec.py", self.spec_path)
        combined = validated.stdout + validated.stderr
        self.assertNotIn(f"{GATE_RESULTS_KEY} must be", combined)
        self.assertNotIn("gate must be one of", combined)

    def test_validator_rejects_unknown_gate_name(self):
        self._write_spec()
        spec = json.loads(self.spec_path.read_text(encoding="utf-8"))
        spec[GATE_RESULTS_KEY] = [
            {"gate": "made-up", "passed": True, "coverageComplete": True, "passId": "blockout"}
        ]
        self.spec_path.write_text(json.dumps(spec), encoding="utf-8")
        validated = run("stage2_spec/validate_sculpt_spec.py", self.spec_path)
        self.assertIn("gate must be one of", validated.stdout + validated.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
