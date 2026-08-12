#!/usr/bin/env python3
"""Tests for the solved-camera CLI path and the projection camera-fitness gate.

Two defects are covered here.

1. `fit_camera_to_correspondences()` was fully implemented and tested as a library
   function, but `solve_camera_pose.py`'s CLI -- the command SKILL.md tells the agent
   to run -- had no way to pass correspondences. Every pipeline-driven camera was
   therefore the heuristic default guess, with `agentFill: true` on orientation and
   position.

2. `bake_projected_texture.py` accepted a missing or placeholder camera with a
   *warning* and emitted a usable-looking descriptor anyway. Projection fidelity is
   camera alignment, so that baked a misalignment into the albedo where later gates
   score it as a material defect.

The solver is verified by round trip: project known 3D points through a known camera,
hand only the resulting pixel pairs to the CLI, and require it to recover the camera.

Pure stdlib. Run: python3 forge/tests/test_camera_projection_gate.py
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

sys.path.insert(0, str(SKILL))
sys.path.insert(0, str(FORGE / "stage3_build"))

from forge.stage1_intake import camera_fitting_math as _math  # noqa: E402
from forge.stage1_intake.camera_fitting_types import CameraParameters  # noqa: E402
import bake_projected_texture as bake  # noqa: E402

TRUE_CAMERA = CameraParameters(
    image_width=256,
    image_height=256,
    fov_degrees=42.0,
    yaw_degrees=12.0,
    pitch_degrees=-7.0,
    roll_degrees=3.0,
    position=(0.15, -0.1, 2.9),
)

# Seven non-coplanar points: enough to clear MINIMUM_CORRESPONDENCES and to avoid the
# degenerate-geometry guard.
WORLD_POINTS = [
    (0.40, 0.30, 0.20),
    (-0.35, 0.25, -0.15),
    (0.10, -0.40, 0.30),
    (-0.20, -0.30, -0.25),
    (0.45, 0.05, -0.30),
    (-0.40, -0.05, 0.35),
    (0.05, 0.45, -0.05),
]


def run(script: str, *args: object) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(FORGE / script), *map(str, args)],
        capture_output=True,
        text=True,
    )


def write_png(path: Path, w: int = 256, h: int = 256) -> None:
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        for x in range(w):
            raw += bytes((200, 120, 60) if (x + y) % 7 else (250, 250, 250))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )


def correspondences_from_truth(points=WORLD_POINTS) -> list[dict]:
    """Project known world points through TRUE_CAMERA to get ground-truth pixel pairs."""
    out = []
    for index, point in enumerate(points):
        projected = _math.project_landmark(point, TRUE_CAMERA)
        assert projected is not None, "fixture points must project through the true camera"
        out.append({"name": f"lm{index}", "world": list(point), "observed": list(projected)})
    return out


class SolveCameraCliTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.image = self.tmp / "ref.png"
        write_png(self.image)
        self.corr = self.tmp / "corr.json"
        self.corr.write_text(json.dumps(correspondences_from_truth()), encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_cli_recovers_the_known_camera(self):
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", self.corr)
        self.assertEqual(result.returncode, 0, result.stderr)
        camera = json.loads(result.stdout)["referenceCamera"]
        self.assertEqual(camera["method"], "numerical-landmark-fit")
        fit = camera["fit"]
        self.assertEqual(fit["convergence"]["status"], "converged")
        self.assertLess(fit["finalReprojectionError"], 1e-3)
        # The fit must be a real improvement, not a restatement of the initial guess.
        self.assertLess(fit["finalReprojectionError"], fit["initialReprojectionError"])
        params = fit["cameraParameters"]
        self.assertAlmostEqual(params["fovDegrees"], TRUE_CAMERA.fov_degrees, delta=1e-3)
        self.assertAlmostEqual(params["yawDegrees"], TRUE_CAMERA.yaw_degrees, delta=1e-3)
        self.assertAlmostEqual(params["pitchDegrees"], TRUE_CAMERA.pitch_degrees, delta=1e-3)
        self.assertAlmostEqual(params["rollDegrees"], TRUE_CAMERA.roll_degrees, delta=1e-3)

    def test_solved_camera_clears_agent_fill_flags(self):
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", self.corr)
        camera = json.loads(result.stdout)["referenceCamera"]
        self.assertFalse(camera["fovDegrees"]["agentFill"])
        for axis in ("yawDegrees", "pitchDegrees", "rollDegrees"):
            self.assertFalse(camera["orientation"][axis]["agentFill"], axis)

    def test_solved_camera_records_the_source_image(self):
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", self.corr)
        camera = json.loads(result.stdout)["referenceCamera"]
        # The CLI resolves the path, which on macOS expands the /var -> /private/var symlink.
        self.assertEqual(camera["sourceImage"], str(self.image.resolve()))

    def test_legacy_path_without_correspondences_still_emits_placeholders(self):
        # Backward compatibility: the heuristic descriptor must survive unchanged.
        result = run("stage1_intake/solve_camera_pose.py", self.image)
        self.assertEqual(result.returncode, 0, result.stderr)
        camera = json.loads(result.stdout)["referenceCamera"]
        self.assertTrue(camera["orientation"]["yawDegrees"]["agentFill"])

    def test_too_few_correspondences_is_an_actionable_error(self):
        short = self.tmp / "short.json"
        short.write_text(json.dumps(correspondences_from_truth(WORLD_POINTS[:3])), encoding="utf-8")
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", short)
        self.assertEqual(result.returncode, 1)
        self.assertIn("need at least", result.stderr)

    def test_collinear_correspondences_are_refused(self):
        # has_degenerate_world_geometry() targets COLLINEAR (and coincident) point
        # sets: points strung along one axis leave the rotation about that axis
        # unconstrained, so the fit would return a confident wrong camera.
        collinear = [(t * 0.1, t * 0.1, t * 0.1) for t in range(1, len(WORLD_POINTS) + 1)]
        line = self.tmp / "line.json"
        line.write_text(json.dumps(correspondences_from_truth(collinear)), encoding="utf-8")
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", line)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("degenerate", result.stderr)

    def test_coplanar_correspondences_are_accepted(self):
        # Deliberate counterpart to the collinear case: a planar target is the classic
        # well-posed calibration configuration (a homography over 4+ coplanar points
        # determines focal length and pose), so it must NOT be treated as degenerate.
        coplanar = [(x, y, 0.0) for x, y, _z in WORLD_POINTS]
        flat = self.tmp / "flat.json"
        flat.write_text(json.dumps(correspondences_from_truth(coplanar)), encoding="utf-8")
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", flat)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(
            json.loads(result.stdout)["referenceCamera"]["fit"]["finalReprojectionError"], 1e-3
        )

    def test_malformed_correspondence_file_is_reported(self):
        bad = self.tmp / "bad.json"
        bad.write_text(json.dumps([{"name": "a", "world": [0, 0], "observed": [1, 2]}]), encoding="utf-8")
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", bad)
        self.assertEqual(result.returncode, 1)
        self.assertIn("world", result.stderr)

    def test_accepts_object_wrapped_correspondences(self):
        wrapped = self.tmp / "wrapped.json"
        wrapped.write_text(
            json.dumps({"correspondences": correspondences_from_truth()}), encoding="utf-8"
        )
        result = run("stage1_intake/solve_camera_pose.py", self.image, "--correspondences", wrapped)
        self.assertEqual(result.returncode, 0, result.stderr)


class CameraFitnessTest(unittest.TestCase):
    def test_missing_camera_is_unfitted(self):
        fitness = bake.camera_fitness(None, max_error_pixels=4.0)
        self.assertFalse(fitness["fitted"])

    def test_opaque_reference_id_is_unfitted(self):
        fitness = bake.camera_fitness({"reference": "cam-7"}, max_error_pixels=4.0)
        self.assertFalse(fitness["fitted"])
        self.assertIn("opaque reference id", fitness["blockers"][0])

    def test_high_reprojection_error_is_unfitted(self):
        camera = {
            "method": "numerical-landmark-fit",
            "fovDegrees": {"agentFill": False},
            "orientation": {
                "yawDegrees": {"agentFill": False},
                "pitchDegrees": {"agentFill": False},
                "rollDegrees": {"agentFill": False},
            },
            "fit": {"finalReprojectionError": 22.5, "convergence": {"status": "converged"}},
        }
        fitness = bake.camera_fitness(camera, max_error_pixels=4.0)
        self.assertFalse(fitness["fitted"])
        self.assertIn("exceeds", fitness["blockers"][0])

    def test_non_converged_fit_is_unfitted(self):
        camera = {
            "method": "numerical-landmark-fit",
            "fovDegrees": {"agentFill": False},
            "orientation": {
                "yawDegrees": {"agentFill": False},
                "pitchDegrees": {"agentFill": False},
                "rollDegrees": {"agentFill": False},
            },
            "fit": {"finalReprojectionError": 0.2, "convergence": {"status": "max-iterations"}},
        }
        self.assertFalse(bake.camera_fitness(camera, max_error_pixels=4.0)["fitted"])


class ProjectionGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.image = self.tmp / "ref.png"
        write_png(self.image)
        self.corr = self.tmp / "corr.json"
        self.corr.write_text(json.dumps(correspondences_from_truth()), encoding="utf-8")
        self.solved = self.tmp / "cam-solved.json"
        run("stage1_intake/solve_camera_pose.py", self.image,
            "--correspondences", self.corr, "--out", self.solved)
        self.placeholder = self.tmp / "cam-placeholder.json"
        run("stage1_intake/solve_camera_pose.py", self.image, "--out", self.placeholder)
        self.addCleanup(self._tmp.cleanup)

    def _bake(self, *extra: object) -> tuple[subprocess.CompletedProcess, dict]:
        result = run(
            "stage3_build/bake_projected_texture.py",
            "--reference-image", self.image,
            "--mesh-id", "blade",
            *extra,
        )
        return result, json.loads(result.stdout)["projectedTextureBake"]

    def test_no_camera_blocks_perspective_bake(self):
        result, payload = self._bake()
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertEqual(payload["gate"], "camera-fitness")
        # A blocked descriptor must not look like a successful run to a caller that
        # only checks the exit code.
        self.assertEqual(result.returncode, 1)

    def test_placeholder_camera_blocks_perspective_bake(self):
        _result, payload = self._bake("--camera", self.placeholder)
        self.assertEqual(payload["status"], "BLOCKED")
        self.assertTrue(
            any("agent-fill placeholder" in cause for cause in payload["cause"]),
            payload["cause"],
        )

    def test_solved_camera_passes(self):
        result, payload = self._bake("--camera", self.solved)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(payload["status"], "BLOCKED")
        self.assertTrue(payload["cameraFitness"]["fitted"])

    def test_orthographic_mode_is_exempt(self):
        # Orthographic front projection does not claim to reproduce the reference
        # viewpoint, so it does not depend on a solved camera.
        result, payload = self._bake("--projection-mode", "orthographic-front-projection")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(payload["status"], "BLOCKED")

    def test_triplanar_mode_is_exempt(self):
        result, payload = self._bake("--projection-mode", "triplanar-fallback")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(payload["status"], "BLOCKED")

    def test_override_proceeds_but_labels_the_result_an_approximation(self):
        result, payload = self._bake("--allow-unfitted-camera")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(payload["status"], "BLOCKED")
        self.assertTrue(
            any("APPROXIMATION" in item for item in payload["limitations"]),
            payload["limitations"],
        )

    def test_error_ceiling_is_configurable(self):
        # Tightening the ceiling below the achieved error must block a camera that
        # passes at the default.
        _ok, passing = self._bake("--camera", self.solved)
        self.assertNotEqual(passing["status"], "BLOCKED")
        _result, payload = self._bake(
            "--camera", self.solved, "--max-reprojection-error", "1e-12"
        )
        self.assertEqual(payload["status"], "BLOCKED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
