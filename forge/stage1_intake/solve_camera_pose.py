#!/usr/bin/env python3
"""Estimate a starting referenceCamera block for a reference image.

This is not camera calibration and it does not solve for the true focal
length, sensor, or 6-DoF pose that produced the photo. A single 2D image
under-constrains that problem. Instead this script emits a reasonable
default guess (FOV) plus image-derived facts (aspect ratio) and explicit
agent-fill placeholders (orientation, position) that the agent is expected
to refine by rendering the fitted mesh from this camera and visually
overlaying it against the reference image until silhouettes line up. The
`agentFill` flag on each field marks what still needs that visual pass.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from . import camera_fitting_math as _math
    from . import camera_fitting_solver as _solver
    from . import camera_fitting_types as _types
    from . import camera_image_helpers as _images
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from forge.stage1_intake import camera_fitting_math as _math
    from forge.stage1_intake import camera_fitting_solver as _solver
    from forge.stage1_intake import camera_fitting_types as _types
    from forge.stage1_intake import camera_image_helpers as _images

CONVERGED_RMS_PIXELS = _types.CONVERGED_RMS_PIXELS
DEFAULT_MAXIMUM_DAMPING_RETRIES = _types.DEFAULT_MAXIMUM_DAMPING_RETRIES
DEFAULT_MAXIMUM_ITERATIONS = _types.DEFAULT_MAXIMUM_ITERATIONS
FINITE_DIFFERENCE_STEPS = _types.FINITE_DIFFERENCE_STEPS
INITIAL_DAMPING = _types.INITIAL_DAMPING
MAXIMUM_DAMPING = _types.MAXIMUM_DAMPING
MAXIMUM_FOV_DEGREES = _types.MAXIMUM_FOV_DEGREES
MINIMUM_CAMERA_DEPTH = _types.MINIMUM_CAMERA_DEPTH
MINIMUM_CORRESPONDENCES = _types.MINIMUM_CORRESPONDENCES
MINIMUM_FOV_DEGREES = _types.MINIMUM_FOV_DEGREES
CameraFitDescriptor = _types.CameraFitDescriptor
CameraInitialization = _types.CameraInitialization
CameraParameters = _types.CameraParameters
DegenerateCorrespondencesError = _types.DegenerateCorrespondencesError
FitCameraParameters = _types.FitCameraParameters
InsufficientCorrespondencesError = _types.InsufficientCorrespondencesError
InvalidCameraDimensionsError = _types.InvalidCameraDimensionsError
InvalidInitialCameraError = _types.InvalidInitialCameraError
LandmarkCorrespondence = _types.LandmarkCorrespondence
NonFiniteCameraInputError = _types.NonFiniteCameraInputError
NormalizedCorrespondence = _types.NormalizedCorrespondence
ResidualDiagnostic = _types.ResidualDiagnostic
ScalarField = _types.ScalarField
SolverLimits = _types.SolverLimits
bmp_size = _images.bmp_size
build_camera = _images.build_camera
clamp = _images.clamp
detect_size = _images.detect_size
estimate_fov = _images.estimate_fov
gif_size = _images.gif_size
jpeg_size = _images.jpeg_size
png_size = _images.png_size
webp_size = _images.webp_size

# Mutable public seams retained for deterministic test callers.
MAXIMUM_ITERATIONS = DEFAULT_MAXIMUM_ITERATIONS
MAXIMUM_DAMPING_RETRIES = DEFAULT_MAXIMUM_DAMPING_RETRIES

__all__ = [
    "CONVERGED_RMS_PIXELS",
    "FINITE_DIFFERENCE_STEPS",
    "INITIAL_DAMPING",
    "MAXIMUM_DAMPING",
    "MAXIMUM_DAMPING_RETRIES",
    "MAXIMUM_FOV_DEGREES",
    "MAXIMUM_ITERATIONS",
    "MINIMUM_CAMERA_DEPTH",
    "MINIMUM_CORRESPONDENCES",
    "MINIMUM_FOV_DEGREES",
    "CameraFitDescriptor",
    "CameraInitialization",
    "DegenerateCorrespondencesError",
    "InsufficientCorrespondencesError",
    "InvalidCameraDimensionsError",
    "InvalidInitialCameraError",
    "LandmarkCorrespondence",
    "NonFiniteCameraInputError",
    "bmp_size",
    "build_camera",
    "clamp",
    "detect_size",
    "estimate_fov",
    "fit_camera_to_correspondences",
    "gif_size",
    "jpeg_size",
    "main",
    "png_size",
    "webp_size",
]


def fit_camera_to_correspondences(
    correspondences: Sequence[LandmarkCorrespondence],
    *,
    initial_camera: CameraInitialization,
) -> CameraFitDescriptor:
    """Fit the seven-DOF camera model after parsing all public numerical input."""
    normalized = _types.normalize_fit_input(correspondences, initial_camera)
    if _math.has_degenerate_world_geometry(normalized.correspondences):
        raise DegenerateCorrespondencesError(correspondence_count=len(normalized.correspondences))
    initial_residuals = _math.residual_components(normalized.correspondences, normalized.initial_camera)
    if initial_residuals is None:
        raise _invalid_initial_camera_error(normalized.correspondences, normalized.initial_camera)
    initial_error = _math.rms_reprojection_error(initial_residuals)
    fit_state = _solver.fit_parameters(normalized.correspondences, normalized.initial_camera, _solver_limits())
    fitted_camera = _camera_parameters_payload(fit_state.parameters)
    residuals = _residual_diagnostics(normalized.correspondences, fit_state.parameters)
    final_error = math.sqrt(sum(item["errorPixels"] * item["errorPixels"] for item in residuals) / len(residuals))
    return {
        "version": "1.0",
        "sourceImage": "landmark-correspondences",
        "solver": "stage1_intake/solve_camera_pose.py",
        "method": "numerical-landmark-fit",
        "imageWidth": fit_state.parameters.image_width,
        "imageHeight": fit_state.parameters.image_height,
        "fovDegrees": {
            "value": fitted_camera["fovDegrees"],
            "source": "numerical-landmark-fit",
            "agentFill": False,
            "rationale": "fitted from ordered 3D-to-2D landmark correspondences",
        },
        "aspect": {
            "value": fit_state.parameters.image_width / fit_state.parameters.image_height,
            "source": "initial-camera-image-dimensions",
            "agentFill": False,
        },
        "orientation": {
            "yawDegrees": _fitted_scalar(fitted_camera["yawDegrees"]),
            "pitchDegrees": _fitted_scalar(fitted_camera["pitchDegrees"]),
            "rollDegrees": _fitted_scalar(fitted_camera["rollDegrees"]),
            "note": "Fitted using pitch, yaw, then roll rotations matching the stage-1 projection convention.",
        },
        "position": {
            "hint": fitted_camera["position"],
            "distance": _fitted_scalar(fitted_camera["position"][2]),
            "note": "Fitted camera position in the existing scene-coordinate convention.",
        },
        "confidence": 0.95 if fit_state.status == "converged" else 0.5,
        "limitations": [
            "The fit holds image dimensions and principal point fixed from the initialization.",
            "The fit does not estimate lens distortion, sensor size, or a multi-view reconstruction.",
        ],
        "note": "Residuals are recomputed from the emitted cameraParameters descriptor in correspondence input order.",
        "fit": {
            "method": "central-difference-damped-least-squares",
            "cameraParameters": fitted_camera,
            "initialReprojectionError": initial_error,
            "finalReprojectionError": final_error,
            "residuals": residuals,
            "convergence": {
                "status": fit_state.status,
                "iterations": fit_state.iterations,
                "acceptedSteps": fit_state.accepted_steps,
                "rejectedSteps": fit_state.rejected_steps,
                "damping": fit_state.damping,
            },
        },
    }


def _solver_limits() -> SolverLimits:
    return SolverLimits(
        maximum_iterations=MAXIMUM_ITERATIONS,
        maximum_damping_retries=MAXIMUM_DAMPING_RETRIES,
        initial_damping=INITIAL_DAMPING,
        maximum_damping=MAXIMUM_DAMPING,
        converged_rms_pixels=CONVERGED_RMS_PIXELS,
        finite_difference_steps=FINITE_DIFFERENCE_STEPS,
    )


def _invalid_initial_camera_error(
    correspondences: Sequence[NormalizedCorrespondence], camera: CameraParameters
) -> InvalidInitialCameraError:
    for correspondence in correspondences:
        if _math.project_landmark(correspondence.world, camera) is None:
            return InvalidInitialCameraError(correspondence_name=correspondence.name)
    return InvalidInitialCameraError(correspondence_name=correspondences[0].name)


def _camera_parameters_payload(camera: CameraParameters) -> FitCameraParameters:
    return {
        "fovDegrees": camera.fov_degrees,
        "yawDegrees": camera.yaw_degrees,
        "pitchDegrees": camera.pitch_degrees,
        "rollDegrees": camera.roll_degrees,
        "position": [camera.position[0], camera.position[1], camera.position[2]],
    }


def _fitted_scalar(value: float) -> ScalarField:
    return {"value": value, "source": "numerical-landmark-fit", "agentFill": False}


def _residual_diagnostics(
    correspondences: Sequence[NormalizedCorrespondence], camera: CameraParameters
) -> list[ResidualDiagnostic]:
    diagnostics: list[ResidualDiagnostic] = []
    for correspondence in correspondences:
        projected = _math.project_landmark(correspondence.world, camera)
        if projected is None:
            raise _invalid_initial_camera_error(correspondences, camera)
        diagnostics.append(
            {
                "name": correspondence.name,
                "errorPixels": math.hypot(
                    projected[0] - correspondence.observed[0],
                    projected[1] - correspondence.observed[1],
                ),
            }
        )
    return diagnostics


@dataclass(frozen=True, slots=True)
class _FileCorrespondence:
    """One 3D-to-2D landmark pair parsed from `--correspondences`.

    Satisfies the LandmarkCorrespondence protocol without importing its dataclass,
    so the file format stays decoupled from the solver's internals.
    """

    name: str
    world: tuple[float, float, float]
    observed: tuple[float, float]


@dataclass(frozen=True, slots=True)
class _CliCameraInitialization:
    """Starting camera for the fit, seeded from image dimensions plus CLI hints."""

    image_width: int
    image_height: int
    fov_degrees: float
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float
    position: tuple[float, float, float]


def _number_triple(value: Any, label: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must be an array of {length} numbers")
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} must contain only numbers")
        out.append(float(item))
    return tuple(out)


def load_correspondences(path: Path) -> list[_FileCorrespondence]:
    """Parse the correspondence file the solver needs.

    Format -- a bare array, or {"correspondences": [...]}:

        [{"name": "muzzle-tip", "world": [0, 0.2, 1.1], "observed": [812, 344]}, ...]

    `world` is in the spec's own object-space coordinate frame; `observed` is in
    pixels on the reference image. At least MINIMUM_CORRESPONDENCES pairs are
    required, and they must not be coplanar-degenerate -- the solver raises for
    both cases rather than returning a confident wrong camera.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("correspondences") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError(
            "correspondence file must be a JSON array, or an object with a 'correspondences' array"
        )
    parsed: list[_FileCorrespondence] = []
    for index, item in enumerate(items):
        label = f"correspondences[{index}]"
        if not isinstance(item, dict):
            raise ValueError(f"{label} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label}.name must be a non-empty string")
        world = _number_triple(item.get("world"), f"{label}.world", 3)
        observed = _number_triple(item.get("observed"), f"{label}.observed", 2)
        parsed.append(
            _FileCorrespondence(
                name=name,
                world=(world[0], world[1], world[2]),
                observed=(observed[0], observed[1]),
            )
        )
    return parsed


def _initial_camera_from_args(image: Path, args: argparse.Namespace) -> _CliCameraInitialization:
    size = detect_size(image.read_bytes())
    if size is None:
        raise ValueError(
            "could not read image dimensions, so the fit has no pixel frame to solve in; "
            "supply a PNG/JPEG/GIF/WebP the probe can decode"
        )
    width, height = size
    aspect = width / height if height else 1.0
    default_fov, _rationale = estimate_fov(round(aspect, 4))
    return _CliCameraInitialization(
        image_width=width,
        image_height=height,
        fov_degrees=args.fov_degrees if args.fov_degrees is not None else default_fov,
        yaw_degrees=args.yaw,
        pitch_degrees=args.pitch,
        roll_degrees=args.roll,
        position=(0.0, args.height_offset, args.distance if args.distance is not None else 2.5),
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path)
    parser.add_argument("--fov-degrees", type=float, default=None, help="Override the default FOV guess")
    parser.add_argument("--yaw", type=float, default=0.0, help="Orientation placeholder, degrees")
    parser.add_argument("--pitch", type=float, default=0.0, help="Orientation placeholder, degrees")
    parser.add_argument("--roll", type=float, default=0.0, help="Orientation placeholder, degrees")
    parser.add_argument("--distance", type=float, default=None, help="Camera distance hint, scene units")
    parser.add_argument("--height-offset", type=float, default=0.0, help="Camera Y offset hint, scene units")
    parser.add_argument(
        "--correspondences",
        type=Path,
        default=None,
        help=(
            f"JSON of >= {MINIMUM_CORRESPONDENCES} 3D-to-2D landmark pairs "
            "({name, world:[x,y,z], observed:[u,v]}). Supplying this SOLVES the camera "
            "instead of emitting placeholder orientation/position."
        ),
    )
    parser.add_argument("--out", type=Path, help="Write the referenceCamera JSON block to this path")
    args = parser.parse_args(argv)
    image = args.image.expanduser().resolve()
    if not image.exists():
        parser.error(f"{image} does not exist")

    if args.correspondences is not None:
        # The solved path. Without it this CLI can only emit a default guess, which is
        # why a projection bake downstream had nothing better than an identity camera
        # to work from.
        correspondence_path = args.correspondences.expanduser().resolve()
        if not correspondence_path.exists():
            parser.error(f"{correspondence_path} does not exist")
        try:
            correspondences = load_correspondences(correspondence_path)
            camera = fit_camera_to_correspondences(
                correspondences,
                initial_camera=_initial_camera_from_args(image, args),
            )
        except InsufficientCorrespondencesError as exc:
            print(
                f"error: need at least {exc.required} correspondences to solve a camera, got {exc.actual}",
                file=sys.stderr,
            )
            return 1
        except DegenerateCorrespondencesError as exc:
            print(
                "error: the supplied correspondences are geometrically degenerate "
                f"({exc.correspondence_count} points with no depth spread); add landmarks that are "
                "not coplanar so the fit is constrained",
                file=sys.stderr,
            )
            return 1
        except (InvalidInitialCameraError, NonFiniteCameraInputError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        # The fitted descriptor keeps the source image, which the fit itself does not know.
        camera["sourceImage"] = str(image)
    else:
        camera = build_camera(image, args)

    text = json.dumps({"referenceCamera": camera}, indent=2, ensure_ascii=False)
    if args.out:
        out_path = args.out.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
