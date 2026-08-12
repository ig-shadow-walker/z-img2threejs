#!/usr/bin/env python3
"""Emit a projection/UV-bake descriptor for photo-projected texturing.

This script does not perform GPU projective texturing and it does not
rasterize or bake any pixels. Actual camera-space projection of the
(ideally de-lit) reference image onto the fitted mesh, and the bake of that
projection into the mesh's UV space, is a Three.js runtime operation (a
projective ShaderMaterial, e.g. the `three-projected-material` technique).
What this script produces is the plan: a validated, versioned descriptor
that records which camera, which source image(s), which mesh, and which
projection settings the Three.js generator/agent should use to actually run
that bake, plus the back/side inference strategy for regions the camera
never saw.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


VALID_PROJECTION_MODES = ("perspective-camera-projection", "orthographic-front-projection", "triplanar-fallback")
VALID_UNSEEN_STRATEGIES = ("mirror-symmetry", "palette-continue", "request-additional-view", "leave-unprojected")

# Projection is the largest single fidelity lever in the pipeline, and it is
# entirely a function of camera alignment: the pixels land where the camera says
# they land. A perspective projection through a default-guess camera therefore
# bakes a misalignment into the albedo, which every later gate then scores as a
# material problem. So an unfitted camera BLOCKS a perspective bake instead of
# emitting a warning next to a descriptor that will be used anyway.
#
# The orthographic and triplanar modes are exempt: neither claims to reproduce
# the reference viewpoint, so neither depends on a solved camera.
CAMERA_DEPENDENT_PROJECTION_MODES = frozenset({"perspective-camera-projection"})

# RMS reprojection error, in reference-image pixels, above which a solved camera is
# not trustworthy enough to project through. The solver converges to well under 1px
# on a well-conditioned correspondence set, so this is generous rather than strict.
DEFAULT_MAX_REPROJECTION_ERROR_PIXELS = 4.0


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _scalar_is_agent_fill(block: Any) -> bool:
    return isinstance(block, dict) and block.get("agentFill") is True


def camera_fitness(camera: dict[str, Any] | None, *, max_error_pixels: float) -> dict[str, Any]:
    """Judge whether a camera block was actually solved, and how well.

    Returns `{"fitted": bool, "blockers": [...], "reprojectionErrorPixels": float|None}`.
    `solve_camera_pose.py` already marks every unsolved field `agentFill: true` and
    tags the solved path `method: numerical-landmark-fit`, so this reads the
    provenance the producer already recorded rather than guessing.
    """
    blockers: list[str] = []
    if camera is None:
        return {
            "fitted": False,
            "reprojectionErrorPixels": None,
            "blockers": [
                "no camera supplied; a perspective projection would use an identity/front "
                "camera assumption and bake that misalignment into the albedo"
            ],
        }
    if set(camera.keys()) == {"reference"}:
        return {
            "fitted": False,
            "reprojectionErrorPixels": None,
            "blockers": [
                f"camera is an opaque reference id ({camera['reference']!r}) with no fitted "
                "parameters to project through"
            ],
        }

    fit = camera.get("fit") if isinstance(camera.get("fit"), dict) else None
    error = fit.get("finalReprojectionError") if fit else None
    error_value = float(error) if isinstance(error, (int, float)) and not isinstance(error, bool) else None

    if camera.get("method") != "numerical-landmark-fit" or fit is None:
        blockers.append(
            "camera was not solved from landmark correspondences (method is "
            f"{camera.get('method')!r}); re-run solve_camera_pose.py with --correspondences"
        )

    orientation = camera.get("orientation") if isinstance(camera.get("orientation"), dict) else {}
    placeholder_fields = [
        name
        for name, key in (
            ("fovDegrees", "fovDegrees"),
            ("orientation.yawDegrees", "yawDegrees"),
            ("orientation.pitchDegrees", "pitchDegrees"),
            ("orientation.rollDegrees", "rollDegrees"),
        )
        if _scalar_is_agent_fill(camera.get(key) if key == "fovDegrees" else orientation.get(key))
    ]
    if placeholder_fields:
        blockers.append(
            "camera still carries agent-fill placeholder values for "
            f"{', '.join(placeholder_fields)}"
        )

    if error_value is not None and error_value > max_error_pixels:
        blockers.append(
            f"camera fit RMS reprojection error {error_value:.3f}px exceeds the "
            f"{max_error_pixels:.3f}px ceiling for projection"
        )

    convergence = fit.get("convergence") if fit and isinstance(fit.get("convergence"), dict) else {}
    if convergence and convergence.get("status") != "converged":
        blockers.append(f"camera fit did not converge (status {convergence.get('status')!r})")

    return {
        "fitted": not blockers,
        "reprojectionErrorPixels": error_value,
        "blockers": blockers,
    }


def load_camera(camera_arg: str | None) -> tuple[dict[str, Any] | None, list[str]]:
    warnings: list[str] = []
    if not camera_arg:
        warnings.append("no --camera reference supplied; projection will use an identity/front camera assumption")
        return None, warnings
    path = Path(camera_arg).expanduser()
    if not path.exists():
        warnings.append(f"--camera path {camera_arg!r} does not exist; recording it as an opaque reference id instead")
        return {"reference": camera_arg}, warnings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append(f"could not parse --camera JSON at {path}: {exc}; recording it as an opaque reference id")
        return {"reference": camera_arg}, warnings
    camera = data.get("referenceCamera", data) if isinstance(data, dict) else None
    if not isinstance(camera, dict):
        warnings.append(f"--camera JSON at {path} did not contain a referenceCamera object")
        return {"reference": camera_arg}, warnings
    return camera, warnings


def build_descriptor(args: argparse.Namespace) -> dict[str, Any]:
    intake = None
    if getattr(args, "manifest", None):
        manifest_path = Path(args.manifest).expanduser()
        intake_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(intake_value, dict):
            raise ValueError("CS2 intake manifest must be a JSON object")
        intake = intake_value
        if not args.reference_image:
            views = intake.get("sourceViews", [])
            if isinstance(views, list) and views and isinstance(views[0], dict):
                source_path = views[0].get("path")
                if isinstance(source_path, str):
                    args.reference_image = source_path
        if intake.get("state") != "proceed":
            return {"projectedTextureBake": {"version": "1.0", "status": "request-input", "reason": f"intake state is {intake.get('state', 'unknown')}"}}
    if not args.reference_image:
        raise ValueError("--reference-image or --manifest with a source view is required")
    camera, camera_warnings = load_camera(args.camera)
    warnings = list(camera_warnings)

    # Fail closed on an unfitted camera for the modes that actually depend on one,
    # matching how generate_threejs_factory.py returns BLOCKED rather than emitting a
    # factory it cannot stand behind.
    max_error = getattr(args, "max_reprojection_error", DEFAULT_MAX_REPROJECTION_ERROR_PIXELS)
    fitness = camera_fitness(camera, max_error_pixels=max_error)
    camera_dependent = args.projection_mode in CAMERA_DEPENDENT_PROJECTION_MODES
    if camera_dependent and not fitness["fitted"] and not getattr(args, "allow_unfitted_camera", False):
        return {
            "projectedTextureBake": {
                "version": "1.0",
                "generator": "stage3_build/bake_projected_texture.py",
                "status": "BLOCKED",
                "gate": "camera-fitness",
                "targetMeshId": args.mesh_id,
                "projectionMode": args.projection_mode,
                "cameraFitness": fitness,
                "cause": fitness["blockers"],
                "nextAction": (
                    "solve the camera against landmark correspondences: "
                    "python3 forge/stage1_intake/solve_camera_pose.py <image> "
                    "--correspondences <pairs.json> --out reference-camera.json, then re-run this "
                    "bake with --camera reference-camera.json"
                ),
                "note": (
                    "Projection fidelity is camera alignment. Baking through a default-guess camera "
                    "writes the misalignment into the albedo, where later gates score it as a "
                    "material defect instead of a camera defect. Use --allow-unfitted-camera only "
                    "for a deliberately approximate bake, or switch to "
                    "--projection-mode orthographic-front-projection/triplanar-fallback, neither of "
                    "which claims to reproduce the reference viewpoint."
                ),
            }
        }
    if camera_dependent and not fitness["fitted"]:
        warnings.append(
            "camera is unfitted and --allow-unfitted-camera was passed: this bake is an "
            f"APPROXIMATION, not a reference match ({'; '.join(fitness['blockers'])})"
        )

    source_images: dict[str, str] = {"reference": str(Path(args.reference_image).expanduser())}
    if args.delit_image:
        source_images["delit"] = str(Path(args.delit_image).expanduser())
    else:
        warnings.append(
            "no --delit-image supplied; projecting the raw reference will bake its lighting into the mesh "
            "unless the runtime applies its own de-lighting pass first"
        )

    unseen_strategy = args.unseen_strategy
    unseen_confidence = {
        "mirror-symmetry": 0.45,
        "palette-continue": 0.3,
        "request-additional-view": 0.0,
        "leave-unprojected": 0.0,
    }[unseen_strategy]

    bake_steps = [
        "load the fitted mesh identified by targetMeshId and its UV layout",
        "load sourceImages.delit if present, otherwise sourceImages.reference, as the projection source texture",
        "construct a projection camera in the Three.js scene from the camera block (fovDegrees, aspect, orientation, position)",
        f"apply {args.projection_mode} to project the source texture onto mesh surfaces facing the projection camera within tolerance",
        f"for surfaces outside the camera's view frustum or facing away, apply the '{unseen_strategy}' strategy",
        f"rasterize the resulting camera-space projection into a {args.texture_size}x{args.texture_size} UV-space texture",
        "flag any UV texels that received no projected sample (fully unseen regions) in the bake output metadata",
        "hand the baked texture back to the material pipeline as the projected albedo input",
    ]

    return {
        "projectedTextureBake": {
            "version": "1.0",
            "generator": "stage3_build/bake_projected_texture.py",
            "status": "descriptor-only; no pixels are baked or rasterized by this script",
            "targetMeshId": args.mesh_id,
            "projectionMode": args.projection_mode,
            "textureSize": args.texture_size,
            "camera": camera,
            "cameraFitness": fitness,
            "sourceImages": source_images,
            "unseenRegionStrategy": {
                "mode": unseen_strategy,
                "confidence": unseen_confidence,
                "note": "Regions the reference camera never saw (back, occluded folds, underside) are inferred, not observed.",
            },
            "runtimeApproach": (
                "actual projective texturing and UV bake happen in the Three.js runtime via a projective "
                "ShaderMaterial (the three-projected-material approach) or an equivalent camera-space "
                "projection shader; this descriptor only records the plan for that step"
            ),
            "bakeSteps": bake_steps,
            "limitations": [
                "this script performs no image sampling, projection math, or UV rasterization",
                "camera accuracy is inherited from whatever produced the camera block; an unrefined camera will misalign the projection",
                "unseen-region inference is a heuristic guess, not observed geometry or texture",
                "the resulting bake still needs a rendered overlay review against the reference image before being trusted",
            ]
            + warnings,
            "note": (
                "Feed this descriptor to the Three.js generator/agent to run the actual projection and bake, "
                "then re-render and visually compare the baked mesh against the reference image before "
                "accepting the result as final."
            ),
        }
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-image", help="Path to the original reference photo")
    parser.add_argument("--manifest", type=Path, help="Validated cs2-intake.json with source view and camera evidence")
    parser.add_argument("--delit-image", help="Path to a de-lit albedo produced by stage1_intake/delight_albedo.py, if available")
    parser.add_argument("--camera", help="Path to a referenceCamera JSON produced by stage1_intake/solve_camera_pose.py")
    parser.add_argument("--mesh-id", required=True, help="Identifier of the target mesh/node to project onto")
    parser.add_argument(
        "--projection-mode",
        choices=VALID_PROJECTION_MODES,
        default="perspective-camera-projection",
        help="Projection technique the Three.js runtime should use (default perspective-camera-projection)",
    )
    parser.add_argument("--texture-size", type=int, default=1024, help="Target baked texture resolution (square)")
    parser.add_argument(
        "--unseen-strategy",
        choices=VALID_UNSEEN_STRATEGIES,
        default="mirror-symmetry",
        help="How to handle mesh regions outside the reference camera's view (default mirror-symmetry)",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=DEFAULT_MAX_REPROJECTION_ERROR_PIXELS,
        help=(
            "RMS reprojection-error ceiling in reference pixels for a solved camera "
            f"(default {DEFAULT_MAX_REPROJECTION_ERROR_PIXELS})"
        ),
    )
    parser.add_argument(
        "--allow-unfitted-camera",
        action="store_true",
        help=(
            "Emit a perspective bake descriptor through an unfitted/placeholder camera. "
            "Records the result as an explicit approximation; never use it for a "
            "reference-matched finish."
        ),
    )
    parser.add_argument("--out", type=Path, help="Write the descriptor JSON to this path")
    args = parser.parse_args(argv)

    if args.texture_size <= 0:
        parser.error("--texture-size must be positive")
    if args.max_reprojection_error <= 0:
        parser.error("--max-reprojection-error must be positive")

    try:
        descriptor = build_descriptor(args)
        text = json.dumps(descriptor, indent=2, ensure_ascii=False)
        if args.out:
            out_path = args.out.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
        print(text)
        # A blocked descriptor is not a successful run: exit non-zero so a caller that
        # only checks the status code cannot mistake it for a usable bake plan.
        if descriptor.get("projectedTextureBake", {}).get("status") == "BLOCKED":
            return 1
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
