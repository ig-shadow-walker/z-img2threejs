#!/usr/bin/env python3
"""Build a labeled (reference, render) corpus for Divine Eye calibration.

WHY THIS EXISTS
---------------
`calibrate_eye.py` states the precondition plainly: "Before the Eye is allowed to
HARD-GATE anything, it runs in report-only mode over a corpus of labeled (reference,
render) pairs... Only then may report-only be flipped to hard-gate."

No corpus existed. `divine_eye.py` hard-gates on `IOU_HARD_MIN = 0.85` and
`SCALE_HARD_MAX = 0.08` -- hand-picked constants its own comments mark "calibratable in
Phase 5" -- and weights the soft ensemble with values marked "provisional". So the
false-reject rate (which burns the bounded correction budget) and the false-accept rate
(which ships wrong geometry) were both unmeasured in both directions.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT
-------------------------------------------------
It generates known-bad pairs by applying controlled, named perturbations to a reference
image, each perturbation standing for a real defect class. That gives the Eye a runnable
regression harness immediately: it proves the Eye RESPONDS IN THE RIGHT DIRECTION to a
known defect, and it will catch a future change that silently desensitizes a signal.

It does NOT authorize flipping any threshold. A synthetic perturbation of a photograph is
not a Three.js render of a reconstructed model: it has the reference's own shading,
noise and material response, so the absolute fidelity numbers it produces are not the
numbers real renders produce. Every pair is stamped
`provenance: "synthetic-perturbation"`, and `calibrate_eye.py` refuses to authorize a
hard-gate flip on a corpus containing any such pair. Real showcase renders are the only
thing that can satisfy the precondition, and they need a browser.

One perturbation is a deliberate NEGATIVE CONTROL. `remove-feature` deletes a small
identity feature, which the Eye is EXPECTED to miss because its signals run on a 64x64
luma grid. Recording that expectation makes the blind spot measurable instead of
anecdotal, and `stage4_review/microscope.py` is what actually catches that class.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_intake"))
from extract_pbr_evidence import load_image, write_png_rgb  # noqa: E402

SYNTHETIC_PROVENANCE = "synthetic-perturbation"
REAL_RENDER_PROVENANCE = "browser-render"

Pixels = list[tuple[int, int, int, int]]


def _clamp_byte(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _sample(pixels: Pixels, width: int, height: int, x: float, y: float) -> tuple[int, int, int]:
    """Nearest-neighbour sample, clamped at the border.

    Nearest-neighbour rather than bilinear on purpose: interpolation would soften every
    edge, so a geometric perturbation would also become an edge perturbation and the
    corpus could not attribute a signal change to one cause.
    """
    sx = max(0, min(width - 1, int(round(x))))
    sy = max(0, min(height - 1, int(round(y))))
    red, green, blue, _alpha = pixels[sy * width + sx]
    return red, green, blue


def _background(pixels: Pixels, width: int, height: int) -> tuple[int, int, int]:
    """Median of the four corners: the fill colour for pixels a transform vacates."""
    corners = [
        pixels[0],
        pixels[width - 1],
        pixels[(height - 1) * width],
        pixels[height * width - 1],
    ]
    channels = []
    for index in range(3):
        values = sorted(corner[index] for corner in corners)
        channels.append((values[1] + values[2]) // 2)
    return channels[0], channels[1], channels[2]


# --------------------------------------------------------------------- perturbations


def perturb_scale(pixels: Pixels, width: int, height: int, factor: float) -> bytes:
    """Shrink or grow the subject about the frame centre. Targets the scale hard gate."""
    fill = _background(pixels, width, height)
    out = bytearray()
    cx, cy = width / 2.0, height / 2.0
    for y in range(height):
        for x in range(width):
            sx = cx + (x - cx) / factor
            sy = cy + (y - cy) / factor
            if 0 <= sx < width and 0 <= sy < height:
                out.extend(_sample(pixels, width, height, sx, sy))
            else:
                out.extend(fill)
    return bytes(out)


def perturb_translate(pixels: Pixels, width: int, height: int, offset: int) -> bytes:
    """Shift the subject. Targets silhouette IoU without changing shape."""
    fill = _background(pixels, width, height)
    out = bytearray()
    for y in range(height):
        for x in range(width):
            sx, sy = x - offset, y - offset
            if 0 <= sx < width and 0 <= sy < height:
                out.extend(_sample(pixels, width, height, sx, sy))
            else:
                out.extend(fill)
    return bytes(out)


def perturb_aspect(pixels: Pixels, width: int, height: int, factor: float) -> bytes:
    """Stretch horizontally only. Targets the proportion / aspect-ratio signal."""
    fill = _background(pixels, width, height)
    out = bytearray()
    cx = width / 2.0
    for y in range(height):
        for x in range(width):
            sx = cx + (x - cx) / factor
            if 0 <= sx < width:
                out.extend(_sample(pixels, width, height, sx, y))
            else:
                out.extend(fill)
    return bytes(out)


def perturb_hue_shift(pixels: Pixels, width: int, height: int, degrees: float) -> bytes:
    """Rotate hue, preserving luma. Targets colour signals specifically.

    Luma-preserving by construction, so every luma-based signal (SSIM, tonal, edge,
    blowout) should be nearly unmoved. That makes it a clean probe of whether the Eye can
    see colour at all -- which today is only `hueZoneParity`, and that is report-only.
    """
    radians = math.radians(degrees)
    cosine, sine = math.cos(radians), math.sin(radians)
    # YIQ hue rotation: keeps Y (luma) exactly, rotates the chroma plane.
    out = bytearray()
    for red, green, blue, _alpha in pixels:
        luma = 0.299 * red + 0.587 * green + 0.114 * blue
        i = 0.596 * red - 0.274 * green - 0.322 * blue
        q = 0.211 * red - 0.523 * green + 0.312 * blue
        i2 = i * cosine - q * sine
        q2 = i * sine + q * cosine
        out.extend(
            (
                _clamp_byte(luma + 0.956 * i2 + 0.621 * q2),
                _clamp_byte(luma - 0.272 * i2 - 0.647 * q2),
                _clamp_byte(luma - 1.106 * i2 + 1.703 * q2),
            )
        )
    return bytes(out)


def perturb_blur(pixels: Pixels, width: int, height: int, radius: int) -> bytes:
    """Box blur. Targets edge overlap and the directional chamfer contour signal."""
    radius = max(1, radius)
    out = bytearray()
    for y in range(height):
        for x in range(width):
            totals = [0, 0, 0]
            count = 0
            for dy in range(-radius, radius + 1):
                sy = y + dy
                if sy < 0 or sy >= height:
                    continue
                for dx in range(-radius, radius + 1):
                    sx = x + dx
                    if sx < 0 or sx >= width:
                        continue
                    pixel = pixels[sy * width + sx]
                    totals[0] += pixel[0]
                    totals[1] += pixel[1]
                    totals[2] += pixel[2]
                    count += 1
            out.extend(_clamp_byte(total / count) for total in totals)
    return bytes(out)


def perturb_flatten(pixels: Pixels, width: int, height: int, _unused: float = 0.0) -> bytes:
    """Replace shading with a flat fill per region. Targets the flat-region QA signal.

    This is the "material is not reacting to light" defect: a dead flat albedo where the
    reference has a gradient.
    """
    fill = _background(pixels, width, height)
    threshold = sum(fill) / 3.0
    # One flat tone for the subject, the original background elsewhere.
    subject_pixels = [
        pixel for pixel in pixels if abs(sum(pixel[:3]) / 3.0 - threshold) > 18
    ]
    if subject_pixels:
        flat = tuple(
            _clamp_byte(sum(pixel[index] for pixel in subject_pixels) / len(subject_pixels))
            for index in range(3)
        )
    else:
        flat = fill
    out = bytearray()
    for red, green, blue, _alpha in pixels:
        if abs((red + green + blue) / 3.0 - threshold) > 18:
            out.extend(flat)
        else:
            out.extend((red, green, blue))
    return bytes(out)


def perturb_remove_feature(
    pixels: Pixels, width: int, height: int, roi_fraction: float = 0.03
) -> bytes:
    """Erase a small square feature by inpainting its surroundings.

    NEGATIVE CONTROL. The Eye is EXPECTED to miss this: at 64x64 luma a feature this size
    is sub-pixel, so it is absent before comparison rather than scored badly. Recording the
    expectation turns the documented blind spot into a measured one.
    """
    side = max(4, int(min(width, height) * roi_fraction))
    x0 = int(width * 0.55)
    y0 = int(height * 0.30)
    # Fill with the median of a ring just outside the erased square, which approximates
    # "this feature was never modelled" rather than "a hole was punched here".
    ring: list[tuple[int, int, int]] = []
    for x in range(max(0, x0 - 2), min(width, x0 + side + 2)):
        for y in (max(0, y0 - 2), min(height - 1, y0 + side + 1)):
            ring.append(pixels[y * width + x][:3])
    if not ring:
        ring = [_background(pixels, width, height)]
    replacement = tuple(
        sorted(sample[index] for sample in ring)[len(ring) // 2] for index in range(3)
    )
    out = bytearray()
    for y in range(height):
        for x in range(width):
            if x0 <= x < x0 + side and y0 <= y < y0 + side:
                out.extend(replacement)
            else:
                out.extend(pixels[y * width + x][:3])
    return bytes(out)


def perturb_identity(pixels: Pixels, width: int, height: int, _unused: float = 0.0) -> bytes:
    """The known-good pair: the render IS the reference."""
    out = bytearray()
    for red, green, blue, _alpha in pixels:
        out.extend((red, green, blue))
    return bytes(out)


# Each entry describes one defect class and separates TWO different questions, because
# collapsing them is how a regression harness turns into a permanently-red wish list:
#
#   expectedDetection -- what the Eye MEASURABLY does today. This is what the regression
#     audit asserts, so the harness is green while behaviour is unchanged and goes red the
#     moment a signal's sensitivity moves in either direction. None means "not asserted".
#   desiredDetection  -- what the Eye SHOULD do. Where it differs from expectedDetection
#     the pair is reported as a KNOWN GAP, so the harness doubles as a live inventory of
#     the Eye's sensitivity holes instead of hiding them behind a passing run.
#
# The expectedDetection values below were measured, not assumed. Three of them are gaps.
PERTURBATIONS: dict[str, dict[str, Any]] = {
    "identity": {
        "fn": perturb_identity,
        "default": 0.0,
        "label": "good",
        "defectClass": "none",
        "expectedDetection": False,
        "desiredDetection": False,
        "rationale": "the render IS the reference; a reject here would be a false reject",
    },
    "scale": {
        "fn": perturb_scale,
        "default": 0.85,
        "label": "bad",
        "defectClass": "wrong-scale",
        "expectedDetection": True,
        "desiredDetection": True,
        "rationale": "targets SCALE_HARD_MAX; measured verdict reject, scaleDelta 0.283",
    },
    "translate": {
        "fn": perturb_translate,
        "default": 24,
        "label": "bad",
        "defectClass": "misplaced-subject",
        "expectedDetection": True,
        "desiredDetection": True,
        "rationale": "targets silhouette IoU without altering shape; measured low-confidence",
    },
    "aspect": {
        "fn": perturb_aspect,
        "default": 1.25,
        "label": "bad",
        "defectClass": "wrong-proportion",
        "expectedDetection": True,
        "desiredDetection": True,
        "rationale": "targets the proportion signal and ASPECT_SOFT_MAX; measured reject",
    },
    "hue-shift": {
        "fn": perturb_hue_shift,
        "default": 60.0,
        "label": "bad",
        "defectClass": "wrong-colour",
        "expectedDetection": False,
        "desiredDetection": True,
        "rationale": (
            "KNOWN GAP, and the most actionable one this corpus found. The perturbation is "
            "luma-preserving, so every luma signal is blind to it by construction. "
            "hueZoneParity scores it 0.000 -- it detects the defect perfectly -- but carries "
            "zero ensemble weight because it is report-only until calibrated, so fidelity "
            "stays at 0.959 and the Eye PASSES a 60-degree hue error. The signal is not "
            "missing, it is disconnected."
        ),
    },
    "blur": {
        "fn": perturb_blur,
        "default": 3,
        "label": "bad",
        "defectClass": "soft-edges",
        "expectedDetection": False,
        "desiredDetection": True,
        "rationale": (
            "KNOWN GAP: measured fidelity 0.890 clears the 0.85 target, so a visibly softened "
            "render passes. Directional chamfer moves only to 0.942, and it is report-only too."
        ),
    },
    "flatten": {
        "fn": perturb_flatten,
        "default": 0.0,
        "label": "bad",
        "defectClass": "material-not-reacting-to-light",
        "expectedDetection": None,
        "desiredDetection": True,
        "requiresShadedReference": True,
        "rationale": (
            "NOT ASSERTED: this perturbation only means anything against a reference that has "
            "real shading. On a flat-filled reference it is close to a no-op (measured "
            "flatRegionScore 0.989), so a failure here would indict the fixture rather than "
            "the Eye. Run it against a photographic reference to make it meaningful."
        ),
    },
    "remove-feature": {
        "fn": perturb_remove_feature,
        "default": 0.03,
        "label": "bad",
        "defectClass": "missing-small-feature",
        "expectedDetection": False,
        "desiredDetection": False,
        "negativeControl": True,
        "rationale": (
            "NEGATIVE CONTROL: measured fidelity 1.0000 -- not merely a low score but a "
            "PERFECT one, because the feature is sub-pixel on every grid the Eye samples, so "
            "it is absent before comparison rather than scored badly. The Eye is not expected "
            "to catch this class at all; stage4_review/microscope.py is the instrument for it. "
            "If this pair ever starts failing, the fixture stopped isolating the blind spot."
        ),
    },
}


def build_corpus(
    reference: Path,
    out_dir: Path,
    *,
    names: list[str] | None = None,
    overrides: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Render each perturbation and return the corpus manifest."""
    width, height, pixels, warnings = load_image(reference)
    overrides = overrides or {}
    selected = names or list(PERTURBATIONS)
    unknown = [name for name in selected if name not in PERTURBATIONS]
    if unknown:
        raise ValueError(
            f"unknown perturbation(s) {unknown}; available: {sorted(PERTURBATIONS)}"
        )
    if "identity" not in selected:
        raise ValueError(
            "a corpus without the identity pair measures only false rejects; include "
            "'identity' so known-good behaviour is measured too"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs: list[dict[str, Any]] = []
    for name in selected:
        spec = PERTURBATIONS[name]
        parameter = overrides.get(name, spec["default"])
        rgb = spec["fn"](pixels, width, height, parameter)
        render_path = out_dir / f"{name}.png"
        write_png_rgb(render_path, width, height, rgb)
        pairs.append(
            {
                "reference": str(reference.resolve()),
                "render": str(render_path.resolve()),
                "label": spec["label"],
                "perturbation": name,
                "parameter": parameter,
                "defectClass": spec["defectClass"],
                "expectedDetection": spec["expectedDetection"],
                "desiredDetection": spec["desiredDetection"],
                "knownGap": (
                    spec["expectedDetection"] is not None
                    and spec["expectedDetection"] != spec["desiredDetection"]
                ),
                "negativeControl": bool(spec.get("negativeControl")),
                "requiresShadedReference": bool(spec.get("requiresShadedReference")),
                "rationale": spec["rationale"],
                # The stamp calibrate_eye.py reads to refuse a threshold flip.
                "provenance": SYNTHETIC_PROVENANCE,
            }
        )

    return {
        "version": "1.0",
        "generator": "stage4_review/build_corpus.py",
        "reference": str(reference.resolve()),
        "resolution": {"width": width, "height": height},
        "pairs": pairs,
        "warnings": warnings,
        "provenance": SYNTHETIC_PROVENANCE,
        "thresholdFlipAuthorized": False,
        "note": (
            "Synthetic perturbations of a reference photograph. Valid as a REGRESSION "
            "harness -- it proves the Eye responds in the right direction to a known defect "
            "class and will catch a change that desensitizes a signal. NOT valid for "
            "choosing threshold values: these images carry the reference's own shading and "
            "noise, so their absolute fidelity numbers are not the numbers a Three.js "
            "render produces. Flipping report-only to hard-gate requires real browser "
            f"renders labelled provenance={REAL_RENDER_PROVENANCE!r}."
        ),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("reference", type=Path, help="reference image to perturb")
    parser.add_argument("--out-dir", type=Path, required=True, help="directory for the renders")
    parser.add_argument(
        "--manifest", type=Path, help="write the corpus manifest JSON here (for calibrate_eye.py)"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=None,
        metavar="NAME",
        help=f"restrict to these perturbations (repeatable); available: {sorted(PERTURBATIONS)}",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="override a perturbation's parameter, e.g. --set scale=0.9",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    reference = args.reference.expanduser().resolve()
    if not reference.exists():
        parser.error(f"{reference} does not exist")

    overrides: dict[str, float] = {}
    for item in args.set:
        if "=" not in item:
            parser.error(f"--set expects NAME=VALUE, got {item!r}")
        name, raw = item.split("=", 1)
        try:
            overrides[name] = float(raw)
        except ValueError:
            parser.error(f"--set value for {name!r} must be numeric, got {raw!r}")

    try:
        manifest = build_corpus(
            reference,
            args.out_dir.expanduser().resolve(),
            names=args.only,
            overrides=overrides,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.manifest:
        path = args.manifest.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
    else:
        print(f"corpus: {len(manifest['pairs'])} pair(s) at {args.out_dir}")
        for pair in manifest["pairs"]:
            if pair["expectedDetection"] is None:
                note = "not-asserted"
            elif pair["knownGap"]:
                note = "KNOWN GAP (desired: detect, actual: pass)"
            elif pair["negativeControl"]:
                note = "negative control (Eye not expected to catch this)"
            elif pair["expectedDetection"]:
                note = "detects"
            else:
                note = "known-good"
            print(f"  {pair['perturbation']:16} {pair['label']:5} {note}")
        gaps = [pair["perturbation"] for pair in manifest["pairs"] if pair["knownGap"]]
        if gaps:
            print(f"known sensitivity gaps: {', '.join(gaps)}")
        print("thresholdFlipAuthorized: False (synthetic provenance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
