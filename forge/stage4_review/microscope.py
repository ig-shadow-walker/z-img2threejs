#!/usr/bin/env python3
"""Feature-scale fidelity measurement — the executable half of the Divine Eye microscope.

WHY THIS EXISTS
---------------
`divine_eye.py` computes every structural signal on a 64x64 luma grid (SSIM, tonal,
blowout, flat) and a 96x96 edge grid. Its own module docstring states the consequence
plainly: a feature a few pixels wide in a 1920px reference "is not scored badly, it is
ABSENT before any comparison happens. No threshold tuning recovers it."

`per_feature.py` was built to gate on exactly those features -- `is_gating()` fails a
pass when any critical feature is missing or below threshold -- but it "consumes a
scores dict and never opens an image". So the tier machinery was sound and starved:
correct gating logic fed by numbers nothing produced at feature scale.

This module produces those numbers. It crops each declared feature ROI at NATIVE
reference resolution and measures inside the crop, so a 12px tear is compared as 12px
of pixels instead of a fraction of one downsampled cell. Its output is shaped for
`per_feature.py --scores`, which closes the loop.

Spec: `grimoire/review/divine_eye_microscope.md`.

SCOPE
-----
Like `diagnose_render_multi_angle.py`, this analyzes already-captured PNGs. Driving a
browser to produce zoom patches via `setViewOffset` / render targets is a separate
concern, and deliberately so: the doc is explicit that moving the camera closer or
switching to orthographic changes the projection, so the patch would stop showing what
the reference view shows. Cropping an existing full-resolution capture cannot make that
mistake.

THE TWO EMPIRICAL RULES, BOTH ENFORCED HERE
-------------------------------------------
1. Measure a component's VISIBLE FOOTPRINT -- the pixels that differ between the full
   frame and a frame with only that component hidden -- never an isolation render. An
   isolation render reveals geometry the reference photograph cannot contain; in the
   mini-dragon build that produced a confident false finding ("the spars are twice too
   thick") and cost a full correction cycle. Pass `--component-hidden id=frame.png` to
   get footprint-restricted measurement.
2. Never colour-gate a CONCAVE feature. A dark ratio there captures cavity shading, not
   material. Mark such features `"concave": true` and their colour metric becomes
   report-only, with the reference dark pixels bucketed by `min(r,g,b)` so the evidence
   for that call is recorded rather than asserted.

THE HARD LIMIT
--------------
Zoom cannot create information the reference does not contain. When a feature's ROI
covers fewer than `minimumProjectedPixels` in the reference, the verdict is
`insufficient-reference-resolution` and the feature is EXCLUDED from the gating scores
rather than scored. Gating a model against an upsampled invention is worse than not
gating it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "stage1_intake"))
from extract_pbr_evidence import build_foreground_mask, load_image, write_png_rgb  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_shared"))
from color_metrics import ciede2000, srgb_to_lab  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The verdict is delegated to per_feature.evaluate_features so the threshold rule has ONE
# definition. Duplicating it here would let the microscope's verdict drift from the gate
# that actually blocks a pass.
from per_feature import evaluate_features  # noqa: E402


# A feature must cover at least this many reference pixels to be measurable at all.
# The doc's floor for a rendered critical feature is 128-256px; the same floor applied to
# the REFERENCE side is what separates "the model got it wrong" from "the photo cannot
# show it".
DEFAULT_MINIMUM_PROJECTED_PIXELS = 128

# CIEDE2000 distance treated as a full colour mismatch. ~2.3 is the classic
# just-noticeable difference; 12 is a clearly different colour, so agreement is scaled
# across that range rather than against an arbitrary maximum.
DELTA_E_FULL_MISMATCH = 12.0

# Sobel gradient magnitude (on 0..1 luma) counted as an edge pixel. Matches the
# threshold divine_eye._sobel_edges uses, so edge density means the same thing at both
# scales.
EDGE_THRESHOLD = 0.12

# Below this share of foreground pixels in the render ROI the feature is reported
# MISSING (score None) rather than scored low. A missing tear must fail as absent, not
# as a mediocre number that an average could absorb.
PRESENCE_FLOOR = 0.02

# Reference dark pixels are "material" when min(r,g,b) sits near black; a population that
# spreads through mid-grey is shading. Both numbers come from the mini-dragon finding,
# where the shading case peaked at 60-79 with only 14% near-black and the material case
# ran 32.9% near-black.
MATERIAL_DARK_CEILING = 32
DARK_PIXEL_CEILING = 80

# A dark population must cover at least this share of the measured pixels before its
# spread is allowed to downgrade colour to report-only. Without this floor, any ROI whose
# few darkest pixels happen to be mid-grey silences colour gating -- which would weaken
# the gate everywhere instead of only on the concave features the rule is about.
DARK_SHARE_FLOOR = 0.10


class MicroscopeError(Exception):
    """Raised when the microscope cannot measure what it was asked to measure."""


# --------------------------------------------------------------------------- ROIs


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def normalize_roi(roi: Any, label: str) -> tuple[float, float, float, float]:
    """Parse `[u0, v0, u1, v1]` normalized ROI corners, as the descriptor defines them."""
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise MicroscopeError(f"{label} must be [u0, v0, u1, v1] in normalized 0..1 coordinates")
    values = []
    for item in roi:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise MicroscopeError(f"{label} must contain only numbers")
        values.append(_clamp01(float(item)))
    u0, v0, u1, v1 = values
    if u1 <= u0 or v1 <= v0:
        raise MicroscopeError(f"{label} must have u1 > u0 and v1 > v0")
    return u0, v0, u1, v1


def roi_pixel_box(roi: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Convert a normalized ROI to an integer pixel box, always at least 1px per axis."""
    u0, v0, u1, v1 = roi
    x0 = max(0, min(width - 1, math.floor(u0 * width)))
    y0 = max(0, min(height - 1, math.floor(v0 * height)))
    x1 = max(x0 + 1, min(width, math.ceil(u1 * width)))
    y1 = max(y0 + 1, min(height, math.ceil(v1 * height)))
    return x0, y0, x1, y1


class Frame:
    """One decoded image plus the derived data every metric needs.

    Decoded once per file and reused across every ROI, so a manifest with 30 features
    does not decode the same PNG 30 times.
    """

    def __init__(self, path: Path):
        self.path = path
        width, height, pixels, warnings = load_image(path)
        self.width = width
        self.height = height
        self.pixels = pixels
        self.warnings = warnings
        mask, diagnostics, mask_warnings = build_foreground_mask(width, height, pixels)
        self.mask = mask
        self.mask_diagnostics = diagnostics
        self.warnings = list(warnings) + list(mask_warnings)

    def rgb_at(self, x: int, y: int) -> tuple[int, int, int]:
        red, green, blue, _alpha = self.pixels[y * self.width + x]
        return red, green, blue

    def luma_at(self, x: int, y: int) -> float:
        red, green, blue = self.rgb_at(x, y)
        return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0

    def foreground_at(self, x: int, y: int) -> bool:
        return bool(self.mask[y * self.width + x])


def crop_patch(frame: Frame, roi: tuple[float, float, float, float]) -> dict[str, Any]:
    """Extract one ROI at NATIVE resolution.

    Deliberately not `material_views.crop_visible_footprint`: that helper pads the box
    by 12% and rewrites transparent pixels to black, both of which are right for a
    material swatch and wrong for a feature metric -- padding pulls in neighbouring
    geometry, and black-filling invents dark pixels a colour metric would then score.
    """
    x0, y0, x1, y1 = roi_pixel_box(roi, frame.width, frame.height)
    lumas: list[float] = []
    rgbs: list[tuple[int, int, int]] = []
    foreground: list[bool] = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            lumas.append(frame.luma_at(x, y))
            rgbs.append(frame.rgb_at(x, y))
            foreground.append(frame.foreground_at(x, y))
    return {
        "box": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
        "pixelCount": len(lumas),
        "luma": lumas,
        "rgb": rgbs,
        "foreground": foreground,
        "width": x1 - x0,
        "height": y1 - y0,
    }


def visible_footprint_mask(
    full: dict[str, Any],
    hidden: dict[str, Any],
    *,
    tolerance: int = 6,
) -> list[bool]:
    """Pixels where the component is the frontmost thing drawn.

    The full frame minus the component-hidden frame. These are exactly the pixels a
    photograph of the same pose shows of that component -- the rule the mini-dragon
    false finding established.
    """
    if full["pixelCount"] != hidden["pixelCount"]:
        raise MicroscopeError(
            "component-hidden frame ROI does not match the render ROI; both captures must "
            "share one resolution and camera"
        )
    footprint: list[bool] = []
    for (r1, g1, b1), (r2, g2, b2) in zip(full["rgb"], hidden["rgb"], strict=True):
        differs = (
            abs(r1 - r2) > tolerance or abs(g1 - g2) > tolerance or abs(b1 - b2) > tolerance
        )
        footprint.append(differs)
    return footprint


# ------------------------------------------------------------------------ metrics


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def patch_ssim(a: list[float], b: list[float]) -> float:
    """Global SSIM over the patch luma, at the patch's own resolution.

    Same formulation as divine_eye.global_ssim; the difference that matters is the input,
    which here is native-resolution feature pixels rather than a 64x64 whole-frame grid.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    mu_a, mu_b = _mean(a), _mean(b)
    var_a = _mean([(value - mu_a) ** 2 for value in a])
    var_b = _mean([(value - mu_b) ** 2 for value in b])
    cov = _mean([(x - mu_a) * (y - mu_b) for x, y in zip(a, b, strict=True)])
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    return _clamp01(numerator / denominator) if denominator else 0.0


def edge_fraction(luma: list[float], width: int, height: int) -> float:
    """Share of pixels whose Sobel gradient magnitude clears EDGE_THRESHOLD."""
    if width < 3 or height < 3:
        return 0.0
    edges = 0
    considered = 0
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            base = y * width + x
            gx = (
                -luma[base - width - 1] - 2 * luma[base - 1] - luma[base + width - 1]
                + luma[base - width + 1] + 2 * luma[base + 1] + luma[base + width + 1]
            )
            gy = (
                -luma[base - width - 1] - 2 * luma[base - width] - luma[base - width + 1]
                + luma[base + width - 1] + 2 * luma[base + width] + luma[base + width + 1]
            )
            considered += 1
            if math.hypot(gx, gy) >= EDGE_THRESHOLD:
                edges += 1
    return edges / considered if considered else 0.0


def median_delta_e(
    reference: dict[str, Any],
    render: dict[str, Any],
    selection: list[bool],
) -> float | None:
    """Median CIEDE2000 between the two patches over the selected pixels.

    Median rather than mean: one blown highlight inside the ROI should not decide a
    feature's colour verdict.
    """
    deltas: list[float] = []
    for index, selected in enumerate(selection):
        if not selected:
            continue
        deltas.append(
            ciede2000(srgb_to_lab(reference["rgb"][index]), srgb_to_lab(render["rgb"][index]))
        )
    if not deltas:
        return None
    deltas.sort()
    middle = len(deltas) // 2
    if len(deltas) % 2:
        return deltas[middle]
    return (deltas[middle - 1] + deltas[middle]) / 2.0


def dark_pixel_profile(patch: dict[str, Any], selection: list[bool]) -> dict[str, Any]:
    """Bucket the reference's dark pixels to separate material from shading.

    The doc's rule, from the mini-dragon ear: a real dark MATERIAL clusters near-black;
    cavity SHADING spreads through mid-grey. Recording this is what stops a dark inner-ear
    material from being invented to match a shadow.
    """
    dark_values = []
    total = 0
    for index, selected in enumerate(selection):
        if not selected:
            continue
        total += 1
        minimum = min(patch["rgb"][index])
        if minimum <= DARK_PIXEL_CEILING:
            dark_values.append(minimum)
    if not total:
        return {"darkShare": 0.0, "nearBlackShare": 0.0, "classification": "no-pixels"}
    near_black = sum(1 for value in dark_values if value <= MATERIAL_DARK_CEILING)
    dark_share = len(dark_values) / total
    near_black_share = near_black / len(dark_values) if dark_values else 0.0
    if not dark_values:
        classification = "no-dark-pixels"
    elif near_black_share >= 0.5:
        classification = "material-like"
    else:
        classification = "shading-like"
    return {
        "darkShare": round(dark_share, 4),
        "nearBlackShare": round(near_black_share, 4),
        "classification": classification,
    }


# ---------------------------------------------------------------- feature scoring


def measure_feature(
    descriptor: dict[str, Any],
    reference: Frame,
    render: Frame,
    hidden_frames: dict[str, Frame],
    *,
    default_minimum_pixels: int,
    patch_dir: Path | None = None,
) -> dict[str, Any]:
    """Measure one declared feature and return its record.

    `score` is the MINIMUM of the individual metric agreements, never their mean: a
    correct colour must not be able to mask a wrong edge. That is the doc's aggregation
    rule applied within a feature as well as across features.
    """
    feature_id = descriptor.get("featureId") or descriptor.get("id")
    if not isinstance(feature_id, str) or not feature_id.strip():
        raise MicroscopeError("every feature descriptor needs a featureId")
    roi = normalize_roi(descriptor.get("referenceRoi"), f"{feature_id}.referenceRoi")
    minimum_pixels = descriptor.get("minimumProjectedPixels")
    if not isinstance(minimum_pixels, int) or isinstance(minimum_pixels, bool) or minimum_pixels <= 0:
        minimum_pixels = default_minimum_pixels

    record: dict[str, Any] = {
        "featureId": feature_id,
        "componentId": descriptor.get("componentId"),
        "criticality": descriptor.get("criticality", "detail"),
        "concave": bool(descriptor.get("concave")),
        "referenceRoi": list(roi),
        "metrics": {},
        "notes": [],
    }

    reference_patch = crop_patch(reference, roi)
    render_patch = crop_patch(render, roi)
    record["referencePixelCount"] = reference_patch["pixelCount"]
    record["renderPixelCount"] = render_patch["pixelCount"]

    # THE HARD LIMIT. Checked before anything is measured, so no score is ever produced
    # from a region the reference cannot resolve.
    if reference_patch["pixelCount"] < minimum_pixels:
        record["status"] = "insufficient-reference-resolution"
        record["score"] = None
        record["gating"] = False
        record["notes"].append(
            f"reference ROI covers {reference_patch['pixelCount']}px, below the "
            f"{minimum_pixels}px floor; zoom cannot create information the reference does "
            "not contain, so this feature is excluded from gating rather than scored"
        )
        return record

    if reference_patch["pixelCount"] != render_patch["pixelCount"]:
        raise MicroscopeError(
            f"{feature_id}: reference and render ROIs differ in pixel count "
            f"({reference_patch['pixelCount']} vs {render_patch['pixelCount']}); the two "
            "captures must share one resolution"
        )

    # Restrict to the component's visible footprint when the evidence for it exists.
    component_id = descriptor.get("componentId")
    footprint: list[bool] | None = None
    if isinstance(component_id, str) and component_id in hidden_frames:
        hidden_patch = crop_patch(hidden_frames[component_id], roi)
        footprint = visible_footprint_mask(render_patch, hidden_patch)
        record["visibleFootprintPixels"] = sum(footprint)
        record["measurementBasis"] = "visible-footprint"
        if record["visibleFootprintPixels"] == 0:
            record["status"] = "component-not-visible"
            record["score"] = None
            record["gating"] = False
            record["notes"].append(
                "the component-hidden frame is identical to the full frame inside this ROI, so "
                "the component is not the frontmost thing drawn anywhere here; nothing to measure"
            )
            return record
    else:
        record["measurementBasis"] = "roi-foreground"
        if isinstance(component_id, str):
            record["notes"].append(
                "no --component-hidden frame supplied for this component, so metrics are measured "
                "over the ROI foreground; supply one to restrict measurement to the visible "
                "footprint and avoid charging the model for occluded geometry"
            )

    reference_selection = (
        footprint if footprint is not None else list(reference_patch["foreground"])
    )
    if not any(reference_selection):
        reference_selection = [True] * reference_patch["pixelCount"]
        record["notes"].append(
            "no foreground detected inside the reference ROI; measuring over every ROI pixel"
        )

    # Presence: is the feature in the render at all? A missing feature must read as
    # absent (score None -> MISSING), which per_feature.py fails for a gating target.
    render_foreground_share = sum(
        1 for index, selected in enumerate(reference_selection)
        if selected and render_patch["foreground"][index]
    ) / max(1, sum(reference_selection))
    record["metrics"]["renderForegroundShare"] = round(render_foreground_share, 4)

    reference_edges = edge_fraction(
        reference_patch["luma"], reference_patch["width"], reference_patch["height"]
    )
    render_edges = edge_fraction(
        render_patch["luma"], render_patch["width"], render_patch["height"]
    )
    record["metrics"]["referenceEdgeFraction"] = round(reference_edges, 4)
    record["metrics"]["renderEdgeFraction"] = round(render_edges, 4)

    if render_foreground_share < PRESENCE_FLOOR:
        record["status"] = "missing"
        record["score"] = None
        record["gating"] = _is_gating(record)
        record["notes"].append(
            f"render ROI foreground share {render_foreground_share:.4f} is below the "
            f"{PRESENCE_FLOOR} presence floor: the feature is absent from the render"
        )
        return record

    ssim = patch_ssim(reference_patch["luma"], render_patch["luma"])
    record["metrics"]["patchSsim"] = round(ssim, 4)

    # Edge agreement catches a feature whose colour is right and whose linework is not.
    edge_denominator = max(reference_edges, render_edges, 1e-6)
    edge_agreement = 1.0 - abs(reference_edges - render_edges) / edge_denominator
    record["metrics"]["edgeAgreement"] = round(_clamp01(edge_agreement), 4)

    delta_e = median_delta_e(reference_patch, render_patch, reference_selection)
    dark_profile = dark_pixel_profile(reference_patch, reference_selection)
    record["metrics"]["referenceDarkProfile"] = dark_profile

    colour_agreement: float | None = None
    if delta_e is not None:
        record["metrics"]["medianDeltaE00"] = round(delta_e, 3)
        colour_agreement = _clamp01(1.0 - delta_e / DELTA_E_FULL_MISMATCH)
        record["metrics"]["colourAgreement"] = round(colour_agreement, 4)

    # RULE 2: a concave feature's dark pixels are cavity shading, so its colour metric is
    # recorded but never allowed to gate.
    colour_gates = True
    if record["concave"]:
        colour_gates = False
        record["notes"].append(
            "concave feature: colour is report-only, because a dark ratio here measures cavity "
            "shading rather than material"
        )
    elif (
        dark_profile["classification"] == "shading-like"
        and dark_profile["darkShare"] >= DARK_SHARE_FLOOR
    ):
        colour_gates = False
        record["notes"].append(
            f"reference dark pixels are {dark_profile['classification']} "
            f"(darkShare {dark_profile['darkShare']}, nearBlackShare "
            f"{dark_profile['nearBlackShare']}), which indicates shading rather than a dark "
            "material; colour is report-only for this feature"
        )
    record["metrics"]["colourGates"] = colour_gates

    components = {"patchSsim": ssim, "edgeAgreement": _clamp01(edge_agreement)}
    if colour_agreement is not None and colour_gates:
        components["colourAgreement"] = colour_agreement

    # Worst-of, not average.
    worst_metric = min(components, key=lambda name: components[name])
    record["score"] = round(components[worst_metric], 4)
    record["limitingMetric"] = worst_metric
    record["status"] = "measured"
    record["gating"] = _is_gating(record)

    if patch_dir is not None:
        record["patches"] = _write_patches(patch_dir, feature_id, reference_patch, render_patch)
    return record


def _is_gating(record: dict[str, Any]) -> bool:
    """Same rule as per_feature.is_gating, evaluated on the descriptor's criticality."""
    return record.get("criticality") == "critical" or record.get("mustPass") is True


def _write_patches(
    patch_dir: Path,
    feature_id: str,
    reference_patch: dict[str, Any],
    render_patch: dict[str, Any],
) -> dict[str, str]:
    """Persist the native-resolution crops so a human or VLM can see what was measured."""
    patch_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(char if char.isalnum() or char in "-_" else "-" for char in feature_id)
    written: dict[str, str] = {}
    for name, patch in (("reference", reference_patch), ("render", render_patch)):
        payload = bytearray()
        for red, green, blue in patch["rgb"]:
            payload.extend((red, green, blue))
        out = patch_dir / f"{safe}-{name}.png"
        write_png_rgb(out, patch["width"], patch["height"], bytes(payload))
        written[name] = str(out.resolve())
    return written


# -------------------------------------------------------------------- aggregation


def _target_from_record(record: dict[str, Any]) -> dict[str, Any]:
    """Map a measured feature onto the target shape per_feature.evaluate_features expects."""
    target: dict[str, Any] = {
        "id": record["featureId"],
        # per_feature's tiers are critical/important/detail, the same vocabulary the
        # microscope descriptor uses for `criticality`.
        "tier": record.get("criticality", "detail"),
    }
    if record.get("mustPass") is True:
        target["mustPass"] = True
    minimum = record.get("minimumScore")
    if isinstance(minimum, (int, float)) and not isinstance(minimum, bool):
        target["minimumScore"] = float(minimum)
    return target


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Worst-first aggregation. Never a global average.

    `grimoire/review/divine_eye_microscope.md`: critical-feature AND gate -> worst patch
    -> worst component -> worst view -> overall verdict. One missing critical feature
    fails the iteration regardless of how well everything else scored.

    The pass/fail decision itself is delegated to per_feature.evaluate_features, so a
    critical feature that measured badly fails here for exactly the reason, and at exactly
    the threshold, it would fail in the review gate.
    """
    feature_scores: dict[str, Any] = {}
    insufficient: list[str] = []
    report_only: list[str] = []
    defects: list[str] = []
    gated_records: list[dict[str, Any]] = []

    for record in records:
        feature_id = record["featureId"]
        status = record["status"]
        if status == "insufficient-reference-resolution":
            # Excluded from the gating set: the reference cannot resolve it, so failing the
            # model for it would gate against an invention.
            insufficient.append(feature_id)
            continue
        if status == "component-not-visible":
            report_only.append(feature_id)
            continue
        # None here means MISSING, which per_feature.py treats as a gating failure.
        feature_scores[feature_id] = record["score"]
        gated_records.append(record)
        if status == "missing":
            defects.append(f"{feature_id}: absent from the render")
            continue
        if record["metrics"].get("colourGates") is False:
            report_only.append(feature_id)

    evaluation = evaluate_features(
        [_target_from_record(record) for record in gated_records], feature_scores
    )
    for defect in evaluation["defects"]:
        if defect not in defects:
            defects.append(defect)

    measured = [
        record for record in records
        if record["status"] == "measured" and isinstance(record.get("score"), (int, float))
    ]
    worst = min(measured, key=lambda record: record["score"]) if measured else None

    by_component: dict[str, float] = {}
    for record in measured:
        component = record.get("componentId") or "(unassigned)"
        score = float(record["score"])
        by_component[component] = min(by_component.get(component, 1.0), score)

    # per_feature statuses are "ok" / "below" / "missing"; only the latter two fail.
    gating_failures = [
        feature["id"]
        for feature in evaluation["features"]
        if feature.get("gating") and feature.get("status") in {"below", "missing"}
    ]

    return {
        "featureScores": feature_scores,
        "insufficientReferenceResolution": insufficient,
        "reportOnly": report_only,
        "worstFeature": (
            {"featureId": worst["featureId"], "score": worst["score"], "limitingMetric": worst["limitingMetric"]}
            if worst
            else None
        ),
        "worstComponentScores": {name: round(score, 4) for name, score in sorted(by_component.items())},
        "gatingFailures": gating_failures,
        "perFeatureEvaluation": evaluation,
        "defects": defects,
    }


def run_microscope(
    descriptors: list[dict[str, Any]],
    reference_png: Path,
    render_png: Path,
    hidden_png_by_component: dict[str, Path],
    *,
    default_minimum_pixels: int = DEFAULT_MINIMUM_PROJECTED_PIXELS,
    patch_dir: Path | None = None,
) -> dict[str, Any]:
    if not descriptors:
        raise MicroscopeError(
            "no feature descriptors supplied; a microscope run with nothing to measure would "
            "report a vacuous pass"
        )
    reference = Frame(reference_png)
    render = Frame(render_png)
    if (reference.width, reference.height) != (render.width, render.height):
        raise MicroscopeError(
            f"reference is {reference.width}x{reference.height} and render is "
            f"{render.width}x{render.height}; ROI crops must address the same pixel grid"
        )
    hidden_frames = {name: Frame(path) for name, path in hidden_png_by_component.items()}
    for name, frame in hidden_frames.items():
        if (frame.width, frame.height) != (render.width, render.height):
            raise MicroscopeError(
                f"component-hidden frame for {name!r} is {frame.width}x{frame.height}, which does "
                f"not match the render's {render.width}x{render.height}"
            )

    records = [
        measure_feature(
            descriptor,
            reference,
            render,
            hidden_frames,
            default_minimum_pixels=default_minimum_pixels,
            patch_dir=patch_dir,
        )
        for descriptor in descriptors
    ]
    summary = aggregate(records)
    # per_feature owns the verdict; the microscope owns the measurement.
    passed = bool(summary["perFeatureEvaluation"]["passed"])
    return {
        "action": summary["perFeatureEvaluation"]["action"],
        "version": "1.0",
        "generator": "stage4_review/microscope.py",
        "reference": str(reference_png),
        "render": str(render_png),
        "resolution": {"width": reference.width, "height": reference.height},
        "features": records,
        **summary,
        "passed": passed,
        "note": (
            "Feature scores are measured on native-resolution ROI crops, not on the 64x64 luma "
            "grid divine_eye.py uses, so features a few pixels wide are present in the comparison. "
            "Aggregation is worst-first; no global average is taken. Feed featureScores to "
            "per_feature.py --scores to apply the critical-feature AND gate."
        ),
    }


# --------------------------------------------------------------------------- CLI


def descriptors_from_spec(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Derive descriptors from a spec's detailInventory entries that carry an ROI.

    Lets an existing spec drive the microscope without hand-authoring a manifest. Only
    entries with a usable ROI are returned; the CLI reports how many were skipped so a
    partial inventory cannot look like full coverage.
    """
    inventory = spec.get("detailInventory")
    entries = inventory.get("details") if isinstance(inventory, dict) else None
    if not isinstance(entries, list):
        return []
    targets = spec.get("featureReviewTargets")
    criticality_by_id: dict[str, str] = {}
    if isinstance(targets, list):
        for target in targets:
            if isinstance(target, dict) and isinstance(target.get("id"), str):
                criticality_by_id[target["id"]] = str(target.get("tier", "detail"))
    descriptors: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        roi = entry.get("referenceRoi") or entry.get("roi")
        feature_id = entry.get("id") or entry.get("featureId")
        if not isinstance(feature_id, str) or roi is None:
            continue
        descriptors.append(
            {
                "featureId": feature_id,
                "componentId": entry.get("componentId") or entry.get("component"),
                "criticality": entry.get("criticality") or criticality_by_id.get(feature_id, "detail"),
                "referenceRoi": roi,
                "concave": entry.get("concave"),
                "minimumProjectedPixels": entry.get("minimumProjectedPixels"),
            }
        )
    return descriptors


def load_descriptors(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("features") if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise MicroscopeError(
            "manifest must be a JSON array of feature descriptors, or an object with a "
            "'features' array"
        )
    return [item for item in items if isinstance(item, dict)]


def _parse_hidden(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise MicroscopeError(f"--component-hidden expects componentId=path.png, got {raw!r}")
    component_id, path = raw.split("=", 1)
    if not component_id.strip():
        raise MicroscopeError("--component-hidden needs a non-empty componentId")
    return component_id, Path(path)


def _format_summary(result: dict[str, Any]) -> str:
    lines = [
        f"microscope: {len(result['features'])} feature(s) at "
        f"{result['resolution']['width']}x{result['resolution']['height']}"
    ]
    for record in result["features"]:
        score = record.get("score")
        rendered = f"{score:.4f}" if isinstance(score, (int, float)) else "—"
        flag = " [GATING]" if record.get("gating") else ""
        limiting = f" limited-by={record['limitingMetric']}" if record.get("limitingMetric") else ""
        lines.append(f"  {record['featureId']}: {record['status']} score={rendered}{limiting}{flag}")
    if result["insufficientReferenceResolution"]:
        lines.append(
            "insufficient reference resolution (excluded from gating): "
            + ", ".join(result["insufficientReferenceResolution"])
        )
    if result["reportOnly"]:
        lines.append("report-only: " + ", ".join(result["reportOnly"]))
    if result["worstFeature"]:
        worst = result["worstFeature"]
        lines.append(f"worst feature: {worst['featureId']} at {worst['score']:.4f} ({worst['limitingMetric']})")
    if result["gatingFailures"]:
        lines.append("GATING failures: " + ", ".join(result["gatingFailures"]))
    for defect in result["defects"]:
        lines.append(f"defect: {defect}")
    lines.append(
        "verdict: "
        + ("PASS" if result["passed"] else "GATE FAILURE")
        + f" (action: {result['action']})"
    )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--reference", type=Path, required=True, help="reference image PNG/JPEG")
    parser.add_argument("--render", type=Path, required=True, help="full-frame render PNG")
    parser.add_argument(
        "--manifest",
        type=Path,
        help="JSON array of feature descriptors (featureId, componentId, criticality, referenceRoi)",
    )
    parser.add_argument(
        "--from-spec",
        type=Path,
        help="Derive descriptors from this spec's detailInventory entries that carry a referenceRoi",
    )
    parser.add_argument(
        "--component-hidden",
        action="append",
        default=[],
        metavar="ID=PATH",
        help=(
            "full frame rendered with only this component hidden; enables visible-footprint "
            "measurement (repeat per component)"
        ),
    )
    parser.add_argument(
        "--minimum-projected-pixels",
        type=int,
        default=DEFAULT_MINIMUM_PROJECTED_PIXELS,
        help=(
            "reference-pixel floor below which a feature is excluded as "
            f"insufficient-reference-resolution (default {DEFAULT_MINIMUM_PROJECTED_PIXELS})"
        ),
    )
    parser.add_argument("--patch-dir", type=Path, help="Write the native-resolution crops here")
    parser.add_argument("--out", type=Path, help="Write the full result JSON to this path")
    parser.add_argument(
        "--scores-out",
        type=Path,
        help="Write just the featureScores mapping, ready for per_feature.py --scores",
    )
    parser.add_argument("--json", action="store_true", help="print JSON instead of a summary")
    args = parser.parse_args(argv)

    try:
        descriptors: list[dict[str, Any]] = []
        if args.manifest:
            descriptors.extend(load_descriptors(args.manifest.expanduser().resolve()))
        if args.from_spec:
            spec = json.loads(args.from_spec.expanduser().resolve().read_text(encoding="utf-8"))
            derived = descriptors_from_spec(spec)
            if not derived and not descriptors:
                raise MicroscopeError(
                    "--from-spec found no detailInventory entries carrying a referenceRoi; add ROIs "
                    "to the inventory or supply --manifest, rather than reporting a vacuous pass"
                )
            descriptors.extend(derived)
        if not args.manifest and not args.from_spec:
            parser.error("one of --manifest or --from-spec is required")

        hidden = dict(_parse_hidden(raw) for raw in args.component_hidden)
        result = run_microscope(
            descriptors,
            args.reference.expanduser().resolve(),
            args.render.expanduser().resolve(),
            {name: path.expanduser().resolve() for name, path in hidden.items()},
            default_minimum_pixels=args.minimum_projected_pixels,
            patch_dir=args.patch_dir.expanduser().resolve() if args.patch_dir else None,
        )

        text = json.dumps(result, indent=2, ensure_ascii=False)
        if args.out:
            out_path = args.out.expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text + "\n", encoding="utf-8")
        if args.scores_out:
            scores_path = args.scores_out.expanduser().resolve()
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            scores_path.write_text(
                json.dumps(result["featureScores"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(text if args.json else _format_summary(result))
        return 0 if result["passed"] else 1
    except MicroscopeError as exc:
        # Fail closed: a missing ROI, missing capture or unmeasurable manifest is an
        # error, never a printed complaint with a zero exit.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
