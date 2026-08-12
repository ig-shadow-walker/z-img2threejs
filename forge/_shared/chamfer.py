#!/usr/bin/env python3
"""Directional chamfer distance between two silhouette boundaries.

WHY, given IoU already exists
-----------------------------
Silhouette IoU measures overlapping AREA, and area is insensitive to boundary SHAPE.
Two silhouettes can share 0.95 IoU while one has a smooth curve where the other has a
serrated edge, because the mismatched pixels sit in a thin band along the contour and a
thin band is a small fraction of a filled region. That is precisely the error class a
procedural reconstruction produces: right mass, wrong edge.

Chamfer distance measures the contour directly -- for every boundary point, how far is
the nearest boundary point in the other image. `divine_eye.py` lists "Directional Chamfer
Distance" as a deferred Phase-3 signal; this is that signal.

DIRECTIONAL, not plain
----------------------
Plain chamfer matches a boundary point to the nearest boundary point of any orientation,
so a horizontal edge can be "matched" by a vertical one that happens to be nearby, and a
rotated or transposed feature scores well. Directional chamfer only matches points whose
local edge orientation agrees, which is what makes it sensitive to a tear edge's angle
rather than only its position.

Bidirectional by default: A->B alone is blind to boundary that exists in B and not in A
(and vice versa), so the reported distance is the max of both directions. Max rather than
mean, because a one-directional match is exactly the asymmetric-error case worth catching.

Pure stdlib, deterministic, zero token.
"""

from __future__ import annotations

import math
from typing import Any

# Edge orientation is defined modulo pi (an edge has no head or tail), so bins span 180
# degrees. Four bins -- horizontal, vertical and the two diagonals -- keep the matching
# meaningfully orientation-aware while bounding cost; more bins make the match stricter and
# the signal noisier on anti-aliased contours.
DEFAULT_ORIENTATION_BINS = 4

# Neighbouring bins are accepted as an orientation match. Without this an edge sitting
# exactly on a bin boundary would fail to match its own counterpart in the other image.
ORIENTATION_BIN_TOLERANCE = 1

# Boundary points are strided down to at most this many per image, deterministically. A
# noisy or textured mask can otherwise produce tens of thousands of boundary pixels and
# turn an O(|A|x|B|) search into a stall. The applied stride is reported so reduced
# coverage is visible rather than silent.
MAX_BOUNDARY_POINTS = 1200

# Distance at which two contours are treated as fully unrelated, as a fraction of the grid
# width. Beyond ~8% of the frame the contours are not describing the same edge.
FULL_MISMATCH_FRACTION = 0.08


def boundary_points(
    mask: list[bool],
    size: int,
    *,
    bins: int = DEFAULT_ORIENTATION_BINS,
) -> list[tuple[int, int, int]]:
    """Return `(x, y, orientation_bin)` for every silhouette boundary pixel.

    A boundary pixel is a foreground pixel with at least one background 4-neighbour.
    Orientation comes from a Sobel gradient over the binary mask; the gradient is normal
    to the edge, so the edge direction is the gradient angle rotated by 90 degrees, taken
    modulo pi and quantized into `bins`.
    """
    points: list[tuple[int, int, int]] = []
    for y in range(size):
        for x in range(size):
            if not mask[y * size + x]:
                continue
            # 4-neighbourhood; the frame border counts as background so a silhouette
            # running off the edge still yields a boundary there.
            if (
                x > 0 and mask[y * size + x - 1]
                and x < size - 1 and mask[y * size + x + 1]
                and y > 0 and mask[(y - 1) * size + x]
                and y < size - 1 and mask[(y + 1) * size + x]
            ):
                continue

            def at(px: int, py: int) -> float:
                if px < 0 or py < 0 or px >= size or py >= size:
                    return 0.0
                return 1.0 if mask[py * size + px] else 0.0

            gx = (
                -at(x - 1, y - 1) - 2 * at(x - 1, y) - at(x - 1, y + 1)
                + at(x + 1, y - 1) + 2 * at(x + 1, y) + at(x + 1, y + 1)
            )
            gy = (
                -at(x - 1, y - 1) - 2 * at(x, y - 1) - at(x + 1, y - 1)
                + at(x - 1, y + 1) + 2 * at(x, y + 1) + at(x + 1, y + 1)
            )
            if gx == 0.0 and gy == 0.0:
                # No measurable gradient: an isolated pixel. Bin 0 rather than dropping it,
                # so it still contributes to the distance.
                orientation = 0
            else:
                # Edge direction is normal to the gradient; mod pi because an edge is
                # undirected.
                angle = (math.atan2(gy, gx) + math.pi / 2.0) % math.pi
                orientation = int(angle / math.pi * bins) % bins
            points.append((x, y, orientation))
    return points


def _stride(points: list[tuple[int, int, int]]) -> tuple[list[tuple[int, int, int]], int]:
    """Deterministically thin a boundary-point list to MAX_BOUNDARY_POINTS."""
    if len(points) <= MAX_BOUNDARY_POINTS:
        return points, 1
    step = math.ceil(len(points) / MAX_BOUNDARY_POINTS)
    return points[::step], step


def _orientations_match(a: int, b: int, bins: int) -> bool:
    """True when two orientation bins agree within tolerance, wrapping around pi."""
    delta = abs(a - b) % bins
    delta = min(delta, bins - delta)
    return delta <= ORIENTATION_BIN_TOLERANCE


def _one_way_distance(
    source: list[tuple[int, int, int]],
    target: list[tuple[int, int, int]],
    bins: int,
) -> float | None:
    """Mean distance from each source point to the nearest orientation-matched target point.

    Returns None when either boundary is empty -- no contour means no contour comparison,
    which must be reported as absent rather than as a perfect or a zero score.
    """
    if not source or not target:
        return None
    by_bin: dict[int, list[tuple[int, int]]] = {}
    for x, y, orientation in target:
        by_bin.setdefault(orientation, []).append((x, y))

    total = 0.0
    for x, y, orientation in source:
        candidates: list[tuple[int, int]] = []
        for offset in range(-ORIENTATION_BIN_TOLERANCE, ORIENTATION_BIN_TOLERANCE + 1):
            candidates.extend(by_bin.get((orientation + offset) % bins, ()))
        if not candidates:
            # No target boundary shares this orientation at all: an unmatched edge
            # direction. Charge the full-mismatch distance rather than skipping it, which
            # would quietly reward a missing edge.
            total += float("inf")
            continue
        best = min((px - x) ** 2 + (py - y) ** 2 for px, py in candidates)
        total += math.sqrt(best)
    return total / len(source)


def directional_chamfer(
    reference_mask: list[bool],
    render_mask: list[bool],
    size: int,
    *,
    bins: int = DEFAULT_ORIENTATION_BINS,
) -> dict[str, Any]:
    """Bidirectional directional chamfer distance and its [0,1] agreement score.

    `agreement` is 1.0 for identical contours and falls to 0.0 at
    FULL_MISMATCH_FRACTION of the grid width. `distance` is in grid pixels, so it is
    directly interpretable ("the contours are off by 3.2px on a 224px grid").
    """
    reference_points, reference_stride = _stride(boundary_points(reference_mask, size, bins=bins))
    render_points, render_stride = _stride(boundary_points(render_mask, size, bins=bins))

    forward = _one_way_distance(reference_points, render_points, bins)
    backward = _one_way_distance(render_points, reference_points, bins)

    if forward is None or backward is None:
        return {
            "distance": None,
            "agreement": None,
            "forward": forward,
            "backward": backward,
            "referenceBoundaryPoints": len(reference_points),
            "renderBoundaryPoints": len(render_points),
            "strides": {"reference": reference_stride, "render": render_stride},
            "orientationBins": bins,
            "note": (
                "one of the silhouettes has no boundary, so there is no contour to compare; "
                "reported as absent rather than as a score"
            ),
        }

    # Max, not mean: a good match in one direction must not hide a missing contour in the
    # other.
    distance = max(forward, backward)
    limit = size * FULL_MISMATCH_FRACTION
    agreement = 0.0 if math.isinf(distance) else max(0.0, 1.0 - distance / limit)

    return {
        "distance": None if math.isinf(distance) else round(distance, 4),
        "agreement": round(agreement, 4),
        "forward": None if math.isinf(forward) else round(forward, 4),
        "backward": None if math.isinf(backward) else round(backward, 4),
        "referenceBoundaryPoints": len(reference_points),
        "renderBoundaryPoints": len(render_points),
        "strides": {"reference": reference_stride, "render": render_stride},
        "orientationBins": bins,
        "fullMismatchDistance": round(limit, 4),
        "note": (
            "bidirectional directional chamfer on the silhouette contour; complements IoU, "
            "which measures overlapping area and is insensitive to boundary shape at matched area"
        ),
    }
