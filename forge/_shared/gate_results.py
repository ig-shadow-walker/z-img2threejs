#!/usr/bin/env python3
"""Shared recording + query contract for the deterministic off-axis review gates.

Tier 1 (`stage4_review/diagnose_render.py`) has always written its verdict into the
spec as `tier1Results`, and that channel is the only reason
`orchestrate_passes.py check` can refuse to unlock the Tier-2 comparison sheet
without passing evidence.

The off-axis gates had no such channel. `turntable_gate.py`,
`self_intersection.py`, `attachment_anchor.py` and `diagnose_render_multi_angle.py`
each computed a real verdict and then printed it to stdout, where nothing
deterministic could read it. SKILL.md calls all four blocking ("Multi-angle or it
didn't happen"), but a failure could only stop a pass if the agent volunteered to
honor it — which is exactly how, in the project's own words, "a hole through a
skull, a hat at hip height and a charm floating below the ground plane survived
eight front-only review rounds". This module gives those gates the channel Tier 1
already had.

One array (`gateResults`), one entry shape, queried by gate name, so adding a gate
later does not add another schema field.

COVERAGE IS PART OF THE VERDICT. Each gate already reports what it did *not* look
at — `missingAzimuths`, `sampledVertexCount`, `unmeasuredAttachments` — and
`self_intersection.py` and `attachment_anchor.py` both carry comments warning that
"0 violations" from a check that never ran is not a pass. So a recorded entry only
satisfies a gate when `passed` AND `coverageComplete` are both true. A clean
verdict over zero sampled vertices is stored (it is evidence of what was run) but
never unlocks anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

GATE_RESULTS_KEY = "gateResults"

GATE_TURNTABLE = "turntable"
GATE_SELF_INTERSECTION = "self-intersection"
GATE_ATTACHMENT_ANCHOR = "attachment-anchor"
GATE_MULTI_ANGLE = "multi-angle"

KNOWN_GATES: frozenset[str] = frozenset(
    {
        GATE_TURNTABLE,
        GATE_SELF_INTERSECTION,
        GATE_ATTACHMENT_ANCHOR,
        GATE_MULTI_ANGLE,
    }
)

# Gates that answer "does this form hold when the camera moves". A non-planar
# subject needs at least one of these, because every other gate scores one camera.
OFF_AXIS_GATES: frozenset[str] = frozenset({GATE_TURNTABLE, GATE_MULTI_ANGLE})

# Gates that answer "is the built structure sound", independent of any camera.
STRUCTURE_GATES: frozenset[str] = frozenset(
    {GATE_SELF_INTERSECTION, GATE_ATTACHMENT_ANCHOR}
)


class GateResultError(Exception):
    """Raised when a gate result cannot be recorded."""


def _int_or_zero(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _list_len(value: Any) -> int:
    return len(value) if isinstance(value, (list, tuple)) else 0


def coverage_for(gate: str, result: dict[str, Any]) -> dict[str, Any]:
    """Extract the gate's own account of what it did and did not measure.

    Every field here is already produced by the gate; this only normalizes it into
    one shape so `satisfies_gate` can refuse a vacuous clean verdict.
    """
    if gate == GATE_TURNTABLE:
        missing = result.get("missingAzimuths")
        measured = _list_len(result.get("angles"))
        return {
            "measuredAzimuths": measured,
            "missingAzimuths": list(missing) if isinstance(missing, (list, tuple)) else [],
            # `covered` is the gate's own azimuth-coverage verdict.
            "complete": bool(result.get("covered")) and _list_len(missing) == 0 and measured > 0,
        }
    if gate == GATE_SELF_INTERSECTION:
        sampled = _int_or_zero(result.get("sampledVertexCount"))
        total = _int_or_zero(result.get("totalVertexCount"))
        return {
            "sampledVertexCount": sampled,
            "totalVertexCount": total,
            # A run that sampled nothing proves nothing, however clean it looks.
            "complete": sampled > 0,
        }
    if gate == GATE_ATTACHMENT_ANCHOR:
        unmeasured = result.get("unmeasuredAttachments")
        count = _int_or_zero(result.get("attachmentCount"))
        return {
            "attachmentCount": count,
            "unmeasuredAttachments": list(unmeasured) if isinstance(unmeasured, (list, tuple)) else [],
            # A spec with no attachments is vacuously complete: there is nothing to
            # measure, and that is a real answer rather than a missing one.
            "complete": _list_len(unmeasured) == 0,
        }
    if gate == GATE_MULTI_ANGLE:
        orbits = _list_len(result.get("angles"))
        return {
            "orbitViewCount": orbits,
            # SKILL.md: a non-planar form must hold from >= 2 angles, so one orbit
            # view beside the fixed view is the floor.
            "complete": orbits >= 2,
        }
    raise GateResultError(f"unknown gate {gate!r}; expected one of {sorted(KNOWN_GATES)}")


def build_entry(
    gate: str,
    result: dict[str, Any],
    *,
    passed: bool,
    pass_id: str | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the compact record stored in the spec.

    The full gate output stays on stdout; the spec keeps the verdict, the coverage
    account, and any evidence paths, so a spec does not grow by a full report per
    gate per pass.
    """
    if gate not in KNOWN_GATES:
        raise GateResultError(f"unknown gate {gate!r}; expected one of {sorted(KNOWN_GATES)}")
    coverage = coverage_for(gate, result)
    complete = bool(coverage.pop("complete"))
    entry: dict[str, Any] = {
        "gate": gate,
        "passed": bool(passed),
        "passId": pass_id or None,
        "coverage": coverage,
        "coverageComplete": complete,
    }
    if evidence:
        entry["evidence"] = evidence
    return entry


def record_gate_result(spec: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Append one gate entry to `spec['gateResults']`, creating the array if absent."""
    results = spec.setdefault(GATE_RESULTS_KEY, [])
    if not isinstance(results, list):
        raise GateResultError(f"spec[{GATE_RESULTS_KEY!r}] must be a list")
    results.append(entry)
    return entry


def gate_entries(spec: dict[str, Any], gate: str, pass_id: str | None = None) -> list[dict[str, Any]]:
    """Every recorded entry for one gate, optionally narrowed to a pass."""
    results = spec.get(GATE_RESULTS_KEY)
    if not isinstance(results, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in results:
        if not isinstance(entry, dict) or entry.get("gate") != gate:
            continue
        if pass_id is not None and entry.get("passId") != pass_id:
            continue
        out.append(entry)
    return out


def satisfies_gate(spec: dict[str, Any], gate: str, pass_id: str | None = None) -> bool:
    """True only when a recorded entry both passed AND actually measured something.

    This is the whole point of storing coverage: `passed` alone is satisfiable by a
    check that never ran.
    """
    return any(
        entry.get("passed") and entry.get("coverageComplete")
        for entry in gate_entries(spec, gate, pass_id)
    )


def satisfies_any_gate(spec: dict[str, Any], gates: frozenset[str], pass_id: str | None = None) -> bool:
    return any(satisfies_gate(spec, gate, pass_id) for gate in gates)


def missing_gate_reasons(
    spec: dict[str, Any],
    gate: str,
    pass_id: str | None = None,
) -> str:
    """Explain *why* a gate is unsatisfied, distinguishing the three real cases.

    "never ran", "ran and failed" and "ran clean but measured nothing" need
    different fixes, so they must not collapse into one message.
    """
    entries = gate_entries(spec, gate, pass_id)
    if not entries:
        return f"{gate}: no recorded result for this pass"
    if any(entry.get("passed") and entry.get("coverageComplete") for entry in entries):
        return ""
    latest = entries[-1]
    if not latest.get("passed"):
        return f"{gate}: recorded result failed"
    coverage = latest.get("coverage") if isinstance(latest.get("coverage"), dict) else {}
    return f"{gate}: recorded result passed but measured nothing usable ({coverage})"


def add_recording_arguments(parser: argparse.ArgumentParser, *, spec_flag: bool = True) -> None:
    """Add the same `--spec/--pass-id/--in-place/--out-spec` quartet Tier 1 uses.

    `spec_flag=False` for a gate that already takes the spec as a positional
    (attachment_anchor.py), so the two cannot collide.
    """
    if spec_flag:
        parser.add_argument(
            "--spec",
            type=Path,
            help="ObjectSculptSpec JSON to record this gate result into",
        )
    parser.add_argument("--pass-id", help="Build pass this gate result belongs to")
    parser.add_argument("--in-place", action="store_true", help="Record the result into the spec")
    parser.add_argument("--out-spec", type=Path, help="Write the spec with the recorded result here")


def apply_recording(
    args: argparse.Namespace,
    gate: str,
    result: dict[str, Any],
    *,
    passed: bool,
    spec_path: Path | None = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Record the gate result into the spec when the caller asked for it.

    Returns the stored entry, or None when no recording was requested. Mirrors
    diagnose_render.py: `--in-place` writes back to the spec, `--out-spec` writes a
    copy, and supplying neither leaves the spec untouched.
    """
    target = spec_path if spec_path is not None else getattr(args, "spec", None)
    if target is None:
        return None
    in_place = bool(getattr(args, "in_place", False))
    out_spec = getattr(args, "out_spec", None)
    if not in_place and not out_spec:
        return None

    resolved = Path(target).expanduser().resolve()
    spec = json.loads(resolved.read_text(encoding="utf-8"))
    entry = build_entry(
        gate,
        result,
        passed=passed,
        pass_id=getattr(args, "pass_id", None),
        evidence=evidence,
    )
    record_gate_result(spec, entry)
    output = resolved if in_place else Path(out_spec).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entry
