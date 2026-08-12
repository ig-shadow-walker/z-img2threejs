#!/usr/bin/env python3
"""Divine Eye calibration harness (Plan 1.3 §5) — fit/validate the Eye on a labeled corpus.

Before the Eye is allowed to HARD-GATE anything, it runs in report-only mode over a
corpus of labeled (reference, render) pairs — known-good (the render should pass) and
known-bad (it must be rejected). This harness runs divine_eye.evaluate on each pair,
tabulates every signal per label class, and checks that the fidelity score cleanly
SEPARATES good from bad. A corpus needs BOTH classes or it only measures false-rejects.

Acceptance (absolute, on the corpus): every known-good pair passes AND every known-bad
pair is rejected. Only then may report-only be flipped to hard-gate. Percentage targets
(Q1/Q4/Q5) apply to a rolling window of ≥20 real builds, not this small corpus.

Note: producing real RENDERS of the showcase demos needs a browser (deferred to the
end-to-end infra). This harness is the deterministic, render-agnostic scaffold: give it
whatever (reference, render) PNG pairs exist and it reports the separation + suggests a
threshold. Pure stdlib + reuses divine_eye. Zero token.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from divine_eye import evaluate  # noqa: E402


def run_corpus(pairs: list[dict]) -> dict[str, Any]:
    """pairs: [{"reference": path, "render": path, "label": "good"|"bad"}, ...].
    Runs the Eye on each and aggregates per-signal stats per label class."""
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        result = evaluate(Path(pair["reference"]), Path(pair["render"]))
        rows.append({
            "label": pair.get("label", "unlabeled"),
            "fidelity": result["fidelity"],
            "verdict": result["verdict"],
            "hardGateFailures": result["hardGateFailures"],
            "signals": result["signals"],
            "reference": pair["reference"],
            "render": pair["render"],
        })
    return {"rows": rows, "signalStats": _signal_stats(rows)}


def _signal_stats(rows: list[dict]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    for label in ("good", "bad"):
        labelled = [r for r in rows if r["label"] == label]
        if not labelled:
            continue
        fids = [r["fidelity"] for r in labelled]
        stats[label] = {
            "count": len(labelled),
            "fidelityMin": round(min(fids), 4),
            "fidelityMax": round(max(fids), 4),
            "fidelityMean": round(sum(fids) / len(fids), 4),
        }
    return stats


def separation(rows: list[dict]) -> dict[str, Any]:
    """Does fidelity cleanly separate good from bad? A clean split exists when the
    worst good fidelity is strictly above the best bad fidelity; the suggested
    threshold sits between them."""
    good = [r["fidelity"] for r in rows if r["label"] == "good"]
    bad = [r["fidelity"] for r in rows if r["label"] == "bad"]
    result: dict[str, Any] = {
        "hasGood": bool(good),
        "hasBad": bool(bad),
        "minGoodFidelity": round(min(good), 4) if good else None,
        "maxBadFidelity": round(max(bad), 4) if bad else None,
    }
    if good and bad:
        min_good, max_bad = min(good), max(bad)
        clean = min_good > max_bad
        result["cleanSeparation"] = clean
        result["suggestedThreshold"] = round((min_good + max_bad) / 2, 4) if clean else None
        # absolute acceptance: every good passes at the current target, every bad rejected
        result["allGoodPass"] = all(
            r["verdict"] == "pass" for r in rows if r["label"] == "good"
        )
        result["allBadRejected"] = all(
            r["verdict"] != "pass" for r in rows if r["label"] == "bad"
        )
        result["corpusAcceptable"] = result["allGoodPass"] and result["allBadRejected"]
    else:
        result["cleanSeparation"] = None
        result["corpusAcceptable"] = False
        result["warning"] = "corpus needs BOTH good and bad pairs (else it only measures false-rejects)"
    return result


SYNTHETIC_PROVENANCE = "synthetic-perturbation"


def provenance_audit(pairs: list[dict]) -> dict[str, Any]:
    """Decide whether this corpus can authorize flipping report-only to hard-gate.

    A synthetic perturbation of a reference photograph is a valid REGRESSION fixture: it
    proves the Eye responds in the right direction to a known defect class. It is not a
    valid basis for choosing threshold VALUES, because it carries the reference's own
    shading, noise and material response, so its absolute fidelity numbers are not the
    numbers a Three.js render produces. Authorizing a threshold change from it would be
    calibrating against the wrong distribution while appearing rigorous.
    """
    counts: dict[str, int] = {}
    for pair in pairs:
        provenance = pair.get("provenance") or "unlabelled"
        counts[provenance] = counts.get(provenance, 0) + 1
    synthetic = counts.get(SYNTHETIC_PROVENANCE, 0)
    unlabelled = counts.get("unlabelled", 0)
    blockers: list[str] = []
    if synthetic:
        blockers.append(
            f"{synthetic} pair(s) are {SYNTHETIC_PROVENANCE}: valid as a regression harness, "
            "not as a basis for threshold values"
        )
    if unlabelled:
        blockers.append(
            f"{unlabelled} pair(s) carry no provenance; label each pair so its origin is "
            "auditable before a threshold decision rests on it"
        )
    return {
        "counts": counts,
        "thresholdFlipAuthorized": not blockers,
        "blockers": blockers,
    }


def expectation_audit(rows: list[dict], pairs: list[dict]) -> dict[str, Any]:
    """Check each pair against its own declared `expectedDetection`.

    This is what makes a synthetic corpus useful despite never authorizing a threshold
    flip. `expectedDetection` records what the Eye MEASURABLY does, so a deviation in
    either direction is a regression: a signal that got less sensitive, or one that got
    more sensitive than its recorded behaviour (which may be a welcome improvement, but is
    still a change worth seeing rather than absorbing silently).

    `knownGaps` is separate and reported, never failed: a pair whose measured behaviour is
    correct-as-recorded but whose DESIRED behaviour differs. Those are the Eye's real
    sensitivity holes, and keeping them visible in a passing run is the point.
    """
    by_render = {pair.get("render"): pair for pair in pairs}
    regressions: list[str] = []
    improvements: list[str] = []
    known_gaps: list[dict[str, Any]] = []
    controls: list[str] = []
    not_asserted: list[str] = []

    for row in rows:
        pair = by_render.get(row.get("render")) or {}
        expected = pair.get("expectedDetection")
        desired = pair.get("desiredDetection")
        name = pair.get("perturbation") or row.get("render")
        detected = row["verdict"] != "pass"

        if pair.get("negativeControl"):
            controls.append(name)
        if expected is None:
            not_asserted.append(
                f"{name}: not asserted ({pair.get('rationale', 'no rationale recorded')})"
            )
            continue
        if detected != expected:
            message = (
                f"{name}: recorded behaviour is "
                f"{'detect' if expected else 'pass'} but the Eye "
                f"{'detected' if detected else 'passed'} it "
                f"(fidelity {row['fidelity']})"
            )
            (improvements if detected else regressions).append(message)
            continue
        if desired is not None and expected != desired:
            known_gaps.append(
                {
                    "perturbation": name,
                    "defectClass": pair.get("defectClass"),
                    "fidelity": row["fidelity"],
                    "rationale": pair.get("rationale"),
                }
            )

    return {
        "negativeControls": controls,
        "notAsserted": not_asserted,
        "regressions": regressions,
        "improvements": improvements,
        "knownGaps": known_gaps,
        # Improvements are surfaced but do not fail: a signal getting sharper is not a
        # defect, though the recorded expectation should then be updated.
        "regressionClean": not regressions,
    }


def calibrate(pairs: list[dict]) -> dict[str, Any]:
    corpus = run_corpus(pairs)
    corpus["separation"] = separation(corpus["rows"])
    corpus["provenance"] = provenance_audit(pairs)
    corpus["expectations"] = expectation_audit(corpus["rows"], pairs)
    corpus["reportOnly"] = True
    # Two independent conditions. The corpus can be statistically acceptable and still be
    # the wrong KIND of data to set a threshold from, so both must hold.
    corpus["thresholdFlipAuthorized"] = bool(
        corpus["separation"].get("corpusAcceptable")
        and corpus["provenance"]["thresholdFlipAuthorized"]
    )
    corpus["note"] = (
        "report-only — flipping to hard-gate requires BOTH corpusAcceptable (all good pass, "
        "all bad rejected) AND provenance.thresholdFlipAuthorized (every pair a real render, "
        "not a synthetic perturbation)"
    )
    return corpus


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path,
                        help="JSON list of {reference, render, label} pairs, or a "
                             "build_corpus.py manifest with a 'pairs' array")
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Exit on the expectation audit (each pair vs its own expectedDetection) instead "
             "of on absolute corpus acceptance. This is the correct mode for a synthetic "
             "corpus, which can never authorize a threshold flip.",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.corpus.read_text(encoding="utf-8"))
        # Accept both the bare list and a build_corpus.py manifest.
        pairs = data.get("pairs") if isinstance(data, dict) else data
        if not isinstance(pairs, list):
            raise ValueError(
                "corpus must be a JSON array of pairs, or an object with a 'pairs' array"
            )
        result = calibrate(pairs)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        sep = result["separation"]
        expectations = result["expectations"]
        print(f"corpus: {result['signalStats']}")
        print(f"separation: clean={sep.get('cleanSeparation')} acceptable={sep.get('corpusAcceptable')} "
              f"suggestedThreshold={sep.get('suggestedThreshold')}")
        print(f"provenance: {result['provenance']['counts']}")
        for blocker in result["provenance"]["blockers"]:
            print(f"  blocker: {blocker}")
        if expectations["negativeControls"]:
            print(f"negative controls: {', '.join(expectations['negativeControls'])}")
        for gap in expectations["knownGaps"]:
            print(f"  KNOWN GAP {gap['perturbation']} ({gap['defectClass']}) fidelity={gap['fidelity']}")
        for item in expectations["regressions"]:
            print(f"  REGRESSION {item}")
        for item in expectations["improvements"]:
            print(f"  IMPROVED {item} — update the recorded expectation")
        for item in expectations["notAsserted"]:
            print(f"  skipped {item}")
        print(f"regressionClean: {expectations['regressionClean']}")
        print(f"thresholdFlipAuthorized: {result['thresholdFlipAuthorized']}")
    if args.regression:
        # Regression mode: every pair behaved as its own descriptor predicted.
        return 0 if result["expectations"]["regressionClean"] else 1
    # Default: exit 0 only when the corpus cleanly validates the Eye AND is the right kind
    # of data to validate it with.
    return 0 if result["thresholdFlipAuthorized"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
