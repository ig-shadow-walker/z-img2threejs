# Gates Reference (full contract)

Read this reference completely before any visual review or `continue` decision. `SKILL.md` keeps
only the executable order and one-line summary; this file defines the mandatory gate behavior.

- **Suitability + reference integrity**: pass / conditional / reject before any planning
  (`grimoire/intake/validation_rubric.md`), AND every reference admitted via
  `forge/stage1_intake/check_reference_admission.py` (rejects empty/fragmented/tiny/duplicate/
  undecodable refs with a reason). Intake understanding cross-checked by
  `forge/stage1_intake/check_intake_correctness.py` (halts on a confident class contradiction).
- **Divine Eye (the harness heart) — deterministic-first, model-last**: the render evaluator is
  `forge/stage4_review/divine_eye.py` — a zero-token multi-signal ensemble (IoU/scale HARD gates;
  proportion/symmetry-parity/pHash/SSIM/edge/blowout/flat/tonal-parity soft) with self-uncertainty
  (`probe` on signal disagreement) and deterministic routing (`continue`/`refine-spec`/`refine-code`/
  `probe`). The VLM (`forge/stage4_review/vlm_gate.py`) is a gated, calibrated, cross-checked
  last layer: **never consulted on a hard-gate failure**, multi-sample-voted, and can rescue a
  soft near-threshold reject but never grant past a hard geometric failure.
- **Multi-angle or it didn't happen**: a non-planar form must hold from ≥2 camera angles.
  `forge/stage4_review/diagnose_render_multi_angle.py` flags `degenerate-view` when an orbited
  silhouette collapses (a flat plane faking a volume). Orbit angles use reference-free
  self-consistency — never scored against a reference angle the photo doesn't cover.
- **Recorded off-axis and placement evidence (schema 2.2+)**: a gate whose verdict is only printed
  cannot block anything. `turntable_gate.py`, `diagnose_render_multi_angle.py`,
  `attachment_anchor.py` and `self_intersection.py` all accept `--pass-id`/`--in-place` and append a
  compact record to `spec.gateResults` (contract: `forge/_shared/gate_results.py`).
  `orchestrate_passes.py check` then requires, for the pass being closed:
  a `turntable` **or** `multi-angle` entry whenever any component is non-planar, and an
  `attachment-anchor` entry whenever the spec declares child appendages. Tier 1 alone is not
  sufficient — it scores one camera on a 64×64 luma grid and therefore passes an off-axis defect by
  construction. This is the gate class whose absence let a hole through a skull, a hat at hip height
  and a charm below the ground plane survive eight front-only review rounds.

  **Coverage is part of the verdict.** Each entry stores the gate's own account of what it did not
  measure — `missingAzimuths`, `sampledVertexCount`, `unmeasuredAttachments`, `orbitViewCount` — and
  a `passed` entry only satisfies the gate when `coverageComplete` is also true. A clean
  self-intersection run over zero sampled vertices, or a multi-angle run with one orbit view, is
  recorded as evidence of what ran and unlocks nothing. The three unsatisfied cases stay distinct in
  the failure message, because "never ran", "ran and failed" and "ran clean but measured nothing"
  need different fixes. Pre-2.2 specs are exempt, mirroring how `requires_topology_classification`
  gates the 2.1 topology fields, so work authored before the requirement is not retroactively broken.
- **Feature-scale fidelity (the microscope)**: `forge/stage4_review/microscope.py` is the instrument
  for details the ensemble cannot see. `divine_eye.py` computes SSIM/tonal/blowout/flat on a 64×64
  luma grid and edges on 96×96, so a feature a few pixels wide in a 1920px reference is **absent
  before any comparison happens** rather than scored badly, and `per_feature.py` cannot compensate
  because it consumes a scores dict and never opens an image. The microscope crops each declared
  feature ROI at native resolution, measures inside the crop, and emits scores in the shape
  `per_feature.py --scores` consumes; the verdict is delegated to `per_feature.evaluate_features` so
  the threshold rule has one definition. Scoring within a feature is **worst-of, never averaged**, so
  a correct colour cannot mask wrong linework; aggregation across features is worst-first, never a
  global mean. Three rules are enforced:
  - **Visible footprint, not isolation.** Measure the pixels that differ between the full frame and a
    frame with only that component hidden (`--component-hidden id=frame.png`). An isolation render
    reveals geometry that is occluded in the real frame; measured that way the mini-dragon's wings
    read `darkShare` 0.4362 against a reference 0.2240 and the spars were thinned to 0.72× on the
    strength of it. Measured correctly at the original radii the same wings read 0.1891 against
    0.2240 — comfortably inside tolerance. **The spars were never too thick.** Keep isolation renders
    for structural questions, where seeing occluded geometry is the point.
  - **Never colour-gate a concave feature.** A dark ratio there measures cavity shading, not
    material. The mini-dragon's ear read 14.3% dark, but bucketed by `min(r,g,b)` only 14% of it was
    near-black — a gradient, i.e. shading — while the wing's dark ran 32.9% near-black. Gating on it
    would have driven a dark inner-ear material to be invented to match a shadow. The microscope
    buckets the reference dark pixels and drops colour to report-only when the population reads
    shading-like, so the evidence for the call is recorded rather than asserted.
  - **Zoom cannot create information the reference does not contain.** A feature whose reference ROI
    falls below its pixel floor returns `insufficient-reference-resolution` and is excluded from the
    gating scores. Failing a model for a feature the photograph cannot resolve gates it against an
    invention, which is worse than not gating it.

  Full rules and the tier structure: `grimoire/review/divine_eye_microscope.md`.
- **Projection needs a solved camera**: projection fidelity is camera alignment — the pixels land
  where the camera says they land — so a perspective bake through an unfitted camera writes the
  misalignment into the albedo, where every later gate scores it as a *material* defect.
  `forge/stage3_build/bake_projected_texture.py` therefore returns `BLOCKED` with exit 1 for
  `perspective-camera-projection` unless the camera was solved: `method: numerical-landmark-fit`, no
  `agentFill` placeholders, converged, and RMS reprojection error within `--max-reprojection-error`
  (default 4px). Solve it with `forge/stage1_intake/solve_camera_pose.py <image> --correspondences
  <pairs.json>` — at least six non-collinear 3D↔2D landmark pairs, which runs the seven-parameter
  damped least-squares fit. Orthographic and triplanar modes are exempt because neither claims to
  reproduce the reference viewpoint. `--allow-unfitted-camera` proceeds but stamps the descriptor an
  explicit approximation; never use it for a reference-matched finish.
- **Report-only signals stay report-only until calibrated**: `hueZoneParity`, `specularWash` and
  `directionalChamfer` are computed and reported but carry no ensemble weight, because a signal that
  has never been measured against known-good/known-bad pairs must not move a verdict.
  `forge/stage4_review/build_corpus.py` generates a labeled corpus by controlled perturbation and
  `forge/stage4_review/calibrate_eye.py` audits it. The corpus is a valid **regression** harness and
  never authorizes a threshold change: every synthetic pair is stamped
  `provenance: synthetic-perturbation`, and `calibrate_eye.py` refuses
  `thresholdFlipAuthorized` on any corpus containing one. Only real browser renders can satisfy the
  precondition. Measured gaps the corpus currently records: a 60° hue error PASSES (hueZoneParity
  scores it 0.000 — the signal detects it perfectly and is simply disconnected), and a radius-3 blur
  PASSES at fidelity 0.890. A deliberate negative control (`remove-feature`) scores fidelity
  **1.0000**, which is the 64×64 blind spot measured rather than argued.
- **CS2 knife review contract**: `forge/stage4_review/cs2_review.py` consumes the manifest and
  versioned scene fixture, then blocks wrong family identity, missing projection coverage,
  painted-region mismatch, critical identity-detail failure, finish/material response failure,
  and degenerate orbit form. It records exactness tier, hidden-region confidence, per-region
  confidence, approximation notes, camera, environment hash, exposure, tone mapping, resolution,
  background, and renderer version.
- **Bounded correction loop (token-burn safety)**: `forge/stage4_review/correction_loop.py`
  guarantees termination (success/repeated-defect/oscillation/plateau/hard-ceiling), escalating to
  `request-input` — never a silent infinite burn.
- **Tier 1 (legacy, still valid)**: "Tier 2 (AI-vision) never runs against a render that has not passed Tier 1." Run `forge/stage4_review/diagnose_render.py` (silhouette IoU/proportion/symmetry/per-part color) and record it (`--spec ... --in-place`) before requesting a comparison sheet; `orchestrate_passes.py check` refuses otherwise.
- **Pre-spec / strict-quality**: blocks code gen until the spec is deep enough for its contract.
- **Screenshot feedback**: `continue` is allowed only with a render + comparison sheet + global
  AI-vision score ≥ threshold (default 0.7) AND every critical feature ≥ its own threshold.
  Details + per-layer scorecard: `grimoire/feedback/render_capture.md`.
- **Action-ready**: build a runtime hierarchy (pivots, sockets, colliders, destruction groups),
  never an inert lump; expose `root.userData.sculptRuntime`. `grimoire/readiness/action_rigging.md`.
- **Assembly gate (structure, not pixels) — every model ships explodable AND clickable**: this is
  a build requirement, not a per-project extra. Name every mesh; flag surface relief
  `userData.explodeWithParent` so it rides its shell; let a named group of *anonymous* meshes be one
  part while a named group of *named* parts stays a container. Explode and part-picking must share
  one definition of "a part" — if they disagree, both are wrong. Separate parts by SCALING the
  layout about the model centre, never by pushing every part the same distance (that translates the
  arrangement without opening any gap). Then run
  `forge/stage4_review/check_part_coverage.py --spec <spec> --manifest <parts.json>`: it FAILS on a
  specified component that was never built and on two components fused onto one mesh; it warns on
  inventoried details that never reached the spec and on meshes belonging to no named part. This is
  the only gate that scores STRUCTURE — every other one scores pixels, and a single fused mesh
  wearing a projected photo passes all of those. Its limit is honest and must be stated when
  reporting: it proves you built what you specified, never that you specified enough.
  Full contract + the two rules it took a wrong pass to learn: `grimoire/build/geometry_patterns.md`.
- **Attachment**: child appendages (branches/limbs/handles/tubes) need `attachment.parentSocket`,
  `localStart`, `localEnd`, `contactType`, `embedDepth`/`overlap`, `gapTolerance` — no mid-air parts.
  `grimoire/readiness/joint_attachment.md`.
- **Material/lighting**: `grimoire/feedback/shading_realism.md` — independent PBR channels
  (never alias albedo into roughness/normal/AO), macro/meso/micro frequency bands, real lights.
- **Detail inventory**: for `moderate`+ subjects strict-quality blocks code gen until the
  `detailInventory` reaches `targetMinDetails` and every detail maps to a real component/material
  entry (gloss needs low-roughness/clearcoat; fasteners need instancing/micro parts).
- **Character track**: when `primaryDomain` is `character`/`hybrid` (or `--character`), the spec
  author auto-builds a stylized humanoid template (head/neck/torso/arms + hair, glasses,
  headphones, face features), flattened to world space under a hidden root, with per-part
  character materials and character build passes (`proportion-lock`, `feature-placement`).
  strict-quality requires a filled `anatomy` block (head-units, proportions, face landmarks) and
  character feature targets. Suitability routing for humans: `grimoire/intake/validation_rubric.md`
  (stylized vs maximum-likeness). Stylized bust, not a face-copy; refine positions per reference.
