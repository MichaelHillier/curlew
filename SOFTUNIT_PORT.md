# Soft-unit / NLL coupling — scoping handoff

**Status: NOT STARTED — this doc scopes the next milestone.** Read this + the memory
`wcsb-geoinr-loss-port` before doing anything. The task for the session that picks this up
is to produce a **design spec** (plan mode; no implementation until the design is approved).

## Why

The per-field GeoINR recipe in `curlew/geology/stratbuilder.py` (unit-only path) is at its
ceiling: stratigraphic ordering is essentially solved (0.1 % column inversions on wcsb), but
per-point fits to the well markers are loose (band accuracy ~0.6). Root cause (validated,
2026-06-12 session): hinge-only inequality losses give zero gradient once satisfied, nothing
calibrates each field's absolute value laterally, and post-hoc global iso-values land
mid-slack. The designed fix — and the **core reason for the GeoINR→curlew integration** — is
GeoINR's **soft-unit / NLL coupling across all scalar fields**, which gives every unit point
a two-sided, always-active likelihood, calibrates field values everywhere, and makes unit
classification consistent across the whole field stack (also fixing the sequential-combine
fragility at depth).

## Reference implementation (study it, do NOT transliterate it)

`D:/Development/GeoINR_Research2/geoinr/model/geological_model.py` —
`GeoINRSoftUnitModel(GeoINRModel)` (~line 337) and its helpers:

- `soft_carve_unit_probs(fields, domains, iso_values, unit_index_to_class, tau)` — the
  differentiable carve: evaluates **all** scalar fields at the query points and converts the
  field stack + **shared learnable `iso_values`** + stratigraphic domain rules into per-unit
  class probabilities with temperature `tau` (`soft_unit_tau`, default 0.05).
- `fit()` — per-epoch: per-field interface/normal losses (`include_unit_constraints=False` —
  the per-field above/below unit losses are *replaced* by the global term), then a
  **balanced per-level sample** of unit points (`soft_unit_points_per_level`, default 1024)
  → log-probs → **cross-entropy against the true unit labels** (`soft_unit_domain_ce`),
  plus an **iso-gap barrier** (`compute_multi_interface_iso_gap_barrier`) stopping adjacent
  iso-values collapsing onto each other.
- The owner's verdict on this code: it works (GeoINR_Research2-level accuracy) but is
  **hard to follow and could be designed better** — a redesign is encouraged; parity of
  *behaviour*, not of structure, is the goal.

## Hard requirements (from the owner)

1. **Extremely well designed**: easy to follow, minimal code. Quality of the design is the
   point of the session, not speed.
2. **Couples ALL scalar fields** (and the surface iso-values) in one likelihood; accurate
   per-point fitting, good training dynamics, robustness.
3. **Do not break `GeoModel.predict`** — its geological machinery (retro-deformation,
   faults, intrusions, overprints, domain boundaries) is the reason GeoINR is being ported
   into curlew at all. Extending GeoINR *with* those features is the goal; a port that
   bypasses or forks predict defeats the purpose.
4. Owner's instinct (to be tested, not assumed): the coupling may be expressible as **just a
   well-designed loss**, keeping predict as-is.

## Design avenues the spec must weigh (at least these)

- **A. `custom_loss` over curlew's native soft predict.** `GeoModel.fit(custom_loss=...)`
  already exists, and `GeoModel.predict(..., to_numpy=False)` already produces
  **differentiable** `softLithoID` / `softStructureID` via the `Overprint` sigmoid weights
  (`lithoSharpness` ≈ 1/tau). An NLL of soft lithology vs true levels would reuse the entire
  geological combine (faults/intrusions included) for free. Must verify: end-to-end
  differentiability of the predict path at training time (watch for `detach()` — e.g.
  iso resolution in `GeoEvent.getIsovalues`/`updateThresh` — and `batchEval` chunking),
  per-epoch cost, and how lithology-class probabilities map onto unit levels
  (`M.level_litho` / `llookup`).
- **B. A dedicated `SoftUnitLoss(LearnableBase)`** (Pebble/custom-loss pattern, lives with
  the builder or in `curlew/geology/`): owns the coupling — evaluates the event fields
  directly (training only), applies a clean re-derivation of the soft carve, and holds
  whatever iso parametrisation is chosen. Predict stays untouched; the learned iso-values
  are pushed into the ordinary `addIsosurface(value=...)` after fit.
- **C. Hybrids / phased training**: warm-start with the current per-field recipe (cheap,
  robust), then switch on the coupling; tau schedule; balanced sampling (already have
  capped per-level pools in `M.level_points`).

**The crux to resolve: where do iso-values live during training?** The soft carve needs
them in the autograd graph. Options to weigh: (a) learnable parameters owned by the
coupling object (NOT scattered through core — the 2026-06-10 refactor removed per-field
learnable isos for invasiveness; do not reintroduce that shape); (b) periodic re-estimation
(EM-style, from the current `estimate_isosurfaces`); (c) fixed after a warm-start. Include
the iso-gap barrier question (GeoINR needed it).

## Constraints carried over from the existing work

- Core (`core.py`, `fields/__init__.py`, `geoevent.py`, `geomodel.py`) stays clean — the
  GeoINR-specific machinery lives in `curlew/fields/geoinr.py` (the `GeoINR` family with
  its gradient-normalised losses) and `curlew/geology/stratbuilder.py`. Extending
  `custom_loss`-adjacent hooks in core is acceptable if genuinely generic.
- `Pebble.__add__` raises on duplicate (group, name) keys — a coupling loss must not
  re-emit per-field groups.
- All-data fitting (no holdout); `scale≈1` fields; magnitude (non-normalised) overturn.
- Validation = wcsb: marker fit (band/basement accuracy vs the GeoINR_Research2 baseline),
  0-inversion stratigraphy, the notebook's structural checks, plus multilayer_fold
  regression (must stay passing) and the pytest suite.

## Process for the scoping session

1. Read the memory (`wcsb-geoinr-loss-port`) + this doc + `SPEC.md` §8.1.
2. Study the reference implementation (files above) and curlew's soft predict path
   (`Overprint.apply` soft weights, `Geode.softLithoID`, `GeoModel.fit(custom_loss=)`).
3. Produce `SOFTUNIT_SPEC.md`: chosen architecture with rationale, rejected alternatives
   (briefly), training procedure (phases, sampling, tau/weights), iso-value treatment,
   exact file/function inventory, validation plan with numeric targets.
4. Use plan mode; surface the genuine decision points to the owner (AskUserQuestion) —
   especially the iso-value treatment and avenue A vs B — before finalising.
5. **No implementation until the spec is approved.**
