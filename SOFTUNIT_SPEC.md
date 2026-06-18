# Soft-unit / NLL coupling — design spec

**Status: DESIGN APPROVED 2026-06-15 (plan mode). No implementation yet.** This spec is the
deliverable of the scoping session handed off by `SOFTUNIT_PORT.md`. Read that doc + the
`wcsb-geoinr-loss-port` memory + `SPEC.md` §8.1 first. Implementation is a separate, spec-gated
session.

Reference studied (for *behaviour*, not structure — redesign is explicit):
`D:/Development/GeoINR_Research2/geoinr/model/geological_model.py` (`GeoINRSoftUnitModel`) and
`.../domain_logic.py` (`soft_carve_unit_probs`, `hard_carve_chunk`, the iso-gap barrier).

---

## 1. Problem & goal

The unit-only / wcsb path of `curlew.geology.stratbuilder.build_geomodel` fits each event's field
independently with **hinge inequalities** (`CSet.iq` → `GeoINR.iq_norm_loss`) plus a **no-overturn**
prior. Stratigraphic ordering is essentially solved (≈0.1 % column inversions), but per-point marker
fits are loose (band acc ≈0.61, basement ≈0.50). Validated cause (2026-06-12): a hinge gives **zero
gradient once satisfied**, nothing calibrates a field's absolute value laterally, and a single
post-hoc global iso lands in the middle of the satisfied slack.

The fix — and the reason GeoINR is being ported into curlew — is GeoINR's **soft-unit / NLL coupling
across all scalar fields**: convert the field stack + shared iso-values into per-point **class
probabilities** and train them with **cross-entropy against the true unit labels**. Every unit point
then carries a two-sided, always-active likelihood; field values are calibrated everywhere; and unit
classification is globally consistent (addressing the sequential-combine fragility at depth).

**Hard requirements (from the owner, carried from `SOFTUNIT_PORT.md`):**
1. Extremely well designed — minimal, easy to follow, robust across geological settings, not
   over-engineered. Quality is the point.
2. Couples **all** scalar fields (and the surface iso-values) in one likelihood.
3. Do **not** break or bypass `GeoModel.predict` — its geological machinery (retro-deformation,
   faults, intrusions, overprints, domain boundaries) is the reason for the port.
4. Core stays clean (`core.py`, `fields/__init__.py`, `geoevent.py`, `geomodel.py`,
   `interactions.py` untouched). New machinery lives in `curlew/geology/`.

---

## 2. Chosen architecture — Avenue B: a dedicated `SoftUnitLoss`

A coupling **loss object** passed to the existing `GeoModel.fit(custom_loss=[...])` hook. It produces
a genuine `(N, C)` per-class probability simplex from the field stack + its own learnable
iso-values, and computes NLL against true unit levels. `GeoModel.predict` is **untouched**.

### 2.1 Why B, not A (reuse `softLithoID`)

`GeoModel.predict(..., to_numpy=False)` does produce a differentiable `softLithoID`, but it is **a
single scalar that linearly blends integer class IDs** — `w·i + (1−w)·prev` within an event
(`geoevent.py:556-558`) and `younger·w + parent·(1−w)` across events (`core.py:754-759`). Two
consequences make it unfit as the coupling signal:

- **It is not a probability simplex.** Cross-entropy needs per-class probabilities; a scalar
  expectation of class IDs cannot express "this point is class L and *not* the in-between classes."
- **The boundary-shell artifact** the owner observed in ParaView/napari: at any contact between two
  regions whose IDs are far apart (e.g. parent ID 5, younger ID 20) the scalar blend sweeps through
  every intermediate integer 6…19, so a categorical colormap paints thin shells of *other* units at
  every boundary — "repeated through the volume, inconsistent with the stratigraphic sequence."

Two further blockers confirm A is unworkable mid-training:

- **`predict(combine=True)` cannot run before isos exist.** `Overprint.updateThresh`
  (`interactions.py:178-199`) resolves the threshold from the iso **name** and asserts it is present;
  on the unit-only path isos are set only *post-fit*. So the combined soft predict is uncallable
  during training.
- **Iso-values are not in the autograd graph through predict.** `getIsovalues`
  (`geoevent.py:925-995`) returns Python floats and `.detach()`es seed evaluations. Learnable
  iso-values must therefore be owned by the coupling object, not by the events' `isosurfaces` dict.

**Rejected A′ (extend `Geode`/`combine`/`predict` to emit `(N, C)` soft probs in core):** would also
cure the `softLithoID` artifact, but it is invasive to core and contradicts the 2026-06-10
"core stays clean" refactor. Noted only as a possible future *generic* improvement.

### 2.2 Why predict's geology is preserved (requirement 3)

The soft carve **legitimately replaces the sequential overprint-combine at training time** with a
global per-point carve — that *is* the fix (GeoINR's global classification vs curlew's fragile
sequential combine). It still drives the **same fields** under the **same retro-deformation**, because
the field stack is obtained through the deformation machinery:

```python
g = ev.predict(x, combine=False, transform=True, to_numpy=False,
               litho=False, isosurfaces=False, props=False)   # → g.scalar = f_ev(x)
```

`combine=False` evaluates only `ev`'s field but `transform=True` still walks the `child.undeform`
chain (`geoevent.py:664-717`), so faults/intrusions retro-deform for free. It needs **no** iso-values
and runs **no** overprint — sidestepping both blockers above. `GeoModel.predict` itself is unchanged
and remains the inference path; after training, the learned iso-values are written into the ordinary
`addIsosurface(value=...)` and predict reproduces the model.

### 2.3 Classify in LEVEL space (consistency)

Classes are **unit levels** (the data's own labels), not `lithoID`. `lithoID` runs opposite to `level`
(the historical ParaView confusion: ascending `lithoID` = younger, ascending `level` = older). By
training in level space the loss never depends on `lithoID` numbering, so it cannot reproduce the
`softLithoID` inconsistency. Post-fit, predict's `lithoID → level` (via `M.level_litho` / `llookup`)
maps back consistently; §7 cross-checks the two assemblies agree.

---

## 3. The soft carve — a per-class re-derivation of `Geode.combine`

The carve is the per-class version of curlew's own sequential combine: the **same** Overprint
sigmoid/threshold form, accumulating an `(N, C)` simplex instead of a scalar. Scoped to the builder's
linear `strati` chain (region-only ↔ erosional ↔ depositional, with `mode='above'` truncation/onlap).
Geology is read from the existing `_FieldMeta`; nothing is re-encoded.

Notation: events oldest→youngest `E_0…E_K`; `f_E(x)` the deformation-correct field value; sharpness
`s = 1/τ` (`τ` = soft-unit temperature, default 0.05 ⇒ `s = 20`, matching GeoINR; ≈ `lithoSharpness`).

### 3.1 Across-event ownership (youngest-first stick-break = the combine, unrolled)

`Geode.combine` applies the youngest event's overprint outermost: `out = w·child + (1−w)·older`.
Unrolling youngest→oldest gives event-ownership probabilities:

```
remaining = ones(N)
for E from youngest to oldest:
    w_E = overprint_weight(E)          # the SAME weight Overprint.apply uses
    claim_E = remaining * w_E
    remaining = remaining * (1 - w_E)
```

`overprint_weight(E)` reproduces `Overprint._soft_weight` exactly (`interactions.py:114-128`):

| event kind (builder)                         | domain          | threshold   | `w_E`                         |
|----------------------------------------------|-----------------|-------------|-------------------------------|
| erosional unconformity (`mode='above'`, own iso) | `f_E` (child)   | `θ_E`       | `σ(s·(f_E − θ_E))`            |
| depositional/region onlapping unconformity `U` | `f_U` (parent)  | `θ_U`       | `σ(s·(f_U − θ_U))`            |
| oldest region-only basement (`base=-inf`)    | —               | —           | `1` (claims all remaining)   |

The erosional event owns **no** class band (its `_FieldMeta.level_litho` is empty); its role is the
`(1 − w_E)` factor that **truncates older events** above the unconformity (without it the basement,
`w=1`, would leak through). Its claimed mass is ≈0 anyway because the younger onlapping package — which
uses the *same* `σ(s·(f_E − θ_E))` and is reached first in youngest-first order — already consumed the
above-unconformity region.

### 3.2 Within-event band probabilities (consecutive sigmoid differences)

A depositional package with bands `b_0` (youngest/top) … `b_m` (oldest/base) separated by contacts at
iso values `θ_1 > θ_2 > … > θ_m` (descending — guaranteed by the monotone reparam, §5):

```
P(b_0) = σ(s·(f_E − θ_1))                          # above top contact
P(b_j) = σ(s·(f_E − θ_{j+1})) − σ(s·(f_E − θ_j))   # 0 < j < m
P(b_m) = 1 − σ(s·(f_E − θ_m))                       # below bottom contact
```

These telescope to 1 and are **non-negative because the reparam keeps `θ` descending** (no clamping
needed). Region-only events have one band ⇒ `P = 1` for their single class. Each band maps to one
level via `_FieldMeta.level_litho`.

### 3.3 Assemble the simplex

```
probs = zeros(N, C)
remaining = ones(N)
for E from youngest to oldest:
    w_E = overprint_weight(E); claim = remaining * w_E
    for (level, p_band) in within_event_band_probs(E):   # empty for erosional
        probs[:, class_index[level]] += claim * p_band
    remaining = remaining * (1 - w_E)
# basement is the oldest event with w=1 ⇒ remaining fully consumed; renormalise defensively
return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
```

Cost note: the carve needs field **values** only (no `∇f`), so it is *cheaper* per epoch than the
current `iq` recipe (which needed `∇f` for normalisation). Only the retained `overturn` term needs
`∇f`. ≈15 events × balanced-sample forward passes ≈ the existing budget (~0.3–0.9 s/epoch GPU).

---

## 4. Training procedure — joint from scratch

One `M.fit(custom_loss=[soft_unit_loss])`. No warm-start, no phasing (owner's call: phasing adds
complexity, and with CE the per-field inequalities are redundant).

- **Per-field losses:** set `iq_norm_weight = 0` (CE supersedes the inequality ordering, mirroring
  GeoINR's `include_unit_constraints=False`); **keep `overturn_weight`** (a geology-agnostic
  monotonicity/polarity prior — also what makes the field younging-consistent so the optional
  trend-init in §5 is meaningful). `HSet` stays all-zero, as today.
- **Per epoch (inside `SoftUnitLoss.__call__`):**
  1. `sample_balanced_unit_indices` — `points_per_level` (default 1024) drawn per level from
     `M.level_points` (already capped), concatenated and shuffled. Balanced sampling stops large units
     dominating the CE.
  2. evaluate the field stack at the sampled coords via per-event `combine=False` predict (§2.2);
  3. `soft_unit_probs` → `(N, C)` (§3); `log` → `F.nll_loss(log_probs, labels)`;
  4. push the term into the returned `Pebble` under the loss's own group, carrying the **field
     optimisers** (so the fields co-train) **and** the **iso-value optimiser** (iso_lr ≈ 10× field lr,
     per GeoINR). `Pebble.__add__` merges these with the per-event `overturn` terms in `M.fit`.
- **Temperature `τ`:** fixed, exposed (default 0.05). No annealing in v1 (minimal); a schedule is a
  later tuning knob if convergence needs it.
- **Optimisers:** the loss owns one optimiser for its iso parameters; field optimisers already exist
  on the fields. `pebble.step()` in `M.fit` steps all.
- **Post-fit:** `soft_unit_loss.write_isosurfaces(M)` pushes the learned iso-values into
  `addIsosurface(value=...)` **oldest-first** (so ascending `lithoID` = younger, per the existing
  convention), then `M.predict` as today. `estimate_isosurfaces` is **not** used on this path (isos are
  learned, not estimated); it stays for the existing non-coupled per-field path.

---

## 5. Iso-value treatment — learnable, monotone, field-free init

The crux of `SOFTUNIT_PORT.md`. Iso-values are **learnable Parameters owned by `SoftUnitLoss`**
(option (a) — never scattered through core; the 2026-06-10 refactor removed per-field learnable isos
for invasiveness and must not be reintroduced).

**Correctness is owned by two geology-agnostic mechanisms, so the init is only a convergence aid:**

1. **Per-package monotone reparam.** Each depositional package's internal contacts are
   `θ_j = θ_0 − cumsum(softplus(φ_j))` (one `φ` vector per package — the `_iso_mono` trick the earlier
   learnable-iso impl proved "truly fixes iso_monotonic"; overturn alone did not). Contacts therefore
   **cannot cross**, and ordering holds in **value space** (younger band = higher value) regardless of
   spatial geometry — robust to overturned folds, steep dips, thrust repeats, etc. (those deform the
   field's *geometry*, not the value-space order of a conformable package). This also makes the §3.2
   band probabilities valid by construction (no clamping). Unconformity isos are single free values.
   **GeoINR's iso-gap barrier is therefore unnecessary**; an *optional* tiny pinch penalty
   (`Σ softplus(φ)` small) keeps a band from collapsing to zero thickness unless the data demands it.
2. **Always-active CE** with learnable isos continuously refines spacing.

**Initialisation (field-free — replaces the circular "estimate_isosurfaces on an untrained field"):**
the init needs only correct *order* (given by the strat column — assumption-free) + rough spacing.

- **Default (robust, minimal): ordinal / uniform gaps.** No spatial assumption. `θ_0` at the top of
  the field range (~+1 for `scale≈1`), uniform `softplus(φ)` gaps down the range.
- **Optional refinement (data-driven): trend-projection median spacing.** Init each contact from the
  **median younging-axis separation** of the two unit point-clouds it divides, where the younging axis
  is the explicit, configurable `CSet.trend` (the **same** vector `overturn` already uses; default
  `+z`) — *not* a hard-coded `+z`. Thickness-aware where younging ≈ trend (e.g. wcsb), degrades
  gracefully elsewhere (still ordered, still spanning the range).
- **The default is decided empirically** by the wcsb A/B in §7 — data-informed, not asserted.

This keeps the single geological assumption (younging direction) visible, overridable, and
single-sourced; everything load-bearing (ordering, simplex validity) is geometry-independent.

**Rejected:** periodic-EM re-estimation (no iso gradient — loses the calibration we want);
fixed-after-warm-start (boundaries pinned mid-slack — defeats the purpose); free isos + GeoINR gap
barrier (superseded by the structural reparam).

---

## 6. File / function inventory

**New — `curlew/geology/softunit.py`:**
- `soft_unit_probs(model, x, iso_values, tau, carve_spec) -> Tensor[(N, C)]` — §3 carve.
- `SoftUnitLoss(LearnableBase)` — owns `iso_values` (monotone reparam, §5), `tau`, `points_per_level`,
  the level→class map and per-field `carve_spec` (built once from `M.field_meta` / `M.level_litho`),
  and the balanced pools + labels (from `M.level_points`). Implements `__call__(pebble, model, C)`
  (returns a `Pebble`) and `write_isosurfaces(model)`.
- helpers: `build_carve_spec(model)`, `build_class_index(model)`, `sample_balanced_unit_indices(...)`,
  the iso reparam (`iso_values()` property resolving `φ → θ`), `iso_init_*` (uniform / trend-median).

**Edit — `curlew/geology/stratbuilder.py`:**
- `attach_soft_unit_loss(M, *, tau=0.05, points_per_level=1024, iso_init='uniform'|'trend', ...)`
  — builds and returns a `SoftUnitLoss` wired to a built model.
- on the coupled build path set `iq_norm_weight = 0` (keep `overturn_weight`). The existing per-field
  path, `_FieldMeta`, and `estimate_isosurfaces` are **left unchanged**.

**Edit — `curlew/geology/__init__.py`:** export `SoftUnitLoss`, `attach_soft_unit_loss`.

**Untouched:** `core.py`, `fields/__init__.py`, `geoevent.py`, `geomodel.py`, `interactions.py`,
`curlew/fields/geoinr.py` (its loss weights are toggled from the builder, not edited).

**Notebook — `examples/wcsb/_build_notebook.py`:** replace the per-field-only fit with the joint
`SoftUnitLoss` fit; add the soft-carve-vs-predict consistency check (§7) and the iso-init A/B; keep
the `FIELD` (Siren/GeoINR) toggle and the `saveVTK` export with the `level` array.

---

## 7. Validation plan (numeric targets)

Validation = wcsb (the established harness `/tmp/wcsb_sb.py` / the notebook), plus regressions.

- **Marker fit (primary):** band accuracy **> 0.61** (beat the per-field ceiling; target the
  GeoINR_Research2 baseline); basement accuracy **> 0.50 and rising** vs the 0.497 per-field result.
  Report per-level accuracy as the notebook does.
- **Stratigraphy:** column inversions **≤ 0.1 %** (no regression vs the per-field path); contacts
  monotone **by construction** (reparam) — assert.
- **Soft-vs-hard consistency (answers the `softLithoID` concern directly):** on the markers,
  `argmax`(soft carve, in level space) agrees with predict's hard `lithoID → level` to within a small
  tolerance — i.e. the training assembly and the inference assembly agree. This is the direct test of
  the sequential-combine fragility the milestone targets.
- **Iso-init A/B:** uniform-gap vs trend-median init on the full wcsb budget — pick the default from
  the result (data-informed). Report band/basement accuracy and convergence for both.
- **Regression:** `multilayer_fold` notebook still passes (its on-surface path is untouched); full
  `pytest ./tests/` green.
- **New `tests/test_softunit.py`:**
  - `soft_unit_probs` rows sum to 1 (simplex);
  - autograd flows to **both** field params **and** `iso_values` (guards a future `detach`);
  - NLL **decreases** on a toy 2-unit / 2-event model with separable points;
  - monotone reparam keeps a package's contacts strictly ordered for arbitrary `φ`;
  - `write_isosurfaces` round-trips: post-fit predict labels match the carve argmax on the training
    points.

---

## 8. Out of scope / future

- **`softLithoID` viz fix** — document-only this milestone (Avenue B doesn't use it; §2.1 explains the
  cause). Exposing the carve's `(N, C)` probs at inference for cleaner volume colouring, or a per-class
  soft output in core (A′), is a separate, optional follow-up.
- **Faults / intrusions / domain boundaries in the carve** — the *deformation* is already handled by
  the per-event `combine=False` predict; only the per-class *assembly* (§3) is currently
  chain-specific. Generalising the assembly to the full event tree (deformation events, `parent2`
  domain boundaries) is a later extension; the wcsb chain is the validation target.
- **τ schedule, alternative samplers, ensemble/holdout validation** — later tuning, not v1.
