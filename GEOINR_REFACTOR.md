# GeoINR inclusion — refactor handoff

**Status: EXECUTED (2026-06-10).** The checklist below was completed: core reverted to the
pre-checkpoint state (`git checkout 01170cd` for the four files — all `sb`/learnable-iso
machinery removed), the GeoINR losses now live in `_GeoINRLossMixin` (`curlew/fields/geoinr.py`,
constructor kwargs `iq_norm_weight`/`overturn_weight`), the unit-only builder uses `CSet.iq` +
post-hoc `estimate_isosurfaces(M)` (`curlew/geology/stratbuilder.py`), and the wcsb notebook was
regenerated. Full test suite passes; a CPU smoke run (cap 150, 15 epochs) verified build → fit →
estimate → predict end-to-end. Kept for the findings below; the deferred soft-unit/NLL milestone
still stands.

**Original plan:** the GeoINR stratigraphic machinery currently works but is **too invasive** to curlew
core. This note is the plan to re-do it **minimally**, isolating GeoINR to `geoinr.py` +
`stratbuilder.py`. Written for a cold-start session — read this + the memory
`wcsb-geoinr-loss-port`, then execute the checklist. Reference: `SPEC.md` (§8.1 Findings).

**Goal right now:** a clean, minimal, well-documented inclusion of the GeoINR *building blocks*.
**NOT** matching the GeoINR_Research2 baseline accuracy — that needs the **soft-unit/NLL coupling
across scalar fields**, which is the explicit, deferred next milestone (do not attempt yet).

---

## Architecture decision (chosen): `iq` + post-hoc iso-values

1. **Stratigraphic constraints = `iq` (point-vs-point), not `sb`.** `sb` is just `iq` with the RHS
   being an iso-value; drop it. Reuse the existing `CSet.iq` (unit-A points `>` unit-B points for
   adjacent units; for an unconformity, onlapping-package points `>` eroded-package points).
2. **GeoINR loss lives in `geoinr.py`, not core.** `GeoINR`/`Siren` **override `loss()`** to add:
   - a **normalized inequality** term `(f(P₁) − f(P₂)) / ‖∇f(P₁)‖` (GeoINR style), and
   - the **no-overturn** term (magnitude — see findings).
   `BaseSF.loss` reverts to generic (no `sb_loss`/`overturn_loss`).
3. **No learnable iso-values.** Remove all the iso machinery from core. After training, **estimate
   each surface post-hoc** (the value that best separates the two units' field-value distributions
   — e.g. midpoint of medians; for an unconformity, between the eroded vs onlapping medians) and set
   it with the **existing** `addIsosurface(value=…)`. Tradeoff (known/accepted): surfaces are
   estimates, unconformities hardest — but it's a tractable 1-D estimate per surface and keeps core
   untouched.

Net: GeoINR ends up only in `geoinr.py` (loss + maybe a small constraint helper) and
`stratbuilder.py` (build with `iq`; estimate + set isosurfaces after fit). `napari_viewer.py`'s
`vertical_exaggeration` is a **general** viewer feature — **keep it** (not GeoINR-specific).

---

## Findings to preserve (validated this session — keep the learnings, redo the code)

- **`scale ≈ 1`** for the unit-only fields (`_FIELD_SCALE`): at `scale=1e2` an iso of ±0.78 is ≈0
  relative to a ±100 field (meaningless) and Adam can't move things; ~1 makes value/iso placement
  meaningful. (With no learnable isos this matters less, but keep the field range ~O(1).)
- **No-overturn uses the gradient *MAGNITUDE* — do NOT normalize by ‖∇f‖.** I briefly normalized it
  to a cosine; that was a bug (made fields non-monotone → crossing contacts + weird §5 geometry).
  The norm is what corrects field polarity. Form: `clamp_max((∇f·younging), 0).abs().mean()`.
- **An eroded unit's top is the unconformity, NOT a depositional contact.** Don't create a contact
  "named after" an eroded unit (the "Pennsylvanian" example). Curlew labels the band *above* an
  isosurface, so a contact is the **top of the older unit below it**.
- **§5 "turn-up in the west" is not a bug** — extrapolation beyond data + the real westward
  basin-margin rise (Penn 91% west; L20 anomalously 83% east, which is what collapsed its thin band).
- **Basement (deepest, region-only) is the hard ceiling:** ~0.13 at `scale=1` (was ~0.29 at
  `scale=1e2`); the deep unconstrained region is truncated by younger unconformities in the
  sequential combine. **Only soft-unit/NLL coupling fixes this** — not a bigger overturn weight.
- Per-field accuracy now (learnable-iso impl, weight 30): band ~0.49, structure ~0.76, basement ~0.13.

---

## What to KEEP vs REVERT

**KEEP** (general / not GeoINR-specific, validated):
- `curlew/visualise/napari_viewer.py` — `vertical_exaggeration` (+ `_apply_ve_volume`, the
  `_to_napari_xyz`/`addMesh` z-scaling). General viewer feature.
- `examples/wcsb/` notebook + `_build_notebook.py` structure (obs-points overlay, 5km/20m grid,
  `batchSize=50000`, §7 asserting only structural invariants). Re-point constraint/iso bits to the
  new approach.
- `stratbuilder.py` partitioning (`derive_scalar_fields`, the chain build) + `scale=1` + magnitude
  overturn weight — but constraints become `iq`, isosurfaces become post-hoc fixed values.

**REVERT** (invasive core — undo my changes):
- `core.py` — `CSet.sb`, `HSet.sb_loss`, `HSet.overturn_loss`.
- `fields/__init__.py` (`BaseSF`) — `iso_values`/`_iso_mono`/`add_iso`/`add_ordered_isos`/`get_iso`,
  the iso group in `init_optim`, and the `sb_loss` + `overturn_loss` blocks in `loss()`.
- `geoevent.py` — `iso_litho`, `addOrderedIsosurfaces`, the `learnable=` branch of `addIsosurface`,
  the `iso_litho` use in `predict` lithoID.
- `geomodel.py` — the `iso_litho` use in the `predict` llookup.

(Quickest path: `git diff <last-good-commit> -- curlew/core.py curlew/fields/__init__.py
curlew/geology/geoevent.py curlew/geology/geomodel.py` to see exactly what to undo. The pre-session
baseline already had `sb`/learnable-isos from the *prior* session, so reverting goes back further
than this session — revert the whole `sb`/learnable-iso mechanism, not just my edits.)

---

## Refactor checklist (ordered)

1. Revert the four core files above to remove `sb` + learnable-iso + in-core GeoINR losses.
2. `geoinr.py`: give `GeoINR`/`Siren` a `loss()` that calls `super().loss()` and adds
   (a) normalized-iq `(f(P₁)−f(P₂))/‖∇f(P₁)‖` over `C.iq` pairs (hinge by relation, active-only),
   (b) magnitude no-overturn over `C.grid` along `C.trend`. Gate both with `HSet` weights that live
   on the GeoINR side (or reuse `iq_loss` + a small GeoINR-only weight) — keep `HSet` core clean.
3. `stratbuilder.py`: build each event with `CSet.iq` (point-vs-point ordering); keep `scale=1`,
   the no-overturn grid, and the partitioning. After `M.fit`, **estimate** each contact/unconformity
   iso (separation value between adjacent unit field-value distributions) and `addIsosurface(value=)`.
4. `examples/wcsb/_build_notebook.py`: re-point §3/§5/§6/§7 to the new names/values; keep napari + the
   obs-points overlay; regenerate `wcsb.ipynb`.
5. Re-validate: all `tests/` pass; wcsb notebook runs; band/structure sane (basement will stay low —
   expected, that's the soft-unit milestone).

## Open notes
- Post-hoc iso estimation method is the main thing to get right (esp. unconformities). Start with
  midpoint-of-medians; refine only if extraction is visibly off.
- Overturn weight ~30 was a reasonable balance with the magnitude term at `scale=1`; re-tune after.
- Deferred milestone (separate, larger): **soft-unit / NLL coupling across scalar fields** — the
  path to GeoINR_Research2-level results.
7