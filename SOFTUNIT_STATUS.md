# SOFTUNIT — implementation status & open items

Status of the unit / NLL coupling on the `geoinr` branch. Design: [SOFTUNIT_SPEC.md](SOFTUNIT_SPEC.md).
How it works: [GEOINR_INTEGRATION.md](GEOINR_INTEGRATION.md).

## Status — validated baseline (ready to commit)

The coupling is implemented, debugged against the GeoINR reference, and green on tests.

- **Code:** `curlew/geology/softunit.py` (`UnitLoss`, `soft_unit_probs`, `build_carve_spec`,
  `monotone_contacts`, `write_isosurfaces`) + `attach_unit_loss` in `stratbuilder.py` (additive).
  **Core untouched** (`core.py`, `fields/__init__.py`, `geoevent.py`, `geomodel.py`,
  `interactions.py`, `fields/geoinr.py`).
- **Tests:** `tests/test_softunit.py` (6) + `tests/test_stratbuilder.py` (2) pass; full
  `pytest ./tests/` = 78 passed / 2 skipped.
- **wcsb numbers** (Siren, `overturn=6`, `cap_per_unit=20000`, uniform iso-init, 3000 epochs):

  | metric | this baseline | GeoINR reference |
  |---|---|---|
  | NLL / loss | **0.815** | 0.78 |
  | band (exact unit) acc | 0.730 | — |
  | structure (package) acc | 0.920 | — |
  | basement (L37) acc | 0.578 | 0.536 |
  | soft-carve ↔ predict | 0.943 | — |
  | column inversions (coarse grid) | 0.417 % | — |

  Reference-comparable. The residual error is the deep, densely-packed packages (Granite
  Wash / Mississippian internals) — genuinely hard, and the weakest levels in the reference too.

**Reproduce:** `python examples/wcsb/_softunit_harness.py 3000 20000 Siren 6`
(args: `epochs cap field overturn`; needs the GPU + `GeoINR_Research2/.../wcsb` data).

### The three fixes that established this baseline (2026-06-16)
1. **Structural carve bug** — erosional events were carve steps that *discarded* mass and
   double-truncated, creating a degenerate CE minimum (worse with training). Fixed: only
   class-owning events are steps; each surface truncates once (GEOINR_INTEGRATION.md §5).
2. **`overturn=30` strangled the coupled fit** (over-smoothed; NLL stuck ~1.06). Relaxed to
   6 via `attach_unit_loss(overturn_weight=...)` → NLL 0.82. (Trade-off: vs predict inversions.)
3. **`cap_per_unit=1500` starved the fields.** On the coupled path use a large cap (balance
   comes from `points_per_level`).

### Simplifications applied this pass
- Removed the **`trend` iso-init** (+ `_trend_unit`, `_level_trend_median`, the §7 A/B cell):
  uniform won decisively and the reparam+CE make init non-load-bearing.
- Removed the **`iso_pinch_weight` / gap barrier**: the monotone reparam makes it unnecessary.
- Renamed **`SoftUnitLoss → UnitLoss`**, **`attach_soft_unit_loss → attach_unit_loss`**, loss
  group `"soft_unit" → "unit"`. (Kept `soft_unit_probs` — it is the *soft* carve — and the
  module file `softunit.py`.)

---

## Open items

### Done this session
- [x] Rename `SoftUnitLoss → UnitLoss` (your item 4).

### Improvements (Phase 2 — notebook / UX)
- [ ] **napari volume should colour by `level`, not `lithoID`** (your item 1). `lithoID`
  numbering (with gaps for surface-only events) makes units *look* out-of-sequence; `level` is
  the data's own convention (ascending = older) and matches the point observations + surfaces.
  The `level` array is already computed for the VTK export — reuse it for `addVolume`, the
  points, and the surface colouring so all three are consistent.
- [ ] **§2 erosional print is confusing** (your item 2). `"iso between L35..36 / L37"` means
  *the unconformity sits between the onlapping package (L35,36, above) and the eroded unit
  (L37, below)*. Reword to e.g. `"L35,36 above unconformity ; L37 eroded below"`. (The
  child-vs-parent diagram you asked for now lives in GEOINR_INTEGRATION.md §2 — optionally
  embed a trimmed version in the notebook.)

### Questions — answered, now documented
- [x] **`_AliasField` design** (your item 3) → GEOINR_INTEGRATION.md §3 (loss-free alias of the
  adjacent unconformity field; no params, `C=None`, so no loss / no `Pebble` clash).
- [x] **Why event 13 has ~35 `iq` pairs** (your item 3) → GEOINR_INTEGRATION.md §4. Your reading
  is correct: Quaternary `>` every older unit ⇒ "all above the unconformity". **But note** they
  are *unused for the loss under coupling* (`iq_norm_weight=0`).

### Known limitations / candidate future work
- [ ] **Vestigial `iq` on the coupled path.** Built but unused (only sizes pools + §3 listing).
  Could skip building them when coupling is intended — a `build_geomodel` refactor (the build
  step doesn't currently know coupling is coming; `attach_unit_loss` is separate).
- [ ] **`overturn` is one global knob** trading marker-fit vs predict-inversions. A split by
  event kind (strong on unconformities, weak on packages) was tried and **did not** recover the
  basement (the leak is iso placement, not unconformity-field overturn), so it was dropped.
  A smarter, data-aware monotonicity prior is open.
- [ ] **Band-accuracy ceiling ~0.73** on the full markers (reference-comparable). Dominated by
  the deep packed bands; one field per package is the same limit GeoINR has.
- [ ] **`softLithoID` volume artifact** — out of scope (SOFTUNIT_SPEC §8); the carve's `(N,C)`
  probs could be exposed at inference for cleaner volume colouring later.

### Add your remaining problems / ideas here
- (drop new items as you find them — this is the Phase-2 agenda)
