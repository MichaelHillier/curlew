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

### Done (the four original items)
- [x] **Rename `SoftUnitLoss → UnitLoss`** (item 4).
- [x] **§2 erosional print reworded** (item 2): now `"L35,36 above unconformity, L37 eroded below"`.
- [x] **`_AliasField` design** (item 3) → documented in GEOINR_INTEGRATION.md §3 (loss-free alias
  of the adjacent unconformity field; no params, `C=None`, so no loss / no `Pebble` clash).
- [x] **Why event 13 has ~35 `iq` pairs** (item 3) → GEOINR_INTEGRATION.md §4. Your reading is
  correct (Quaternary `>` every older unit ⇒ "all above the unconformity"); note they are *unused
  for the loss under coupling* (`iq_norm_weight=0`).

### Decided not to do (for now)
- **napari volume colour by `level`** (item 1) — owner keeps `lithoID` for now; no change planned.
- **Stop building the vestigial `iq` on the coupled path** — accepted as-is for now.

### Known limitations
Moved to **GEOINR_INTEGRATION.md §7** (the durable reference): vestigial `iq`, the single global
`overturn` knob, the deep-band accuracy ceiling, and the `softLithoID` volume artifact.

### Add your remaining problems / ideas here
- (Phase-2 agenda — drop new items as you find them)
