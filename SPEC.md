# SPEC — Integrating GeoINR INRs and strat-column model building into Curlew

Self-contained specification for porting selected pieces of `GeoINR_Research2`
(`D:\Development\GeoINR_Research2`) into `curlew` (`D:\Development\curlew`).

This is a *first integration pass*. It deliberately ports the minimum needed to
reproduce two reference datasets end-to-end inside Curlew, reusing Curlew's
existing machinery (events, constraints, transforms, isosurfaces) rather than
GeoINR's bespoke training/model stack.

---

## 1. Goal

1. Bring two of the GeoINR neural fields — the plain MLP (`INR`) and the
   sine-activation network (`Siren`) — into Curlew as first-class scalar fields.
2. Provide an automated builder that turns a stratigraphic-column CSV
   (e.g. [strat_col_v2.csv](../GeoINR_Research2/geoinr/data/wcsb/strat_col_v2.csv))
   plus point observations into a ready-to-fit `curlew.geology.geomodel.GeoModel`.
3. Verify both end-to-end against `multilayer_fold` (simple, single field) and
   `wcsb` (complex, many fields).

The deep motivation: GeoINR's strat-column → constraint pipeline
([point_constraints.py](../GeoINR_Research2/geoinr/input/point_constraints.py))
is convoluted. Curlew already has the right primitives (graph of events,
`CSet` constraints, seed isosurfaces). We move the *partitioning logic* over in
its clean form and discard the rest.

---

## 2. Locked decisions (from requirements interview)

| Topic | Decision |
|---|---|
| MVP scope | **Fields + strat-column→GeoModel builder.** wcsb's many fields are constructed automatically. |
| Field packaging | **Two classes** — a plain-MLP field and a Siren field, each a `curlew.fields.BaseNF`. |
| Above/below strategy | **Pairwise inequalities + seed isosurfaces.** Any contact with on-surface points (depositional *or* erosional) gets its iso from a seed. A **learnable iso-value** is a *fallback* used **only** when an unconformity has no on-surface points (the case for all wcsb unconformities) — it then serves as both the `Overprint` threshold and the extraction iso. Not engaged when seeds exist. |
| Minimalism | Highest priority: the implementation must stay **minimal and readable**. Prefer Curlew-native primitives over new machinery; only add the learnable-iso path because wcsb genuinely lacks on-surface unconformity points. |
| Unconformity representation | Each erosional surface is its **own** `strati(mode='above')` GeoEvent with its own neural field (GeoINR-style), sitting between its older and younger packages. Above/below inequalities constrain *this* field. |
| Single-unit basement | A depositional package with one unit and no interfaces (e.g. Precambrian) is a **region-only** `strati` event: a neural field with no `eq`/seed/gradient constraints. Its points serve only as below-references for the overlying unconformity. |
| Normalization | **Curlew `Transform`.** Isometric and anisotropic (vertical-exaggeration) scaling expressed as a `Transform` matrix on the `GeoModel`. No scaler classes ported. |
| `multilayer_fold` variant | **3D** (`interface.vtp` + `normal.vtp`). |
| Constraint mapping | Interface points → **equality trace** (`CSet.eq`) **and** **seed** for `addIsosurface`; normals → **gradient** (`CSet.gp`/`gv`, polarity-enforcing); unit points → **pairwise inequalities** (`CSet.iq`). |
| Builder output | **Ready `GeoModel`** with events, CSets, isosurfaces, and the normalization `Transform` attached. |
| wcsb tractability | Load the **full** markers file, **cap points per unit**, and **re-sample a different random subset each training step**. Sampling must be **imbalance-aware**: every unit (including rare ones) must stay represented. |

---

## 3. Source → target reference

### GeoINR source (read-only reference)
- INRs: [geoinr/model/inr.py](../GeoINR_Research2/geoinr/model/inr.py) — classes `INR`, `Siren`; layers in [geoinr/model/mlp/layers.py](../GeoINR_Research2/geoinr/model/mlp/layers.py) (`Perceptron`, `SineLayer`).
- Clean strat-column **partitioning only**: [geoinr/input/stratigraphy.py](../GeoINR_Research2/geoinr/input/stratigraphy.py) — reuse `load_strat_column_csv`, `derive_scalar_fields`, and (optionally) `derive_horizons` for naming. **Do _not_ reuse `derive_relational_constraints`**: it re-implements the convoluted cross-field above/below logic from `point_constraints.py`. The relational (above/below/equality) constraints are generated from the bespoke rules written out in §4.2 "Constraints per event", not from this function.
- Convoluted legacy pipeline (do **not** port): [geoinr/input/point_constraints.py](../GeoINR_Research2/geoinr/input/point_constraints.py). Read only to confirm relation semantics.
- Data loading / formats: [geoinr/input/readers.py](../GeoINR_Research2/geoinr/input/readers.py); scalers (reference only): [geoinr/input/scalers.py](../GeoINR_Research2/geoinr/input/scalers.py).

### Curlew target (where things land)
- Field classes: [curlew/fields/geoinr.py](curlew/fields/geoinr.py) — currently a `GeoINR(BaseNF)` stub; extend here. Field base/contract: [curlew/fields/__init__.py](curlew/fields/__init__.py) (`BaseNF`, `initField`, `evaluate`, `loss`, `fit`).
- Builder: new module, `curlew/geology/stratbuilder.py` (proposed), exposing `build_geomodel(...)`.
- Events / model: [curlew/geology/__init__.py](curlew/geology/__init__.py) (`strati`), [curlew/geology/geomodel.py](curlew/geology/geomodel.py) (`GeoModel`).
- Constraints: [curlew/core.py](curlew/core.py) (`CSet`: `gp`/`gv`, `eq`, `iq`).
- Isosurfaces / seeds: [curlew/geology/geoevent.py](curlew/geology/geoevent.py) (`addIsosurface(seed=...)`, `getIsovalues`).
- Coordinate transform: [curlew/geometry.py](curlew/geometry.py) (`Transform`).

---

## 4. Minimum feature

### 4.1 Field classes — `curlew/fields/geoinr.py`

Two `BaseNF` subclasses. Each must implement `initField(**kwargs)` (build the
`nn.Module`, push to `curlew.device`/`curlew.dtype`, call `self.init_optim(...)`)
and `evaluate(self, x) -> (N, output_dim)`. Loss/fit inherit from `BaseNF`
unchanged (Curlew computes the constraint loss from the bound `CSet`).

- **`GeoINR`** (plain MLP, ports `INR`):
  - `initField(hidden_dim=256, num_hidden_layers=3, activation=nn.Softplus(beta=20), learning_rate=1e-3)`.
  - Linear stack `input_dim → hidden_dim → … → output_dim`; activation between hidden layers; last layer linear. Kaiming-uniform init (`fan_in`, ReLU gain), matching `Perceptron`.
  - Replaces the existing stub. Keep the docstring's Hillier et al. 2023 reference.
- **`Siren`** (ports `Siren`):
  - `initField(hidden_dim=256, num_hidden_layers=3, omega0=2.0, omega=30.0, learning_rate=1e-4)`.
  - `SineLayer`-style: first layer `is_first` weight init `U(-1/in, 1/in)`; subsequent `U(-√(6/in)/omega, √(6/in)/omega)`; final linear with matching bounded init. `forward = sin(omega0 * linear(x))` per layer.

Constraints / out of scope for this pass: `lipschitz`/spectral-norm, RFF
encoding, `RFFMLP`, `SupportResidualSiren`, progressive controllers. None are
implemented and **no dedicated parameters are added for them** — in particular
there is no `lipschitz` argument on either field. Unknown kwargs are absorbed by
`initField(**kwargs)` and ignored.

**Contract check:** must construct via Curlew factories, e.g.
`strati('s', C=GeoINR('f', input_dim=3, H=HSet(...)))` and
`strati('s', C=Siren('f', input_dim=3, H=HSet(...)))`, and train through
`GeoModel.fit`. All fields require `H` positionally per `BaseNF` (consistent with
`NFF`/`FSF`); do not make `H` optional.

### 4.2 Strat-column → GeoModel builder — `curlew/geology/stratbuilder.py`

Reuse **only the partitioning** from GeoINR's `stratigraphy.py` (port or vendor
`load_strat_column_csv` + `derive_scalar_fields`, plus `derive_horizons` for
contact naming; these are clean and dependency-light — `csv` + dataclasses only).
**Do not** reuse `derive_relational_constraints` — the above/below/equality
relations come from the bespoke rules in "Constraints per event" below, which use
points from younger/older *events* exactly as specified there and nowhere else.

Public entry point (returns a ready model):

```
build_geomodel(strat_col_csv, observations, *, field="GeoINR",
               scale="isometric", transform=None, cap_per_unit=None,
               field_kwargs=None) -> GeoModel
```

**Partitioning** (`derive_scalar_fields`): each row is a unit with a `relation`
(`conformal` | `baselap` | `eroded`) and a monotonic `level`. The parser yields
three kinds of scalar field — and the builder must handle all three plus their
edge cases, because the kind determines both the Curlew event wiring and the
constraint set (§ "Constraints per event"):

1. **depositional (one or more contacts/interfaces)** — a conformal/baselap package of two or more
   units; carries one internal contact (interface) per unit boundary.
2. **depositional (region-only)** — a package with a single unit and **no
   internal interfaces**. This is the basement edge case (see below): there is no
   surface to interpolate, so the field carries no contact/normal constraints.
3. **erosional** — a single unconformity surface. In wcsb these have **no
   on-surface observation points** (only unit markers exist), so the surface
   geometry is learned indirectly from inequalities (see below).

**Edge case — oldest event is a single-unit basement eroded from above.** In
wcsb the oldest unit (Precambrian) is a depositional basement with no internal
interfaces, immediately overlain by an erosional unconformity that erodes it.
The partitioner must emit, oldest-first: a *region-only depositional* field
(Precambrian) followed by an *erosional* field (the unconformity). The basement
field has a single unit and zero interfaces; it must still become a valid Curlew
event (region-only, see §2) so the basement lithology can be labelled and its
points can act as below-references for the unconformity.

**Mapping to Curlew events.** Build the event list **oldest → youngest**
(`GeoModel` expects oldest-first; GeoINR strat rows are youngest-first, so
reverse). One neural field (`GeoINR` or `Siren`) per scalar field. Each kind maps
to a `strati` event:
- depositional (one or more contacts) → `strati(name, C=<field>)`; `onlap=True` when the
  package base is a `baselap` contact (older/parent field defines the basal
  unconformity geometry → `defaultDomain='parent'`).
- depositional (region-only basement) → `strati(name, C=<field>)` with no
  isosurfaces and a CSet that carries no `eq`/seed/`gv` (region-only).
- erosional → its **own** `strati(name, C=<field>, mode='above', base=<iso>)`
  event, placed in the column between its older and younger packages. If the
  unconformity has on-surface points, `base` is a **seed** isosurface like any
  contact. Only when it has none (every wcsb unconformity) does `base` fall back
  to a **learnable iso-value** owned by the event, serving as both the `Overprint`
  threshold and the extraction iso. The younger/older inequalities (below) pull
  the field above/below it.

This means an unconformity between package A (older) and package B (younger)
produces **three** events in Curlew: `A → unconformity → B`, not a single base
threshold on B. (Curlew's idiomatic single-field `base`-threshold pattern, as in
the `hutton` synthetic, is intentionally *not* used here — see open item in §8.)

**Constraints per event** (`CSet`, one per event). The constraint set depends on
the field kind. `CSet.iq` is `(N, [(P1, P2, '>'|'<'), …])` enforcing
`field(P1) rel field(P2)`; Curlew samples `N` random pairs from each `(P1, P2)`
pool per evaluation, so all pairing below is expressed as point-set pools.

*Depositional (one or more contacts) packages.* Layer these constraints; more is better,
and the inequality terms are what guarantee correct polarity when normals are
sparse or absent:
- **On-contact `eq` traces** — for each internal contact, the points sampling it
  form an `eq` group: all share one (unknown) iso-value. This is what makes a
  contact a clean level set for iso-surface extraction. Also
  `addIsosurface(name, seed=<those points>)` so the iso-value is recovered at
  predict time. One `eq` group + one isosurface per contact.
- **Normals → `gv` gradients** (where present, e.g. multilayer_fold) — `gp`/`gv`
  gradient constraints, polarity-enforcing (younging up-section). Use `gov`
  (orientation only) only if a dataset's normals lack reliable sense.
- **Inequalities between contacts → `iq`** (always useful; *essential when no
  normals are available*) — for every pair of contacts in the package, the
  younger contact's points must have **larger** field values than the older
  contact's points: pool P1 = younger-contact points, P2 = older-contact points,
  relation `'>'` (equivalently older `<` younger). This fixes the ordering /
  polarity of the field even without gradients. `eq` says "same value within a
  contact"; this says "correct order *between* contacts".
- **Unit points → `iq`** (where present) — for each unit, pairs against the
  bounding contacts: unit points `>` the contact below them and `<` the contact
  above them. (Generated by the bespoke rules here — *not* by
  `derive_relational_constraints`.)

*Depositional (region-only basement).* No `eq`, no seed isosurface, no `gv` — there
is no surface to fit. The field is effectively free; its only role is to label the
basement region and to supply its unit points as below-references to the overlying
erosional event. (Verify a Curlew event with an unconstrained field trains/predicts
cleanly — see §8.)

*Erosional unconformities.* If the surface has on-surface points, treat it like a
depositional contact (`eq` trace + seed isosurface, optional `gv`). When it has
none — every wcsb unconformity — constraints are **inequalities only**:
- **Above** — points from **younger _stratigraphic/depositional_ events only**
  (both their contact points and their unit points). Younger *erosional* events
  are excluded from the above-set.
- **Below** — points from **any older event** (its unit points, and any contact
  points), i.e. everything older in the column.
- These are expressed directly as pairwise `iq`: pool P1 = above-set, P2 =
  below-set, relation `'>'`. The fallback learnable `base` (extraction iso /
  `Overprint` threshold) is then trained to sit between the two pools. The exact
  mechanism for the learnable-scalar reference is an open item (§8); keep it
  minimal — prefer deriving `base` from the trained pools over adding new
  constraint machinery, and engage it only when seeds are unavailable.

**Normalization (`Transform`):** compute data bounds, build a single homogeneous
scale+translate matrix into `[-1, 1]`:
- `scale="isometric"` → one scale = `2 / max(extent_x, extent_y, extent_z)` on all axes.
- `scale="geo_extensive"` → separate xy scale and z scale (vertical exaggeration);
  z gets `2 / extent_z`, xy gets `2 / max(extent_x, extent_y)`.
Attach as `GeoModel(events, transform=Transform(matrix))`. The same `Transform`
applies to constraint positions (Curlew handles global→model rebinding) and to
`predict` inputs. No GeoINR scaler classes are ported.

### 4.3 Observation loader

A thin loader producing per-level point arrays (coords + level), supporting the
formats already in the data dirs:
- `.vtp` (e.g. `cleaner_markers.vtp`, `interface.vtp`, `normal.vtp`) — points,
  a `level` point-array, optional `normals` point-array. (pyvista is GeoINR's
  reader; in Curlew keep VTK/pyvista an **optional, lazily-imported** dependency
  per the repo convention — core must run on numpy/torch/tqdm.)
- `.csv` with `x,y[,z],level` and optional `is_interface`, `nx,ny[,nz]`.

The loader groups points by `level` so the builder can assemble per-interface
seeds/traces and per-unit inequality pairs.

### 4.4 Imbalance-aware per-step sampling

wcsb units are highly imbalanced (some levels have a handful of points, others
have ~millions). Requirements:
- **Load full** markers, then **cap each unit** at `cap_per_unit` points held in
  memory (uniform random subsample per unit at load).
- **Per training step**, re-draw a fresh random subset, **stratified by unit** so
  every unit — including rare ones — contributes at least a minimum number of
  points/pairs each step. Do not let a few huge units dominate the `iq` pair pool.
- Implement by controlling how `iq` pairs (and `eq`/gradient batches) are
  sampled: build pairs per (unit, interface) group and draw a balanced quota from
  each group, rather than sampling globally. (`CSet.iq`'s `N` is global; the
  balancing happens in how `P1`/`P2` pools are constructed and re-sampled.)

---

## 5. Datasets to test

### 5.1 multilayer_fold (basic — single scalar field)
- Dir: [geoinr/data/multilayer_fold/](../GeoINR_Research2/geoinr/data/multilayer_fold/).
- Strat column: [strat_col.csv](../GeoINR_Research2/geoinr/data/multilayer_fold/strat_col.csv) — six units A–F, all `conformal` ⇒ **one** depositional field, five internal contacts.
- Observations (3D): `interface.vtp` (contact points + level) and `normal.vtp`
  (oriented normals + level).
- Curlew model: a single `strati` event with a `Siren` (or `GeoINR`) field;
  five seed isosurfaces (one per contact); `eq` traces per contact; `gv` gradient
  constraints from normals; isometric `Transform`.

### 5.2 wcsb (complex — many scalar fields)
- Dir: [geoinr/data/wcsb/](../GeoINR_Research2/geoinr/data/wcsb/).
- Strat column: [strat_col_v2.csv](../GeoINR_Research2/geoinr/data/wcsb/strat_col_v2.csv)
  — 36 units mixing `conformal`/`baselap`/`eroded` ⇒ many depositional packages
  interleaved with erosional unconformities ⇒ many Curlew `strati` events.
- Observations: `cleaner_markers.vtp` — **unit points only** (level labels), no
  interface points, no normals.
- Curlew model: one event per derived scalar field, ordered oldest→youngest.
  Erosional unconformities are their own `strati(mode='above')` events with a
  learnable iso; depositional packages sit between them. The oldest event is the
  region-only Precambrian basement (single unit, no interfaces). Constraints are
  **`iq` inequalities only** (no on-contact or normal data exists): for each
  unconformity, younger depositional-event points are above and older-event points
  below its learnable iso; depositional packages additionally get unit↔contact and
  contact↔contact inequalities where multiple units share a package.
- Scaling: anisotropic (`geo_extensive`) — large xy extent vs thin z.
- Tractability: full load + `cap_per_unit` + imbalance-aware per-step resampling
  (§4.4).

---

## 6. End-to-end verification — Jupyter notebooks

The primary deliverable for each dataset is a **Jupyter notebook** (under
`examples/<dataset>/<dataset>.ipynb`), runnable top-to-bottom in the `curlew`
conda env (§9). Each notebook must make the pipeline *visible*, not just assert
on it. The three visual outputs below are required in every dataset notebook;
lightweight `assert`s are embedded so a clean run also functions as a regression
check.

**Required notebook structure (both datasets):**
1. **Load & normalize** — read the strat column and observations; build the
   `Transform`; show the parsed stratigraphic column (units, relations, levels) as
   a table.
2. **Scalar-field partitioning (visual)** — render how units/contacts are
   partitioned into Curlew events: a labelled diagram/table of the derived fields
   (depositional multi-contact / region-only basement / erosional) in
   oldest→youngest order, colour-coded, with each field's units and contacts. A
   matplotlib figure (and, for the model graph, `GeoModel`'s `_repr_svg_`).
3. **Constraint relations (visual)** — show the constraints used *between
   geological features*: per event, plot `eq` traces, `gv` gradient arrows where
   present, and the `iq` above/below pairings as connectors between the relevant
   point sets (e.g. younger-above vs older-below clouds for each unconformity).
   This must make the bespoke above/below rules (§4.2) legible at a glance.
4. **Fit** — `M.fit(...)`; plot the loss history. For wcsb, log per-unit sample
   counts per step to show every unit (incl. rare ones) stays represented.
5. **Predict & 3D view (napari)** — `M.predict(grid)` → `Geode`; open the
   **napari** viewer showing the predicted scalar field / lithology volume and the
   extracted iso-surfaces (`Geode.getSurfaces`). Include a static screenshot
   fallback (matplotlib section) so the notebook renders without a live Qt session.
6. **Checks** — the dataset-specific assertions below.

### 6.1 multilayer_fold notebook
- `M = build_geomodel('…/multilayer_fold/strat_col.csv', obs, field='Siren', scale='isometric')`.
- One depositional event, five contacts; partitioning view shows a single field.
- Constraint view shows the five `eq` traces and the `gv` normals.
- **Checks:**
   - The five recovered iso-values (`getIsovalues` from seeds) are **strictly
     monotonic** in stratigraphic order.
   - At held-out interface points, predicted scalar ≈ the corresponding contact
     iso-value (within tolerance).
   - Predicted gradient direction at normal points aligns with the supplied
     normals (positive dot product / small angular error).

### 6.2 wcsb notebook
- `M = build_geomodel('…/wcsb/strat_col_v2.csv', obs, field='GeoINR', scale='geo_extensive', cap_per_unit=…)`.
- Partitioning view shows the full event chain: oldest = region-only Precambrian
  basement, then its erosional unconformity, then alternating
  depositional/erosional events youngest-last; each unconformity its own
  `mode='above'` event with a learnable iso.
- Constraint view shows, per unconformity, the younger-depositional above-set vs
  older below-set inequality pairings (no `eq`/`gv` — none exist).
- **Checks:**
   - Builder emits the expected events in the expected order.
   - Every unit is represented in the sampled constraints each step (per-unit
     counts > 0, rare units included).
   - Inequality satisfaction on held-out unit points exceeds a threshold (e.g. > 0.9).
   - Each unconformity's learnable iso lands **between** its above/below clouds.
   - Recovered iso-values are monotonic across the modelled sequence per field.
   - Predicted unit at known unit points matches the label for a high fraction of
     held-out points, including the basement region.

### 6.3 Field-port smoke test (independent of datasets; `pytest` in `tests/`)
- Construct `strati('s', C=GeoINR('f', input_dim=3))` and the `Siren` equivalent;
  bind a tiny synthetic `CSet` (a few `gv` + `iq`); `GeoModel([...]).fit(small)`;
  assert loss is finite and decreases. Confirms the `BaseNF` contract
  (`initField`/`evaluate`/optimiser) is satisfied for both classes.

---

## 7. Explicitly out of scope (future passes)

- Learned iso-values are **not** the default. Any contact with on-surface
  points — depositional *or* erosional — gets its iso from a seed. The learnable
  iso is a **fallback used only when an unconformity has no on-surface points**
  (true for every wcsb unconformity, but not a general assumption). Keep this path
  minimal: if seeds are available it must not be engaged.
- RFF encoding, `RFFMLP`, `SupportResidualSiren`, progressive/support controllers.
- Lipschitz / spectral normalization.
- GeoINR's training loop, distributed/sharded training, diagnostics, VTK export
  pipeline — Curlew's `fit`/`predict`/`Geode` replace these.
- Porting the legacy `GeologicalConstraints` pipeline; only the clean
  `stratigraphy.py` partitioning is reused.

---

## 8. Open items to resolve during implementation

- Confirm Curlew `Transform` anisotropic scaling round-trips constraint vectors
  (`gv`) correctly under non-uniform z scaling (gradient direction vs magnitude).
- Define the exact balanced-quota rule for `iq` pair construction (min pairs per
  unit per step, and how `cap_per_unit` interacts with it).
- Choose where the observation loader lives (likely `curlew/io.py` extension,
  keeping pyvista optional/guarded).
- **Learnable-iso fallback mechanism** (erosional events with no on-surface
  points). Pick the *minimal* option: derive `base` post-hoc from the trained
  above/below pools, or a small learnable scalar with a hinge loss. Must not be
  engaged when seeds exist, and must not complicate the common (seeded) path.
- **Region-only basement event.** Confirm a Curlew `strati` event whose field has
  an empty/near-empty `CSet` trains and predicts without error, and that its
  lithology ID is assigned correctly beneath the oldest unconformity.
- **Three-event unconformity wiring.** Confirm `A (older) → erosional → B
  (younger)` overprints correctly: the erosional event's `mode='above'` + iso
  truncates A where B's package sits above the unconformity, for both the seeded
  and learnable-iso cases.

### 8.1 Findings (from the depositional `multilayer_fold` pass)

These resolve/refine the open items above and should guide the wcsb pass:

- **Build constraints in *model* coords; transform seeds yourself.** Construct every
  `CSet` with `crs='model'` and map `addIsosurface(seed=...)` points into model
  coords before attaching. `GeoModel.T` (global→model) is applied by `predict` but
  **not** when `getIsovalues` evaluates seeds through the event's `forward`, so
  mixing global constraints (auto-rebound) with global seeds (not rebound) misaligns
  recovered iso-values. Keep constraints, seeds and `gv` all in model space.

- **The field magnitude is an unconstrained gauge.** Normalized `gv`, mean-normalised
  `eq` and sign-only `iq` are all scale-free, so the field range can drift or collapse
  to a trivial minimum. In the depositional path, `eq_loss≈10` (with the default
  output `scale`) holds contacts apart; too-small `scale` collapses them, too-large
  `eq` destabilises. **For wcsb (fields are `iq`-only — no `eq`/`gv`) this is acute:**
  both offset and scale are free, so the learnable-iso fallback must *actively anchor
  the gauge* (e.g. a hinge placing the iso between the above/below pools) or the field
  collapses / the net learns a shortcut. A principled option worth porting is GeoINR's
  residual normalization: divide scalar differences (`f(p_i) − iso`, where `iso` is the
  learnable value or the per-interface mean) by **‖∇f‖**, so residuals stay meaningful
  when the gradient is small across the domain — optionally implement GeoINR-style
  `eq`/`iq` losses this way.

- **`derive_scalar_fields` does NOT emit the region-only basement.** It only appends a
  depositional field when it has ≥1 horizon, so a lone basement (no internal contact)
  yields no field. The builder must **synthesize the basement event itself**
  (oldest-first, region-only, empty/near-empty `CSet`) before the first unconformity —
  do not rely on the partitioner for it.

- **§4.4 is largely handled by the loss already.** The `iq` loss samples `ns` index
  pairs **with replacement, per `(P1,P2)` pool, each step**, and `ns` is global — so
  per-pair / per-interface balance is automatic regardless of point counts. Remaining
  wcsb work: `cap_per_unit` at **load**, and construct **one pool per unit/contact** so
  rare units are represented. Store only **one direction per pair** (`younger > older`);
  the reverse is implied — do not add `<` duplicates.

- **Oscillation is a smoothness problem, not a field-type problem.** `Siren`
  (`omega0=2, omega=30`) is init-sensitive on sparse data: it fits normals perfectly
  but can oscillate between them (and a "good" CPU seed can be wild on GPU, since CUDA
  RNG ≠ CPU RNG). Keep `Siren` for fitting capacity — switching to softplus/`GeoINR`
  removes oscillation but, via INR spectral bias, reduces ability to fit points (only
  fine for simple sets like `multilayer_fold`). The right fix is a **global smoothness
  prior**: curlew's `mono_loss`, or a "no-overturn" trend term (≈ curlew's flat/trend
  loss / GeoINR's no-overturn) that penalises gradients pointing in −z. Add this rather
  than lowering capacity.

- **Tooling available for the wcsb notebook.** The observation loader landed in
  `curlew/io.py` (`loadObservations` / `Observations`, pyvista lazy/guarded).
  `GeoModel.fit(..., history=True)` returns per-epoch detached `Pebble`s for the loss
  curve (progress bar is now `tqdm.auto`) — use it instead of a `custom_loss` callback.
  For napari, add **separate layers** (`addVolume` for scalar/units + per-event
  `G.contour`→`addMesh` for surfaces; `addGeode`'s bundled surface path was
  unreliable), all in world coords. Fit on **all data — no train/validation holdout**
  (validation belongs to later ensemble work).

---

## 9. Environment & dependencies

Target env: the existing `curlew` conda env
(`C:\Users\mhillier\.conda\envs\curlew\python.exe`). Already present and
sufficient for the core/fields/builder work: `numpy`, `torch`, `tqdm`,
`matplotlib`, `scipy`, `pandas`, `napari 0.7.0` with a working **PyQt6** backend,
`ipykernel`, `plyfile`.

**Must install** (not present, needed for the notebooks):
- `pyvista` — reads the `.vtp` observation files (`cleaner_markers.vtp`,
  `interface.vtp`, `normal.vtp`); pulls in `vtk`. Keep it a lazily-imported,
  *optional* dependency inside Curlew (core must still run on numpy/torch/tqdm).

**Recommended:**
- `ipywidgets` — progress bars and napari/notebook interop in the notebooks.

**Optional:**
- `jupyterlab` (or `notebook`) — only if running the notebooks outside VS Code;
  `ipykernel` already covers the VS Code notebook UI.

**Not needed:** `scikit-learn` (the GeoINR scalers are replaced by Curlew
`Transform`); any extra Qt binding (PyQt6 already resolves via `qtpy`).

Install into this env (conda-forge preferred for the VTK stack):
```
conda install -n curlew -c conda-forge pyvista ipywidgets
# optional: conda install -n curlew -c conda-forge jupyterlab
```

**Also:** the env currently has the **pip-installed `curlew` 1.1**, not this
repo. Develop against the repo (1.2/dev) with an editable install so the
notebooks exercise the new code:
```
C:\Users\mhillier\.conda\envs\curlew\python.exe -m pip install -e d:\Development\curlew
```
