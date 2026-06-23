# skmb — handoff notes (seed-iso hybrid path)

Working notes for refining the **skmb** example and the **seed-iso** (interface + unit hybrid)
path in a fresh session. Companion to [GEOINR_INTEGRATION.md](../../GEOINR_INTEGRATION.md)
(the readable walkthrough) and [SOFTUNIT_SPEC.md](../../SOFTUNIT_SPEC.md) (the wcsb design).

## 1. Status (what works)

skmb combines **interface points** (`markers.vtp`) and **unit points** (`units.vtp`) with
baselap-onto-conformal splits and doubled unconformities. Built by
`build_geomodel(..., field="Siren")` → `attach_unit_loss(..., seed_isos=True)` →
`M.fit(custom_loss=[uloss])`. Surfaces are **seeded from interfaces** (no learnable isos);
unit points drive the soft-unit (NLL/CE) carve.

Last verified (500 epochs, full data, GPU): band **0.819**, structure **0.938**,
section voids **0%**, Precambrian margin-cut **0.7%** (down from 5.7%). All correct
structurally; the suite is green (95 passed). Remaining issue: small modelling artifacts at
the basin **pinch-out margin** (see §4).

## 2. The constraint architecture (seed-iso path)

This is the crux to understand/validate/simplify. Each event's neural field carries several
loss terms (all gradient-normalised, all in `curlew.fields.geoinr.GeoINR.loss`):

| term | what it pins | where built |
|---|---|---|
| `eq` (interface) | the field == its contact value at on-surface points | `_build_*_event` |
| `iq` within-package (unit) | younger band unit pts > older band unit pts | `_build_depositional_iq_event` |
| `iq` within-package (interface) | younger contact pts > older contact pts | `_build_depositional_iq_event` |
| `iq` unconformity nesting | this unconf surface > each older unconf surface (one pair each) | `_build_erosional_event` |
| `iq` package-top nesting | this package's top contact > each older unconf surface | `_build_depositional_iq_event` |
| `iq` erosional ordering | onlapping pkg pts > eroded + every older unit level | `_build_erosional_event` |
| `overturn` | field monotone-up on a resampled grid | field loss |
| CE / NLL (carve) | unit points classified by the seeded stick-break | `softunit.UnitLoss` |

`attach_unit_loss(seed_isos=True)` keeps `iq_norm_weight=1` on **every** field (the wcsb
learnable path zeros it) and relaxes `overturn` to 6.

### The single unifying principle (KEY for simplification)

All four `iq` "ordering/nesting" terms are instances of **one** idea:

> In each field, every surface point sits **above** every *older* surface point (and above
> every older surface it could onlap/truncate against), in that field's value.

- *within-package* → orders a package's own contacts;
- *unconformity nesting* → keeps a younger unconformity above older unconformity surfaces;
- *package-top nesting* → keeps a package's top (a baselap onlap threshold) above older surfaces.

They were added incrementally while debugging, so they are **separate code paths**. A fresh
session should be able to **collapse them into one constraint builder**: for each field,
collect (its own contact levels) + (every older surface level it must stay above), and emit
the `young>old` pairs in one place. This would cut code and make the intent obvious.

### Why the nesting is needed (validates the user's framing)

The user's pattern `c,c,b,c,c,b,eroded`: a baselap unit can onlap **both** the youngest interface
of the next (older) conformal package **and** the next-older unconformity — **spatially varying**.
The clarification (user): the lower package's youngest interface itself **can terminate (onlap)
on the next-older unconformity**, so in the regions where the package top has pinched out the
baselap onlaps **directly onto the unconformity**, and elsewhere onto the package top. In the
combine, the onlapping package claims where its *onlap-domain field* (the parent — a package
**top iso** or an **unconformity iso**) exceeds that threshold. If that threshold surface dips
below an older surface at the basin margin, the onlapping package **erodes below it** (e.g.
younger units appear below the Precambrian). The nesting forbids the threshold surface from
crossing older surfaces — so whichever surface the baselap onlaps **in a given region**, the
geometry stays ordered. **Open question for refinement:** the current build wires a baselap-
onto-conformal package to a *single* onlap threshold (the lower package's top iso), and relies
on the combine + nesting to resolve the right surface per region. Whether that faithfully
reproduces the *spatially-varying* onlap (package-top in some regions, unconformity in others)
should be validated — it may need the onlap to fall through to the unconformity where the
package top has terminated.
The **per-pair** form (one pair per older surface, not one pooled set) matters because the
margin points are sparse and would otherwise be under-sampled. **Package-top nesting was the
decisive fix** — the baselap-onto-conformal packages (Winnipegosis, Torquay…) onlap a package
*top*, which the unconformity nesting did not cover.

## 3. Session history (fixes, with the diagnostic that motivated each)

1. **Doubled unconformity void** (Sub_Cantuar L12 sandwiched between two unconformities) →
   synthesize a region-only event for the sandwiched unit. (`orphaned: []`, section voids 0%.)
2. **Precambrian cut, attempt 1** — added unconformity nesting (own surface > pooled older
   surfaces). Field-violations → ~0 but predict still cut ~1.3% (under-sampled margin).
3. **Per-pair nesting** + **within-package interface ordering** (user's suggestion) +
   `iq_norm` on for all seed-mode fields. Improved band placement (per-unconf "respected"
   80–99%), but margin cut still 5.7%.
4. **Attribution diagnostic** → cutting events were the **baselap-onto-conformal packages** →
   **package-top nesting**. Margin cut 5.7% → **0.7%**, accuracy held.

## 4. Open items / next steps

### a. Remaining margin artifacts (small) — most promising: overturn SAMPLING
At the extreme pinch-out margin (shallow Precambrian, NE), ~0.7% of Precambrian and ~2.8% of
Sub_Miss surface points are still mis-labelled — surfaces genuinely converge to zero thickness
there. The **highest-priority lever (user's strong steer): the no-overturn sampling strategy.**

State of play (verified in code):
- It **already resamples fresh points each epoch** (`Grid.draw` → `Grid.sample` →
  `np.random.choice`), so it is not "fixed sampling".
- The grid (`_model_grid`, `reg_grid_n=48`) **does span the full model domain** (a 48³ ≈ 110k
  lattice over the whole `[-1,1]` box — checked: the per-epoch sample's extent equals the data
  extent). So it is **not** restricted to a tiny spatial region.
- BUT it draws **uniform-random from that fixed coarse lattice** (`sampleArgs={'N': draw}`), so
  the prior is only ever evaluated at those 110k fixed nodes, clumpily. The network never sees
  the continuum between nodes, and a coarse lattice (~22 km nodes at skmb scale) can miss
  finer-scale overturns — exactly where the margin pinch-outs live.

Recommended change: switch the overturn grid to **Poisson-disk sampling** (curlew already
supports it — `Grid.sample(poissonDisk=(r, k, seed))`; pass it via `_model_grid`'s
`sampleArgs`, `seed=None` to re-draw each epoch) for **even** domain coverage, and make the
lattice **finer / model-scaled** so the field "sees" the whole domain over training. Per the
user: `reg_samples` should **scale with the model's geographic coverage** (≈ **500–10,000**,
larger models → more), and *smaller* per-epoch samples seem to learn the no-overturn prior
*more* robustly than large ones (unverified but consistent with their experience).

Other (lower-priority) levers: per-event `overturn_weight` (tried on wcsb, did **not** help
there; may differ); more epochs (verified only to 500).

### b. Training time (~90 min) — the main practical concern
Driven by **265 `iq` pairs × 1024 samples/epoch**, + `overturn` (5000 × 17 fields), all needing
**second-order autograd**. (Clarification: the eq/iq/overturn losses are functions of the
field's *spatial* gradient ∇f — a **first**-order derivative w.r.t. the input coordinates, e.g.
the no-overturn penalises the downward component of ∇f. But back-propagating that loss to the
**network weights** means differentiating *through* ∇f — the gradient of a gradient — so curlew
builds ∇f with `create_graph=True` and pays a **second-order** autograd cost. The CE/NLL carve
uses field *values* only, so it is first-order and cheaper.) The youngest unconformities
dominate: Sub_Lea_Park has **47 pairs** because its erosional "below" side is the *entire* older
column (every older unit level + every older unconf surface). Reduction levers (cheap, likely
also *better* per user's insight
that smaller samples train the priors better):
- `iq_samples` 1024 → **256** (≈4× less `iq` cost; user: smaller samples also train better);
- `reg_samples`: scale with model coverage (≈500–10,000) + Poisson sampling — see §4a;
- subsample / pool the erosional below-side (don't emit a pair per older unit *level* — a
  handful of representative older levels, or one pooled "deep" reference, may suffice now that
  nesting handles the surfaces explicitly);
- the **within-package unit** `iq` is largely redundant with the within-package **interface**
  `iq` + CE on this path — consider dropping it in seed mode.

### c. Complexity (growing concern — agreed)
The seed-iso path now has ~6 loss terms. **Consolidate** the four ordering/nesting `iq` terms
into the single principle in §2 (one builder, one relation tag). This is the highest-value
cleanup: less code, clearer intent, and it makes (b) easy (one place to set sample counts).

## 5. How to re-verify (diagnostic)

The margin diagnostic used this session (drop into a script, run on GPU):
- **cut rate**: predict at each unconformity's interface points; fraction labelled
  *much-younger* than the surface (e.g. `predict_level < eroded_level - 6`), split by margin
  (`z > quantile(z, 0.80)`). This is the geologically-impossible "cut".
- **attribution**: on the cut margin points, `collections.Counter(geode.structureID names)`
  tells you *which event* is doing the cutting (that is what pointed to the baselap packages).
- **regression**: sample unit points per level, `M.predict`, compare predicted lithology/
  structure to `M.level_litho` / the owning event (band / structure accuracy).

Build with `cap_per_unit=8000, iq_samples=1024`, `field_kwargs=dict(omega0=30,omega=30,
learning_rate=1e-4,hidden_dim=256)`, `attach_unit_loss(tau=0.05, points_per_level=1024,
overturn_weight=6, seed_isos=True)`, fit 500 epochs (≈ enough to see the geometry).

## 6. Recommended fresh-session plan

1. **Overturn sampling → Poisson** (§4a) — the user's strongest steer and likely the biggest
   quality win for the residual margin artifacts: switch `_model_grid` to Poisson-disk sampling,
   re-drawn each epoch, with a model-scaled lattice and `reg_samples` ≈ 500–10,000. Cheap, and
   plausibly *reduces* cost too (smaller, better-spread samples).
2. **Consolidate** the ordering/nesting `iq` into one builder (§2, §4c) — no behaviour change,
   just structure. Re-run the suite + the §5 diagnostic to confirm parity.
3. **Cut cost** (§4b): `iq_samples`→256, trim the erosional below-side, drop the redundant
   within-package unit `iq`. Confirm the margin cut stays low and accuracy holds — measure the
   training-time win.
4. **Validate** the spatially-varying baselap onlap (§2 open question) if margin issues persist.
