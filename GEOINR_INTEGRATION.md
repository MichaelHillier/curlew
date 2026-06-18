# GeoINR → Curlew integration — how it works

A readable, illustrated walkthrough of the runtime behaviour. For the *decisions* see
[SPEC.md](SPEC.md); for the soft-unit *design rationale* see [SOFTUNIT_SPEC.md](SOFTUNIT_SPEC.md);
for *current status & open items* see [SOFTUNIT_STATUS.md](SOFTUNIT_STATUS.md).

The integration turns a **stratigraphic column + point observations** into a fitted
Curlew `GeoModel`, using GeoINR's neural fields and (for unit-only data like wcsb) GeoINR's
**unit / NLL coupling**, but reusing Curlew's native event graph, constraints and `predict`.

```
strat_col.csv + observations
        │  build_geomodel()                       (curlew/geology/stratbuilder.py)
        ▼
   GeoModel  =  graph of GeoEvents (oldest → youngest)
        │  fit
        ├── on-surface data  → per-field eq/iq/grad losses        (multilayer_fold)
        └── unit-only data   → attach_unit_loss() joint CE        (wcsb)
        ▼  predict  (sequential overprint combine — unchanged Curlew machinery)
   Geode  (scalar, lithoID, structureID, surfaces, …)
```

## File map

| Concern | Where |
|---|---|
| Neural fields (`GeoINR` MLP, `Siren`) + their losses | [curlew/fields/geoinr.py](curlew/fields/geoinr.py) |
| Strat-column → event-chain builder | [curlew/geology/stratbuilder.py](curlew/geology/stratbuilder.py) |
| Unit / NLL coupling (carve, `UnitLoss`) | [curlew/geology/softunit.py](curlew/geology/softunit.py) |
| Event, overprint, combine (Curlew core — untouched) | [geoevent.py](curlew/geology/geoevent.py), [interactions.py](curlew/geology/interactions.py), [core.py](curlew/core.py) |
| wcsb notebook | [examples/wcsb/_build_notebook.py](examples/wcsb/_build_notebook.py) |

---

## 1. The event chain

`build_geomodel` parses the column (youngest-first), partitions it into **scalar fields**
(one neural field per *package* or *unconformity*), and reverses to oldest→youngest. For
unit-only data it emits an **alternating chain** with synthesized *region-only* events at
both ends. wcsb (abbreviated):

```
 oldest                                                              youngest
 ┌──────────┐ ┌────────────┐ ┌────────┐ ┌────────────┐      ┌────────────┐ ┌──────────┐
 │ region   │ │ erosional  │ │ depo   │ │ erosional  │ ...  │ erosional  │ │ region   │
 │Precambri.│ │Precamb.Unc.│ │Cambrian│ │Sil/Ord Unc.│      │BellyRiv Unc│ │Quaternary│
 │ L37      │ │  (no unit) │ │ L35,36 │ │  (no unit) │      │  (no unit) │ │  L2      │
 └──────────┘ └────────────┘ └────────┘ └────────────┘      └────────────┘ └──────────┘
   alias of ───┘                                                    └─── alias of
```

Three event kinds (each one `strati(...)` `GeoEvent`, each owning one neural field):

- **depositional package** — a run of conformable units (e.g. Cambrian = levels 35,36).
  Owns one field; its internal contacts become learnable isosurfaces.
- **erosional unconformity** — a single surface that *truncates* older geology. Owns a
  field but **no lithology** (it's a surface, not a unit); just a threshold.
- **region-only** — basement / dropped-top unit. One lithology, **no constraints of its
  own**; its field is an *alias* (§3).

`M.field_meta[i]` (a `_FieldMeta`) describes event `M.events[i]`: its `kind`,
`is_unconformity`, the unit `levels` it owns, and bookkeeping for the constraints.

---

## 2. Onlap: `child` vs `parent` (the key structural idea)

Every generative event has an `Overprint` that decides, point-by-point, where this event's
rock replaces the older rock below it. The **domain** of that decision is one scalar field,
chosen by `Overprint.defaultDomain`:

**`defaultDomain='child'` — erosional truncation.** The boundary is *this event's own*
field. The unconformity cuts the older beds; its surface geometry is its own.

```
   ▔▔▔▔ erosion surface  =  CHILD field iso (this event) ▔▔▔▔
   ╲ ╲ ╲   older beds truncated at the surface   ╲ ╲ ╲
```

**`defaultDomain='parent'` — onlap.** The boundary is the *parent* (the older unconformity
it sits on). Younger beds lap onto — and pinch out against — that older surface.

```
        this package (younger beds)
   ────────────────────────────────────
   ──────────────────────────────╮            beds ONLAP and pinch out
   ────────────────────────────╮ │            against the unconformity
   ~~~ unconformity = PARENT field iso ~~~      ← base follows the PARENT surface
   ╱ ╱ older, already-truncated beds ╱ ╱
```

In the builder ([`strati(..., onlap=True)`](curlew/geology/__init__.py)):

| event | mode | `defaultDomain` | overprint domain field |
|---|---|---|---|
| erosional unconformity | `above` | `child` | its **own** field |
| depositional package (onlaps the unconformity below it) | `above` | `parent` | the **unconformity** field below it |
| region-only basement (oldest) | `above`, `base=-inf` | `child` | — (claims everything below) |
| region-only top (onlaps youngest unconformity) | `above` | `parent` | the youngest **unconformity** field |

This table *is* the rule the carve (§5) and `predict` both follow.

---

## 3. Region-only events & `_AliasField`

A region-only event (basement, dropped-top unit) has **one lithology and no data of its
own** — it only needs to label its region and supply its points as references. Rather than
give it a random untrained network, its field is a loss-free **`_AliasField`** of the
*adjacent unconformity's* field:

```
  Precambrian (basement, L37)        evaluates  ≡  Precambrian-Unconformity field
                                                     (continued below the iso)
```

Why an alias and not the same object?
- **GeoINR's basement treatment:** the basement domain is classified by the *unconformity's*
  field + iso, so continuing that fitted field downward is the geologically meaningful scalar.
- **No double-optimisation / no Pebble clash:** the alias owns **no parameters** and its
  `C is None`, so `loss()` returns an empty `Pebble`. (Curlew's `Pebble` refuses duplicate
  loss keys when `GeoModel.fit` sums per-event losses, so a *shared field object* would
  raise — a loss-free alias sidesteps that.)

Valid because the region-only event is stratigraphically adjacent to its target with no
deformation between them, so their reference frames coincide. (This is why a region-only
event shows **no constraints / no loss** when you inspect it — that's by design.)

---

## 4. Constraints per event — and why `iq` looks the way it does

For unit-only data there are no on-surface points, so each field is constrained by
**point-vs-point inequalities** (`CSet.iq`). For an **erosional** event the rule is:

> the package that **onlaps** the unconformity sits **above** the eroded unit **and every
> older unit** — so the surface is pinned below the young package and above all old rock.

In code ([`_build_erosional_event`](curlew/geology/stratbuilder.py)) the *above* side is
**pooled into one tensor** and there is **one pair per older level**:

```
iq[1] = [ (above_pool, level_B_points, '>')  for B in (eroded unit … oldest) ]
          └ lhs ┘     └─── rhs ───┘    └ '>' ┘
```

So for the **youngest** unconformity (*Belly River*, event 13) you see **~35 pairs**, each
`(Quaternary_points  >  one older unit's points)`. **Your reading is exactly right:** the
single onlapping unit (Quaternary, the dropped top) is required to be above *every* older
unit, which forces the unconformity's field below Quaternary and above all the rock it
eroded into. (`'>'` everywhere because "younger is higher-valued".) The `_FieldMeta.iq_relations`
records the same thing in level form for the notebook's §3 listing.

> ⚠️ **Under the coupling these `iq` constraints are built but _not used for the loss_.**
> `attach_unit_loss` sets `iq_norm_weight = 0` — the cross-entropy (§5) supersedes the
> inequalities. They survive only to **size the per-level pools** and feed the §3 inspection.
> (A future cleanup could skip building them on the coupled path; see SOFTUNIT_STATUS.md.)

A depositional package instead gets adjacent-band pairs (band *j* `>` band *j+1*); a
region-only event gets none.

---

## 5. The unit / NLL coupling — training carve vs inference combine

This is the heart of the wcsb path, and the key idea is a **duality**:

| | **Training** (`UnitLoss`) | **Inference** (`GeoModel.predict`) |
|---|---|---|
| what | global per-point **soft carve** → `(N,C)` simplex | Curlew's **sequential overprint combine** |
| robust to | deep/ambiguous points (one global decision) | — (fragile if fields aren't monotone) |
| signal | cross-entropy vs true unit **levels** | hard `lithoID` |
| code | [`soft_unit_probs`](curlew/geology/softunit.py) | [`Geode.combine`](curlew/core.py) (unchanged) |

They are kept consistent: the carve uses the **same fields** (evaluated through
`ev.predict(combine=False, transform=True)`, so faults/deformation still apply) and the
**same iso-values**, which are written back after fitting so `predict` reproduces the carve.

### The carve (`soft_unit_probs`) — a per-class re-derivation of `combine`

Youngest→oldest stick-break. `remaining` starts at 1; each **class-owning** event claims
a share and passes the rest down. `s = 1/τ` (τ = 0.05 ⇒ s = 20).

```
remaining = 1
for each class-owning event E, youngest → oldest:
    w = σ( s · ( f_domain(E) − θ_domain(E) ) )     # domain per the §2 table
    claim = remaining · w
        depositional → split `claim` across its bands by f_E and its contacts θ₁>…>θₘ
        region-only  → all of `claim` to its one class
    remaining = remaining · (1 − w)
# the oldest basement (w=1) takes whatever `remaining` is left
return  probs / probs.sum()                          # already ≈1
```

```
            f_BellyRiver                f_Precambrian-Unc.
 Quaternary ─►σ(...)              Cambrian ─►σ(...)            Precambrian
   claim = remaining·w              claim = remaining·w          claim = remaining (w=1)
   remaining ·= (1−w)               remaining ·= (1−w)
   └────────────── each unconformity truncates exactly ONCE ──────────────┘
```

**Why erosional events are _not_ steps here (the bug that was fixed):** an unconformity's
truncation is *already* applied by the package that onlaps it (whose `w` uses that same
unconformity field+iso, per §2). Treating the erosional event as its own step too would
(a) truncate twice `(1−w)²` and (b) discard `remaining·w·(1−w)` of probability that then
gets renormalised — which lets the CE cheat by driving `w→0.5` to inflate the true-class
prob without separating units (a degenerate minimum that got *worse* with training). So the
carve only steps class-owning events; each surface truncates once. (wcsb's chain is strictly
alternating with zero orphan unconformities, so this is exact.)

### Iso-values: learned, monotone, written back

Iso-values are **learnable Parameters owned by `UnitLoss`**, not scattered through core:
- each **unconformity** has one free threshold;
- each **package**'s internal contacts use a **monotone reparam**
  `θⱼ = θ₀ − cumsum(softplus(φ))` ([`monotone_contacts`](curlew/geology/softunit.py)) — so
  contacts are descending **by construction** (a band can pinch to zero thickness but two
  contacts can never cross), which also makes the band probabilities valid without clamping.
- init is **uniform** (θ₀≈+1, even gaps); correctness is owned by the reparam + CE, so the
  init is only a convergence aid.

After `M.fit(custom_loss=[uloss])`, `uloss.write_isosurfaces(M)` pushes the learned values
into ordinary `addIsosurface(value=...)`, **oldest-contact-first** (so ascending `lithoID`
= younger), and `predict` works as usual. **`estimate_isosurfaces` is not used on this path.**

---

## 6. Tuning knobs (coupled path)

| knob | where | effect |
|---|---|---|
| `overturn_weight` | `attach_unit_loss(...)` | **the** trade-off: low → better marker fit + lower NLL, but more deep-region inversions in `predict`; high → cleaner volume, looser fit. Default **6** (per-field path uses 30). |
| `cap_per_unit` | `build_geomodel(...)` | distinct points each field sees. **Large/None** on the coupled path (balance comes from `points_per_level`, not the cap). |
| `points_per_level` | `attach_unit_loss(...)` | balanced points/level/epoch (default 1024). |
| `tau` | `attach_unit_loss(...)` | carve sharpness `s=1/τ` (default 0.05). |

See [SOFTUNIT_STATUS.md](SOFTUNIT_STATUS.md) for the measured numbers and open items.
