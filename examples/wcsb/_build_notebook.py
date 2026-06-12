"""
Generator for examples/wcsb/wcsb.ipynb (SPEC §6.2).

Run once to (re)write the notebook. Kept in-repo so the notebook is reproducible
and reviewable as plain source.

wcsb is the complex reference dataset: 36 units (conformal / baselap / eroded)
observed as **unit points only** (`cleaner_markers.vtp` level labels) — no contact
points, no normals. It exercises the unit-only path of the builder, which ports
GeoINR's per-field recipe into Curlew:

  * one neural field per event (Siren or softplus-MLP), assembled by Curlew's
    native predict/combine (oldest→youngest onlap/truncation chain);
  * each field is constrained purely by **point-vs-point inequalities** (`CSet.iq`:
    younger unit points above older ones), consumed by the GeoINR-side
    gradient-normalised inequality loss, residual = (f(P1) - f(P2))/‖∇f(P1)‖ —
    which prevents field collapse;
  * a **no-overturn** regularizer keeps each field monotone in the younging
    direction so the unconformity surfaces nest;
  * surfaces are **estimated post-fit** (`estimate_isosurfaces`: midpoint of the
    adjacent units' median field values) and set as ordinary fixed iso-values;
  * the region-only events (basement / top unit) carry loss-free **aliases** of the
    adjacent unconformity's field (GeoINR's basement treatment), so the deep scalar
    is structured rather than an untrained network;
  * `cap_per_unit` subsampling keeps rare units represented (SPEC §4.4); the per-epoch
    constraint budget matches GeoINR (`IQ_SAMPLES`~1024 ≈ full pools, 5000
    regularization samples).

The notebook is built for **inspection**: it renders the per-field constraints and
the per-field scalar fields (so each field can be judged on its own) as well as the
combined model, and reports per-unit prediction accuracy. Switch `FIELD` between
Siren and the softplus MLP to compare.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# `wcsb` — strat-column → GeoModel (Curlew / GeoINR integration, complex case)

Unit-only path of the GeoINR → Curlew integration (SPEC §6.2): the Western Canada
Sedimentary Basin column (36 units, conformal/baselap/eroded) observed as **unit
points only** (`cleaner_markers.vtp`). `build_geomodel` produces an **alternating
chain** of erosional unconformities and depositional packages with synthesized
region-only events at both ends (Precambrian basement at the bottom; the dropped
top baselap unit at the top), each a neural field constrained by unit ordering.

**Loss (GeoINR's per-field `pairwise=False` recipe, on the field classes):** younger
unit points must sit above older unit points (`CSet.iq`), with the residual
`(f(P1) - f(P2))/‖∇f(P1)‖` normalised by the gradient so the scale-free field cannot
collapse; a **no-overturn** term keeps the field monotone upward. Surfaces are not
learned: after fitting, `estimate_isosurfaces` places each contact/unconformity at the
separation value between the adjacent units' field-value distributions (midpoint of
medians) — that value is the `Overprint` threshold and the extraction iso.

This notebook is built for **inspection**: it renders the per-field constraints and
per-field scalar fields (each field judged on its own), the combined model, and
per-unit accuracy. Pipeline: **load → partition → constraints → fit → per-field fields
→ combined predict/3D → checks**.""")

# ---------------------------------------------------------------- 0. setup
md("## 0. Setup")
code(r"""%matplotlib inline
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import curlew
from curlew.io import loadObservations
from curlew.geology.stratbuilder import build_geomodel, estimate_isosurfaces

DEVICE = os.environ.get("CURLEW_DEVICE", "cuda")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but unavailable; set CURLEW_DEVICE=cpu to run on CPU.")
curlew.device = DEVICE
torch.manual_seed(0); np.random.seed(0)

CANDIDATES = [
    r"D:/Development/GeoINR_Research2/geoinr/data/wcsb",
    os.path.join(os.path.dirname(os.getcwd()), "data", "wcsb"),
    "./data/wcsb",
]
DATA = next((p for p in CANDIDATES if os.path.exists(os.path.join(p, "strat_col_v2.csv"))), None)
assert DATA is not None, f"Could not locate wcsb data; tried {CANDIDATES}"
STRAT_CSV = os.path.join(DATA, "strat_col_v2.csv")

# --- build knobs (reproducible) ---------------------------------------------
# FIELD: 'Siren' (omega 30) or 'GeoINR' (softplus MLP). Compare both.
FIELD = "Siren"
if FIELD == "Siren":
    FIELD_KWARGS = dict(omega0=30.0, omega=30.0, learning_rate=1e-4, hidden_dim=256)
else:  # softplus MLP — beta in ~[20, 50]
    FIELD_KWARGS = dict(activation=nn.Softplus(beta=40), learning_rate=1e-3, hidden_dim=256)
CAP_PER_UNIT = 1500     # cap per unit at load (imbalance-aware, SPEC §4.4)
IQ_SAMPLES   = 1024     # points drawn per inequality pool each step (~full pool at the cap;
                        # GeoINR uses ALL constraint points per epoch — max_points=0)
N_EPOCHS     = 3000     # GeoINR's count; per-field accuracy plateaus around here

# per-field GeoINR loss weights (consumed by the field classes, not HSet).
# NB: GeoINR's original overturn weight (1) does NOT transfer to curlew's sequential
# combine: measured here, weight 1 gives 66.5% of columns an older-above-younger
# inversion (basement leaks through non-monotone deep fields) vs 0.1% at weight 30,
# for only +0.014 band accuracy. Lower it only if you also change the combine.
IQ_WEIGHT       = 1.0   # gradient-normalised inequality (GeoINR's unit weight)
OVERTURN_WEIGHT = 30.0  # no-overturn magnitude
FIELD_KWARGS.update(iq_norm_weight=IQ_WEIGHT, overturn_weight=OVERTURN_WEIGHT)
print("device:", curlew.device, "| field:", FIELD, "| data:", DATA)""")

# ---------------------------------------------------------------- 1. load
md(r"""## 1. Load & normalize

Read the strat column and the **unit** observations. `build_geomodel` caps each unit
at `CAP_PER_UNIT` points, builds the event chain, and attaches an **anisotropic**
(`geo_extensive`) normalization `Transform` (the basin's xy extent is ~300× its thin
z, so z is vertically exaggerated into a comparable model range).""")
code(r"""obs = loadObservations(units=os.path.join(DATA, "cleaner_markers.vtp"))
print("observations:", len(obs), "unit points |", obs.ndim, "D |",
      f"{len(obs.levels('unit'))} levels {obs.levels('unit')[0]}..{obs.levels('unit')[-1]}")

M = build_geomodel(STRAT_CSV, obs, field=FIELD, scale="geo_extensive",
                   cap_per_unit=CAP_PER_UNIT, iq_samples=IQ_SAMPLES, seed=0,
                   field_kwargs=FIELD_KWARGS)
T = M.normalization""")
code(r"""try:
    import pandas as pd
    display(pd.DataFrame([{"index": i, "name": u.name, "relation": u.relation, "level": u.level}
                          for i, u in enumerate(M.strat_units)]))
except Exception:
    for i, u in enumerate(M.strat_units):
        print(i, u.name, u.relation, u.level)

import itertools
lo, hi = obs.bounds(); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
mc = T.apply(np.array(list(itertools.product(*zip(lo, hi))), dtype=float))
for ax, nm in enumerate("xyz"[:obs.ndim]):
    print(f"  {nm}: world extent {hi[ax]-lo[ax]:11.1f}  →  model [{mc[:,ax].min():6.3f}, {mc[:,ax].max():6.3f}]")""")

# ---------------------------------------------------------------- 2. partition
md(r"""## 2. Scalar-field partitioning (visual)

The builder reverses the partitioner to oldest→youngest and emits an alternating
chain: a region-only **basement**, then for each unconformity an **erosional** event
(`mode='above'`) followed by the **depositional** package that **onlaps** it, capped by
a region-only event for the dropped top unit. The two region-only events carry no
constraints of their own — their fields are **aliases** of the adjacent unconformity's
field (basement = the Precambrian Unconformity field continued below its iso; top unit =
the youngest unconformity's field above its iso). Each colour is one event; erosional
events own no lithology (they are surfaces covered by the onlapping package).""")
code(r"""print("derived events (oldest → youngest):")
for ev, meta in zip(M.events, M.field_meta):
    extra = (f"iso between L{min(meta.above_levels)}..{max(meta.above_levels)} / L{meta.eroded_level}"
             if meta.is_unconformity else (f"levels={meta.levels}" if meta.levels else ""))
    print(f"  eid={ev.eid:2d}  {meta.kind:11s}  onlap={ev.overprint.defaultDomain:6s}  "
          f"n_iq={meta.n_iq:2d}  {meta.name[:34]:34s}  {extra}")

units = M.strat_units; n = len(units)
ev_of_level = {L: k for k, meta in enumerate(M.field_meta) for L in meta.levels}
cmap = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(4.4, 8))
for i, u in enumerate(units):
    y = n - 1 - i; k = ev_of_level.get(u.level, -1)
    ax.add_patch(plt.Rectangle((0, y), 1, 1, facecolor=cmap(k % 20) if k >= 0 else "lightgray", edgecolor="k"))
    ax.text(0.5, y + 0.5, f"L{u.level} {u.name[:32]} ({u.relation})", ha="center", va="center", fontsize=6)
ax.set_xlim(0, 1); ax.set_ylim(0, n); ax.axis("off"); ax.set_title("Strat column → owning event")
plt.tight_layout(); plt.show()""")
md("**GeoModel event graph** (`_repr_svg_`):")
code(r"""M""")

# ---------------------------------------------------------------- 3. constraints
md(r"""## 3. Constraint relations (per field)

There are **no** `eq` traces or `gv` normals — each field is driven purely by
**point-vs-point inequalities** (`CSet.iq`): for a depositional package, each band's
points sit above the band directly below; for an unconformity, every level of the
onlapping (adjacent-younger) package sits above the eroded unit **and every older
level** (the explicit "everything older" side is what keeps the deep unconstrained
region — the basement — below the surface; the no-overturn term alone is too weak
there). The cell lists, **per field**, the exact level orderings recorded at build
time. No iso-values exist yet — they are estimated after fitting (§4).""")
code(r"""# exact per-field constraints recorded at build time (meta.iq_relations: list of
# (above, below_level, '>') where `above` is one level, or a tuple of levels for the
# pooled above side of an unconformity). Each says "<above> points sit above <below> points".
def lab(a):  # an above side is a single level or a pooled tuple of levels
    return "+".join(f"L{x}" for x in a) if isinstance(a, tuple) else f"L{a}"
print("per-field iq constraints  (above  >  below) :")
for ev, meta in zip(M.events, M.field_meta):
    if not meta.iq_relations:
        print(f"  {meta.name[:34]:34s} [{meta.kind}] : (no constraints)")
        continue
    by_above = {}   # above (level or tuple) -> [below_levels]
    for a, b, _ in meta.iq_relations:
        by_above.setdefault(a, []).append(b)
    print(f"  {meta.name[:34]:34s} [{meta.kind}]  ({meta.n_iq} pairs)")
    for a in by_above:
        bs = sorted(by_above[a])
        rng_s = f"L{bs[0]}..L{bs[-1]}" if len(bs) > 2 else ",".join(f"L{b}" for b in bs)
        print(f"      {lab(a)}  >  {rng_s}")""")

# ---------------------------------------------------------------- 4. fit
md(r"""## 4. Fit & post-hoc surface estimation

`M.fit(history=True)` trains all fields jointly and returns a per-epoch loss
breakdown. The unit-only path learns **no** surfaces during the fit — afterwards,
`estimate_isosurfaces(M)` places each contact/unconformity at the **midpoint of the
median field values** of the two unit populations it separates and sets it as an
ordinary fixed iso (this must run before `M.predict`). Below the loss curve we
confirm **every unit** appears in ≥1 inequality (so it is resampled — `IQ_SAMPLES`
points — every step; SPEC §4.4); the constraints come straight from
`meta.iq_relations`, so the count is exact.""")
code(r"""loss, pebble, history = M.fit(N_EPOCHS, early_stop=None, best=False, history=True)
print("final loss:", loss); print(pebble)

isos = estimate_isosurfaces(M)     # REQUIRED on the unit-only path (before any predict)
print(f"\nestimated {len(isos)} surface iso-values (midpoint-of-medians):")
for k, v in isos.items():
    print(f"  {k[:48]:48s} {v:+.3f}")

losses = [p.total() for p in history]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.4))
ax[0].plot(np.arange(1, len(losses) + 1), losses, lw=1); ax[0].set_yscale("log")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("total loss (log)"); ax[0].grid(True, alpha=0.3)
ax[0].set_title(f"wcsb training loss ({FIELD})")

# per-unit representation: # inequality pairs each level appears in (either side;
# an erosional pair's above side is a pooled tuple of levels)
rep = {L: 0 for L in obs.levels("unit")}
for meta in M.field_meta:
    for a, b, _ in meta.iq_relations:
        for x in (a if isinstance(a, tuple) else (a,)):
            rep[x] += 1
        rep[b] += 1
levels_sorted = sorted(rep)
ax[1].bar([str(L) for L in levels_sorted], [rep[L] for L in levels_sorted])
ax[1].set_xlabel("unit level"); ax[1].set_ylabel("# iq pairs"); ax[1].tick_params(axis="x", labelrotation=90, labelsize=6)
ax[1].set_title(f"per-unit representation (min={min(rep.values())})")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- 5. per-field fields
md(r"""## 5. Per-field scalar fields (inspection)

Each event's field, evaluated **on its own** (`combine=False`) on a vertical section,
with its estimated iso-surface(s) contoured. This shows whether each field is
*individually* geologically reasonable (monotone, surface at the right level) —
independently of how the sequential combine assembles them. (Note: a "turn-up" at the
west edge is extrapolation beyond the data plus the real westward basin-margin rise,
not a fitting bug.)""")
code(r"""lo, hi = obs.bounds(); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
center = 0.5 * (lo + hi)
nx, nz = 200, 130
xs = np.linspace(lo[0], hi[0], nx); zs = np.linspace(lo[2], hi[2], nz)
XX, ZZ = np.meshgrid(xs, zs); YY = np.full_like(XX, center[1])
SEC = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3).astype(float)

# pick a few representative events spanning the column
sel = [ev for ev in M.events if ev.field.C is not None][::2][:6]
fig, axes = plt.subplots(2, 3, figsize=(15, 7)); axes = axes.ravel()
for ax, ev in zip(axes, sel):
    # NB: GeoEvent.predict works in MODEL coords (it does not apply the global→model
    # transform M.T that GeoModel.predict applies). Map the world section into model
    # space first, else the field is evaluated at raw world coordinates (garbage).
    g = ev.predict(T.apply(SEC), combine=False)   # this field only, model coords
    S = g.scalar.reshape(nz, nx)
    pcm = ax.pcolormesh(xs, zs, S, shading="auto", cmap="viridis")
    isov = sorted(ev.getIsovalues().values())
    if isov:
        ax.contour(xs, zs, S, levels=isov, colors="k", linewidths=1.0)
    ax.set_title(ev.name[:30], fontsize=8); ax.set_xlabel("x"); ax.set_ylabel("z")
for ax in axes[len(sel):]: ax.axis("off")
fig.suptitle(f"per-field scalar fields ({FIELD}) — black = estimated iso-surfaces")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- 6. combined / 3D
md(r"""## 6. Combined model — predict & 3D view

`M.predict` assembles all fields via the onlap/truncation combine. The matplotlib
section shows the combined scalar + lithology; the optional napari view shows the
lithology/structure volumes, the **observation points** (coloured by their true unit so
you can judge the fit directly), and the extracted surfaces. Comparing this with §5 shows
where the **combine** (not the individual fields) places unit boundaries.

The predict grid is **high-resolution** (5 km horizontal, 20 m vertical — `predict`
chunks it by `curlew.batchSize`), and the napari viewer applies a **vertical
exaggeration** (the basin is ~300× wider than it is deep) so the layering is visible
rather than squashed onto one plane.""")
code(r"""from curlew.geometry import Grid
RES_XY, RES_Z = 5000.0, 20.0
curlew.batchSize = 50000       # predict is chunked by this; lower if you hit GPU OOM, raise for speed
ext = hi - lo
G = Grid(dims=tuple(ext), step=(RES_XY, RES_XY, RES_Z), center=tuple(center))
print(f"grid cells: {int(np.prod(G.shape)):,}  shape {G.shape}  (RES_XY={RES_XY:.0f} m, RES_Z={RES_Z:.0f} m)")
geode = M.predict(G)
print("grid", geode.grid.shape, "| scalar range", float(np.nanmin(geode.scalar)), "→", float(np.nanmax(geode.scalar)),
      "| lithologies", len(np.unique(geode.lithoID)))

sec = M.predict(SEC)
fig, ax = plt.subplots(2, 1, figsize=(12, 7))
pcm = ax[0].pcolormesh(xs, zs, sec.scalar.reshape(nz, nx), shading="auto", cmap="viridis")
fig.colorbar(pcm, ax=ax[0]); ax[0].set_title("combined scalar field (x–z @ mid-y)")
pcm2 = ax[1].pcolormesh(xs, zs, sec.lithoID.reshape(nz, nx), shading="auto", cmap="tab20")
fig.colorbar(pcm2, ax=ax[1]); ax[1].set_title("lithology ID (combined)")
for a in ax: a.set_xlabel("x"); a.set_ylabel("z")
plt.tight_layout(); plt.show()""")
code(r"""SHOW_NAPARI = True
if SHOW_NAPARI:
    from curlew.visualise.napari_viewer import NapariViewer
    from matplotlib.colors import Normalize

    # vertical_exaggeration stretches z on every layer (the basin is ~300x wider than deep)
    nv = NapariViewer(title=f"wcsb ({FIELD})", ndisplay=3, vertical_exaggeration=100.0)

    # lithology / structure volumes (pin contrast limits so the points below colour-match)
    idmin, idmax = float(geode.lithoID.min()), float(geode.lithoID.max())
    nv.addVolume("lithology", G.reshape(geode.lithoID.astype(float)), grid=G,
                 rendering="attenuated_mip", colormap=curlew.ccstrat,
                 contrast_limits=(idmin, idmax))
    nv.addVolume("structure", G.reshape(geode.structureID.astype(float)), grid=G,
                 rendering="additive", blending="additive", colormap=curlew.ccstrat, opacity=0.35)

    # --- observation points: the data the model was fit to -------------------------------
    # Coloured by their TRUE unit lithology with the SAME colormap + contrast limits as the
    # lithology volume, so a well-fit model shows each point sitting in volume of the same
    # colour (mismatched colours flag where the model misfits the data). Subsampled for
    # responsiveness — raise N_SHOW to plot more; adjust `size` (world metres) to taste.
    N_SHOW = 150000
    rng = np.random.default_rng(0)
    pidx = rng.choice(len(obs.coords), size=min(N_SHOW, len(obs.coords)), replace=False)
    Pworld = obs.coords[pidx].astype(float)
    true_id = np.array([M.llookup.get(M.level_litho.get(int(L)), 0) for L in obs.level[pidx]], float)
    pt_rgba = curlew.ccstrat(Normalize(vmin=idmin, vmax=idmax)(true_id))
    nv.addPoints("observations (true unit)", Pworld, rgb=pt_rgba,
                 size=3 * RES_XY, border_color="black")

    # per-event estimated iso-surfaces
    nC = sum(len(e.isosurfaces) for e in M.events) or 1; j = 0
    for e in M.events:
        if not e.isosurfaces: continue
        mask = geode.structureID == e.eid
        if mask.sum() == 0: continue
        for name, iso in e.getIsovalues().items():
            try:
                verts, faces = G.contour(geode.fields[e.name], iso=iso, mask=mask, erodeMask=-2)
                if len(verts): nv.addMesh(f"{e.name[:18]}:{name[:14]}", verts=verts, faces=faces,
                                          rgb=np.asarray(curlew.ccramp(j / nC)))
            except Exception as exc:
                print("skip", e.name, name, exc)
            j += 1
    nv.show()
    print("napari: lithology + structure volumes + observation points + per-event iso-surfaces")""")
md(r"""### Export for ParaView

Write the predicted volume to a VTK file for detailed inspection (`curlew.io.saveVTK`;
`.vti` since the grid is axis-aligned — use `.vts` for rotated grids). Two unit arrays
are written:

- **`level`** — the predicted unit's stratigraphic level, *the same convention as the
  observation data* (ascending = older; basement = max level; `-1` where the predicted
  lithology is not a unit). This is the array to colour by / compare against the data.
- **`lithoID`** — curlew's internal lithology id (ascending = younger; ids are assigned
  oldest→youngest with gaps for surface-only events). The id→name legend is embedded as
  the `lithoID_legend` field-data array (Spreadsheet view → Field Data) and printed below.

Note the napari colormap (`curlew.ccstrat`) is a deliberately *shuffled* ramp (so thin
adjacent bands stay distinguishable) — apparent out-of-sequence colours in napari are
not evidence of out-of-sequence units; check `level` here instead.""")
code(r"""from curlew.io import saveVTK
VTK_PATH = "wcsb_prediction.vti"

# stratigraphic level per voxel (the data's own convention: ascending = older)
lut = np.full(max(M.llookup.values()) + 1, -1, dtype=np.int32)
for L, key in M.level_litho.items():
    lut[M.llookup[key]] = L
saveVTK(VTK_PATH, geode, extra={"level": lut[geode.lithoID]})
print("wrote", VTK_PATH, f"({np.prod(G.shape):,} cells)")
print("\nlithoID legend (id -> unit / band):")
for i, n in sorted(geode.lithoLookup.items()):
    print(f"  {i:3d}  {n}")""")

# ---------------------------------------------------------------- 7. checks
md(r"""## 7. Checks & diagnostics (SPEC §6.2)

The **structural** checks (event chain, per-unit representation) verify the
SPEC §4.2/§4.4/§8.1 builder work and are asserted. The **fit-quality** diagnostics
(inequality satisfaction, iso placement, per-package iso ordering) and the
**per-point label accuracy** (lithology band / structure / basement) are *reported*
for inspection — with this per-field model + the sequential combine the accuracy is
bounded (key points near unconformities are mislabelled, as expected for the
non-coupled recipe); the soft-unit/NLL coupling is the path to higher per-point
accuracy.""")
code(r"""metas = M.field_meta
def fvals(ev, pts):
    with torch.no_grad():
        return ev.forward(torch.tensor(np.asarray(pts, float), dtype=curlew.dtype,
                                       device=curlew.device)).detach().cpu().numpy().reshape(-1)

results = {}
# 1. event chain
kinds = [m.kind for m in metas]
results["event_chain"] = bool(kinds[0] == "region-only" and kinds[-1] == "region-only"
                              and sum(m.is_unconformity for m in metas) >= 1
                              and sum(m.kind == "depositional" for m in metas) >= 1)
print(f"1. event chain: {len(metas)} events "
      f"({sum(k=='region-only' for k in kinds)} region-only, {sum(m.is_unconformity for m in metas)} erosional, "
      f"{sum(k=='depositional' for k in kinds)} depositional) -> {results['event_chain']}")

# 2. every unit represented
results["every_unit_represented"] = bool(min(rep.values()) > 0)
print(f"2. every unit in >=1 iq pair: min={min(rep.values())} -> {results['every_unit_represented']}")

# 3. inequality satisfaction on unit points (fraction of cross-pairs correctly ordered)
# Field values per (event, level) from the builder's capped model-coord pools.
sat = []
for meta in metas:
    if not meta.iq_relations: continue
    ev = M[meta.name]
    lv_set = set()
    for a, b, _ in meta.iq_relations:
        lv_set.update(a if isinstance(a, tuple) else (a,)); lv_set.add(b)
    vals = {L: fvals(ev, M.level_points[L]) for L in lv_set}
    for a, b, _ in meta.iq_relations:
        va = np.concatenate([vals[x] for x in (a if isinstance(a, tuple) else (a,))])
        sat.append(float(np.mean(va[:, None] > vals[b][None, :])))
iq_sat = float(np.mean(sat))
results["constraint_satisfaction"] = bool(iq_sat > 0.9)
print(f"3. inequality satisfaction (mean over pairs): {iq_sat:.3f} -> {results['constraint_satisfaction']}")

# 4. estimated unconformity iso between the adjacent clouds (medians, matching the estimator)
between = True
for meta in metas:
    if not meta.is_unconformity: continue
    ev = M[meta.name]
    iso = ev.getIsovalue(meta.iso_name)
    a = np.median(fvals(ev, np.concatenate([M.level_points[L] for L in meta.above_levels
                                            if len(M.level_points.get(L, ())) > 0])))
    b = np.median(fvals(ev, M.level_points[meta.eroded_level]))
    between &= bool(b < iso < a)
results["iso_between"] = bool(between)
print(f"4. each unconformity iso between adjacent clouds -> {results['iso_between']}")

# 5. monotonic isos per depositional field (younger contact above older)
mono = True
for meta in metas:
    if meta.kind != "depositional" or len(meta.contacts) < 2: continue
    ev = M[meta.name]
    iv = np.array([ev.getIsovalue(c.name) for c in meta.contacts]); d = np.diff(iv)
    mono &= bool(np.all(d < 0))
results["iso_monotonic"] = bool(mono)
print(f"5. per-package iso monotonicity -> {results['iso_monotonic']}")

print("\n" + "=" * 48)
# assert only the structural builder invariants; report the fit-quality diagnostics
# (satisfaction, iso ordering) — these depend on the fit and the post-hoc estimates.
STRUCTURAL = ("event_chain", "every_unit_represented")
for k, v in results.items():
    print(f"  {('PASS' if v else ('FAIL' if k in STRUCTURAL else 'report')):6s} {k}")
assert all(results[k] for k in STRUCTURAL), "A structural (builder) check failed."
print("=" * 48)""")

code(r"""# --- reported per-point accuracy (not asserted: per-field ceiling) ---
rng = np.random.default_rng(3); pts, lev = [], []
for L in obs.levels("unit"):
    idx = np.where((~obs.is_interface) & (obs.level == L))[0]
    idx = rng.choice(idx, size=min(400, len(idx)), replace=False)
    pts.append(obs.coords[idx].astype(float)); lev.append(np.full(len(idx), L))
pts = np.vstack(pts); lev = np.concatenate(lev); g = M.predict(pts)
litho = np.array([g.lithoLookup.get(int(i), "?") for i in g.lithoID])
struct = np.array([g.structureLookup.get(int(s), "?") for s in g.structureID])
lev_struct = {L: m.name for m in metas for L in m.levels}
lit_ok = np.array([litho[k] == M.level_litho[int(lev[k])] for k in range(len(lev))])
str_ok = np.array([struct[k] == lev_struct[int(lev[k])] for k in range(len(lev))])
base_level = max(obs.levels("unit"))
print(f"label (band) accuracy   : {lit_ok.mean():.3f}")
print(f"structure (package) acc : {str_ok.mean():.3f}")
print(f"basement (L{base_level}) accuracy : {lit_ok[lev == base_level].mean():.3f}")
print("\nper-level structure accuracy (for inspection):")
for L in obs.levels("unit"):
    m = lev == L
    print(f"  L{L:2d} {M.strat_units[[u.level for u in M.strat_units].index(L)].name[:30]:30s} "
          f"struct={str_ok[m].mean():.2f} band={lit_ok[m].mean():.2f}")""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = Path(__file__).with_name("wcsb.ipynb")
nbf.write(nb, str(out))
print("wrote", out)
