"""
Generator for examples/multilayer_fold/multilayer_fold.ipynb (SPEC §6.1).

Run once to (re)write the notebook. Kept in-repo so the notebook is reproducible
and reviewable as plain source. The notebook itself is the deliverable.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md(r"""# `multilayer_fold` — strat-column → GeoModel (Curlew / GeoINR integration)

End-to-end verification of the depositional path of the GeoINR → Curlew
integration (SPEC §6.1). A six-unit conformable sequence (A–F) is parsed from a
stratigraphic-column CSV, partitioned into a **single depositional scalar field**
with five internal contacts, fitted with a **Siren** neural field, and checked.

Pipeline: **load & normalize → partition (visual) → constraints (visual) → fit →
predict & 3D view → checks**.

The builder (`curlew.geology.stratbuilder.build_geomodel`) and observation loader
(`curlew.io.loadObservations`) are the newly ported pieces; everything else reuses
Curlew's native events / `CSet` constraints / seed isosurfaces / `Transform`.""")

# ---------------------------------------------------------------- 0. setup
md("## 0. Setup")
code(r"""%matplotlib inline
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import curlew
from curlew.io import loadObservations
from curlew.geology.stratbuilder import build_geomodel

# --- compute on the GPU -----------------------------------------------------
# This notebook runs on the GPU. Override with the env var CURLEW_DEVICE=cpu
# only for headless/CI execution on a machine without a CUDA device.
DEVICE = os.environ.get("CURLEW_DEVICE", "cuda")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU requested but torch.cuda.is_available() is False. "
        "Install a CUDA-enabled torch / driver, or set CURLEW_DEVICE=cpu to run on CPU."
    )
curlew.device = DEVICE
# float64 is Curlew's default; drop to float32 to save GPU memory if needed.
# curlew.dtype = torch.float32

# deterministic run (network initialisation)
torch.manual_seed(0)
np.random.seed(0)

# locate the GeoINR reference dataset
CANDIDATES = [
    r"D:/Development/GeoINR_Research2/geoinr/data/multilayer_fold",
    os.path.join(os.path.dirname(os.getcwd()), "data", "multilayer_fold"),
    "./data/multilayer_fold",
]
DATA = next((p for p in CANDIDATES if os.path.exists(os.path.join(p, "strat_col.csv"))), None)
assert DATA is not None, f"Could not locate multilayer_fold data; tried {CANDIDATES}"
STRAT_CSV = os.path.join(DATA, "strat_col.csv")
print("device:", curlew.device, "| dtype:", curlew.dtype)
print("data:", DATA)""")

# ---------------------------------------------------------------- 1. load
md(r"""## 1. Load & normalize

Read the strat column and the point observations (`interface.vtp` contact points
+ `normal.vtp` oriented normals) and build the model on **all** observations — a
single curlew model is fit to all available data (no train/validation holdout).
`build_geomodel` computes the data bounds and attaches an **isometric**
normalization `Transform` into `[-1, 1]`.""")
code(r"""obs = loadObservations(
    interfaces=os.path.join(DATA, "interface.vtp"),
    normals=os.path.join(DATA, "normal.vtp"),
)
print("observations:", len(obs), "points |", obs.ndim, "D")
print("interface points:", obs.interface_points()[0].shape[0],
      "across levels", obs.levels("interface"))
print("normal points:", obs.normal_points()[0].shape[0])

# Siren hyperparameters are forwarded to the field's `initField` via `field_kwargs`.
# `omega0`/`omega` set the sine-layer frequencies (lower = smoother field);
# learning_rate=1e-4 is the Siren default.
SIREN_KWARGS = dict(omega0=2.0, omega=30.0, learning_rate=1e-4, hidden_dim=256)
M = build_geomodel(STRAT_CSV, obs, field="Siren", scale="isometric",
                   field_kwargs=SIREN_KWARGS)
T = M.normalization""")

code(r"""# parsed stratigraphic column (youngest-first, as in the CSV)
try:
    import pandas as pd
    col = pd.DataFrame([
        {"index": i, "name": u.name, "relation": u.relation, "level": u.level}
        for i, u in enumerate(M.strat_units)
    ])
    display(col)
except Exception:
    for i, u in enumerate(M.strat_units):
        print(i, u.name, u.relation, u.level)

print("\nnormalization Transform (global → model):")
print(np.array2string(T.matrix, precision=4, suppress_small=True))""")

md(r"""**Reading the transform.** It is a single homogeneous *scale + translate* that
maps world coordinates into the model box. For `scale="isometric"` the diagonal is
one shared factor `s = 2 / max(extent_x, extent_y, extent_z)` (here `≈0.0083`), and
the translation re-centres the data on the origin. So the **longest** axis is mapped
to exactly `[-1, 1]` (`extent · s = 2`), and the shorter axes map to proportionally
**smaller** sub-ranges (they keep the same units, so the model stays undistorted).
`scale="geo_extensive"` instead uses a separate xy and z factor (vertical
exaggeration). The cell below prints the actual per-axis ranges after normalization.""")
code(r"""lo, hi = obs.bounds()
lo = np.asarray(lo, float); hi = np.asarray(hi, float)
# transform the 8 bounding-box corners into model space
import itertools
corners = np.array(list(itertools.product(*zip(lo, hi))), dtype=float)
mc = T.apply(corners)
for ax, name in enumerate("xyz"[:obs.ndim]):
    print(f"  {name}: world [{lo[ax]:8.2f}, {hi[ax]:8.2f}]  (extent {hi[ax]-lo[ax]:7.2f})"
          f"   →  model [{mc[:,ax].min():6.3f}, {mc[:,ax].max():6.3f}]")
print("\nLongest world axis maps to the full [-1, 1]; shorter axes stay proportionally smaller.")""")

# ---------------------------------------------------------------- 2. partitioning
md(r"""## 2. Scalar-field partitioning (visual)

How the six units are partitioned into Curlew events. All units are `conformal`,
so the partitioner emits **one depositional field** carrying five internal
contacts (the tops of B–F). The diagram is oldest→youngest.""")
code(r"""print("derived scalar fields (oldest → youngest):")
for meta in M.field_meta:
    cs = sorted(meta.contacts, key=lambda c: c.unit_index, reverse=True)  # oldest→youngest
    print(f"  {meta.name}  kind={meta.kind}  units={meta.unit_indices}")
    for c in cs:
        print(f"     contact '{c.name}'  level={c.level}  n_points={c.n_points}")

# stratigraphic-column diagram, colour-coded by scalar field
units = M.strat_units
n = len(units)
field_of = {ui: k for k, meta in enumerate(M.field_meta) for ui in meta.unit_indices}
cmap = plt.get_cmap("tab10")
fig, ax = plt.subplots(figsize=(3.4, 5))
for i, u in enumerate(units):              # i=0 youngest at top
    y = n - 1 - i
    fid = field_of.get(i, -1)
    ax.add_patch(plt.Rectangle((0, y), 1, 1,
                 facecolor=cmap(fid % 10) if fid >= 0 else "lightgray",
                 edgecolor="k"))
    ax.text(0.5, y + 0.5, f"{u.name}  (lvl {u.level}, {u.relation})",
            ha="center", va="center", fontsize=8)
ax.set_xlim(0, 1); ax.set_ylim(0, n); ax.axis("off")
ax.set_title("Strat column → scalar field\n(colour = Curlew event)")
plt.tight_layout(); plt.show()""")

md("**GeoModel event graph** (`_repr_svg_`):")
code(r"""M""")

# ---------------------------------------------------------------- 3. constraints
md(r"""## 3. Constraint relations (visual)

The constraints that wire the geological features together. They are plotted here
in **world (global) coordinates** for clarity:

- **`eq` traces / seed isosurfaces** — interface points coloured by contact (each
  contact is one equal-value level set + one seed isosurface), with the **`gv`**
  bedding normals (younging direction) drawn as arrows;
- **`iq` inequalities** — one per contact pair (`C(5,2) = 10`), each enforcing the
  younger contact's field value `>` the older contact's (the reverse `<` is then
  automatic, so only one direction is stored). The `iq pools` count is printed below.

With a GeoINR-family field (as here) both are consumed by GeoINR's
**gradient-normalised** losses on the field itself — interface: `|Δf|/‖∇f‖` between
points of the same contact; inequality: the hinged `(f(P1)−f(P2))/‖∇f(P1)‖` — so the
residuals behave like distances and the field cannot shrink its output range to
satisfy them trivially. Only the bedding-normal (`grad_loss`) term uses the generic
`HSet`.

**On the normals.** The builder stores constraints in *model* coordinates, but the
normalization here is **isometric** (a uniform scale + translate), which **preserves
directions** — so the bedding-normal vectors are identical in world and model space
(only their attachment points are rescaled). The model uses the raw unit normals as
`gv`; nothing rotates them. (Under an *anisotropic* `geo_extensive` scaling the
direction would change and would need the inverse-transpose mapping — but
`multilayer_fold` is isometric and `wcsb` carries no normals.)""")
code(r"""ev = M.events[0]
C = ev.field.C            # bound CSet (model coords)
contacts = sorted(M.field_meta[0].contacts, key=lambda c: c.unit_index, reverse=True)
nco, nvec, _ = obs.normal_points()
nvec_u = nvec / np.linalg.norm(nvec, axis=1, keepdims=True)
arrow_len = 0.12 * float(np.max(hi - lo))   # ~12% of the largest extent

# eq traces (interface points by contact) + gv normals — world coordinates
fig = plt.figure(figsize=(6.5, 5.5))
ax = fig.add_subplot(projection="3d")
for k, c in enumerate(contacts):
    p = obs.coords[obs.is_interface & (obs.level == c.level)]
    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=16, color=plt.get_cmap("viridis")(k/4),
               label=f"{c.name} (lvl {c.level})")
ax.quiver(nco[:, 0], nco[:, 1], nco[:, 2],
          nvec_u[:, 0], nvec_u[:, 1], nvec_u[:, 2],
          length=arrow_len, color="crimson", linewidth=0.7, normalize=True,
          label="bedding normals (gv)")
ax.set_title("eq traces (by contact) + gv normals  [world coords]")
ax.legend(fontsize=7, loc="upper left")
ax.set_box_aspect((hi[0]-lo[0], hi[1]-lo[1], hi[2]-lo[2]))
ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
plt.tight_layout(); plt.show()

n_iq = 0 if C.iq is None else len(C.iq[1])
print(f"CSet: eq traces={0 if C.eq is None else len(C.eq)}  "
      f"gv normals={0 if C.gp is None else C.gp.shape[0]}  iq pools={n_iq}")
print("HSet (generic terms):", {k: getattr(ev.field.H, k) for k in
      ['eq_loss', 'grad_loss', 'iq_loss']})
print("GeoINR loss weights:", {k: getattr(ev.field, k) for k in
      ['eq_norm_weight', 'iq_norm_weight', 'overturn_weight']})""")


# ---------------------------------------------------------------- 4. fit
md(r"""## 4. Fit

Train the Siren field on all bound constraints with a single `M.fit(...)` call.
`M.fit` returns the final loss and a `Pebble` (the final per-term breakdown). Passing
`history=True` additionally returns one detached `Pebble` **per epoch** — that's the
loss curve. `M.fit` shows a `tqdm` progress bar while it runs (`vb=True` by default).""")
code(r"""N_EPOCHS = 2000
loss, pebble, history = M.fit(N_EPOCHS, early_stop=None, best=False, history=True)
print("final loss:", loss)
print(pebble)                                   # final per-term breakdown

# history is a list of per-epoch Pebbles; total() gives each epoch's loss
losses = [p.total() for p in history]
plt.figure(figsize=(6, 3.2))
plt.plot(np.arange(1, len(losses) + 1), losses, lw=1)
plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("total loss (log)")
plt.title("multilayer_fold — training loss"); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- 5. predict / 3D
md(r"""## 5. Predict & 3D view (WebGL HTML viewer)

Evaluate the fitted model on a regular grid → `Geode`, then build a **self-contained
WebGL HTML viewer** (`curlew.visualise.html_viewer`) and open it in the browser. The
viewer renders the predicted **lithology volume** (boundary faces between units), one
extracted **iso-surface mesh per contact** (`G.contour`), the **interface observation
points**, and the **bedding normals** as toggleable vectors — all sharing one
unit-visibility mask, palette, vertical-exaggeration and X/Y/Z slicer. The point and
normal layers are derived straight from the `Observations` object by
`write_geomodel_viewer(obs=...)`. This model is small, so the *self-contained* mode embeds
all geometry into a single `.html` (no server needed; opens straight from `file://`).

The file is written into `outputs/` (git-ignored). `OPEN_BROWSER` is on by default; set
it `False` for headless/CI runs. A matplotlib cross-section follows as a static fallback
so the notebook still renders something in a non-interactive context.""")
code(r"""from curlew.geometry import Grid
lo, hi = obs.bounds()
lo = np.asarray(lo, float); hi = np.asarray(hi, float)
ext = hi - lo
center = 0.5 * (lo + hi)
G = Grid(dims=tuple(ext), step=tuple(ext / 48.0), center=tuple(center))
geode = M.predict(G)
print("grid shape:", geode.grid.shape, "| scalar range:",
      float(np.nanmin(geode.scalar)), "→", float(np.nanmax(geode.scalar)))""")

code(r"""from pathlib import Path
from curlew.visualise.html_viewer import write_geomodel_viewer, serve_and_open

OUT = Path("outputs"); OUT.mkdir(exist_ok=True)

html = write_geomodel_viewer(
    OUT / "multilayer_fold_viewer.html", M, geode, G,
    obs=obs, color_by="lithoID", erode_mask=-4,
    mode="self_contained", title="multilayer_fold", initial_z_exaggeration=1,
)
print("wrote", html)
serve_and_open(html)   # opens the .html directly in the default browser (file://)""")

code(r"""# static matplotlib cross-section (y–z plane at mid-x), with contact iso-contours
ny = nz = 160
ys = np.linspace(lo[1], hi[1], ny)
zs = np.linspace(lo[2], hi[2], nz)
YY, ZZ = np.meshgrid(ys, zs)
XX = np.full_like(YY, center[0])
pts = np.stack([XX, YY, ZZ], axis=-1).reshape(-1, 3).astype(float)
sec = M.predict(pts)
S = sec.scalar.reshape(nz, ny)
isovals = [ev.getIsovalue(c.name) for c in contacts]

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
pcm = ax[0].pcolormesh(ys, zs, S, shading="auto", cmap="viridis")
ax[0].contour(ys, zs, S, levels=sorted(isovals), colors="k", linewidths=1.2)
fig.colorbar(pcm, ax=ax[0]); ax[0].set_title("scalar field (y–z @ mid-x)\nblack = contact iso-levels")
ax[0].set_xlabel("y"); ax[0].set_ylabel("z")
L = sec.lithoID.reshape(nz, ny)
pcm2 = ax[1].pcolormesh(ys, zs, L, shading="auto", cmap="tab20")
fig.colorbar(pcm2, ax=ax[1]); ax[1].set_title("lithology ID (stratigraphic units)")
ax[1].set_xlabel("y"); ax[1].set_ylabel("z")
plt.tight_layout(); plt.show()""")

# ---------------------------------------------------------------- 6. checks
md(r"""## 6. Checks (SPEC §6.1)

1. The five recovered iso-values (from seeds) are **strictly monotonic** in
   stratigraphic order.
2. At the interface points, the predicted scalar ≈ the contact iso-value
   (each point lands nearest its own contact; mean error small vs contact spacing).
3. Predicted gradient at normal points **aligns** with the supplied normals
   (positive dot / small angular error).""")
code(r"""results = {}

# --- Check 1: monotonic iso-values in stratigraphic order (oldest → youngest) ---
contacts_strat = sorted(M.field_meta[0].contacts, key=lambda c: c.unit_index, reverse=True)
isovals = np.array([ev.getIsovalue(c.name) for c in contacts_strat])
d = np.diff(isovals)
mono = bool(np.all(d > 0) or np.all(d < 0))
print("Check 1 — recovered iso-values (oldest→youngest):",
      np.array2string(isovals, precision=3))
print(f"         strictly monotonic: {mono}")
results["monotonic_iso"] = mono

# --- Check 2: predicted ≈ iso at the (all) interface points the model was fit to ---
min_gap = float(np.min(np.abs(d)))
errs, nearest_ok = [], []
for c in M.field_meta[0].contacts:
    iso = ev.getIsovalue(c.name)
    hp = obs.coords[obs.is_interface & (obs.level == c.level)].astype(float)
    if len(hp) == 0:
        continue
    s = M.predict(hp).scalar
    errs.extend(np.abs(s - iso).tolist())
    nearest = [contacts_strat[int(np.argmin(np.abs(isovals - v)))].name == c.name for v in s]
    nearest_ok.extend(nearest)
errs = np.array(errs)
mean_err = float(errs.mean()); rmse = float(np.sqrt((errs ** 2).mean()))
acc = float(np.mean(nearest_ok))
ok2 = (acc >= 0.9) and (mean_err < 0.5 * min_gap)
print(f"\nCheck 2 — interface points (n={len(errs)}): "
      f"mean|err|={mean_err:.3f}, rmse={rmse:.3f}, min contact gap={min_gap:.3f}")
print(f"         mean error = {100*mean_err/min_gap:.0f}% of min gap; "
      f"nearest-contact accuracy = {acc:.2f}")
print(f"         pass (acc≥0.9 and mean<0.5·gap): {ok2}")
results["interface_predict"] = ok2

# --- Check 3: predicted gradient aligns with supplied normals ---
nco, nvec, _ = obs.normal_points()
nvec = nvec / (np.linalg.norm(nvec, axis=1, keepdims=True) + 1e-12)
g = ev.gradient(T.apply(nco.astype(float)), normalize=True, transform=False)
dots = np.sum(g * nvec, axis=1)
ang = np.degrees(np.arccos(np.clip(dots, -1, 1)))
ok3 = bool(dots.mean() > 0.95 and (dots > 0).mean() == 1.0)
print(f"\nCheck 3 — gradient vs normals: mean dot={dots.mean():.3f}, "
      f"min dot={dots.min():.3f}, mean angle={ang.mean():.2f} deg, frac>0={(dots>0).mean():.2f}")
print(f"         pass (mean dot>0.95, all positive): {ok3}")
results["gradient_alignment"] = ok3

print("\n" + "=" * 48)
print("SPEC §6.1 CHECKS:", "ALL PASS ✅" if all(results.values()) else "FAILURE ❌")
for k, v in results.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
assert all(results.values()), "One or more §6.1 checks failed."
print("=" * 48)""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).with_name("multilayer_fold.ipynb")
nbf.write(nb, str(out))
print("wrote", out)
