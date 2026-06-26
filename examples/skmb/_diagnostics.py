"""
skmb geology diagnostics — quantify the issues SKMB_HANDOFF.md discusses.

Trains the seed-iso model and reports, per unconformity surface:
  * **cut rate** — fraction of surface points predicted as a *much-younger* unit (a younger
    event eroding below an older surface — geologically impossible), split by basin **margin**;
  * **attribution** — on the cut margin points, which event is doing the cutting;
  * **band / structure accuracy** — regression check on unit points.

Usage:  python _diagnostics.py [epochs] [overturn] [iq_samples] [reg_samples]
Set CURLEW_DEVICE=cpu to run on CPU (uses a small net so it is fast but under-fit).
"""
import os
import sys
import collections

import numpy as np
import torch

import curlew
from curlew.io import loadObservations
from curlew.geology.stratbuilder import build_geomodel, attach_unit_loss

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OVT = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0  # seed-iso path wants a stiffer no-overturn
                                                          # prior than wcsb's coupled 6 (the interface
                                                          # eq adds deep-region curvature) -> robust to
                                                          # over-training (vol-inv falls with epochs)
IQ = int(sys.argv[3]) if len(sys.argv) > 3 else 256       # iq samples/pair (4x cheaper than 1024)
REG = int(sys.argv[4]) if len(sys.argv) > 4 else 2000     # model-scaled Poisson overturn (reproducible)

curlew.device = os.environ.get("CURLEW_DEVICE", "cuda")
torch.manual_seed(0); np.random.seed(0)
HID = 32 if curlew.device == "cpu" else 256

DATA = r"D:/Development/GeoINR_Research2/geoinr/data/skmb"
obs = loadObservations(interfaces=DATA + "/markers.vtp", units=DATA + "/units.vtp")
M = build_geomodel(DATA + "/strat_column.csv", obs, field="Siren", scale="geo_extensive",
                   cap_per_unit=8000, iq_samples=IQ, reg_samples=REG, seed=0,
                   field_kwargs=dict(omega0=30.0, omega=30.0, learning_rate=1e-4, hidden_dim=HID))
uloss = attach_unit_loss(M, tau=0.05, points_per_level=1024, overturn_weight=OVT, seed_isos=True)
print(f"epochs={EPOCHS} overturn={OVT} iq_samples={IQ} reg_samples={REG} | events={len(M.field_meta)}")
M.fit(EPOCHS, early_stop=None, best=False, vb=False, custom_loss=[uloss])

imask = obs.is_interface
name2level = {key: L for L, key in M.level_litho.items()}


def predict_levels_and_struct(world_pts):
    g = M.predict(world_pts)
    id2level = {i: name2level.get(n, -1) for i, n in g.lithoLookup.items()}
    lev = np.array([id2level.get(int(i), -1) for i in g.lithoID])
    st = np.array([g.structureLookup.get(int(s), "?") for s in g.structureID])
    return lev, st


print("\nunconformity surface respected / cut (predict at interface points):")
print("  level   respected(>=L-1)      cut(<L-6, much-younger)")
cut_blame = collections.Counter()
for L in sorted([m.eroded_level for m in M.field_meta if m.is_unconformity]):
    w = obs.coords[imask & (obs.level == L)]
    if len(w) == 0:
        continue
    pl, ps = predict_levels_and_struct(w)
    z = w[:, 2]; margin = z > np.quantile(z, 0.80)
    resp = pl >= L - 1; cut = pl < L - 6
    print(f"   L{L:2d}    all {resp.mean()*100:5.1f}% margin {resp[margin].mean()*100:5.1f}%"
          f"   |  all {cut.mean()*100:4.1f}% margin {cut[margin].mean()*100:4.1f}%  (n={len(w)})")
    for nm in ps[margin & cut]:
        cut_blame[nm] += 1
if cut_blame:
    print("\n  events doing the margin cutting (structure on cut points):")
    for nm, c in cut_blame.most_common(8):
        print(f"     {c:4d}  {nm}")

# regression: band / structure accuracy on unit points
rng = np.random.default_rng(3); pts, lev = [], []
for L in [L for L in obs.levels("unit") if L in M.level_litho]:
    ix = np.where((~imask) & (obs.level == L))[0]
    ix = rng.choice(ix, size=min(400, len(ix)), replace=False)
    pts.append(obs.coords[ix].astype(float)); lev.append(np.full(len(ix), L))
pts = np.vstack(pts); lev = np.concatenate(lev); g = M.predict(pts)
litho = np.array([g.lithoLookup.get(int(i), "?") for i in g.lithoID])
struct = np.array([g.structureLookup.get(int(s), "?") for s in g.structureID])
ls = {L: m.name for m in M.field_meta for L in m.levels}
band = np.mean([litho[k] == M.level_litho[int(lev[k])] for k in range(len(lev))])
strA = np.mean([struct[k] == ls[int(lev[k])] for k in range(len(lev))])
print(f"\nband accuracy {band:.3f} | structure accuracy {strA:.3f}")

# --- VOLUME quality: grid age-inversions (the deep-region 'island' artifacts that the
# point metrics above miss; level ascending == older, so going UP the level should DECREASE).
from curlew.geometry import Grid
lo, hi = obs.bounds(); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
ext = np.maximum(hi - lo, 1e-6); center = 0.5 * (lo + hi)
res = np.array([90, 90, 70])
G = Grid(dims=tuple(ext), step=tuple(ext / res), center=tuple(center))
curlew.batchSize = 100000
gg = M.predict(G)
id2level = {i: name2level.get(n, -1) for i, n in gg.lithoLookup.items()}
levv = np.array([id2level.get(int(i), -1) for i in gg.lithoID])
L3 = G.reshape(levv).astype(float)
# find the vertical (z) index axis robustly: z varies only along it (axis-aligned grid)
zc = G.reshape(G.coords()[:, 2])
zaxis = int(np.argmax([abs(np.nanmean(np.diff(zc, axis=a))) for a in range(3)]))
up_inc = np.nanmean(np.diff(zc, axis=zaxis)) > 0     # does increasing index go UP?
Lz = np.moveaxis(L3, zaxis, -1)
if not up_inc:
    Lz = Lz[..., ::-1]
up, dn = Lz[..., 1:], Lz[..., :-1]                    # up = the voxel above (larger z)
valid = (up >= 0) & (dn >= 0)
inv = valid & (up > dn)                                # older (larger level) sitting ABOVE younger
col_inv = inv.any(axis=-1); col_unit = (Lz >= 0).any(axis=-1)
print(f"volume age-inversions (grid {res[0]}x{res[1]}x{res[2]}): "
      f"{100*col_inv.sum()/max(col_unit.sum(),1):.2f}% of unit columns | "
      f"{int(inv.sum())} inverted adjacencies ({100*inv.sum()/max(valid.sum(),1):.3f}% of pairs)")
