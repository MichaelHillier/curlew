"""
wcsb validation harness for the unit (soft-carve / NLL) coupling (SOFTUNIT_SPEC §7).

Usage:  python _softunit_harness.py [epochs] [cap] [field] [overturn]
Reports marker band/structure/basement accuracy, soft-carve-vs-predict agreement,
and (coarse-grid) column inversions. Not a pytest — a manual numeric check.
"""
import os
import sys
import time

import numpy as np
import torch

import curlew
from curlew.io import loadObservations
from curlew.geology.stratbuilder import build_geomodel, attach_unit_loss
from curlew.geology.softunit import soft_unit_probs

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
FIELD = sys.argv[3] if len(sys.argv) > 3 else "Siren"
OVT = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0        # coupled overturn weight

curlew.device = os.environ.get("CURLEW_DEVICE", "cuda")
torch.manual_seed(0)
np.random.seed(0)

DATA = r"D:/Development/GeoINR_Research2/geoinr/data/wcsb"
STRAT = os.path.join(DATA, "strat_col_v2.csv")
obs = loadObservations(units=os.path.join(DATA, "cleaner_markers.vtp"))

if FIELD == "Siren":
    FKW = dict(omega0=30.0, omega=30.0, learning_rate=1e-4, hidden_dim=256)
else:
    FKW = dict(activation=torch.nn.Softplus(beta=40), learning_rate=1e-3, hidden_dim=256)

print(f"=== wcsb unit-coupling harness: field={FIELD} epochs={EPOCHS} cap={CAP} ovt={OVT} ===")
M = build_geomodel(STRAT, obs, field=FIELD, scale="geo_extensive",
                   cap_per_unit=CAP, iq_samples=1024, seed=0, field_kwargs=FKW)
soft = attach_unit_loss(M, tau=0.05, points_per_level=1024, overturn_weight=OVT)

t = time.time()
loss, pebble = M.fit(EPOCHS, early_stop=None, best=False, vb=False, custom_loss=[soft])
dt = time.time() - t
print(f"fit: {dt:.1f}s ({dt/max(EPOCHS,1)*1000:.1f} ms/epoch)  final loss {loss:.4f}")
print(pebble)

soft.write_isosurfaces(M)

# --- marker accuracy (mirrors notebook §7) ---
rng = np.random.default_rng(3)
pts, lev = [], []
for L in obs.levels("unit"):
    idx = np.where((~obs.is_interface) & (obs.level == L))[0]
    idx = rng.choice(idx, size=min(400, len(idx)), replace=False)
    pts.append(obs.coords[idx].astype(float)); lev.append(np.full(len(idx), L))
pts = np.vstack(pts); lev = np.concatenate(lev)
g = M.predict(pts)
litho = np.array([g.lithoLookup.get(int(i), "?") for i in g.lithoID])
struct = np.array([g.structureLookup.get(int(s), "?") for s in g.structureID])
metas = M.field_meta
lev_struct = {L: m.name for m in metas for L in m.levels}
lit_ok = np.array([litho[k] == M.level_litho[int(lev[k])] for k in range(len(lev))])
str_ok = np.array([struct[k] == lev_struct[int(lev[k])] for k in range(len(lev))])
base_level = max(obs.levels("unit"))
print(f"\nband accuracy      : {lit_ok.mean():.4f}   (target > 0.61)")
print(f"structure accuracy : {str_ok.mean():.4f}")
print(f"basement (L{base_level}) acc : {lit_ok[lev == base_level].mean():.4f}   (target > 0.50)")

# --- soft-carve vs predict consistency (SOFTUNIT_SPEC §7) ---
# evaluate on the same markers, in model coords
Xm = M.T(torch.tensor(pts, dtype=curlew.dtype, device=curlew.device))
with torch.no_grad():
    probs = soft_unit_probs(M, Xm, soft.resolve_isos(), soft.tau, soft.spec)
carve_level = np.array([soft.spec.class_levels[i] for i in probs.argmax(1).cpu().numpy()])
lut = {M.llookup[key]: L for L, key in M.level_litho.items()}
pred_level = np.array([lut.get(int(i), -1) for i in g.lithoID])
agree = (carve_level == pred_level).mean()
carve_acc = (carve_level == lev).mean()
print(f"\nsoft-carve vs predict agreement : {agree:.4f}")
print(f"soft-carve marker accuracy      : {carve_acc:.4f}")

# --- coarse-grid column inversions (ascending level = older; older should sit below) ---
from curlew.geometry import Grid
lo, hi = obs.bounds(); lo = np.asarray(lo, float); hi = np.asarray(hi, float)
center = 0.5 * (lo + hi); ext = hi - lo
curlew.batchSize = 50000
G = Grid(dims=tuple(ext), step=(ext[0] / 60, ext[1] / 60, ext[2] / 80), center=tuple(center))
geo = M.predict(G)
lutarr = np.full(max(M.llookup.values()) + 1, -1, dtype=np.int64)
for L, key in M.level_litho.items():
    lutarr[M.llookup[key]] = L
level_vox = lutarr[geo.lithoID]
vol = G.reshape(level_vox)  # (nx, ny, nz), z fastest-last per curlew convention
nx, ny, nz = vol.shape
cols = vol.reshape(-1, nz)
inv_cols = 0; tot_cols = 0
for c in cols:
    u = c[c >= 0]
    if len(u) < 2:
        continue
    tot_cols += 1
    # going up (increasing z index) level should be non-increasing (younger up)
    if np.any(np.diff(u.astype(int)) > 0):
        inv_cols += 1
frac = inv_cols / max(tot_cols, 1)
print(f"\ncolumn inversions : {frac*100:.3f}% ({inv_cols}/{tot_cols})   (target <= 0.1%)")
