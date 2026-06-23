"""
skmb validation harness — the **interface + unit** hybrid (seed-iso) coupling.

skmb (Saskatchewan/Manitoba basin, 45 units) is the example that combines BOTH
on-contact (interface) points *and* interior unit points, and that exercises the
**baselap-onto-conformal** pattern (a baselap unit laps onto the top of the conformal
package below it, not an unconformity). Contacts are pinned directly by the interface
data (``eq`` traces + seed isosurfaces) — no learnable iso-values — while the unit points
drive the soft-unit / NLL coupling (``attach_unit_loss(seed_isos=True)``).

Usage:  python _softunit_harness.py [epochs] [cap] [field]
Asserts the four baselap-onto-conformal onlap relationships and the build/fit/predict
round-trip, and reports unit-point band/structure accuracy + soft-carve↔predict agreement.
Not a pytest — a manual numeric check. Set CURLEW_DEVICE=cpu to run on CPU.
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

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
CAP = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
FIELD = sys.argv[3] if len(sys.argv) > 3 else "Siren"

curlew.device = os.environ.get("CURLEW_DEVICE", "cuda")
torch.manual_seed(0)
np.random.seed(0)

CANDIDATES = [
    r"D:/Development/GeoINR_Research2/geoinr/data/skmb",
    os.path.join(os.path.dirname(os.getcwd()), "data", "skmb"),
    "./data/skmb",
]
DATA = next((p for p in CANDIDATES if os.path.exists(os.path.join(p, "strat_column.csv"))), None)
assert DATA is not None, f"Could not locate skmb data; tried {CANDIDATES}"
STRAT = os.path.join(DATA, "strat_column.csv")

# markers.vtp = on-contact (interface) points; units.vtp = interior unit points
obs = loadObservations(interfaces=os.path.join(DATA, "markers.vtp"),
                       units=os.path.join(DATA, "units.vtp"))

if FIELD == "Siren":
    FKW = dict(omega0=30.0, omega=30.0, learning_rate=1e-4, hidden_dim=256)
else:
    FKW = dict(activation=torch.nn.Softplus(beta=40), learning_rate=1e-3, hidden_dim=256)

print(f"=== skmb hybrid (seed-iso) harness: field={FIELD} epochs={EPOCHS} cap={CAP} ===")
print(f"observations: {int(obs.is_interface.sum())} interface + {int((~obs.is_interface).sum())} unit points")
M = build_geomodel(STRAT, obs, field=FIELD, scale="geo_extensive",
                   cap_per_unit=CAP, iq_samples=1024, seed=0, field_kwargs=FKW)

# --- assert the four baselap-onto-conformal onlap relationships ---------------
onlap_pkgs = [m for m in M.field_meta if m.onlap_package]
print(f"\nbaselap-onto-conformal packages: {len(onlap_pkgs)}")
for m in onlap_pkgs:
    lower = M[m.onlap_event]
    assert m.onlap_iso_id.endswith("::top"), m.onlap_iso_id
    # the onlap threshold must be a *seeded* top isosurface on the lower (conformal) package
    lower_meta = next(x for x in M.field_meta if x.name == m.onlap_event)
    assert lower_meta.kind == "depositional" and lower_meta.top_iso is not None
    assert lower_meta.top_iso in lower.isosurfaces
    print(f"  {m.name[:28]:28s} baselaps onto top of {m.onlap_event[:34]:34s} ({lower_meta.top_iso})")
assert len(onlap_pkgs) == 4, f"expected 4 baselap-onto-conformal packages, got {len(onlap_pkgs)}"

# --- fit: seed-iso coupling (contacts pinned by interfaces, units drive NLL) ---
soft = attach_unit_loss(M, tau=0.05, points_per_level=1024, seed_isos=True)
print(f"\ncoupling: {soft.n_classes} level-classes | {len(list(soft._params))} iso params (0 = seeded) "
      f"| {len(soft.spec.seed_map)} seeded thresholds | tau={soft.tau}")

t = time.time()
loss, pebble = M.fit(EPOCHS, early_stop=None, best=False, vb=False, custom_loss=[soft])
dt = time.time() - t
print(f"fit: {dt:.1f}s ({dt/max(EPOCHS,1)*1000:.1f} ms/epoch)  final loss {loss:.4f}")

# seeds already resolve at predict time (write_isosurfaces is a no-op on the seeded path)

# --- unit-point band/structure accuracy ---------------------------------------
# evaluate over levels modelled as a band (M.level_litho). Every observed unit is modelled --
# including the Sub_Cantuar unit sandwiched between two unconformities (its own region-only
# event) -- so this filter is a defensive no-op for skmb.
rng = np.random.default_rng(3)
eval_levels = [L for L in obs.levels("unit") if L in M.level_litho]
skipped = [L for L in obs.levels("unit") if L not in M.level_litho]
if skipped:
    print(f"(skipping unmodelled levels {skipped})")
pts, lev = [], []
for L in eval_levels:
    idx = np.where((~obs.is_interface) & (obs.level == L))[0]
    idx = rng.choice(idx, size=min(400, len(idx)), replace=False)
    pts.append(obs.coords[idx].astype(float)); lev.append(np.full(len(idx), L))
pts = np.vstack(pts); lev = np.concatenate(lev)
g = M.predict(pts)
litho = np.array([g.lithoLookup.get(int(i), "?") for i in g.lithoID])
struct = np.array([g.structureLookup.get(int(s), "?") for s in g.structureID])
lev_struct = {L: m.name for m in M.field_meta for L in m.levels}
lit_ok = np.array([litho[k] == M.level_litho[int(lev[k])] for k in range(len(lev))])
str_ok = np.array([struct[k] == lev_struct[int(lev[k])] for k in range(len(lev))])
print(f"\nband accuracy      : {lit_ok.mean():.4f}")
print(f"structure accuracy : {str_ok.mean():.4f}")

# --- interface reconstruction: the field should be ~constant on each contact ---
# (the eq loss pins it; small spread => the seed iso is a well-defined surface).
spreads = []
for meta in M.field_meta:
    ev = M[meta.name]
    for name, (fld, seed) in ev.isosurfaces.items():
        if not isinstance(seed, np.ndarray):
            continue
        with torch.no_grad():
            v = ev.forward(torch.tensor(np.asarray(seed, float), dtype=curlew.dtype,
                                        device=curlew.device)).cpu().numpy().reshape(-1)
        spreads.append(v.std())
print(f"interface field spread (std over seed pts): median {np.median(spreads):.3f}  (smaller = tighter contacts)")

# --- soft-carve vs predict agreement ------------------------------------------
Xm = M.T(torch.tensor(pts, dtype=curlew.dtype, device=curlew.device))
with torch.no_grad():
    probs = soft_unit_probs(M, Xm, soft.resolve_isos(M), soft.tau, soft.spec)
carve_level = np.array([soft.spec.class_levels[i] for i in probs.argmax(1).cpu().numpy()])
lut = {M.llookup[key]: L for L, key in M.level_litho.items()}
pred_level = np.array([lut.get(int(i), -1) for i in g.lithoID])
print(f"soft-carve vs predict agreement : {(carve_level == pred_level).mean():.4f}")
print(f"soft-carve marker accuracy      : {(carve_level == lev).mean():.4f}")
print("\nOK")
