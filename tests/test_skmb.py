"""
Tests for the skmb additions: the **baselap-onto-conformal** partition split and the
**seed-iso** (interface + unit hybrid) soft-unit coupling.

  * ``derive_scalar_fields`` splits a conformal run at a ``baselap`` unit and flags the
    upper package as ``onlap_package`` (it onlaps the package below, not an unconformity);
    a column whose baselaps are all followed by ``eroded`` (wcsb-style) is unchanged.
  * a tiny hybrid model (interface + unit points, one baselap-onto-conformal) builds via the
    chain path; the upper package onlaps the lower package's seeded **top** isosurface;
  * the seeded ``UnitLoss`` owns no learnable iso-values, resolves every threshold from the
    interface seeds, carves a valid simplex, and the upper package's carve op weights against
    the lower package's ``::top`` threshold; autograd from the NLL reaches the field params.
"""
import numpy as np
import pytest
import torch

import curlew


# A → B(baselap) | C → D | Base(eroded): B baselaps onto the conformal package {C, D}.
HYBRID_CSV = """name,relation,level
A,conformal,1
B,baselap,2
C,conformal,3
D,conformal,4
Base,eroded,5
"""

# wcsb-style: every baselap is followed by an eroded unit (onlaps an unconformity).
WCSB_LIKE_CSV = """name,relation,level
Top,baselap,1
UnconfA,eroded,2
P1a,conformal,3
P1b,baselap,4
UnconfB,eroded,5
Basement,eroded,6
"""


def _load_units(tmp_path, text):
    from curlew.geology.stratbuilder import load_strat_column_csv
    p = tmp_path / "col.csv"
    p.write_text(text)
    return load_strat_column_csv(str(p))


# ---------------------------------------------------------------------------
# partitioner
# ---------------------------------------------------------------------------
def test_partition_splits_baselap_onto_conformal(tmp_path):
    from curlew.geology.stratbuilder import derive_scalar_fields
    units = _load_units(tmp_path, HYBRID_CSV)
    sf = derive_scalar_fields(units)
    dep = [s for s in sf if s.kind == "depositional"]
    levels = lambda s: sorted(units[i].level for i in s.unit_indices)

    # the run A,B,C,D is split at the baselap B into {A,B} (upper) and {C,D} (lower)
    upper = next(s for s in dep if s.onlap_package)
    lower = next(s for s in dep if levels(s) == [3, 4])
    assert levels(upper) == [1, 2]
    assert not lower.onlap_package
    # exactly one baselap-onto-conformal package here
    assert sum(s.onlap_package for s in sf) == 1


def test_partition_wcsb_like_unchanged(tmp_path):
    """A column whose baselaps are all followed by ``eroded`` has no onlap_package split."""
    from curlew.geology.stratbuilder import derive_scalar_fields
    units = _load_units(tmp_path, WCSB_LIKE_CSV)
    sf = derive_scalar_fields(units)
    assert sum(s.onlap_package for s in sf) == 0


# ---------------------------------------------------------------------------
# hybrid (seed-iso) build + carve
# ---------------------------------------------------------------------------
def _make_hybrid(tmp_path):
    """A tiny flat layer-cake with BOTH interface and unit points + a baselap-onto-conformal."""
    curlew.device = "cpu"
    np.random.seed(0)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    csv_path = tmp_path / "col.csv"
    csv_path.write_text(HYBRID_CSV)

    # z bands: unit L occupies (50-10L, 60-10L); marker level L is the top of unit L.
    # markers exist for levels 2..5 (level 1 is the column top, no contact above it).
    unit_rows = ["x,y,z,level"]
    for L in range(1, 6):
        z0 = 50 - 10 * L
        for _ in range(150):
            x, y = rng.uniform(0, 1000, 2)
            z = z0 + rng.uniform(1.0, 9.0)
            unit_rows.append(f"{x},{y},{z},{L}")
    iface_rows = ["x,y,z,level"]
    for L in range(2, 6):
        z = 60 - 10 * L  # boundary between unit L and the (younger) unit above it
        for _ in range(80):
            x, y = rng.uniform(0, 1000, 2)
            iface_rows.append(f"{x},{y},{z},{L}")

    (tmp_path / "units.csv").write_text("\n".join(unit_rows))
    (tmp_path / "ifaces.csv").write_text("\n".join(iface_rows))

    from curlew.io import loadObservations
    from curlew.geology.stratbuilder import build_geomodel, attach_unit_loss

    obs = loadObservations(interfaces=str(tmp_path / "ifaces.csv"),
                           units=str(tmp_path / "units.csv"))
    assert obs.is_interface.any() and (~obs.is_interface).any()
    M = build_geomodel(str(csv_path), obs, field="GeoINR", scale="isometric",
                       cap_per_unit=120, iq_samples=64, reg_samples=128, reg_grid_n=10, seed=0,
                       field_kwargs=dict(hidden_dim=64, num_hidden_layers=2,
                                         activation=torch.nn.Softplus(beta=40), learning_rate=1e-3))
    loss = attach_unit_loss(M, points_per_level=120, seed_isos=True)
    return M, loss


@pytest.fixture
def hybrid(tmp_path):
    return _make_hybrid(tmp_path)


def test_hybrid_uses_chain_path_with_seeded_surfaces(hybrid):
    """Interface+unit data with structure → chain path; every surface is seeded."""
    M, _ = hybrid
    assert getattr(M, "level_points", None) is not None       # chain path ran
    # every depositional contact and every unconformity carries a seed isosurface
    for meta in M.field_meta:
        ev = M[meta.name]
        if meta.is_unconformity:
            assert meta.iso_name in ev.isosurfaces
        elif meta.kind == "depositional":
            for c in meta.contacts:
                assert c.name in ev.isosurfaces


def test_hybrid_baselap_onlaps_lower_package_top(hybrid):
    """The baselap package onlaps the conformal package below it via a seeded top iso."""
    M, _ = hybrid
    onlaps = [m for m in M.field_meta if m.onlap_package]
    assert len(onlaps) == 1
    up = onlaps[0]
    low = next(m for m in M.field_meta if m.name == up.onlap_event)
    assert low.kind == "depositional"                          # onlaps a package, not an unconformity
    assert up.onlap_iso_id.endswith("::top")
    assert low.top_iso is not None and low.top_iso in M[low.name].isosurfaces


def test_seed_carve_simplex_and_no_iso_params(hybrid):
    """Seeded coupling owns no iso params; the carve still forms a valid simplex."""
    from curlew.geology.softunit import soft_unit_probs
    M, loss = hybrid
    assert loss.seed_isos and len(list(loss._params)) == 0
    assert loss.optim is None and len(loss.spec.seed_map) > 0

    coords, _ = loss.sample_balanced_unit_indices()
    with torch.no_grad():
        probs = soft_unit_probs(M, coords, loss.resolve_isos(M), loss.tau, loss.spec)
    assert probs.shape == (coords.shape[0], loss.n_classes)
    assert torch.all(probs >= 0)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_seed_carve_op_weights_against_lower_top(hybrid):
    """The baselap package's carve op truncates against the lower package's ::top threshold."""
    M, loss = hybrid
    up = next(m for m in M.field_meta if m.onlap_package)
    op = next(o for o in loss.spec.ops if o.band_event == up.name)
    assert op.weight_iso == up.onlap_iso_id            # == "<lower>::top"
    assert op.weight_iso in loss.spec.seed_map         # resolvable from the lower package's seed


def test_seed_grad_flows_to_fields(hybrid):
    """Autograd from the seeded NLL reaches the coupled field params (the fields co-train)."""
    from curlew.geology.softunit import soft_unit_probs
    M, loss = hybrid
    coords, labels = loss.sample_balanced_unit_indices()
    probs = soft_unit_probs(M, coords, loss.resolve_isos(M), loss.tau, loss.spec)
    torch.nn.functional.nll_loss(probs.clamp_min(1e-12).log(), labels).backward()

    dep = next(m for m in M.field_meta if m.onlap_package)
    fparams = list(M[dep.name].getField(0).parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in fparams), \
        "no gradient reached the field parameters through the seeded carve"


# ---------------------------------------------------------------------------
# surface nesting (younger unconformity must not erode below an older one)
# ---------------------------------------------------------------------------
# A,UncYoung(eroded),B,UncOld(eroded): two unconformities, young over old.
NESTING_CSV = """name,relation,level
A,conformal,1
UncYoung,eroded,2
B,conformal,3
UncOld,eroded,4
"""


def _make_nesting(tmp_path):
    """Two-unconformity hybrid (interface + unit points) to exercise surface nesting."""
    curlew.device = "cpu"
    np.random.seed(0); torch.manual_seed(0)
    rng = np.random.default_rng(0)
    (tmp_path / "col.csv").write_text(NESTING_CSV)

    unit_rows = ["x,y,z,level"]
    for L in range(1, 5):
        z0 = 40 - 10 * L
        for _ in range(120):
            x, y = rng.uniform(0, 1000, 2)
            unit_rows.append(f"{x},{y},{z0 + rng.uniform(1.0, 9.0)},{L}")
    iface_rows = ["x,y,z,level"]
    for L in range(2, 5):       # markers for levels 2..4 (level 1 is the column top)
        z = 50 - 10 * L
        for _ in range(80):
            x, y = rng.uniform(0, 1000, 2)
            iface_rows.append(f"{x},{y},{z},{L}")
    (tmp_path / "u.csv").write_text("\n".join(unit_rows))
    (tmp_path / "i.csv").write_text("\n".join(iface_rows))

    from curlew.io import loadObservations
    from curlew.geology.stratbuilder import build_geomodel, attach_unit_loss
    obs = loadObservations(interfaces=str(tmp_path / "i.csv"), units=str(tmp_path / "u.csv"))
    M = build_geomodel(str(tmp_path / "col.csv"), obs, field="GeoINR", scale="isometric",
                       cap_per_unit=80, iq_samples=64, reg_samples=64, reg_grid_n=8, seed=0,
                       field_kwargs=dict(hidden_dim=32, num_hidden_layers=2,
                                         activation=torch.nn.Softplus(beta=40), learning_rate=1e-3))
    loss = attach_unit_loss(M, points_per_level=80, seed_isos=True)
    return M, loss


def test_seed_unconformity_surface_nesting(tmp_path):
    """A younger unconformity gets a per-older-surface nesting inequality; the seed path keeps
    the gradient-normalised inequality ON for every constrained field."""
    M, _ = _make_nesting(tmp_path)
    ero = [m for m in M.field_meta if m.is_unconformity]
    assert len(ero) >= 2
    young = min(ero, key=lambda m: m.eroded_level)   # smaller level = younger
    old = max(ero, key=lambda m: m.eroded_level)
    tag = "older-unconformity-surface"
    assert any(r[1] == tag for r in young.iq_relations), "younger unconformity missing nesting pair"
    assert not any(r[1] == tag for r in old.iq_relations), "oldest unconformity should have no older surface"

    # seed path keeps the inequality active on every field that carries constraints
    for m in M.field_meta:
        f = M[m.name].getField(0)
        if hasattr(f, "iq_norm_weight"):
            assert f.iq_norm_weight == 1.0


def test_seed_package_top_nesting(hybrid):
    """A baselap-onto-conformal package onlaps the lower package's TOP iso; that top contact
    must nest above older unconformity surfaces, so the upper package cannot erode below them
    (the basin-margin Precambrian-cut fix)."""
    M, _ = hybrid
    up = next(m for m in M.field_meta if m.onlap_package)
    lower = next(m for m in M.field_meta if m.name == up.onlap_event)   # the onlap-target package
    assert lower.kind == "depositional" and lower.top_iso is not None
    top_level = lower.levels[0]                                          # youngest level = top contact
    nest = [r for r in lower.iq_relations
            if r[0] == top_level and r[1] == "older-unconformity-surface"]
    assert nest, "onlap-target package missing top-contact nesting vs older surfaces"


# ---------------------------------------------------------------------------
# Goal 1 — Poisson-disk overturn sampling (seed-iso path only; wcsb unchanged)
# ---------------------------------------------------------------------------
def _grids(M):
    """Per-field no-overturn grid sampleArgs (None for alias/region-only fields)."""
    out = []
    for m in M.field_meta:
        C = M[m.name].getField(0).C
        out.append(None if (C is None or C.grid is None) else C.grid.sampleArgs)
    return [s for s in out if s is not None]


def _wcsb_like_model(tmp_path):
    """A tiny unit-only (no interface) model — the wcsb seed_isos=False chain path."""
    import torch
    from curlew.io import loadObservations
    from curlew.geology.stratbuilder import build_geomodel
    curlew.device = "cpu"
    np.random.seed(0); torch.manual_seed(0); rng = np.random.default_rng(0)
    (tmp_path / "c.csv").write_text(NESTING_CSV)
    rows = ["x,y,z,level"]
    for L in range(1, 5):
        for _ in range(80):
            x, y = rng.uniform(0, 1000, 2)
            rows.append(f"{x},{y},{(4 - L) * 10 + rng.uniform(1, 9)},{L}")
    (tmp_path / "u.csv").write_text("\n".join(rows))
    obs = loadObservations(units=str(tmp_path / "u.csv"))
    assert not obs.is_interface.any()
    return build_geomodel(str(tmp_path / "c.csv"), obs, field="GeoINR", scale="isometric",
                          cap_per_unit=60, iq_samples=64, reg_samples=200, reg_grid_n=8, seed=0,
                          field_kwargs=dict(hidden_dim=16, num_hidden_layers=2,
                                            activation=torch.nn.Softplus(beta=40)))


def test_seed_path_uses_poisson_overturn(hybrid):
    """The seed-iso (interface present) path samples the no-overturn grid with Poisson-disk."""
    M, _ = hybrid
    grids = _grids(M)
    assert grids and all("poissonDisk" in s for s in grids), \
        "seed-iso overturn grids should use Poisson-disk sampling"


def test_wcsb_path_keeps_uniform_overturn(tmp_path):
    """The unit-only (no interface) wcsb path is untouched — uniform-from-lattice sampling."""
    M = _wcsb_like_model(tmp_path)
    grids = _grids(M)
    assert grids and all("N" in s and "poissonDisk" not in s for s in grids), \
        "wcsb overturn grids must stay uniform (seed_isos=False path unchanged)"


# ---------------------------------------------------------------------------
# Deep-region anchors — the per-level below-side + within-package unit-point band iq are kept
# on BOTH paths. A 2026-06-23 experiment dropped them on the seed-iso path to cut cost; it left
# marker accuracy unchanged but reintroduced deep-region volume-inversion islands (the artifacts
# the point metrics miss), so they stay. These tests lock that in (regression guard).
# ---------------------------------------------------------------------------
def test_within_package_unit_band_iq_kept(hybrid):
    """The within-package unit-point band ordering is present on the seed-iso path (a deep-region
    anchor; dropping it caused volume-inversion islands)."""
    M, _ = hybrid
    # lower package {C(3), D(4)}: the unit-band pair (3>4) must be present
    lower = next(m for m in M.field_meta
                 if m.kind == "depositional" and sorted(m.levels) == [3, 4])
    assert (3, 4, ">") in lower.iq_relations, \
        "seed path must keep the within-package unit-point band ordering (deep-region anchor)"


def test_erosional_below_side_per_level(tmp_path):
    """The erosional below-side keeps **one pair per older level** (the deep-region anchors), not
    a pooled 'deep' reference — on both the wcsb and the seed-iso path."""
    from curlew.geology.stratbuilder import _build_erosional_event
    from curlew.fields.geoinr import GeoINR
    from curlew.geometry import Grid
    rng = np.random.default_rng(0)
    # eroded level 2, with older levels 3,4,5,6 -> eroded + 4 older = 5 per-level below pairs
    level_pts = {L: rng.uniform(-1, 1, (40, 3)) for L in range(1, 7)}
    g = Grid(dims=(2, 2, 2), step=(0.5, 0.5, 0.5), sampleArgs={"N": 16})
    _, meta = _build_erosional_event(
        name="U", eroded_level=2, above_levels=[1], all_levels=list(range(1, 7)),
        level_pts=level_pts, ndim=3, field_cls=GeoINR, field_kwargs=dict(hidden_dim=8),
        iq_samples=16, grid=g, trend=np.array([0.0, 0.0, 1.0]))
    below = [r for r in meta.iq_relations if isinstance(r[1], int)]
    assert len(below) == 5, "below-side must keep one pair per older level (no pooling)"
    assert not any(r[1] == "older-units-pooled" for r in meta.iq_relations)


# ---------------------------------------------------------------------------
# Goal 2 — the single surface-ordering principle (one builder)
# ---------------------------------------------------------------------------
def test_surface_ordering_iq_single_principle():
    """``_surface_ordering_iq`` emits adjacent within-field ordering + top-vs-each-older nesting
    from one place — the §2 'every surface above every older surface' principle."""
    from curlew.geology.stratbuilder import _surface_ordering_iq, _NEST_TAG
    own = [(10, np.zeros((3, 3))), (20, np.zeros((3, 3))), (30, np.zeros((3, 3)))]  # young->old
    older = [np.zeros((3, 3)), np.zeros((3, 3))]                                    # 2 older surfaces
    pairs, rels = _surface_ordering_iq(own_contacts=own, top_surface=(5, np.zeros((3, 3))),
                                       older_surfaces=older)
    # 2 adjacent own pairs (10>20, 20>30) + 2 nesting pairs (top 5 > each older)
    assert rels[:2] == [(10, 20, ">"), (20, 30, ">")]
    assert rels[2:] == [(5, _NEST_TAG, ">"), (5, _NEST_TAG, ">")]
    assert len(pairs) == 4
    # a single-surface (erosional) field: no within-field pairs, only nesting
    p2, r2 = _surface_ordering_iq(own_contacts=[], top_surface=((2,), np.zeros((3, 3))),
                                  older_surfaces=older)
    assert len(p2) == 2 and all(r[1] == _NEST_TAG for r in r2)
