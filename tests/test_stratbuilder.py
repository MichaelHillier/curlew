"""
Smoke test for the unit-only strat-column builder (SPEC §6.3): builds a tiny
layer-cake model from unit points only, fits briefly, estimates isosurfaces and
predicts. Verifies the alias-field wiring for the region-only events.
"""
import numpy as np
import pytest

import curlew


STRAT_CSV = """name,relation,level
TopUnit,baselap,1
UnitA,eroded,2
UnitB,conformal,3
UnitC,baselap,4
Basement,eroded,5
"""


@pytest.fixture
def unit_only_model(tmp_path):
    """A tiny wcsb-style unit-only model: 5 flat-lying units, ~120 points/unit."""
    curlew.device = "cpu"
    np.random.seed(0)

    csv_path = tmp_path / "strat_col.csv"
    csv_path.write_text(STRAT_CSV)

    # layer-cake unit points: level L occupies z in [(5-L)*10, (5-L)*10 + 10)
    rows = ["x,y,z,level"]
    rng = np.random.default_rng(0)
    for L in range(1, 6):
        for _ in range(120):
            x, y = rng.uniform(0, 1000, 2)
            z = (5 - L) * 10.0 + rng.uniform(0.5, 9.5)
            rows.append(f"{x},{y},{z},{L}")
    obs_path = tmp_path / "units.csv"
    obs_path.write_text("\n".join(rows))

    from curlew.io import loadObservations
    from curlew.geology.stratbuilder import build_geomodel

    obs = loadObservations(units=str(obs_path))
    import torch
    torch.manual_seed(0)
    M = build_geomodel(str(csv_path), obs, field="Siren", scale="geo_extensive",
                       cap_per_unit=100, iq_samples=64, reg_samples=256,
                       reg_grid_n=12, seed=0,
                       field_kwargs=dict(hidden_dim=32, num_hidden_layers=2,
                                         omega0=30.0, omega=30.0, learning_rate=1e-4))
    return M, obs


def test_alias_fields(unit_only_model):
    """Region-only events alias the adjacent unconformity's field."""
    from curlew.geology.stratbuilder import _AliasField

    M, obs = unit_only_model
    metas = M.field_meta
    kinds = [m.kind for m in metas]
    assert kinds[0] == "region-only" and kinds[-1] == "region-only"

    # basement aliases the oldest unconformity; top unit aliases the youngest
    erosional = [m.name for m in metas if m.is_unconformity]
    basement = M[metas[0].name].getField(0)
    top = M[metas[-1].name].getField(0)
    assert isinstance(basement, _AliasField)
    assert basement._target is M[erosional[0]].getField(0)
    assert isinstance(top, _AliasField)
    assert top._target is M[erosional[-1]].getField(0)
    # aliases evaluate identically to their target but produce no loss
    pts = M.level_points[5][:10]
    a = M[metas[0].name].predict(pts, combine=False).scalar
    t = M[erosional[0]].predict(pts, combine=False).scalar
    assert np.allclose(a, t)
    assert len(basement.loss().losses) == 0


def test_fit_estimate_predict(unit_only_model):
    """Build → fit → estimate isosurfaces → predict, with sane labels."""
    from curlew.geology.stratbuilder import estimate_isosurfaces

    M, obs = unit_only_model
    # fit must run without Pebble duplicate-key errors (alias fields are loss-free)
    M.fit(3, early_stop=None, best=False, vb=False)

    isos = estimate_isosurfaces(M)
    n_expected = (sum(1 for m in M.field_meta if m.is_unconformity)
                  + sum(len(m.contacts) for m in M.field_meta if m.kind == "depositional"))
    assert len(isos) == n_expected

    g = M.predict(obs.coords[::25].astype(float))
    assert len(np.unique(g.lithoID)) >= 2  # labels assigned
    # lithoID is numbered so ascending = younger; spot-check the legend covers all units
    assert all(key in M.llookup for key in set(M.level_litho.values()))
