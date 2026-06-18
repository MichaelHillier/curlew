"""
Tests for the soft-unit / NLL coupling (SOFTUNIT_SPEC §7).

Builds a tiny wcsb-style unit-only model and exercises the coupling loss:

  * ``soft_unit_probs`` rows form a simplex (sum to 1);
  * autograd flows to **both** the field params **and** the learnable iso-values;
  * the monotone reparam keeps a package's contacts strictly ordered for arbitrary ``φ``;
  * NLL **decreases** when the coupled model is trained on separable layer-cake points;
  * ``write_isosurfaces`` round-trips: post-fit ``predict`` labels (lithoID → level) match
    the soft-carve argmax (level space) on the training points.
"""
import numpy as np
import pytest
import torch

import curlew


STRAT_CSV = """name,relation,level
TopUnit,baselap,1
UnitA,eroded,2
UnitB,conformal,3
UnitC,baselap,4
Basement,eroded,5
"""


def _make_model(tmp_path):
    """A tiny separable layer-cake unit-only model (5 flat units), unfitted."""
    curlew.device = "cpu"
    np.random.seed(0)
    torch.manual_seed(0)

    csv_path = tmp_path / "strat_col.csv"
    csv_path.write_text(STRAT_CSV)

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
    from curlew.geology.stratbuilder import build_geomodel, attach_unit_loss

    obs = loadObservations(units=str(obs_path))
    M = build_geomodel(str(csv_path), obs, field="GeoINR", scale="isometric",
                       cap_per_unit=100, iq_samples=64, reg_samples=256, reg_grid_n=12, seed=0,
                       field_kwargs=dict(hidden_dim=64, num_hidden_layers=2,
                                         activation=torch.nn.Softplus(beta=40),
                                         learning_rate=1e-3))
    loss = attach_unit_loss(M, points_per_level=100)
    return M, loss


def _all_points(loss):
    """All pooled (model-coord) training points and their class labels (no subsample)."""
    coords, labels = [], []
    for c, pts in loss._pool.items():
        coords.append(pts)
        labels.append(torch.full((pts.shape[0],), c, dtype=torch.long, device=pts.device))
    return torch.cat(coords, 0), torch.cat(labels, 0)


def _carve_nll(M, loss, coords, labels):
    from curlew.geology.softunit import soft_unit_probs
    probs = soft_unit_probs(M, coords, loss.resolve_isos(), loss.tau, loss.spec)
    return torch.nn.functional.nll_loss(probs.clamp_min(1e-12).log(), labels)


@pytest.fixture
def built(tmp_path):
    """A freshly built, unfitted coupled model + loss."""
    return _make_model(tmp_path)


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    """Build → record pre-NLL → fit jointly → record post-NLL → write_isosurfaces (once)."""
    M, loss = _make_model(tmp_path_factory.mktemp("softunit"))
    coords, labels = _all_points(loss)
    with torch.no_grad():
        pre = float(_carve_nll(M, loss, coords, labels))
    M.fit(300, early_stop=None, best=False, vb=False, custom_loss=[loss])
    with torch.no_grad():
        post = float(_carve_nll(M, loss, coords, labels))
    isos = loss.write_isosurfaces(M)
    return dict(M=M, loss=loss, pre=pre, post=post, isos=isos)


# ---------------------------------------------------------------------------
def test_iq_off_overturn_kept(built):
    """attach_unit_loss zeros iq_norm on coupled fields but keeps (relaxed) overturn."""
    M, _ = built
    for meta in M.field_meta:
        f = M[meta.name].getField(0)
        if hasattr(f, "iq_norm_weight"):   # coupled (non-alias) fields
            assert f.iq_norm_weight == 0.0
            assert f.overturn_weight > 0.0


def test_simplex_sums_to_one(built):
    """soft_unit_probs rows form a valid probability simplex."""
    from curlew.geology.softunit import soft_unit_probs

    M, loss = built
    coords, _ = loss.sample_balanced_unit_indices()
    with torch.no_grad():
        probs = soft_unit_probs(M, coords, loss.resolve_isos(), loss.tau, loss.spec)
    assert probs.shape == (coords.shape[0], loss.n_classes)
    assert torch.all(probs >= 0)
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)


def test_grad_flows_to_fields_and_isos(built):
    """Autograd from the NLL reaches BOTH iso-values and field params (guards a future detach)."""
    M, loss = built
    coords, labels = loss.sample_balanced_unit_indices()
    nll = _carve_nll(M, loss, coords, labels)
    nll.backward()

    # iso-value parameters (owned by the loss)
    assert len(list(loss._params)) > 0
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in loss._params), \
        "no gradient reached the learnable iso-values"

    # a coupled field's network parameters
    dep = next(m for m in M.field_meta if m.kind == "depositional")
    fparams = list(M[dep.name].getField(0).parameters())
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in fparams), \
        "no gradient reached the field parameters"


def test_monotone_reparam_orders_contacts():
    """
    θ_j = θ_0 − cumsum(softplus(φ)) keeps contacts ordered for arbitrary φ.

    The structural guarantee is **non-crossing**: gaps are softplus(φ) ≥ 0, so contacts
    are non-increasing and can pinch to zero thickness but never invert (a band collapses
    rather than re-ordering — SOFTUNIT_SPEC §5). For ordinary (non-degenerate) φ the
    gaps are strictly positive, so contacts strictly descend.
    """
    from curlew.geology.softunit import monotone_contacts

    torch.manual_seed(1)
    # cannot cross, for arbitrary φ (incl. extreme negatives that pinch a band to zero)
    for _ in range(20):
        theta0 = torch.randn(()) * 5
        phi = torch.randn(7) * 5
        contacts = monotone_contacts(theta0, phi)
        assert torch.all(torch.diff(contacts) <= 0), "contacts must not cross (non-increasing)"
        assert contacts[0] <= theta0, "first contact cannot exceed the anchor"

    # strictly descending for ordinary φ (gaps strictly positive)
    for _ in range(20):
        theta0 = torch.randn(())
        phi = torch.randn(7)
        contacts = monotone_contacts(theta0, phi)
        assert torch.all(torch.diff(contacts) < 0), "ordinary φ should give strictly ordered contacts"
        assert contacts[0] < theta0


def test_nll_decreases(fitted):
    """Joint training on separable layer-cake points reduces the cross-entropy."""
    assert fitted["post"] < fitted["pre"]
    assert fitted["post"] < 1.0   # converges well on a clean, separable toy


def test_write_isosurfaces_round_trip(fitted):
    """Post-fit predict labels (lithoID → level) match the soft-carve argmax (level space)."""
    from curlew.geology.softunit import soft_unit_probs

    M, loss = fitted["M"], fitted["loss"]
    coords, _ = _all_points(loss)

    with torch.no_grad():
        probs = soft_unit_probs(M, coords, loss.resolve_isos(), loss.tau, loss.spec)
    carve_level = np.array([loss.spec.class_levels[i] for i in probs.argmax(1).cpu().numpy()])

    g = M.predict(coords.cpu().numpy(), coords="model")
    lut = {M.llookup[key]: L for L, key in M.level_litho.items()}  # lithoID -> level
    pred_level = np.array([lut.get(int(i), -1) for i in g.lithoID])

    # the two assemblies use the SAME learned thresholds + fields, so they agree
    # (up to the soft/hard boundary shells) — this is the soft-vs-hard consistency check
    assert (carve_level == pred_level).mean() >= 0.95
    assert len(np.unique(carve_level)) >= 2   # non-degenerate (not one class everywhere)
