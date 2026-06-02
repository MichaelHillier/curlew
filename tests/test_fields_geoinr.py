"""
Field-port smoke test (SPEC §6.3): confirm the GeoINR `GeoINR` (plain MLP) and
`Siren` (sine-activation) fields satisfy the `curlew.fields.BaseNF` contract —
they construct via the Curlew `strati` factory, bind a small `CSet` (a few `gv`
gradient + `iq` inequality constraints), and train through `GeoModel.fit` with a
finite, decreasing loss.
"""

import numpy as np
import pytest
import torch

import curlew
from curlew import HSet
from curlew.core import CSet
from curlew.fields import GeoINR, Siren
from curlew.geology import strati
from curlew.geology.geomodel import GeoModel


def _tiny_cset(seed=0):
    """A tiny synthetic 3D constraint set: gradient points up (+z), and high-z
    points must have a larger scalar value than low-z points."""
    rng = np.random.default_rng(seed)

    # gradient constraints: younging direction is +z everywhere
    gp = rng.uniform(-1.0, 1.0, size=(16, 3))
    gv = np.tile([0.0, 0.0, 1.0], (16, 1))

    # inequality pools: a high-z cloud must exceed a low-z cloud
    hi = rng.uniform(-1.0, 1.0, size=(40, 3)); hi[:, 2] = rng.uniform(0.4, 1.0, 40)
    lo = rng.uniform(-1.0, 1.0, size=(40, 3)); lo[:, 2] = rng.uniform(-1.0, -0.4, 40)

    return CSet(gp=gp, gv=gv, iq=(16, [(hi, lo, '>')]))


@pytest.mark.parametrize("field_cls", [GeoINR, Siren])
def test_field_port_smoke(field_cls):
    curlew.default_dim = 3
    torch.manual_seed(0)

    C = _tiny_cset()
    H = HSet(grad_loss=1.0, iq_loss=1.0)

    # construct via the Curlew factory with a pre-built field (SPEC §4.1 contract)
    field = field_cls('s', H=H, C=C, input_dim=3, hidden_dim=64, num_hidden_layers=2)
    event = strati('s', C=field)
    M = GeoModel([event])

    # loss before any training
    loss0 = sum(F.loss().total().item() for F in M.events)
    assert np.isfinite(loss0)

    loss_final, pebble = M.fit(200, early_stop=None, best=True, vb=False)

    assert np.isfinite(loss_final), f"{field_cls.__name__}: loss is not finite"
    assert loss_final < loss0, (
        f"{field_cls.__name__}: loss did not decrease ({loss0:.4g} -> {loss_final:.4g})"
    )
