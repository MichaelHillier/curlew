"""
Soft-unit / NLL coupling for the unit-only strat builder (SOFTUNIT_SPEC.md).

The unit-only path of :func:`curlew.geology.stratbuilder.build_geomodel` fits each
event's field independently with hinge inequalities (``CSet.iq``) plus a no-overturn
prior, then estimates surfaces post-hoc. Inequalities give zero gradient once
satisfied, nothing calibrates a field's absolute value laterally, and a single
post-hoc iso lands mid-slack — so per-point marker fits are loose.

This module ports GeoINR's fix: a **soft-unit / NLL coupling** across all scalar
fields. It converts the field stack plus shared, learnable iso-values into per-point
**class probabilities** (a genuine ``(N, C)`` simplex) and trains them with
cross-entropy against the true unit *levels*. Every unit point then carries a
two-sided, always-active likelihood; field values are calibrated everywhere; and
classification is globally consistent.

Design (Avenue B of SOFTUNIT_SPEC §2): a dedicated :class:`UnitLoss` passed to
:meth:`curlew.geology.geomodel.GeoModel.fit` via its ``custom_loss`` hook.
``GeoModel.predict`` is **untouched** — the field stack is obtained *through* the
deformation machinery (``ev.predict(combine=False, transform=True)`` walks the
``child.undeform`` chain), and after training the learned iso-values are written into
the ordinary ``addIsosurface(value=...)`` so predict reproduces the model.

The carve (:func:`soft_unit_probs`, SOFTUNIT_SPEC §3) is a **per-class re-derivation
of** :meth:`curlew.core.Geode.combine`: the same youngest-first stick-break and the
same ``Overprint`` sigmoid/threshold form, accumulating an ``(N, C)`` simplex instead
of a scalar. Classes are unit **levels** (not ``lithoID``), so the loss never depends
on ``lithoID`` numbering. Iso-values are learnable Parameters owned by the loss, with
a per-package monotone reparam (``θ_j = θ_0 − cumsum(softplus(φ))``) that keeps a
package's contacts ordered in value space by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dcfield

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import curlew
from curlew import _tensor
from curlew.core import LearnableBase, Pebble

EPS = 1e-12


# ---------------------------------------------------------------------------
# Monotone iso reparam (SOFTUNIT_SPEC §5)
# ---------------------------------------------------------------------------
def monotone_contacts(theta0: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Resolve a package's internal contacts from the monotone reparam:

        ``θ_j = θ_0 − cumsum(softplus(φ))_j``   (j = 1 … m, descending).

    The gaps ``softplus(φ)`` are strictly positive, so the returned contacts are
    **strictly decreasing and cannot cross** for arbitrary ``φ`` — younger band =
    higher value, in value space, independent of spatial geometry. This is what
    makes the §3.2 band probabilities valid (non-negative) by construction.

    Parameters
    ----------
    theta0 : torch.Tensor
        Scalar anchor (top of the package, above the first contact).
    phi : torch.Tensor
        ``(m,)`` free parameters; ``softplus(phi)`` are the contact gaps.

    Returns
    -------
    torch.Tensor
        ``(m,)`` contact values ``θ_1 > θ_2 > … > θ_m``.
    """
    return theta0 - torch.cumsum(F.softplus(phi), dim=0)


def _inv_softplus(g: np.ndarray) -> np.ndarray:
    """Inverse of ``softplus`` for positive gaps ``g`` (so ``softplus(φ) == g``)."""
    g = np.clip(np.asarray(g, float), 1e-4, None)
    return np.log(np.expm1(g))


# ---------------------------------------------------------------------------
# Carve spec — built once from the model's _FieldMeta (SOFTUNIT_SPEC §3, §6)
# ---------------------------------------------------------------------------
@dataclass
class _CarveOp:
    """
    One **class-owning** event's contribution to the youngest-first stick-break carve.
    Erosional unconformities are NOT ops (they own no class); each is applied once as
    the onlap threshold of the event that overlies it (see :func:`build_carve_spec`).

    ``kind``         : ``'depositional'`` | ``'region'``.
    ``weight_event`` : name of the event whose field is the overprint **domain** for
                       this event's weight ``w_E`` — the unconformity it onlaps
                       (``None`` ⇒ ``w_E = 1``, the oldest basement claiming all remaining).
    ``weight_iso``   : iso-id (str) of the onlap threshold ``θ`` for ``w_E`` (``None`` ⇒ ``w_E = 1``).
    ``band_event``   : name of the event whose **own** field carves its internal bands
                       (depositional only; ``None`` otherwise).
    ``band_classes`` : class ids of the bands this event owns, youngest→oldest.
    ``contact_isos`` : iso-ids ``θ_1 … θ_m`` of the depositional contacts
                       (``len == len(band_classes) - 1``).
    """

    kind: str
    weight_event: str = None
    weight_iso: str = None
    band_event: str = None
    band_classes: list = _dcfield(default_factory=list)
    contact_isos: list = _dcfield(default_factory=list)


@dataclass
class CarveSpec:
    """
    Pre-computed recipe for the soft carve, derived once from ``model.field_meta``.

    Attributes
    ----------
    ops : list[_CarveOp]
        Per-event carve operations in **youngest→oldest** order (stick-break order).
    n_classes : int
        Number of unit-level classes (simplex width ``C``).
    level_to_class : dict[int, int]
        Maps each unit level to its dense class id.
    class_levels : list[int]
        Class id → unit level (ascending level).
    iso_plan : list[tuple]
        Learnable-iso plan: ``('free', iso_id)`` for an erosional unconformity, or
        ``('package', event_name, m, levels)`` for a depositional package's
        ``θ_0`` + ``φ`` reparam.
    """

    ops: list
    n_classes: int
    level_to_class: dict
    class_levels: list
    iso_plan: list


def build_class_index(model):
    """
    Map each unit **level** to a dense class id (SOFTUNIT_SPEC §2.3).

    Classes are unit levels — the data's own labels — not ``lithoID`` (which runs
    opposite to ``level``). Returns ``(level_to_class, class_levels)`` where
    ``class_levels[c]`` is the unit level of class ``c`` (ascending level).
    """
    levels = sorted({L for meta in model.field_meta for L in meta.levels})
    level_to_class = {L: i for i, L in enumerate(levels)}
    return level_to_class, levels


def build_carve_spec(model) -> CarveSpec:
    """
    Derive the :class:`CarveSpec` from a unit-only model's ``field_meta``.

    Walks the events oldest→youngest (the build order, matching ``model.events``) and
    builds one carve op per **class-owning** event (depositional packages + the basement
    / top region), recording the learnable-iso plan. A class-owning event's onlap weight
    uses the **nearest older unconformity** field + iso (curlew's onlap ``Overprint``
    resolves the same parent threshold), so that unconformity's truncation is applied
    exactly once; the oldest basement region uses ``w = 1``. Erosional unconformities are
    **not** ops — they own no class and their truncation is the onlap weight above — they
    only contribute a learnable iso (see :class:`_CarveOp`).

    Parameters
    ----------
    model : curlew.geology.geomodel.GeoModel
        A model built by :func:`curlew.geology.stratbuilder.build_geomodel`'s
        unit-only path (so ``field_meta`` / ``level_points`` are present).

    Returns
    -------
    CarveSpec
    """
    level_to_class, class_levels = build_class_index(model)
    ops, iso_plan = [], []
    last_unconf_name = None
    last_unconf_iso = None

    for meta in model.field_meta:
        ev = model[meta.name]
        if meta.is_unconformity:
            # An erosional unconformity owns NO class and is NOT a carve op: its
            # truncation is applied exactly once by the class-owning event that onlaps
            # it (whose onlap weight uses this same unconformity field + iso). Adding a
            # separate erosional op would double-truncate the surface AND discard
            # remaining·w·(1−w) of mass into the renormaliser — a degenerate CE minimum
            # (push w→0.5 to shrink the simplex sum) that does not separate units. So we
            # only register its learnable iso; the onlapping op references it below.
            iso_id = f"{meta.name}::unconf"
            iso_plan.append(("free", iso_id))
            last_unconf_name, last_unconf_iso = meta.name, iso_id
        elif meta.kind == "depositional":
            levels = list(meta.levels)  # young → old
            classes = [level_to_class[L] for L in levels]
            m = len(levels) - 1
            contact_isos = [f"{meta.name}::c{j}" for j in range(m)]
            if m > 0:
                iso_plan.append(("package", meta.name, m, levels))
            ops.append(_CarveOp(
                kind="depositional", weight_event=last_unconf_name, weight_iso=last_unconf_iso,
                band_event=meta.name, band_classes=classes, contact_isos=contact_isos,
            ))
        else:  # region-only (basement / dropped top unit)
            classes = [level_to_class[L] for L in meta.levels]
            onlap = (ev.overprint is not None
                     and getattr(ev.overprint, "defaultDomain", "child") == "parent"
                     and last_unconf_name is not None)
            ops.append(_CarveOp(
                kind="region",
                weight_event=last_unconf_name if onlap else None,
                weight_iso=last_unconf_iso if onlap else None,
                band_classes=classes,
            ))

    ops.reverse()  # youngest → oldest (stick-break order)
    return CarveSpec(ops=ops, n_classes=len(class_levels),
                     level_to_class=level_to_class, class_levels=class_levels,
                     iso_plan=iso_plan)


# ---------------------------------------------------------------------------
# The soft carve (SOFTUNIT_SPEC §3)
# ---------------------------------------------------------------------------
def soft_unit_probs(model, x, iso_values: dict, tau: float, carve_spec: CarveSpec) -> torch.Tensor:
    """
    Per-class re-derivation of :meth:`curlew.core.Geode.combine` (SOFTUNIT_SPEC §3).

    Accumulates an ``(N, C)`` probability simplex with the same youngest-first
    stick-break and the same ``Overprint`` sigmoid/threshold form curlew uses, but
    per class instead of a scalar:

        ``remaining = 1``; for E youngest→oldest:
            ``w_E = σ(s·(domain_E − θ_E))``  (``s = 1/τ``; ``w = 1`` for the basement)
            ``claim = remaining · w_E``; distribute ``claim`` over E's band classes;
            ``remaining ← remaining · (1 − w_E)``.

    Within a depositional package the bands are consecutive sigmoid differences using
    the package's **own** field and contacts ``θ_1 > … > θ_m`` (descending by the §5
    reparam, so the band probs are non-negative without clamping).

    Field values are obtained through ``ev.predict(combine=False, transform=True)``,
    which walks the deformation chain (faults/intrusions retro-deform for free) without
    needing iso-values or running any overprint — so ``GeoModel.predict`` is untouched.

    Parameters
    ----------
    model : curlew.geology.geomodel.GeoModel
        The model being trained.
    x : torch.Tensor
        ``(N, d)`` query points in **model** coordinates.
    iso_values : dict[str, torch.Tensor]
        Maps each iso-id to a (grad-carrying) scalar threshold (see
        :meth:`UnitLoss.resolve_isos`).
    tau : float
        Soft-unit temperature; sharpness ``s = 1/τ``.
    carve_spec : CarveSpec
        Pre-computed carve recipe.

    Returns
    -------
    torch.Tensor
        ``(N, C)`` row-normalised probability simplex.
    """
    if tau <= 0:
        raise ValueError("tau must be > 0")
    s = 1.0 / tau

    # evaluate each referenced event's field once (deformation-correct, combine=False)
    field_cache = {}

    def field_of(name):
        if name not in field_cache:
            g = model[name].predict(x, combine=False, transform=True, to_numpy=False,
                                    litho=False, props=False, isosurfaces=False)
            field_cache[name] = g.scalar.reshape(-1)
        return field_cache[name]

    n = x.shape[0]
    probs = torch.zeros((n, carve_spec.n_classes), device=curlew.device, dtype=curlew.dtype)
    remaining = torch.ones(n, device=curlew.device, dtype=curlew.dtype)

    for op in carve_spec.ops:
        # overprint weight w_E (the SAME form Overprint._soft_weight uses)
        if op.weight_event is None:
            w = torch.ones(n, device=curlew.device, dtype=curlew.dtype)  # basement: claims all
        else:
            w = torch.sigmoid(s * (field_of(op.weight_event) - iso_values[op.weight_iso]))
        claim = remaining * w

        classes = op.band_classes
        if op.kind == "depositional" and len(classes) > 1:
            f = field_of(op.band_event)
            # σ(s·(f − θ_j)) for each descending contact θ_1 … θ_m
            sig = [torch.sigmoid(s * (f - iso_values[k])) for k in op.contact_isos]
            probs[:, classes[0]] = probs[:, classes[0]] + claim * sig[0]            # P(b_0)
            for j in range(1, len(classes) - 1):
                probs[:, classes[j]] = probs[:, classes[j]] + claim * (sig[j] - sig[j - 1])
            probs[:, classes[-1]] = probs[:, classes[-1]] + claim * (1.0 - sig[-1])  # P(b_m)
        elif classes:
            # single-band depositional or region-only: split claim across its level(s)
            share = claim / float(len(classes))
            for c in classes:
                probs[:, c] = probs[:, c] + share

        remaining = remaining * (1.0 - w)

    # every op owns ≥1 class and no mass is discarded, so the sum is already ~1; the
    # clamp/divide only guards float drift (no degenerate renormalisation pressure).
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(EPS)


# ---------------------------------------------------------------------------
# Iso-value initialisation (SOFTUNIT_SPEC §5)
# ---------------------------------------------------------------------------
def iso_init_values(model, carve_spec: CarveSpec) -> tuple:
    """
    Field-free **uniform** initial iso-values (SOFTUNIT_SPEC §5). The init needs only a
    correct *order* (from the strat column) + rough spacing — correctness is owned by the
    monotone reparam and the always-active CE, so the init is a convergence aid only and
    needs no spatial assumption:

    - each depositional package starts at ``θ_0 = +1`` (top of the ~unit field range) with
      uniform ``softplus(φ)`` gaps down the range;
    - each erosional unconformity iso starts at 0 (mid-range).

    (A data-driven 'trend' init was trialled and dropped: uniform won the wcsb A/B
    decisively, and since the reparam + CE own ordering and spacing the init is not
    load-bearing — so the extra machinery wasn't worth keeping.)

    Returns ``(free_init, pkg_init)`` where ``free_init[iso_id] = value`` and
    ``pkg_init[event_name] = (theta0, gaps)`` (``gaps`` a strictly positive ``(m,)`` array).
    """
    free_init, pkg_init = {}, {}
    for entry in carve_spec.iso_plan:
        if entry[0] == "free":
            free_init[entry[1]] = 0.0
        else:  # ('package', name, m, levels)
            _, name, m, _levels = entry
            pkg_init[name] = (1.0, np.full(m, 2.0 / (m + 1)))
    return free_init, pkg_init


# ---------------------------------------------------------------------------
# The coupling loss (SOFTUNIT_SPEC §2, §4, §6)
# ---------------------------------------------------------------------------
class UnitLoss(LearnableBase):
    """
    Unit (soft-carve / NLL) coupling loss (Avenue B of SOFTUNIT_SPEC).

    Owns the learnable iso-values (per-package monotone reparam + per-unconformity
    free values, §5), the level→class map and per-event carve recipe (built once from
    ``model.field_meta``), and the balanced per-level point pools + labels (from
    ``model.level_points``). Passed to :meth:`curlew.geology.geomodel.GeoModel.fit`
    via ``custom_loss=[...]``: each epoch it samples a balanced batch, evaluates the
    field stack through ``predict(combine=False)``, carves an ``(N, C)`` simplex
    (:func:`soft_unit_probs`) and returns its NLL against the true unit levels in a
    :class:`~curlew.core.Pebble` carrying the **iso-value optimiser** (the per-field
    optimisers are already in the pebble via the retained ``overturn`` terms, so the
    fields co-train under the same backward pass).

    Use :func:`curlew.geology.stratbuilder.attach_unit_loss` to build one wired to a
    model (it also sets ``iq_norm_weight = 0`` and relaxes ``overturn_weight`` on the
    coupled fields). After fitting, call :meth:`write_isosurfaces` to push the learned
    iso-values into the ordinary ``addIsosurface(value=...)`` so ``GeoModel.predict``
    reproduces the model.
    """

    def __init__(self, model, *, tau: float = 0.05, points_per_level: int = 1024,
                 iso_lr: float = None, weight: float = 1.0, name: str = "unit"):
        """
        Parameters
        ----------
        model : curlew.geology.geomodel.GeoModel
            A model built by the unit-only path (``field_meta`` / ``level_points`` set).
        tau : float, optional
            Soft-unit temperature; sharpness ``s = 1/τ`` (default 0.05 ⇒ ``s = 20``,
            matching GeoINR). Fixed (no annealing in v1).
        points_per_level : int, optional
            Points sampled per level each epoch (balanced sampling, default 1024).
        iso_lr : float, optional
            Learning rate for the iso-value optimiser. Defaults to ~10× a coupled
            field's learning rate (GeoINR convention).
        weight : float, optional
            Weight on the NLL term (default 1.0).
        name : str, optional
            Loss group name (default ``"unit"``).
        """
        super().__init__()
        assert getattr(model, "level_points", None) is not None, (
            "UnitLoss requires a model built by build_geomodel's unit-only path "
            "(model.level_points missing)."
        )
        self.name = name
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError("tau must be > 0")
        self.points_per_level = int(points_per_level)
        self.weight = float(weight)

        # carve recipe + class index (SOFTUNIT_SPEC §3)
        self.spec = build_carve_spec(model)
        self.n_classes = self.spec.n_classes

        # learnable iso parameters from the plan (monotone reparam, §5)
        self._params = nn.ParameterList()
        self._free_iso = {}   # iso_id -> param index
        self._pkg = {}        # event name -> (theta0_idx, phi_idx, [contact_ids])
        free_init, pkg_init = iso_init_values(model, self.spec)
        for entry in self.spec.iso_plan:
            if entry[0] == "free":
                _, iso_id = entry
                self._params.append(nn.Parameter(_tensor(float(free_init[iso_id]))))
                self._free_iso[iso_id] = len(self._params) - 1
            else:  # ('package', name, m, levels)
                _, ename, m, _levels = entry
                theta0, gaps = pkg_init[ename]
                self._params.append(nn.Parameter(_tensor(float(theta0))))
                t0_idx = len(self._params) - 1
                self._params.append(nn.Parameter(_tensor(_inv_softplus(gaps))))
                phi_idx = len(self._params) - 1
                self._pkg[ename] = (t0_idx, phi_idx, [f"{ename}::c{j}" for j in range(m)])

        # balanced per-class pools (model coords) + labels (SOFTUNIT_SPEC §4)
        self._init_pools(model)

        # iso-value optimiser (iso_lr ≈ 10× field lr per GeoINR)
        if iso_lr is None:
            iso_lr = self._default_iso_lr(model)
        self.init_optim(lr=iso_lr)

    # -- iso resolution --------------------------------------------------------
    def resolve_isos(self) -> dict:
        """
        Resolve every iso-id to a (grad-carrying) scalar threshold tensor.

        Free (unconformity) isos return their Parameter directly; a package's contacts
        are computed via the monotone reparam :func:`monotone_contacts`, so they always
        descend and carry gradient to ``θ_0`` and ``φ``.
        """
        out = {iso_id: self._params[idx] for iso_id, idx in self._free_iso.items()}
        for _name, (t0_idx, phi_idx, contact_ids) in self._pkg.items():
            contacts = monotone_contacts(self._params[t0_idx], self._params[phi_idx])
            for j, cid in enumerate(contact_ids):
                out[cid] = contacts[j]
        return out

    # -- sampling --------------------------------------------------------------
    def _init_pools(self, model):
        """Stash each class's (model-coord) unit points as a device tensor for balanced sampling."""
        self._pool = {}  # class id -> (n, d) tensor
        for L, c in self.spec.level_to_class.items():
            pts = model.level_points.get(L, None)
            if pts is not None and len(pts) > 0:
                self._pool[c] = _tensor(np.asarray(pts, float))

    def sample_balanced_unit_indices(self):
        """
        Draw ``points_per_level`` points per class (all if the pool is smaller),
        concatenate and shuffle (SOFTUNIT_SPEC §4). Balanced sampling stops large units
        dominating the cross-entropy.

        Returns
        -------
        coords : torch.Tensor
            ``(M, d)`` model-coordinate batch.
        labels : torch.Tensor
            ``(M,)`` class ids.
        """
        coords_list, labels_list = [], []
        for c, pts in self._pool.items():
            n = pts.shape[0]
            if n <= self.points_per_level:
                sel = torch.arange(n, device=pts.device)
            else:
                sel = torch.randperm(n, device=pts.device)[:self.points_per_level]
            coords_list.append(pts.index_select(0, sel))
            labels_list.append(torch.full((sel.numel(),), c, dtype=torch.long, device=curlew.device))
        coords = torch.cat(coords_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        perm = torch.randperm(coords.shape[0], device=coords.device)
        return coords.index_select(0, perm), labels.index_select(0, perm)

    def _default_iso_lr(self, model):
        for meta in model.field_meta:
            opt = getattr(model[meta.name].getField(0), "optim", None)
            if opt is not None:
                return 10.0 * opt.param_groups[0]["lr"]
        return 1e-3

    # -- the custom-loss callable ---------------------------------------------
    def forward(self, pebble, model, C):
        """
        Compute the soft-unit NLL for one epoch and return a :class:`~curlew.core.Pebble`.

        Signature matches ``GeoModel.fit``'s ``custom_loss`` hook
        (``f(pebble, model, C) -> Pebble``); ``pebble`` (per-event terms) and ``C`` are
        unused. The returned pebble carries the **iso-value optimiser** under this
        loss's group; the per-field optimisers are already present via the retained
        per-event ``overturn`` terms, so ``GeoModel.fit``'s single ``total().backward()``
        co-trains the fields and the iso-values.
        """
        coords, labels = self.sample_balanced_unit_indices()
        iso_values = self.resolve_isos()
        probs = soft_unit_probs(model, coords, iso_values, self.tau, self.spec)
        nll = F.nll_loss(probs.clamp_min(EPS).log(), labels)

        out = Pebble()
        out.push(self.name, "nll_loss", nll, weight=self.weight, optim=self.optim)
        return out

    # -- post-fit --------------------------------------------------------------
    def write_isosurfaces(self, model) -> dict:
        """
        Push the learned iso-values into the events' ordinary ``addIsosurface(value=...)``
        so ``GeoModel.predict`` reproduces the model (SOFTUNIT_SPEC §4, post-fit).

        Mirrors :func:`curlew.geology.stratbuilder.estimate_isosurfaces`' insertion
        order — each erosional unconformity gets its single iso; each depositional
        package's contacts are inserted **oldest-first** so ascending ``lithoID`` =
        younger (curlew numbers isos in insertion order). Uses the learned ``θ`` instead
        of the post-hoc medians, so the carve's assembly and predict's hard assembly use
        the *same* thresholds (the §7 soft-vs-hard cross-check).

        Returns
        -------
        dict
            Maps each isosurface lithology key (``"<event>_<iso name>"``) to its value.
        """
        iso_values = {k: float(v.detach().cpu()) for k, v in self.resolve_isos().items()}
        out = {}
        for meta in model.field_meta:
            ev = model[meta.name]
            if meta.is_unconformity:
                v = iso_values[f"{meta.name}::unconf"]
                ev.addIsosurface(meta.iso_name, value=v)
                out[f"{meta.name}_{meta.iso_name}"] = v
            elif meta.kind == "depositional" and len(meta.contacts) >= 1:
                # contacts[j] ↔ θ_{j+1} (descending); insert oldest-first
                vals = [iso_values[f"{meta.name}::c{j}"] for j in range(len(meta.contacts))]
                for contact, v in zip(reversed(meta.contacts), reversed(vals)):
                    ev.addIsosurface(contact.name, value=v)
                    out[f"{meta.name}_{contact.name}"] = v
        return out
