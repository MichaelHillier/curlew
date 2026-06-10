"""
Strat-column → :class:`~curlew.geology.geomodel.GeoModel` builder.

Turns a stratigraphic-column CSV plus point observations into a ready-to-fit
``GeoModel``, reusing Curlew's native primitives (events, ``CSet`` constraints,
seed isosurfaces, ``Transform``). The stratigraphic *partitioning* is ported
verbatim from GeoINR's clean, dependency-light ``stratigraphy.py``
(``load_strat_column_csv`` / ``derive_horizons`` / ``derive_scalar_fields``);
the relational (above/below/equality) constraints are generated here from the
bespoke rules in the integration spec, **not** from GeoINR's convoluted
``derive_relational_constraints``.

Two construction paths are supported, chosen automatically from the observations:

- **On-surface path** (e.g. ``multilayer_fold``) — when contact (interface) points
  exist, each depositional package is one ``strati`` event with ``eq`` traces, seed
  isosurfaces and ``gv`` gradients (see :func:`_build_depositional_event`).
- **Unit-only path** (e.g. ``wcsb``) — when only unit (level-labelled) points exist,
  the column is built as an alternating chain of **erosional** unconformities and
  **depositional** packages, with synthesized **region-only** events at both ends
  (the eroded basement at the bottom, the dropped top baselap unit at the top).
  Erosional events truncate older geology at their own iso (``mode='above'``);
  depositional / region-only packages **onlap** the unconformity directly below them
  (``onlap=True``). Because no on-surface points exist, every surface is a **learnable
  iso-value** (:meth:`~curlew.geology.geoevent.GeoEvent.addIsosurface` with
  ``learnable=``) constrained by the units **above and below** it via the
  gradient-normalised stratigraphic-bound loss (``CSet.sb`` / ``HSet.sb_loss``), with a
  **no-overturn** regularizer (``HSet.overturn_loss``) keeping the field monotone in the
  younging direction. This is GeoINR's per-field ``pairwise=False`` recipe; the learnable
  iso serves as both the ``Overprint`` threshold and the extraction iso. ``cap_per_unit``
  subsamples each unit so rare units stay represented (SPEC §4.4).
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass, field as _dcfield
from pathlib import Path

import numpy as np

from curlew.core import CSet, HSet
from curlew.geology import strati
from curlew.geology.geomodel import GeoModel
from curlew.geometry import Transform, Grid
from curlew.fields.geoinr import GeoINR, Siren

# field-name → class, used to resolve the ``field=`` argument
_FIELD_TYPES = {"GeoINR": GeoINR, "Siren": Siren}

# Relations a strat-column row may declare. ``conformal``/``baselap`` are
# *stratigraphic* (a depositional contact); ``eroded`` is an unconformity.
SUPPORTED_RELATIONS = ("conformal", "baselap", "eroded")
STRATIGRAPHIC_RELATIONS = {"conformal", "baselap"}


# ---------------------------------------------------------------------------
# Strat-column partitioning (ported from GeoINR geoinr/input/stratigraphy.py)
# ---------------------------------------------------------------------------
@dataclass
class UnitRow:
    """One row of a stratigraphic column: a named unit with a relation and level."""

    name: str
    relation: str
    level: int


@dataclass
class HorizonRow:
    """A contact (horizon) at the base of a unit. The oldest unit has no horizon."""

    unit_index: int
    unit_level: int
    unit_name: str
    relation: str
    kind: str  # "stratigraphic" | "unconformity"
    display_name: str


@dataclass
class ScalarFieldRow:
    """
    A derived scalar field: a package of one or more units sharing one implicit
    function. ``kind`` is ``"depositional"`` or ``"erosional"``;
    ``horizon_unit_indices`` are the units whose base contact this field carries.
    """

    field_index: int
    name: str
    kind: str
    unit_indices: list
    horizon_unit_indices: list

    @property
    def feature_count(self) -> int:
        return len(self.horizon_unit_indices)


def _normalize_relation(value, *, row_number=None) -> str:
    relation = str(value).strip().lower()
    if relation not in SUPPORTED_RELATIONS:
        loc = f" on row {row_number}" if row_number is not None else ""
        raise ValueError(
            f"Unsupported relation '{value}'{loc}. "
            f"Supported relations are: {', '.join(SUPPORTED_RELATIONS)}."
        )
    return relation


def _parse_level(value, *, row_number) -> int:
    try:
        return int(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"Invalid level '{value}' on row {row_number}.") from exc


def _require_columns(fieldnames):
    if not fieldnames:
        raise ValueError("Stratigraphic column CSV is missing a header row.")
    normalized = {name.strip().lower(): name for name in fieldnames}
    missing = [n for n in ("name", "relation", "level") if n not in normalized]
    if missing:
        raise ValueError(
            f"Stratigraphic column CSV is missing required columns: {', '.join(missing)}."
        )
    return normalized


def load_strat_column_csv(path) -> list:
    """
    Parse a stratigraphic-column CSV into a list of :class:`UnitRow`.

    The CSV must have (case-insensitive) ``name``, ``relation`` and ``level``
    columns. Rows are youngest-first, matching GeoINR's convention.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the strat-column CSV.

    Returns
    -------
    list[UnitRow]
        One row per unit, in file (youngest-first) order.
    """
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = _csv.DictReader(handle)
        columns = _require_columns(reader.fieldnames)
        units = []
        for row_number, row in enumerate(reader, start=2):
            units.append(
                UnitRow(
                    name=str(row[columns["name"]] or "").strip(),
                    relation=_normalize_relation(row[columns["relation"]], row_number=row_number),
                    level=_parse_level(row[columns["level"]], row_number=row_number),
                )
            )
    return units


def derive_horizons(units: list) -> list:
    """
    Derive the list of contacts (horizons) from a unit list.

    Every unit except the oldest (index 0) contributes one horizon — its base
    contact, named ``"<unit> Top"`` (conformal/baselap) or ``"<unit> Unconformity"``
    (eroded).
    """
    horizons = []
    for unit_index, unit in enumerate(units):
        if unit_index == 0:
            continue
        kind = "unconformity" if unit.relation == "eroded" else "stratigraphic"
        suffix = "Unconformity" if kind == "unconformity" else "Top"
        base = unit.name.strip() or f"Level {unit.level}"
        horizons.append(
            HorizonRow(
                unit_index=unit_index,
                unit_level=unit.level,
                unit_name=unit.name,
                relation=unit.relation,
                kind=kind,
                display_name=f"{base} {suffix}",
            )
        )
    return horizons


def _append_scalar_field(fields, *, kind, unit_indices, horizon_unit_indices):
    idx = len(fields)
    fields.append(
        ScalarFieldRow(
            field_index=idx,
            name=f"scalar_field{idx}",
            kind=kind,
            unit_indices=list(unit_indices),
            horizon_unit_indices=list(horizon_unit_indices),
        )
    )


def derive_scalar_fields(units: list) -> list:
    """
    Partition a unit list (youngest-first) into scalar fields.

    Consecutive units are grouped into packages bounded by ``eroded`` units. Each
    ``eroded`` unit becomes its own ``erosional`` field; the run of non-eroded
    units after it becomes a ``depositional`` field whose ``horizon_unit_indices``
    are the conformal/baselap tops within the run. A field is appended only when it
    has at least one stratigraphic horizon.

    Returned in youngest-first order (reverse for oldest-first ``GeoModel`` order).
    """
    fields = []
    n_units = len(units)
    if n_units == 0:
        return fields

    i = 0
    if units[0].relation == "conformal":
        end = 0
        while end + 1 < n_units and units[end + 1].relation != "eroded":
            end += 1
        horizon_unit_indices = [
            idx for idx in range(1, end + 1) if units[idx].relation in STRATIGRAPHIC_RELATIONS
        ]
        if horizon_unit_indices:
            _append_scalar_field(
                fields, kind="depositional",
                unit_indices=list(range(0, end + 1)),
                horizon_unit_indices=horizon_unit_indices,
            )
        i = end + 1
    elif units[0].relation == "baselap":
        i = 1

    while i < n_units:
        relation = units[i].relation
        if relation == "eroded":
            horizon_unit_indices = [i] if i > 0 else []
            attach_unit_to_erosion = i == n_units - 1 or (
                i + 1 < n_units and units[i + 1].relation == "eroded"
            )
            _append_scalar_field(
                fields, kind="erosional",
                unit_indices=[i] if attach_unit_to_erosion else [],
                horizon_unit_indices=horizon_unit_indices,
            )
            if attach_unit_to_erosion:
                i += 1
                continue
            end = i
            while end + 1 < n_units and units[end + 1].relation != "eroded":
                end += 1
            package_horizon_unit_indices = [
                idx for idx in range(i + 1, end + 1) if units[idx].relation in STRATIGRAPHIC_RELATIONS
            ]
            if package_horizon_unit_indices:
                _append_scalar_field(
                    fields, kind="depositional",
                    unit_indices=list(range(i, end + 1)),
                    horizon_unit_indices=package_horizon_unit_indices,
                )
            i = end + 1
            continue

        end = i
        while end + 1 < n_units and units[end + 1].relation != "eroded":
            end += 1
        package_horizon_unit_indices = [
            idx for idx in range(i + 1, end + 1) if units[idx].relation in STRATIGRAPHIC_RELATIONS
        ]
        if package_horizon_unit_indices:
            _append_scalar_field(
                fields, kind="depositional",
                unit_indices=list(range(i, end + 1)),
                horizon_unit_indices=package_horizon_unit_indices,
            )
        i = end + 1

    return fields


# ---------------------------------------------------------------------------
# Normalization transform
# ---------------------------------------------------------------------------
def _normalization_transform(bounds, scale: str, ndim: int) -> Transform:
    """
    Build a homogeneous global→model transform mapping the data bounds into
    ``[-1, 1]``.

    ``scale="isometric"`` uses one scale on all axes (``2/max_extent``);
    ``scale="geo_extensive"`` uses a separate xy scale and z scale (vertical
    exaggeration).
    """
    lo, hi = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    extent = np.maximum(hi - lo, 1e-12)
    center = 0.5 * (hi + lo)

    s = np.ones(ndim)
    if scale == "isometric":
        s[:] = 2.0 / extent.max()
    elif scale == "geo_extensive":
        xy = 2.0 / extent[: max(ndim - 1, 1)].max()
        s[: ndim - 1] = xy
        s[ndim - 1] = 2.0 / extent[ndim - 1]
    else:
        raise ValueError(f"Unknown scale mode '{scale}'. Use 'isometric' or 'geo_extensive'.")

    # homogeneous (ndim+1) matrix: scale then translate centre to origin
    M = np.eye(ndim + 1)
    M[:ndim, :ndim] = np.diag(s)
    M[:ndim, ndim] = -s * center
    return Transform(M)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
@dataclass
class _ContactInfo:
    """Per-contact bookkeeping carried on each built event for visualisation/checks."""

    name: str
    unit_index: int
    level: int
    n_points: int


@dataclass
class _FieldMeta:
    """Description of one built event (one derived scalar field)."""

    name: str
    kind: str
    unit_indices: list
    contacts: list = _dcfield(default_factory=list)  # list[_ContactInfo], youngest-first
    inequalities: list = _dcfield(default_factory=list)  # list[(nameA, nameB, '>'|'<')] feature pairs
    # --- unit-only (wcsb) chain metadata ---
    is_unconformity: bool = False     # True for erosional events
    levels: list = _dcfield(default_factory=list)        # unit levels owned by this event (lithology bands)
    above_levels: list = _dcfield(default_factory=list)  # erosional: younger (above) unit levels
    below_levels: list = _dcfield(default_factory=list)  # erosional: older (below) unit levels
    iso_name: str = None              # name of the seed isosurface (unconformity threshold)
    level_litho: dict = _dcfield(default_factory=dict)   # level -> predicted lithology name
    n_iq: int = 0                     # number of inequality pools on this event
    sb_relations: list = _dcfield(default_factory=list)  # list[(level, iso_name, '>'|'<')] exact sb constraints


def _default_hset(has_normals: bool, has_iq: bool) -> HSet:
    """
    A minimal HSet for a depositional field: only the active terms are non-zero.

    ``eq_loss`` is up-weighted because the trace loss is mean-normalised (``var/mu²``)
    and the field's absolute level is an unconstrained gauge, so a weight of 1 leaves
    contacts loosely pinned; ~10 tightens them without collapsing the field's range
    (very large weights destabilise the gauge).
    """
    H = HSet().zero()
    H.eq_loss = 10.0
    if has_normals:
        H.grad_loss = 1.0
    if has_iq:
        H.iq_loss = 1.0
    return H


def _build_depositional_event(sf, units, obs, Xm, ndim, field_cls, field_kwargs,
                              hset, iq_samples):
    """
    Build a ``strati`` event for a depositional (one-or-more-contact) scalar field.

    Constraints (all expressed in *model* coordinates, ``crs='model'``):

    - one ``eq`` trace + one seed isosurface per contact,
    - ``gv`` gradient constraints from bedding normals (where present),
    - ``iq`` inequalities ordering younger contacts above older ones.
    """
    # contacts in youngest→oldest order (smaller unit_index == younger)
    contact_indices = sorted(sf.horizon_unit_indices)
    iface_by_level = obs.by_level("interface")
    level_to_idx = {u.level: i for i, u in enumerate(units)}

    eq_traces = []
    iso_specs = []          # (name, model_points)
    contacts = []           # _ContactInfo youngest-first
    contact_pools = []      # model-coord point pools, youngest-first
    for ui in contact_indices:
        unit = units[ui]
        pts_global = iface_by_level.get(unit.level, np.empty((0, ndim)))
        # global → model (Xm holds all obs already transformed; re-transform here
        # keeps this helper independent of obs row order)
        m = (obs.is_interface) & (obs.level == unit.level)
        pts_model = Xm[m]
        if pts_model.shape[0] == 0:
            # depositional contact with no on-surface points is out of scope here
            raise ValueError(
                f"Contact for unit '{unit.name}' (level {unit.level}) has no interface "
                f"points; depositional contacts require on-surface observations."
            )
        name = unit.name.strip() or f"level{unit.level}"
        eq_traces.append(pts_model)
        iso_specs.append((name, pts_model))
        contact_pools.append(pts_model)
        contacts.append(_ContactInfo(name=name, unit_index=ui, level=unit.level,
                                     n_points=pts_model.shape[0]))

    # gradient (normal) constraints — assign normals to this field by level, or all
    # normals when they carry no level (single-field datasets like multilayer_fold).
    nco, nvec, nlev = obs.normal_points()
    field_levels = {units[u].level for u in sf.unit_indices}
    if nco.shape[0] > 0:
        if np.all(nlev < 0):
            sel = np.ones(nco.shape[0], dtype=bool)
        else:
            sel = np.array([int(l) in field_levels for l in nlev])
    else:
        sel = np.zeros(0, dtype=bool)
    gp = gv = None
    if sel.any():
        # model coordinates for the normal positions
        nmask = obs._has_normal()
        gp = Xm[nmask][sel]
        gvec = nvec[sel].astype(float)
        # normalise to unit poles; under isometric scaling direction is preserved,
        # so the global-space normal direction is valid in model space.
        gvec = gvec / (np.linalg.norm(gvec, axis=1, keepdims=True) + 1e-12)
        gv = gvec

    # inequality constraints — younger contact > older contact, for every pair
    iq_pairs = []
    contact_inequalities = []  # (younger_name, older_name, '>') for the relation matrix
    for a in range(len(contact_pools)):
        for b in range(a + 1, len(contact_pools)):
            # contact_pools[a] is younger than contact_pools[b]
            iq_pairs.append((contact_pools[a], contact_pools[b], ">"))
            contact_inequalities.append((contacts[a].name, contacts[b].name, ">"))
    # unit-point inequalities (where unit points exist): unit pts > contact below,
    # unit pts < contact above. multilayer_fold has none, so this is usually empty.
    unit_by_level = obs.by_level("unit")
    if unit_by_level:
        # map each contact to a model-coord pool keyed by level
        pool_by_level = {c.level: p for c, p in zip(contacts, contact_pools)}
        umask = (~obs.is_interface) & (~obs._has_normal())
        for ui in sf.unit_indices:
            lvl = units[ui].level
            if lvl not in unit_by_level:
                continue
            up = Xm[umask & (obs.level == lvl)]
            if up.shape[0] == 0:
                continue
            # contact directly below this unit shares the unit's own level (its base)
            if lvl in pool_by_level:
                iq_pairs.append((up, pool_by_level[lvl], ">"))

    # assemble the constraint set (model coordinates)
    C = CSet(crs="model")
    if eq_traces:
        C.eq = eq_traces
    if gp is not None:
        C.gp, C.gv = gp, gv
    if iq_pairs:
        C.iq = (int(iq_samples), iq_pairs)

    H = hset.copy() if hset is not None else _default_hset(gp is not None, bool(iq_pairs))

    fkw = dict(field_kwargs or {})
    fobj = field_cls(sf.name, H=H, C=C, input_dim=ndim, **fkw)
    ev = strati(sf.name, C=fobj)
    for name, pts_model in iso_specs:
        ev.addIsosurface(name, seed=pts_model)

    meta = _FieldMeta(name=sf.name, kind=sf.kind,
                      unit_indices=list(sf.unit_indices), contacts=contacts,
                      inequalities=contact_inequalities)
    return ev, meta


# ---------------------------------------------------------------------------
# Unit-only (wcsb) chain — erosional / region-only / seed-iso branches
# ---------------------------------------------------------------------------
# When observations carry only unit (level-labelled) points (no contact points,
# no normals), the column is built as an alternating chain of erosional
# unconformities and depositional packages. Helpers below construct each branch.

#: Name given to every erosional event's learnable iso (its threshold). The
#: name only needs to be unique *within* an event, so it can be reused across
#: events: resolution always happens on the specific event/parent.
_UNCONF_ISO = "unconformity"

#: Default per-field loss weights for the unit-only path. The no-overturn term is heavily
#: up-weighted vs ``sb_loss``: curlew assembles the model with a sequential onlap/truncation
#: combine (not GeoINR's global per-point carve), so a strongly monotone field is what keeps
#: the *unconstrained deep region* (e.g. the basement) reliably below every younger
#: unconformity's iso and so not wrongly truncated. ``overturn_loss`` uses the gradient
#: *magnitude* (GeoINR — the norm drives polarity correction), so it couples to the field
#: scale; that is fine because the unit-only path pins the scale ~1 (:data:`_FIELD_SCALE`),
#: making the term naturally O(1). (Per-package iso *ordering* is now guaranteed separately, by
#: the monotone iso group — :meth:`~curlew.geology.geoevent.GeoEvent.addOrderedIsosurfaces` —
#: not by this term.) ``30`` is a balanced point at 1500 epochs: band ~0.49, structure ~0.76;
#: the basement (deepest, region-only) stays low (~0.1, climbing slowly with weight) — that is
#: the per-field/combine ceiling, whose real fix is soft-unit/NLL coupling, not a bigger weight.
_SB_WEIGHT = 1.0
_OVERTURN_WEIGHT = 30.0
#: Iso-values learn faster than the field (GeoINR uses ~10x the field learning rate).
_ISO_LR_MULT = 10.0

#: Field output range for the unit-only path. Both ``Siren`` and ``GeoINR`` evaluate as
#: ``scale * mlp(x)``, and the raw network ``mlp(x)`` is O(1) at initialisation, so this
#: ``scale`` *is* the field's value range (``BaseNF``'s default is ``1e2``). It is held at
#: ~1 here so every field lives in ``[-1, 1]`` and the learnable iso-values can be placed
#: at fixed, **field-type-independent** positions in that range (see :func:`_iso_inits`).
#: Keeping ``scale`` ~1 is what makes the [-1, 1] iso-init meaningful (at ``1e2`` an iso of
#: 0.78 is ≈0 *relative to* a ±100 field, so the iso placement is lost) and keeps the two
#: loss terms comparable: ``sb_loss`` is scale-free (its residual is ``‖∇f‖``-normalised) while
#: ``overturn_loss`` is magnitude-based and so naturally O(1) only when the field range is ~1.
_FIELD_SCALE = 1.0


def _iso_inits(n_iso: int) -> np.ndarray:
    """
    Initial values for ``n_iso`` learnable iso-values, spread evenly across the field's
    output range ``[-1, 1]`` (set by :data:`_FIELD_SCALE`), youngest (high) → oldest (low).

    A **single** interface initialises at ``0`` (the middle of the range); ``k`` interfaces
    sit at the ``k`` interior dividers of ``[-1, 1]`` partitioned into ``k + 1`` equal bands,
    so each unit band starts with an equal share of the range. This replaces the old
    field-evaluation-based init, which depended on the field's (now fixed) output range and
    so produced wildly different iso-values between ``Siren`` and ``GeoINR``.
    """
    n = int(n_iso)
    return np.linspace(1.0, -1.0, n + 2)[1:n + 1]


def _sb_hset() -> HSet:
    """
    HSet for a unit-only field: the GeoINR-style **stratigraphic-bound** loss
    (``sb_loss``) plus the **no-overturn** regularizer (``overturn_loss``). Both
    residuals are gradient-normalised, so the scale-free field cannot collapse to
    satisfy the bounds trivially.
    """
    return HSet().zero(sb_loss=_SB_WEIGHT, overturn_loss=_OVERTURN_WEIGHT)


def _base_lr(field_kwargs):
    """Field learning rate (the iso-values get ``_ISO_LR_MULT`` x this)."""
    return float((field_kwargs or {}).get("learning_rate", 1e-3))


def _model_grid(Xm, ndim, n=24, draw=1000):
    """A coarse :class:`~curlew.geometry.Grid` spanning the data in *model* space, used to
    sample the no-overturn regularizer. ``draw`` random points are sampled per step."""
    lo, hi = Xm.min(axis=0), Xm.max(axis=0)
    ext = np.maximum(hi - lo, 1e-6)
    center = 0.5 * (lo + hi)
    return Grid(dims=tuple(ext), step=tuple(ext / float(n)), center=tuple(center),
                sampleArgs={"N": int(draw)})


def _trend_up(ndim):
    """Younging-up trend vector (unit +z) in model coordinates."""
    t = np.zeros(ndim)
    t[-1] = 1.0
    return t


def _finalize_field_optim(fobj, field_kwargs):
    """Rebuild the field optimiser so the learnable iso-values (added after the field was
    constructed) are optimised, at ``_ISO_LR_MULT`` x the field learning rate."""
    lr = _base_lr(field_kwargs)
    fobj.init_optim(lr=lr, iso_lr=lr * _ISO_LR_MULT)


def _cap_level_pools(obs, Xm, cap_per_unit, rng):
    """
    Group unit-point model coordinates by level, capping each level at
    ``cap_per_unit`` points (uniform random subsample) so rare units stay
    represented and huge units do not dominate (SPEC §4.4).

    Returns ``dict[int, np.ndarray]`` mapping each unit level to its (capped)
    ``(n, d)`` model-coordinate array.
    """
    unit_mask = (~obs.is_interface) & (~obs._has_normal())
    pools = {}
    for L in obs.levels("unit"):
        idx = np.where(unit_mask & (obs.level == L))[0]
        if (cap_per_unit is not None) and (len(idx) > cap_per_unit):
            idx = rng.choice(idx, size=int(cap_per_unit), replace=False)
        pools[L] = Xm[idx]
    return pools


def _make_field(field_cls, name, ndim, field_kwargs, C=None, H=None):
    """
    Construct a neural field with the given (optional) CSet and HSet.

    The field output ``scale`` defaults to :data:`_FIELD_SCALE` (~1) so the learnable
    iso-values can be initialised at fixed positions in a known range (see
    :func:`_iso_inits`); an explicit ``scale`` in ``field_kwargs`` still wins.
    """
    fkw = dict(field_kwargs or {})
    fkw.setdefault("scale", _FIELD_SCALE)
    return field_cls(name, H=H if H is not None else _sb_hset(),
                     C=C, input_dim=ndim, **fkw)


def _build_region_only_event(name, levels, level_pts, ndim, field_cls, field_kwargs,
                             onlap_iso=None):
    """
    Build a **region-only** event (SPEC §2): a single-lithology package with a
    free (unconstrained) field — used for the basement and the dropped top unit.

    Its field carries no constraints; it exists only to label its region and to
    supply its unit points as above/below references to the adjacent unconformity.
    When ``onlap_iso`` is given the event onlaps the unconformity directly below
    it (``base=<iso name>``, ``onlap=True``); otherwise it is the oldest event and
    keeps the default ``base=-inf``.
    """
    fobj = _make_field(field_cls, name, ndim, field_kwargs, C=None, H=HSet().zero())
    if onlap_iso is not None:
        ev = strati(name, C=fobj, mode="above", base=onlap_iso, onlap=True)
    else:
        ev = strati(name, C=fobj)
    meta = _FieldMeta(name=name, kind="region-only", unit_indices=[],
                      levels=list(levels),
                      level_litho={L: name for L in levels})
    return ev, meta


def _build_erosional_event(name, eroded_level, above_levels, all_levels, level_pts,
                           ndim, field_cls, field_kwargs, sb_samples, grid, trend):
    """
    Build an **erosional** unconformity event (SPEC §4.2): its own ``strati``
    event (``mode='above'``) that truncates older geology above its iso.

    With no on-surface points the surface is learned from **stratigraphic-bound**
    constraints (``CSet.sb``) relative to a single **learnable iso-value** (GeoINR's
    ``pairwise=False`` recipe), constrained from **both** sides:

    - **above** (``'>'``): the immediately-**younger** package (``above_levels``, which
      onlaps this unconformity). Using just the adjacent package pins the iso at the
      right stratigraphic level (just above the eroded unit) without the deep iso being
      pulled around by far-younger levels.
    - **below** (``'<'``): **every** level older than the unconformity (SPEC §4.2 — "below
      = everything older in the column"). This is essential: the no-overturn regularizer
      alone is too weak to keep the field monotone in the *unconstrained* deep region, so
      without an explicit below-constraint a distant older unit (e.g. the basement) floats
      above the iso and gets wrongly truncated by this unconformity.

    Residuals are normalised by ``‖∇f‖`` and the above/below sides are balanced in the
    loss (so a lone deep below-level is not swamped). The **no-overturn** regularizer
    keeps successive surfaces nesting. The learnable iso doubles as the ``Overprint``
    threshold and the extraction iso; one ``sb`` pool per level keeps every unit
    represented (SPEC §4.4).
    """
    above = list(above_levels)
    below_levels = [L for L in sorted(all_levels) if L >= eroded_level]  # eroded unit + all older

    def pool(L):
        return level_pts.get(L, np.empty((0, ndim)))

    sb_entries = ([(pool(L), _UNCONF_ISO, ">") for L in above if len(pool(L))]
                  + [(pool(L), _UNCONF_ISO, "<") for L in below_levels if len(pool(L))])
    sb_relations = ([(L, _UNCONF_ISO, ">") for L in above if len(pool(L))]
                    + [(L, _UNCONF_ISO, "<") for L in below_levels if len(pool(L))])

    C = CSet(crs="model")
    if sb_entries:
        C.sb = (int(sb_samples), sb_entries)
    C.grid = grid.copy()
    C.trend = trend
    fobj = _make_field(field_cls, name, ndim, field_kwargs, C=C, H=_sb_hset())

    ev = strati(name, C=fobj, mode="above", base=_UNCONF_ISO)
    # single interface → initialise the iso at the middle of the field's output range (0;
    # see _iso_inits). The field starts ~scale-normalised, so 0 sits between the above
    # units (driven >0) and the below units (driven <0); the sb loss then refines it.
    next_younger = max(above_levels) if above_levels else eroded_level
    ev.addIsosurface(_UNCONF_ISO, learnable=float(_iso_inits(1)[0]))
    _finalize_field_optim(fobj, field_kwargs)

    meta = _FieldMeta(name=name, kind="erosional", unit_indices=[],
                      is_unconformity=True, iso_name=_UNCONF_ISO,
                      above_levels=list(above_levels), below_levels=list(below_levels),
                      n_iq=len(sb_entries), sb_relations=sb_relations)
    meta.iso_above_level = next_younger
    meta.iso_below_level = eroded_level
    return ev, meta


def _build_depositional_sb_event(name, levels_young_to_old, unit_name, level_pts,
                                 ndim, field_cls, field_kwargs, sb_samples,
                                 grid, trend, onlap_iso=None):
    """
    Build a **depositional** package from unit points only (SPEC §4.2, wcsb).

    ``levels_young_to_old`` are the package's unit levels (ascending = youngest→oldest).
    The internal contacts are one **monotone group** of learnable iso-values (ordered by
    construction — see :meth:`~curlew.geology.geoevent.GeoEvent.addOrderedIsosurfaces` — so
    two contacts can pinch to zero thickness but never cross). Each contact is **named after
    the older unit whose top it is** (so an eroded top unit never names a contact — its true
    top is the unconformity), while its lithology label is the younger band immediately above
    it. Each unit is bound (``CSet.sb``) **above** the contact directly below it and **below**
    the contact directly above it; a **no-overturn** regularizer keeps the field monotone. The
    oldest band is the event's base lithology (the event name); younger bands become isosurface
    lithologies. When ``onlap_iso`` is given the package onlaps the unconformity below it.
    """
    levels = list(levels_young_to_old)
    k = len(levels)

    def pool(L):
        return level_pts.get(L, np.empty((0, ndim)))

    # Contact j sits between band j (younger) and band j+1 (older): it is the TOP of band j+1,
    # so name it after that older unit (matching curlew's "isosurface = formation top" reading).
    # Its lithology label is the band immediately ABOVE it (band j, the younger one) — curlew
    # assigns the band above an isosurface that isosurface's lithology. These differ precisely
    # for an eroded top unit, whose real top is the unconformity (a separate event) and so never
    # names a depositional contact (e.g. there is no "Pennsylvanian" contact).
    contact_names = [f"{unit_name(levels[j + 1])} Top" for j in range(k - 1)]   # iso/surface keys
    contact_litho = [unit_name(levels[j]) for j in range(k - 1)]                # band above (lithology)
    sb_entries, sb_relations = [], []
    for m in range(k):
        if len(pool(levels[m])) == 0:
            continue
        if m < k - 1:               # band m is above the contact below it
            sb_entries.append((pool(levels[m]), contact_names[m], ">"))
            sb_relations.append((levels[m], contact_names[m], ">"))
        if m > 0:                   # band m is below the contact above it
            sb_entries.append((pool(levels[m]), contact_names[m - 1], "<"))
            sb_relations.append((levels[m], contact_names[m - 1], "<"))

    C = CSet(crs="model")
    if sb_entries:
        C.sb = (int(sb_samples), sb_entries)
    C.grid = grid.copy()
    C.trend = trend
    fobj = _make_field(field_cls, name, ndim, field_kwargs, C=C, H=_sb_hset())

    if onlap_iso is not None:
        ev = strati(name, C=fobj, mode="above", base=onlap_iso, onlap=True)
    else:
        ev = strati(name, C=fobj)

    # initialise the k-1 contact isos evenly across the field's output range,
    # younger (high) → older (low); a single contact → 0 (see _iso_inits). Independent
    # of field type because the field output range is fixed by _FIELD_SCALE.
    inits = _iso_inits(k - 1)

    oldest = levels[-1]
    level_litho = {oldest: name}              # base lithology (oldest band = event name)
    contacts = []
    # one monotone group for the package's conformable contacts: the learnable iso-values are
    # ordered by construction, so two contacts can pinch to zero thickness but never cross
    # (which is what made a thin/sparse band's contacts swap before). ``litho=`` keeps each
    # contact's lithology attached to the (younger) band above it while the contact is named
    # after the older unit whose top it is.
    if contact_names:
        ev.addOrderedIsosurfaces(contact_names, [float(v) for v in inits], litho=contact_litho)
    for j, cname in enumerate(contact_names):
        level_litho[levels[j]] = f"{name}_{contact_litho[j]}"
        contacts.append(_ContactInfo(name=cname, unit_index=-1, level=levels[j],
                                     n_points=len(pool(levels[j]))))
    _finalize_field_optim(fobj, field_kwargs)

    meta = _FieldMeta(name=name, kind="depositional", unit_indices=[],
                      levels=levels, contacts=contacts, level_litho=level_litho,
                      n_iq=len(sb_entries), sb_relations=sb_relations)
    return ev, meta


def _build_unit_only_chain(units, sfields, obs, Xm, ndim, field_cls, field_kwargs,
                           sb_samples, cap_per_unit, seed):
    """
    Build the wcsb-style event chain (oldest→youngest) from unit points only.

    The partitioner emits an alternating youngest-first sequence of erosional and
    depositional fields. Reversed to oldest-first, this builder:

    1. synthesizes a **region-only basement** from the oldest erosional field's
       attached unit(s) (``derive_scalar_fields`` does not emit it — SPEC §8.1),
    2. builds each **erosional** event (learnable iso, ``sb`` above/below constraints,
       no-overturn) — see :func:`_build_erosional_event`,
    3. builds each **depositional** package (learnable per-contact isos, ``sb`` band
       constraints, no-overturn), onlapping the unconformity below it,
    4. synthesizes a **region-only** event for the dropped youngest baselap unit
       (top of column), onlapping the youngest unconformity.

    Returns ``(events, field_meta, level_litho)``.
    """
    rng = np.random.default_rng(seed)
    level_pts = _cap_level_pools(obs, Xm, cap_per_unit, rng)
    level_name = {u.level: u.name.strip() or f"level{u.level}" for u in units}
    grid = _model_grid(Xm, ndim)        # shared no-overturn grid (model coords)
    trend = _trend_up(ndim)             # younging-up direction (+z)

    used_names = set()
    def uniq(base):
        name = base
        k = 1
        while name in used_names:
            k += 1
            name = f"{base} ({k})"
        used_names.add(name)
        return name

    def unit_name(level):  # used for contact (isosurface) names
        return level_name.get(level, f"level{level}")

    sfields_old = list(reversed(sfields))  # oldest → youngest
    nsf = len(sfields_old)
    all_levels = sorted(level_pts.keys())
    events, metas, level_litho = [], [], {}

    for i, sf in enumerate(sfields_old):
        if sf.kind == "erosional":
            # the oldest erosional eroded the basement: synthesize a region-only basement
            # event first (the eroded unit(s) become its lithology).
            if i == 0:
                base_levels = sorted({units[u].level for u in sf.unit_indices})
                bname = uniq(unit_name(base_levels[-1]) if base_levels else "Basement")
                bev, bmeta = _build_region_only_event(
                    bname, base_levels, level_pts, ndim, field_cls, field_kwargs)
                events.append(bev); metas.append(bmeta); level_litho.update(bmeta.level_litho)

            # the younger package that onlaps this unconformity (the dropped top unit for
            # the youngest erosional, synthesized after the loop). This sets the local iso
            # placement; the below side is "everything older", computed in the builder.
            if i + 1 < nsf and sfields_old[i + 1].kind == "depositional":
                above_levels = sorted({units[u].level for u in sfields_old[i + 1].unit_indices})
            else:
                above_levels = [units[0].level]

            eroded_level = units[sf.horizon_unit_indices[0]].level
            ename = uniq(f"{unit_name(eroded_level)} Unconformity")
            ev, meta = _build_erosional_event(
                ename, eroded_level, above_levels, all_levels, level_pts, ndim,
                field_cls, field_kwargs, sb_samples, grid, trend)
            events.append(ev); metas.append(meta)
        else:  # depositional package
            levels = sorted({units[u].level for u in sf.unit_indices})  # young → old
            onlap_iso = _UNCONF_ISO if (metas and metas[-1].is_unconformity) else None
            pname = uniq(unit_name(levels[-1]))   # base (oldest) band names the event
            ev, meta = _build_depositional_sb_event(
                pname, levels, unit_name, level_pts, ndim, field_cls, field_kwargs,
                sb_samples, grid, trend, onlap_iso=onlap_iso)
            events.append(ev); metas.append(meta); level_litho.update(meta.level_litho)

    # synthesize the dropped youngest unit (a top baselap package the partitioner
    # skips): a region-only event onlapping the youngest unconformity.
    top = units[0]
    if top.level not in level_litho:
        onlap_iso = _UNCONF_ISO if (metas and metas[-1].is_unconformity) else None
        qname = uniq(unit_name(top.level))
        qev, qmeta = _build_region_only_event(
            qname, [top.level], level_pts, ndim, field_cls, field_kwargs, onlap_iso=onlap_iso)
        events.append(qev); metas.append(qmeta); level_litho.update(qmeta.level_litho)

    return events, metas, level_litho


def build_geomodel(strat_col_csv, observations, *, field="GeoINR",
                   scale="isometric", transform=None, cap_per_unit=None,
                   field_kwargs=None, hset=None, iq_samples=256, seed=0) -> GeoModel:
    """
    Build a ready-to-fit :class:`~curlew.geology.geomodel.GeoModel` from a
    stratigraphic-column CSV and point observations.

    The construction path is chosen automatically from the observations:

    - **On-surface path** (contact points present, e.g. ``multilayer_fold``): each
      depositional package is one ``strati`` event with ``eq`` traces, seed
      isosurfaces, ``gv`` gradients (where normals exist) and ``iq`` inequalities.
    - **Unit-only path** (only level-labelled unit points, e.g. ``wcsb``): the
      column is built as an alternating chain of erosional unconformities and
      depositional packages with synthesized region-only events at both ends. Every
      surface is a **learnable iso-value** constrained by units above/below it via the
      gradient-normalised stratigraphic-bound loss (``CSet.sb``) plus a no-overturn
      regularizer — GeoINR's per-field recipe (see :func:`_build_unit_only_chain`).

    A single homogeneous normalization :class:`~curlew.geometry.Transform` is
    attached to the model.

    Parameters
    ----------
    strat_col_csv : str | os.PathLike
        Path to the strat-column CSV (``name, relation, level``).
    observations : curlew.io.Observations
        Point observations (load via :func:`curlew.io.loadObservations`).
    field : str | type
        Neural field to use for each event: ``"GeoINR"`` or ``"Siren"`` (or a
        ``BaseNF`` subclass).
    scale : str
        Normalization mode: ``"isometric"`` (uniform) or ``"geo_extensive"``
        (vertical exaggeration; large xy extent vs thin z — used for wcsb).
    transform : curlew.geometry.Transform, optional
        Explicit global→model transform; overrides ``scale`` when given.
    cap_per_unit : int, optional
        Cap each unit at this many points (uniform random subsample) on the
        unit-only path so rare units stay represented and huge units do not
        dominate the constraint pools (SPEC §4.4). ``None`` keeps all points. Unused
        on the on-surface path.
    field_kwargs : dict, optional
        Extra keyword arguments forwarded to each field's ``initField`` (e.g.
        ``hidden_dim``, ``learning_rate``). On the unit-only path the learnable
        iso-values use ``10x`` the ``learning_rate`` given here.
    hset : curlew.core.HSet, optional
        Hyperparameters for every event on the on-surface path; a minimal default
        is used when omitted. (The unit-only path uses a stratigraphic-bound +
        no-overturn HSet.)
    iq_samples : int
        Number of random points drawn per constraint pool each evaluation (``iq``
        pairs on the on-surface path; ``sb`` bound points on the unit-only path).
    seed : int
        Random seed for unit subsampling.

    Returns
    -------
    curlew.geology.geomodel.GeoModel
        Model with events, CSets, isosurfaces and the normalization transform
        attached. The parsed column and per-field metadata are stored on
        ``M.strat_units``, ``M.horizons``, ``M.scalar_fields``, ``M.field_meta``
        and (unit-only path) ``M.level_litho`` for inspection/visualisation.
    """
    field_cls = _FIELD_TYPES[field] if isinstance(field, str) else field

    units = load_strat_column_csv(strat_col_csv)
    horizons = derive_horizons(units)
    sfields = derive_scalar_fields(units)

    ndim = observations.ndim
    if transform is None:
        transform = _normalization_transform(observations.bounds(), scale, ndim)
    Xm = transform.apply(observations.coords.astype(float))

    level_litho = {}
    if bool(observations.is_interface.any()):
        # ---- on-surface path (multilayer_fold): depositional events only ----
        events, field_meta = [], []
        for sf in reversed(sfields):  # GeoModel expects oldest→youngest
            if sf.kind != "depositional" or not sf.horizon_unit_indices:
                raise NotImplementedError(
                    "The on-surface path supports depositional packages with contact "
                    f"points only; field '{sf.name}' is '{sf.kind}'. Datasets with only "
                    "unit points (no interface points) use the unit-only chain instead."
                )
            ev, meta = _build_depositional_event(
                sf, units, observations, Xm, ndim, field_cls, field_kwargs, hset, iq_samples
            )
            events.append(ev)
            field_meta.append(meta)
    else:
        # ---- unit-only path (wcsb): learnable-iso / sb / no-overturn chain ----
        events, field_meta, level_litho = _build_unit_only_chain(
            units, sfields, observations, Xm, ndim, field_cls, field_kwargs,
            iq_samples, cap_per_unit, seed,
        )

    M = GeoModel(events, transform=transform, name=Path(strat_col_csv).stem)

    # stash parsing/build metadata for inspection and notebook visualisation
    M.strat_units = units
    M.horizons = horizons
    M.scalar_fields = sfields
    M.field_meta = field_meta
    M.observations = observations
    M.normalization = transform
    M.level_litho = level_litho
    return M
