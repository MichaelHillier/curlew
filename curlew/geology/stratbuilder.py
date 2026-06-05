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

This module currently implements the **depositional path** (conformal/baselap
packages with one or more contacts), which is sufficient to build the
``multilayer_fold`` reference dataset end-to-end. Erosional unconformities and
region-only basements are a later pass and raise ``NotImplementedError``.
"""

from __future__ import annotations

import csv as _csv
from dataclasses import dataclass, field as _dcfield
from pathlib import Path

import numpy as np

import curlew
from curlew.core import CSet, HSet
from curlew.geology import strati
from curlew.geology.geomodel import GeoModel
from curlew.geometry import Transform
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


def build_geomodel(strat_col_csv, observations, *, field="GeoINR",
                   scale="isometric", transform=None, cap_per_unit=None,
                   field_kwargs=None, hset=None, iq_samples=256) -> GeoModel:
    """
    Build a ready-to-fit :class:`~curlew.geology.geomodel.GeoModel` from a
    stratigraphic-column CSV and point observations.

    The strat column is parsed and partitioned into scalar fields
    (:func:`derive_scalar_fields`); each depositional field becomes a ``strati``
    event with seed isosurfaces, ``eq`` traces, ``gv`` gradients (where normals
    exist) and ``iq`` inequalities ordering its contacts. A single homogeneous
    normalization :class:`~curlew.geometry.Transform` is attached to the model.

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
        (vertical exaggeration).
    transform : curlew.geometry.Transform, optional
        Explicit global→model transform; overrides ``scale`` when given.
    cap_per_unit : int, optional
        Reserved for the imbalance-aware wcsb path; unused on the depositional path.
    field_kwargs : dict, optional
        Extra keyword arguments forwarded to each field's ``initField`` (e.g.
        ``hidden_dim``, ``learning_rate``).
    hset : curlew.core.HSet, optional
        Hyperparameters for every event; a minimal default is used when omitted.
    iq_samples : int
        Number of random pairs drawn per inequality pool each evaluation.

    Returns
    -------
    curlew.geology.geomodel.GeoModel
        Model with events, CSets, isosurfaces and the normalization transform
        attached. The parsed column and per-field metadata are stored on
        ``M.strat_units``, ``M.horizons``, ``M.scalar_fields`` and ``M.field_meta``
        for inspection/visualisation.
    """
    field_cls = _FIELD_TYPES[field] if isinstance(field, str) else field

    units = load_strat_column_csv(strat_col_csv)
    horizons = derive_horizons(units)
    sfields = derive_scalar_fields(units)

    ndim = observations.ndim
    if transform is None:
        transform = _normalization_transform(observations.bounds(), scale, ndim)
    Xm = transform.apply(observations.coords.astype(float))

    events = []
    field_meta = []
    # GeoModel expects oldest→youngest; derive_scalar_fields is youngest-first.
    for sf in reversed(sfields):
        if sf.kind != "depositional":
            raise NotImplementedError(
                "build_geomodel currently implements the depositional path only; "
                f"field '{sf.name}' has kind '{sf.kind}' (erosional/region-only is a later pass)."
            )
        if not sf.horizon_unit_indices:
            raise NotImplementedError(
                "Region-only basement events are a later pass; "
                f"field '{sf.name}' has no contacts."
            )
        ev, meta = _build_depositional_event(
            sf, units, observations, Xm, ndim, field_cls, field_kwargs, hset, iq_samples
        )
        events.append(ev)
        field_meta.append(meta)

    M = GeoModel(events, transform=transform, name=Path(strat_col_csv).stem)

    # stash parsing/build metadata for inspection and notebook visualisation
    M.strat_units = units
    M.horizons = horizons
    M.scalar_fields = sfields
    M.field_meta = field_meta
    M.observations = observations
    M.normalization = transform
    return M
