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
  isosurfaces and ``gv`` gradients (see :func:`_build_depositional_event`). With
  GeoINR-family fields the ``eq``/``iq`` constraints are consumed by GeoINR's
  gradient-normalised interface/inequality losses, consistent with the unit-only path.
- **Unit-only path** (e.g. ``wcsb``) — when only unit (level-labelled) points exist,
  the column is built as an alternating chain of **erosional** unconformities and
  **depositional** packages, with synthesized **region-only** events at both ends
  (the eroded basement at the bottom, the dropped top baselap unit at the top).
  Erosional events truncate older geology at their own iso (``mode='above'``);
  depositional / region-only packages **onlap** the unconformity directly below them
  (``onlap=True``). Because no on-surface points exist, each field is constrained
  purely by **point-vs-point inequalities** (``CSet.iq``: younger unit points ``>``
  older unit points) consumed by the GeoINR gradient-normalised inequality loss
  plus a **no-overturn** regularizer, both implemented on the field classes
  themselves (see :class:`curlew.fields.geoinr.GeoINR` — the unit-only
  path therefore requires ``GeoINR`` or ``Siren`` fields). Surfaces are **not**
  learned: after ``M.fit``, call :func:`estimate_isosurfaces` to estimate each
  contact/unconformity iso-value from the separation between the adjacent units'
  field-value distributions and set it via the ordinary
  :meth:`~curlew.geology.geoevent.GeoEvent.addIsosurface`. ``cap_per_unit``
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
from curlew.fields import BaseSF
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

    ``onlap_package`` is ``True`` when the package's **base** unit is a ``baselap``
    sitting on a *conformal* (depositional) package rather than an unconformity — the
    "baselap-onto-conformal" pattern. Such a package onlaps the **top of the package
    directly below it** instead of an erosional unconformity (see
    :func:`_build_unit_only_chain`).
    """

    field_index: int
    name: str
    kind: str
    unit_indices: list
    horizon_unit_indices: list
    onlap_package: bool = False

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


def _append_scalar_field(fields, *, kind, unit_indices, horizon_unit_indices,
                         onlap_package=False):
    idx = len(fields)
    fields.append(
        ScalarFieldRow(
            field_index=idx,
            name=f"scalar_field{idx}",
            kind=kind,
            unit_indices=list(unit_indices),
            horizon_unit_indices=list(horizon_unit_indices),
            onlap_package=bool(onlap_package),
        )
    )


def _package_end(units, start, n_units) -> int:
    """
    Last unit index of the depositional package beginning at ``start`` (youngest-first).

    A package extends to older units until either the **next** unit is ``eroded`` (the
    unconformity bounds it from below, exclusive) or the **current** unit is ``baselap``
    (a baselap unit is the *base* of its package, inclusive). The latter is what splits a
    "baselap-onto-conformal" run into two packages — the baselap unit ends the upper one,
    and the next older unit starts the lower one.
    """
    end = start
    while (end + 1 < n_units and units[end + 1].relation != "eroded"
           and units[end].relation != "baselap"):
        end += 1
    return end


def _onlaps_package(units, end, n_units) -> bool:
    """
    True when the package whose base (oldest unit) is ``units[end]`` baselaps onto a
    *conformal* package below it: its base is ``baselap`` and the next older unit is
    depositional (``conformal``/``baselap``), not an unconformity.
    """
    return (units[end].relation == "baselap" and end + 1 < n_units
            and units[end + 1].relation in STRATIGRAPHIC_RELATIONS)


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
        end = _package_end(units, 0, n_units)
        horizon_unit_indices = [
            idx for idx in range(1, end + 1) if units[idx].relation in STRATIGRAPHIC_RELATIONS
        ]
        if horizon_unit_indices:
            _append_scalar_field(
                fields, kind="depositional",
                unit_indices=list(range(0, end + 1)),
                horizon_unit_indices=horizon_unit_indices,
                onlap_package=_onlaps_package(units, end, n_units),
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
            end = _package_end(units, i, n_units)
            package_horizon_unit_indices = [
                idx for idx in range(i + 1, end + 1) if units[idx].relation in STRATIGRAPHIC_RELATIONS
            ]
            if package_horizon_unit_indices:
                _append_scalar_field(
                    fields, kind="depositional",
                    unit_indices=list(range(i, end + 1)),
                    horizon_unit_indices=package_horizon_unit_indices,
                    onlap_package=_onlaps_package(units, end, n_units),
                )
            i = end + 1
            continue

        end = _package_end(units, i, n_units)
        package_horizon_unit_indices = [
            idx for idx in range(i + 1, end + 1) if units[idx].relation in STRATIGRAPHIC_RELATIONS
        ]
        if package_horizon_unit_indices:
            _append_scalar_field(
                fields, kind="depositional",
                unit_indices=list(range(i, end + 1)),
                horizon_unit_indices=package_horizon_unit_indices,
                onlap_package=_onlaps_package(units, end, n_units),
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
    levels: list = _dcfield(default_factory=list)        # unit levels owned by this event (lithology bands, youngest-first)
    above_levels: list = _dcfield(default_factory=list)  # erosional: younger (above) unit levels
    below_levels: list = _dcfield(default_factory=list)  # erosional: older (below) unit levels
    eroded_level: int = None          # erosional: level of the eroded unit (directly below the iso)
    iso_name: str = None              # name of the isosurface (unconformity threshold; set post-fit)
    level_litho: dict = _dcfield(default_factory=dict)   # level -> predicted lithology name
    n_iq: int = 0                     # number of inequality pairs on this event
    # exact iq constraints: list[(above, below_level, '>')] where ``above`` is a single
    # level (depositional adjacent-band pairs) or a tuple of levels (erosional events
    # pool the onlapping package's levels into one above-side pool)
    iq_relations: list = _dcfield(default_factory=list)
    # --- onlap / seed-iso metadata (baselap-onto-conformal + interface-seeded path) ---
    onlap_package: bool = False       # base is a baselap onto a conformal package (not an unconformity)
    onlap_event: str = None           # name of the event this one onlaps (unconformity or lower package); None for basement
    onlap_iso_id: str = None          # carve iso-id of the onlap threshold on ``onlap_event``
    top_iso: str = None               # name of this package's top isosurface (set when a younger package onlaps it)
    seed_isos: bool = False           # surfaces are seeded from interface points (values not learned)


def _default_hset(has_normals: bool, has_iq: bool) -> HSet:
    """
    A minimal HSet for a depositional field built from a **non-GeoINR** field type
    (GeoINR-family fields use their own gradient-normalised losses instead — see
    :func:`_build_depositional_event`). Only the active terms are non-zero.

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

    For GeoINR-family fields (and no explicit ``hset``) the ``eq``/``iq`` constraints
    are consumed by GeoINR's **gradient-normalised** interface and inequality losses
    (``eq_norm_weight`` / ``iq_norm_weight`` on the field — consistent with the
    unit-only path) instead of the generic curlew terms; only the normal (gradient)
    loss stays on the ``HSet``. Other field types keep :func:`_default_hset`.
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

    fkw = dict(field_kwargs or {})
    if hset is not None:
        H = hset.copy()
    elif isinstance(field_cls, type) and issubclass(field_cls, GeoINR):
        # GeoINR-family: gradient-normalised interface/inequality losses on the field
        # (consistent with the unit-only path); generic HSet keeps only the normal loss
        H = HSet().zero()
        if gp is not None:
            H.grad_loss = 1.0
        fkw.setdefault("eq_norm_weight", 1.0)
        fkw.setdefault("iq_norm_weight", 1.0)
    else:
        H = _default_hset(gp is not None, bool(iq_pairs))
    fobj = field_cls(sf.name, H=H, C=C, input_dim=ndim, **fkw)
    ev = strati(sf.name, C=fobj)
    for name, pts_model in iso_specs:
        ev.addIsosurface(name, seed=pts_model)

    meta = _FieldMeta(name=sf.name, kind=sf.kind,
                      unit_indices=list(sf.unit_indices), contacts=contacts,
                      inequalities=contact_inequalities)
    return ev, meta


# ---------------------------------------------------------------------------
# Unit-only (wcsb) chain — erosional / region-only / depositional branches
# ---------------------------------------------------------------------------
# When observations carry only unit (level-labelled) points (no contact points,
# no normals), the column is built as an alternating chain of erosional
# unconformities and depositional packages. Helpers below construct each branch.

#: Name given to every erosional event's iso (its threshold; value set post-fit
#: by :func:`estimate_isosurfaces`). The name only needs to be unique *within* an
#: event, so it can be reused across events: resolution always happens on the
#: specific event/parent.
_UNCONF_ISO = "unconformity"

#: Default per-field loss weights for the unit-only path (consumed by the GeoINR-side
#: loss terms — see :class:`curlew.fields.geoinr.GeoINR` — NOT by ``HSet``,
#: which stays all-zero here). The no-overturn term is heavily up-weighted vs the
#: normalised inequality: curlew assembles the model with a sequential onlap/truncation
#: combine (not GeoINR's global per-point carve), so a strongly monotone field is what keeps
#: the *unconstrained deep region* (e.g. the basement) reliably below every younger
#: unconformity's iso and so not wrongly truncated. ``overturn`` uses the gradient
#: *magnitude* (GeoINR — the norm drives polarity correction), so it couples to the field
#: scale; that is fine because the unit-only path pins the scale ~1 (:data:`_FIELD_SCALE`),
#: making the term naturally O(1). GeoINR's original weight (1) does NOT transfer to
#: curlew's combine — measured head-to-head on wcsb (Siren, iq 1024, 3000 epochs):
#: weight 1 → 66.5% of grid columns contain an older-above-younger inversion (basement
#: leaking through non-monotone deep unconformity fields) vs 0.1% at weight 30, for only
#: +0.014 band accuracy — so 30 stays the default.
_IQ_WEIGHT = 1.0
_OVERTURN_WEIGHT = 30.0

#: Gradient-normalised interface (equality) weight, used when a field is fit to on-contact
#: (interface) points — the seed-iso / hybrid path (skmb). Matches the on-surface path's
#: default. The eq loss pins each contact's field value to its interface points, so the
#: seeded isosurface reads a stable, data-driven iso-value (no learnable iso needed).
_EQ_WEIGHT = 1.0

#: Field output range for the unit-only path. Both ``Siren`` and ``GeoINR`` evaluate as
#: ``scale * mlp(x)``, and the raw network ``mlp(x)`` is O(1) at initialisation, so this
#: ``scale`` *is* the field's value range (``BaseNF``'s default is ``1e2``). It is held at
#: ~1 here so every field lives in ``[-1, 1]``, keeping the two loss terms comparable:
#: the normalised inequality is scale-free (its residual is ``‖∇f‖``-normalised) while the
#: no-overturn term is magnitude-based and so naturally O(1) only when the field range is ~1.
_FIELD_SCALE = 1.0


def _model_grid(Xm, ndim, n=24, draw=1000, poisson=False):
    """A :class:`~curlew.geometry.Grid` spanning the data in *model* space, used to sample
    the no-overturn regularizer each epoch (``Grid.draw`` re-samples per step).

    Two sampling regimes (chosen by ``poisson``):

    - **uniform** (``poisson=False``, the wcsb unit-only path): ``draw`` points are picked
      uniformly at random from a fixed ``n``-per-axis lattice (``sampleArgs={'N': draw}``).
      Kept exactly as before so the seed_isos=False path is unchanged.
    - **Poisson-disk** (``poisson=True``, the seed-iso / hybrid skmb path): ``draw`` points
      are drawn with a **model-scaled** minimum separation so they cover the whole domain
      **evenly** instead of clumping on a coarse lattice, and the candidate lattice is made
      **fine** (step ≈ r/2) so the field effectively sees the continuum over training. The
      radius ``r`` is sized so ~``draw`` points pack the data volume; ``seed=None`` re-draws
      a fresh, evenly-spread set each epoch. Coarse-lattice under-coverage was the residual
      basin-margin artefact source (SKMB_HANDOFF §4a).
    """
    lo, hi = Xm.min(axis=0), Xm.max(axis=0)
    ext = np.maximum(hi - lo, 1e-6)
    center = 0.5 * (lo + hi)
    if not poisson:
        return Grid(dims=tuple(ext), step=tuple(ext / float(n)), center=tuple(center),
                    sampleArgs={"N": int(draw)})
    # model-scaled Poisson radius: ~``draw`` points evenly spread through the data volume
    # (V / (c·r^d) ≈ draw ⇒ r ≈ (V/draw)^(1/d); the 0.75 keeps r just below the packing
    # limit so ``draw`` points are reliably placed — see the timing in SKMB_HANDOFF §4a).
    V = float(np.prod(ext))
    r = 0.75 * (V / max(int(draw), 1)) ** (1.0 / ndim)
    # candidate lattice fine enough to place points at separation r (step ≈ r/2), capped so
    # the per-epoch Poisson pass stays cheap; this is the "finer / model-scaled lattice".
    n_axis = np.clip(np.ceil(ext / (0.5 * r)), 2, 64).astype(int)
    step = tuple(ext / n_axis)
    return Grid(dims=tuple(ext), step=step, center=tuple(center),
                sampleArgs={"poissonDisk": (float(r), int(draw), None)})


def _trend_up(ndim):
    """Younging-up trend vector (unit +z) in model coordinates."""
    t = np.zeros(ndim)
    t[-1] = 1.0
    return t


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


def _make_field(field_cls, name, ndim, field_kwargs, C=None, strat_loss=False, eq_loss=False):
    """
    Construct a neural field with the given (optional) CSet.

    The ``HSet`` is all-zero — on the unit-only path the loss comes entirely from
    the GeoINR-side terms, enabled with ``strat_loss=True`` (``iq_norm_weight`` /
    ``overturn_weight``; see :class:`curlew.fields.geoinr.GeoINR`). ``eq_loss=True``
    additionally enables the gradient-normalised **interface** loss (``eq_norm_weight``)
    when the field carries on-contact (``CSet.eq``) traces — the seed-iso / hybrid path.
    The field output ``scale`` defaults to :data:`_FIELD_SCALE` (~1) so the no-overturn
    magnitude term stays O(1); explicit ``field_kwargs`` entries still win.
    """
    fkw = dict(field_kwargs or {})
    fkw.setdefault("scale", _FIELD_SCALE)
    if strat_loss:
        fkw.setdefault("iq_norm_weight", _IQ_WEIGHT)
        fkw.setdefault("overturn_weight", _OVERTURN_WEIGHT)
    if eq_loss:
        fkw.setdefault("eq_norm_weight", _EQ_WEIGHT)
    return field_cls(name, H=HSet().zero(), C=C, input_dim=ndim, **fkw)


class _AliasField(BaseSF):
    """
    Read-only alias of another field: evaluates **identically** to ``target`` but owns no
    parameters, no constraints, and produces no loss. Used for the **region-only** events
    (the basement and the dropped top unit), whose region is bounded by the adjacent
    unconformity: GeoINR classifies those domains with the *unconformity's* field and
    iso-value, so the alias continues that fitted, data-aligned field into the region
    instead of filling it with an untrained (random) network. Labels are unaffected —
    the adjacent event's ``Overprint`` decides the boundary — only the scalar output
    becomes meaningful.

    The alias keeps its own ``name`` (the region-only event's), so ``Geode.fields``
    entries stay distinct from the target's, and its ``C`` is ``None`` so ``loss()``
    returns an empty :class:`~curlew.core.Pebble` — important because ``Pebble`` refuses
    duplicate loss keys when ``GeoModel.fit`` sums the per-event losses (the target's
    loss is already counted by the event that owns it). Valid because the aliasing event
    is stratigraphically adjacent to its target with no deformation events between them
    (their reference frames coincide).
    """

    def __init__(self, name, target):
        # plain attribute (not a registered submodule): the target's parameters must
        # remain owned/optimised by its own event only
        object.__setattr__(self, "_target", target)
        super().__init__(name=name, input_dim=target.input_dim)

    def initField(self, **kwargs):
        self.optim = None  # nothing to optimise

    def forward(self, x, transform=True):
        return self._target.forward(x, transform=transform)

    @property
    def scale(self):
        return getattr(self._target, "scale", 1.0)


def _build_region_only_event(name, levels, level_pts, ndim, field_cls, field_kwargs,
                             onlap_iso=None, alias_of=None):
    """
    Build a **region-only** event (SPEC §2): a single-lithology package — used for
    the basement and the dropped top unit.

    Its field carries no constraints; the event exists only to label its region and
    to supply its unit points as above/below references to the adjacent unconformity.
    When ``alias_of`` is given (a fitted field of the adjacent unconformity event) the
    event's field is a loss-free :class:`_AliasField` of it, so the region's scalar is
    the unconformity field continued past its iso (GeoINR's basement treatment) rather
    than an untrained network. When ``onlap_iso`` is given the event onlaps the
    unconformity directly below it (``base=<iso name>``, ``onlap=True``); otherwise it
    is the oldest event and keeps the default ``base=-inf``.
    """
    if alias_of is not None:
        fobj = _AliasField(name, alias_of)
    else:
        fobj = _make_field(field_cls, name, ndim, field_kwargs, C=None)
    if onlap_iso is not None:
        ev = strati(name, C=fobj, mode="above", base=onlap_iso, onlap=True)
    else:
        ev = strati(name, C=fobj)
    meta = _FieldMeta(name=name, kind="region-only", unit_indices=[],
                      levels=list(levels),
                      level_litho={L: name for L in levels})
    return ev, meta


# the relation-matrix tag used for every "stay above an older external surface" nesting pair
_NEST_TAG = "older-unconformity-surface"


def _surface_ordering_iq(own_contacts, top_surface, older_surfaces):
    """
    The **single surface-ordering principle** of the seed-iso path (SKMB_HANDOFF §2):

    > In each field, every surface point sits **above** every *older* surface point — both
    > the field's own younger contacts and every older external surface it onlaps/truncates
    > against.

    This one builder emits all of a field's seed-iso (interface-point) ordering / nesting
    inequalities, collapsing what used to be three separate code paths (within-package
    contact ordering, unconformity nesting, package-top nesting) into the same relation:

    * **within-field ordering** — the field's own contacts, young→old, each above the next
      (adjacent pairs only; transitivity orders the rest);
    * **nesting** — the field's representative *top* surface above **every** older external
      surface, **one pair each** so the sparse margin of every older surface (where surfaces
      converge) gets its own full ``iq_samples`` draw rather than a fraction of one pooled set.

    Erosional events pass ``own_contacts=[]`` (a single surface) and their own unconformity
    surface as ``top_surface``; depositional packages pass their internal contacts as
    ``own_contacts`` and the package **top** as ``top_surface``.

    Parameters
    ----------
    own_contacts : list[(key, pts)]
        The field's own internal contacts, young→old (model coords). ``key`` tags the
        relation's younger side.
    top_surface : tuple(key, pts) | None
        The field's topmost surface that must nest above the older external surfaces
        (``None`` or empty ``pts`` ⇒ no nesting pairs).
    older_surfaces : list[pts]
        Older external (unconformity) surfaces this field must stay above (one array each).

    Returns
    -------
    (iq_pairs, iq_relations)
        ``iq_pairs`` : list[(youngerPts, olderPts, '>')];
        ``iq_relations`` : list[(youngKey, oldKey, '>')].
    """
    iq_pairs, iq_relations = [], []
    # within-field: adjacent own contacts, younger above older (transitivity does the rest)
    for (yk, yp), (ok, op) in zip(own_contacts, own_contacts[1:]):
        yp, op = np.asarray(yp, float), np.asarray(op, float)
        if len(yp) and len(op):
            iq_pairs.append((yp, op, ">"))
            iq_relations.append((yk, ok, ">"))
    # nesting: the top surface above every older external surface (one pair each)
    if top_surface is not None:
        tkey, tpts = top_surface
        tpts = np.asarray(tpts, float)
        if len(tpts):
            for older in older_surfaces or ():
                older = np.asarray(older, float)
                if len(older):
                    iq_pairs.append((tpts, older, ">"))
                    iq_relations.append((tkey, _NEST_TAG, ">"))
    return iq_pairs, iq_relations


def _build_erosional_event(name, eroded_level, above_levels, all_levels, level_pts,
                           ndim, field_cls, field_kwargs, iq_samples, grid, trend,
                           iface_pts=None, below_iface=None):
    """
    Build an **erosional** unconformity event (SPEC §4.2): its own ``strati``
    event (``mode='above'``) that truncates older geology above its iso.

    When ``iface_pts`` (on-contact points lying on the unconformity surface, model
    coords) is given, the iso is **seeded** from them and an ``eq`` trace pins the
    field to that surface (the seed-iso / hybrid path) — so the unconformity's value is
    read directly from the interface data rather than estimated/learned. Otherwise the
    iso is named-only and its value is set post-fit (wcsb path).

    ``below_iface`` (a list of every **older** unconformity's on-surface point array, model
    coords) adds a **surface-nesting** inequality — **one pair per older surface** so each
    (e.g. the Precambrian) gets its own full ``iq_samples`` draw rather than a fraction of one
    pooled set (the margin, where surfaces converge, is otherwise under-sampled and still cut).
    Each pair orders this unconformity's own surface points ``>`` an older surface's points
    *in this field*, so the older surface stays **below this iso** and is not eroded by it.
    The seed-iso path needs this because the deep, unit-sparse region carries no soft-unit (CE)
    signal, so nothing otherwise stops a younger unconformity from floating above its iso right
    at an older surface and cutting below it (e.g. below the Precambrian).

    With no on-surface points the surface is constrained purely by **point-vs-point
    inequalities** (``CSet.iq``), consumed by the GeoINR gradient-normalised
    inequality loss: the immediately-**younger** package (which onlaps this
    unconformity) is ordered ``>`` **every** level older than the unconformity
    (eroded unit + all older — SPEC §4.2). The full older side is essential: the
    no-overturn regularizer alone is too weak to keep the field monotone in the
    *unconstrained* deep region, so without explicit ordering a distant older unit
    (e.g. the basement) floats high and gets wrongly truncated by this unconformity.
    The above side is **pooled into one** (the onlapping package is only 1–3 capped
    levels, so balance is safe) while the below side keeps **one pool per level** so every
    older unit stays represented (SPEC §4.4) — pairs scale with ``|below|`` rather than
    ``|above| × |below|``, which is what keeps large per-epoch sample counts tractable.
    These per-level below pairs are the **deep-region anchors** (they keep the unconformity
    field below every older unit's points across the data-sparse deep region); the
    **no-overturn** regularizer keeps successive surfaces nesting.

    No iso-value exists at build time — the ``Overprint`` threshold is the iso
    *name* (resolved at predict), and its value is estimated after fitting by
    :func:`estimate_isosurfaces` (between the onlapping package's and the eroded
    unit's field-value distributions).
    """
    def pool(L):
        return level_pts.get(L, np.empty((0, ndim)))

    above = [L for L in above_levels if len(pool(L))]
    below = [L for L in sorted(all_levels) if L >= eroded_level and len(pool(L))]  # eroded + all older

    iq_pairs, iq_relations = [], []
    if above and below:
        above_pool = np.concatenate([pool(A) for A in above], axis=0)
        # One pair per older level (eroded + all older). These per-level inequalities are the
        # **deep-region anchors**: they pin the unconformity field below every older unit's
        # points, which (with the no-overturn prior) keeps it well-behaved across the
        # data-sparse deep regions. A 2026-06-23 experiment to pool them into one "deep"
        # reference cut training cost but caused **volume age-inversion islands** there
        # (2.20% of grid columns vs 0.47% with the full set) — so the full per-level set stays.
        iq_pairs = [(above_pool, pool(B), ">") for B in below]
        # relation entries carry the pooled above side as a tuple of levels
        iq_relations = [(tuple(above), B, ">") for B in below]

    has_iface = iface_pts is not None and len(iface_pts) > 0

    # surface nesting (the single §2 principle): this unconformity's own surface sits above
    # every older unconformity's surface (one pair each), so the older surfaces stay below
    # this iso and are not eroded. An erosional event carries a single surface, so it has no
    # within-field contacts — only the nesting pairs.
    if has_iface:
        nest_pairs, nest_rels = _surface_ordering_iq(
            own_contacts=[], top_surface=((eroded_level,), iface_pts),
            older_surfaces=below_iface)
        iq_pairs = list(iq_pairs) + nest_pairs
        iq_relations = list(iq_relations) + nest_rels

    C = CSet(crs="model")
    if iq_pairs:
        C.iq = (int(iq_samples), iq_pairs)
    if has_iface:
        C.eq = [np.asarray(iface_pts, float)]
    C.grid = grid.copy()
    C.trend = trend
    fobj = _make_field(field_cls, name, ndim, field_kwargs, C=C, strat_loss=True,
                       eq_loss=has_iface)

    ev = strati(name, C=fobj, mode="above", base=_UNCONF_ISO)
    if has_iface:
        ev.addIsosurface(_UNCONF_ISO, seed=np.asarray(iface_pts, float))

    meta = _FieldMeta(name=name, kind="erosional", unit_indices=[],
                      is_unconformity=True, iso_name=_UNCONF_ISO,
                      above_levels=list(above_levels), below_levels=list(below),
                      eroded_level=eroded_level,
                      n_iq=len(iq_pairs), iq_relations=iq_relations,
                      seed_isos=has_iface)
    return ev, meta


def _build_depositional_iq_event(name, levels_young_to_old, unit_name, level_pts,
                                 ndim, field_cls, field_kwargs, iq_samples,
                                 grid, trend, onlap_iso=None, iface_by_level=None,
                                 below_iface=None):
    """
    Build a **depositional** package from unit points (SPEC §4.2, wcsb).

    ``levels_young_to_old`` are the package's unit levels (ascending = youngest→oldest).
    Each pair of **adjacent** bands is ordered with a point-vs-point inequality
    (``CSet.iq``: younger band points ``>`` older band points), consumed by the GeoINR
    gradient-normalised inequality loss; a **no-overturn** regularizer keeps the field
    monotone. Each contact is **named after the older unit whose top it is** (a contact is
    the top of the unit below it). Note curlew labels the band immediately *above* an
    isosurface with the isosurface's name, so the band lithology *keys* read like surface
    names (``"<pkg>_<older unit> Top"``) — the ``level_litho`` map keeps the
    level→lithology bookkeeping exact. The oldest band is the event's base lithology (the
    event name). When ``onlap_iso`` is given the package onlaps the surface below it.

    When ``iface_by_level`` is given (the seed-iso / hybrid path), each internal contact
    is additionally given an ``eq`` trace + **seed isosurface** from the on-contact points
    at its (older-unit) level — pinning the contact's value to the interface data instead
    of estimating it post-fit. Contacts with no on-contact points are left value-free.
    """
    levels = list(levels_young_to_old)
    k = len(levels)
    iface_by_level = iface_by_level or {}

    def pool(L):
        return level_pts.get(L, np.empty((0, ndim)))

    def iface(L):
        return iface_by_level.get(L, np.empty((0, ndim)))

    # Contact j sits between band j (younger) and band j+1 (older): it is the TOP of band
    # j+1, so name it after that older unit (a contact is the top of the unit below it).
    contact_names = [f"{unit_name(levels[j + 1])} Top" for j in range(k - 1)]

    # adjacent-band ordering: band j above band j+1 (transitivity orders the whole package).
    # These unit-point pairs are kept on BOTH paths (a 2026-06-23 experiment to drop them on
    # the seed-iso path — as "redundant" with the interface ordering + CE — left band accuracy
    # unchanged at the markers but reintroduced deep-region volume-inversion islands between
    # them, 2.20% of grid columns vs 0.47% kept; the within-package *interface* ordering below
    # is additional, pinning the seeded contacts in value space).
    iq_pairs, iq_relations = [], []
    for j in range(k - 1):
        if len(pool(levels[j])) and len(pool(levels[j + 1])):
            iq_pairs.append((pool(levels[j]), pool(levels[j + 1]), ">"))
            iq_relations.append((levels[j], levels[j + 1], ">"))

    # on-contact (interface) traces + seeds: contact j lives on the top of unit levels[j+1]
    eq_traces, seed_specs = [], []
    for j in range(k - 1):
        ipts = iface(levels[j + 1])
        if len(ipts):
            pts = np.asarray(ipts, float)
            eq_traces.append(pts)
            seed_specs.append((contact_names[j], pts))
    has_iface = len(eq_traces) > 0

    # surface ordering / nesting (the single §2 principle, one builder):
    #   * within-package — each internal contact (the top of unit ``levels[j+1]``), young→old,
    #     sits ABOVE the next-older one, so the seeded contacts keep stratigraphic order and
    #     cannot cross where the interface data defines them (the eq traces pin each value);
    #   * package-top nesting — the package's TOP surface (top of its youngest unit, the
    #     surface a younger baselap-onto-conformal package onlaps) sits ABOVE every OLDER
    #     unconformity surface, so the package's field stays below that threshold at the older
    #     surfaces and a younger package onlapping the top cannot reach down and erode below an
    #     older surface (e.g. the Precambrian) at the basin margin.
    own_contacts = [(levels[j + 1], iface(levels[j + 1])) for j in range(k - 1)]  # young→old
    nest_pairs, nest_rels = _surface_ordering_iq(
        own_contacts=own_contacts, top_surface=(levels[0], iface(levels[0])),
        older_surfaces=below_iface)
    iq_pairs += nest_pairs
    iq_relations += nest_rels

    C = CSet(crs="model")
    if iq_pairs:
        C.iq = (int(iq_samples), iq_pairs)
    if eq_traces:
        C.eq = eq_traces
    C.grid = grid.copy()
    C.trend = trend
    fobj = _make_field(field_cls, name, ndim, field_kwargs, C=C, strat_loss=True,
                       eq_loss=has_iface)

    if onlap_iso is not None:
        ev = strati(name, C=fobj, mode="above", base=onlap_iso, onlap=True)
    else:
        ev = strati(name, C=fobj)
    for cname, pts in seed_specs:
        ev.addIsosurface(cname, seed=pts)

    oldest = levels[-1]
    level_litho = {oldest: name}              # base lithology (oldest band = event name)
    contacts = []
    for j, cname in enumerate(contact_names):
        # band j (younger) sits above contact j, so its lithology key is the contact's
        # llookup key (curlew names the band above an iso after the iso)
        level_litho[levels[j]] = f"{name}_{cname}"
        contacts.append(_ContactInfo(name=cname, unit_index=-1, level=levels[j],
                                     n_points=len(pool(levels[j]))))

    meta = _FieldMeta(name=name, kind="depositional", unit_indices=[],
                      levels=levels, contacts=contacts, level_litho=level_litho,
                      n_iq=len(iq_pairs), iq_relations=iq_relations,
                      seed_isos=has_iface)
    return ev, meta


def _build_unit_only_chain(units, sfields, obs, Xm, ndim, field_cls, field_kwargs,
                           iq_samples, cap_per_unit, seed, reg_samples, reg_grid_n):
    """
    Build the strat-column event chain (oldest→youngest) from point observations.

    The partitioner emits an alternating youngest-first sequence of erosional and
    depositional fields. Reversed to oldest-first, this builder:

    1. synthesizes a **region-only basement** from the oldest erosional field's
       attached unit(s) (``derive_scalar_fields`` does not emit it — SPEC §8.1),
    2. builds each **erosional** event (``iq`` above-vs-below ordering, no-overturn)
       — see :func:`_build_erosional_event`,
    3. builds each **depositional** package (``iq`` adjacent-band ordering,
       no-overturn), onlapping the surface below it — an **unconformity** (``eroded``
       neighbour) or, for a **baselap-onto-conformal** package (``sf.onlap_package``),
       the **top of the conformal package directly below it**,
    4. synthesizes a **region-only** event for the dropped youngest baselap unit
       (top of column), onlapping the youngest unconformity.

    When the observations carry **on-contact (interface) points**, every surface is
    additionally **seeded** from them (``eq`` traces + seed isosurfaces) so its iso-value
    is read directly from the interface data — the seed-iso / hybrid path. With only unit
    points (wcsb) the surfaces stay value-free; call :func:`estimate_isosurfaces` (or use
    the learnable :class:`~curlew.geology.softunit.UnitLoss`) after fitting.

    Returns ``(events, field_meta, level_litho, level_pts)``.
    """
    assert isinstance(field_cls, type) and issubclass(field_cls, GeoINR), (
        "The chain path requires a GeoINR-family field ('GeoINR' or 'Siren'): its "
        "inequality / interface constraints need the gradient-normalised inequality, "
        "interface and no-overturn losses these fields implement."
    )
    rng = np.random.default_rng(seed)
    level_pts = _cap_level_pools(obs, Xm, cap_per_unit, rng)
    level_name = {u.level: u.name.strip() or f"level{u.level}" for u in units}
    # shared no-overturn grid (model coords). The **seed-iso / hybrid** path (interface
    # points present, e.g. skmb) samples it with model-scaled **Poisson-disk** spacing for
    # even, fine domain coverage — the residual basin-margin fix (SKMB_HANDOFF §4a); the
    # **wcsb** unit-only path (no interfaces) keeps the original uniform-from-lattice draw.
    has_iface = bool(obs.is_interface.any())
    grid = _model_grid(Xm, ndim, n=reg_grid_n, draw=reg_samples, poisson=has_iface)
    trend = _trend_up(ndim)             # younging-up direction (+z)

    # on-contact (interface) points by level, in model coords (empty when none present)
    iface_by_level = {}
    if bool(obs.is_interface.any()):
        imask = obs.is_interface
        for L in obs.levels("interface"):
            iface_by_level[L] = Xm[imask & (obs.level == L)]
    eroded_levels = sorted(u.level for u in units if u.relation == "eroded")  # for surface nesting

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

    last_erosional = None
    for i, sf in enumerate(sfields_old):
        if sf.kind == "erosional":
            # the younger package that onlaps this unconformity (the dropped top unit for
            # the youngest erosional, synthesized after the loop). This sets the local iso
            # placement; the below side is "everything older", computed in the builder.
            if i + 1 < nsf and sfields_old[i + 1].kind == "depositional":
                above_levels = sorted({units[u].level for u in sfields_old[i + 1].unit_indices})
            else:
                above_levels = [units[0].level]

            eroded_level = units[sf.horizon_unit_indices[0]].level
            ename = uniq(f"{unit_name(eroded_level)} Unconformity")
            # on-surface points of every OLDER unconformity, one array each (for surface nesting)
            below_iface = [iface_by_level[L] for L in eroded_levels
                           if L > eroded_level and len(iface_by_level.get(L, ()))]
            ev, meta = _build_erosional_event(
                ename, eroded_level, above_levels, all_levels, level_pts, ndim,
                field_cls, field_kwargs, iq_samples, grid, trend,
                iface_pts=iface_by_level.get(eroded_level),
                below_iface=below_iface)

            # the oldest erosional eroded the basement: synthesize a region-only basement
            # event below it (the eroded unit(s) become its lithology). Its field is an
            # alias of this unconformity's field — GeoINR classifies the basement domain
            # with the basement interface's own field + iso, so the deep scalar is the
            # fitted unconformity field continued below its iso, not an untrained network.
            if i == 0:
                base_levels = sorted({units[u].level for u in sf.unit_indices})
                bname = uniq(unit_name(base_levels[-1]) if base_levels else "Basement")
                bev, bmeta = _build_region_only_event(
                    bname, base_levels, level_pts, ndim, field_cls, field_kwargs,
                    alias_of=ev.getField(0))
                events.append(bev); metas.append(bmeta); level_litho.update(bmeta.level_litho)
            elif sf.unit_indices:
                # a unit sandwiched between two consecutive (doubled) unconformities: its
                # material is preserved between the **older** unconformity below it (the
                # previously-built event, which it ONLAPS) and **this** unconformity above it
                # (built next, which TRUNCATES it). Without this region the band between the
                # two iso surfaces is claimed by no event — a void. Its field aliases the
                # older unconformity's (the surface it sits on), like the basement/top regions.
                sand_levels = sorted({units[u].level for u in sf.unit_indices})
                sname = uniq(unit_name(sand_levels[-1]))
                onlap_prev = bool(metas and metas[-1].is_unconformity and last_erosional is not None)
                sev, smeta = _build_region_only_event(
                    sname, sand_levels, level_pts, ndim, field_cls, field_kwargs,
                    onlap_iso=_UNCONF_ISO if onlap_prev else None,
                    alias_of=last_erosional.getField(0) if onlap_prev else None)
                if onlap_prev:
                    smeta.onlap_event = metas[-1].name
                    smeta.onlap_iso_id = f"{metas[-1].name}::unconf"
                events.append(sev); metas.append(smeta); level_litho.update(smeta.level_litho)

            events.append(ev); metas.append(meta)
            last_erosional = ev
        else:  # depositional package
            levels = sorted({units[u].level for u in sf.unit_indices})  # young → old
            # what this package onlaps: the unconformity directly below it, or — for a
            # baselap-onto-conformal package — the top of the conformal package below.
            onlap_iso = onlap_event = onlap_iso_id = None
            if metas and metas[-1].is_unconformity:
                onlap_iso = _UNCONF_ISO
                onlap_event = metas[-1].name
                onlap_iso_id = f"{onlap_event}::unconf"
            elif sf.onlap_package:
                assert metas and metas[-1].kind == "depositional", (
                    "baselap-onto-conformal package expects the conformal package it "
                    "onlaps to have been built immediately before it."
                )
                lower_ev, lower_meta = events[-1], metas[-1]
                top_level = lower_meta.levels[0]   # youngest level of the lower package
                top_pts = iface_by_level.get(top_level)
                assert top_pts is not None and len(top_pts) > 0, (
                    f"baselap-onto-conformal: the lower package '{lower_meta.name}' needs "
                    f"on-contact points at its top (level {top_level}) to seed the onlap "
                    "surface; this pattern requires interface observations."
                )
                top_name = f"{unit_name(top_level)} Top"
                lower_ev.addIsosurface(top_name, seed=np.asarray(top_pts, float))
                lower_meta.top_iso = top_name
                onlap_iso, onlap_event = top_name, lower_meta.name
                onlap_iso_id = f"{onlap_event}::top"

            pname = uniq(unit_name(levels[-1]))   # base (oldest) band names the event
            # on-surface points of every unconformity OLDER than this package (one array each),
            # so the package's top iso (a baselap onlap threshold) nests above them.
            pkg_below = [iface_by_level[L] for L in eroded_levels
                         if L > levels[-1] and len(iface_by_level.get(L, ()))]
            ev, meta = _build_depositional_iq_event(
                pname, levels, unit_name, level_pts, ndim, field_cls, field_kwargs,
                iq_samples, grid, trend, onlap_iso=onlap_iso,
                iface_by_level=iface_by_level, below_iface=pkg_below)
            meta.onlap_package = bool(sf.onlap_package)
            meta.onlap_event, meta.onlap_iso_id = onlap_event, onlap_iso_id
            events.append(ev); metas.append(meta); level_litho.update(meta.level_litho)

    # synthesize the dropped youngest unit (a top baselap package the partitioner
    # skips): a region-only event onlapping the youngest unconformity, with its field
    # aliasing that unconformity's field (its region = above the unconformity's iso).
    top = units[0]
    if top.level not in level_litho:
        onlap = bool(metas and metas[-1].is_unconformity and last_erosional is not None)
        qname = uniq(unit_name(top.level))
        qev, qmeta = _build_region_only_event(
            qname, [top.level], level_pts, ndim, field_cls, field_kwargs,
            onlap_iso=_UNCONF_ISO if onlap else None,
            alias_of=last_erosional.getField(0) if onlap else None)
        events.append(qev); metas.append(qmeta); level_litho.update(qmeta.level_litho)

    return events, metas, level_litho, level_pts


# ---------------------------------------------------------------------------
# Post-fit isosurface estimation (unit-only path)
# ---------------------------------------------------------------------------
def estimate_isosurfaces(M) -> dict:
    """
    Estimate and set every surface iso-value of a **unit-only** model from its
    fitted fields.

    The unit-only path has no on-surface observations, so surfaces cannot be seeded
    or fixed at build time. Instead, after ``M.fit`` this estimates each surface as
    the value that separates the field-value distributions of the two unit-point
    populations it lies between — the **midpoint of their medians** — and sets it
    with the ordinary :meth:`~curlew.geology.geoevent.GeoEvent.addIsosurface`:

    - a **depositional contact** separates its two adjacent bands (closest band with
      points on each side, so empty levels are skipped). A package's contacts are then
      **clamped to stratigraphic order** (strictly decreasing youngest→oldest): where two
      bands' field-value distributions overlap so much that their raw midpoints invert,
      the offending contact is pulled just below the one above it, so the inseparable
      band **pinches out** instead of crossing — crossed contacts would label bands out
      of stratigraphic sequence;
    - an **unconformity** separates the onlapping (adjacent-younger) package from the
      eroded unit directly below it.

    The estimated value becomes both the event's ``Overprint`` threshold (the iso
    *name* was wired at build time and resolves at predict) and the extraction iso.
    Call this **after** ``M.fit`` and **before** ``M.predict``; surfaces are estimates
    (unconformities are the hardest), and re-calling after further fitting simply
    overwrites them.

    Parameters
    ----------
    M : curlew.geology.geomodel.GeoModel
        A fitted model built by :func:`build_geomodel`'s unit-only path.

    Returns
    -------
    dict
        Maps each isosurface's lithology key (``"<event>_<iso name>"``) to the
        estimated value (in field/model units).
    """
    level_pts = getattr(M, "level_points", None)
    assert level_pts is not None, (
        "estimate_isosurfaces requires a model built by build_geomodel's unit-only path "
        "(M.level_points missing)."
    )

    def median_value(ev, levels):
        """Median field value of ``ev`` over the pooled unit points of ``levels`` (model coords)."""
        pts = np.concatenate([level_pts[L] for L in levels
                              if len(level_pts.get(L, ())) > 0], axis=0)
        g = ev.predict(pts, combine=False, to_numpy=True, litho=False,
                       props=False, isosurfaces=False)
        return float(np.median(g.scalar))

    def closest_with_points(levels):
        """First level in ``levels`` (given closest-first) that has unit points."""
        for L in levels:
            if len(level_pts.get(L, ())) > 0:
                return L
        return None

    out = {}
    for meta in M.field_meta:
        if meta.kind == "region-only":
            continue
        ev = M[meta.name]
        if meta.is_unconformity:
            above = [L for L in meta.above_levels if len(level_pts.get(L, ())) > 0]
            below = closest_with_points(sorted(meta.below_levels))  # eroded unit (or closest older)
            assert above and below is not None, (
                f"Cannot estimate iso for '{meta.name}': no unit points above/below it."
            )
            iso = 0.5 * (median_value(ev, above) + median_value(ev, [below]))
            ev.addIsosurface(meta.iso_name, value=iso)
            out[f"{meta.name}_{meta.iso_name}"] = iso
        else:
            # contact j separates band levels[j] (younger) from levels[j+1] (older)
            values = []
            for j, contact in enumerate(meta.contacts):
                younger = closest_with_points(meta.levels[j::-1])      # j, j-1, ..., 0
                older = closest_with_points(meta.levels[j + 1:])       # j+1, j+2, ...
                assert younger is not None and older is not None, (
                    f"Cannot estimate contact '{contact.name}' on '{meta.name}': "
                    "no unit points on one side."
                )
                values.append(0.5 * (median_value(ev, [younger]) + median_value(ev, [older])))
            # clamp to stratigraphic order (strictly decreasing young -> old): if two bands
            # overlap enough that their raw midpoints invert, pull the lower contact just
            # below the one above it, so the inseparable band pinches out rather than
            # crossing (crossed isos would label the bands out of stratigraphic sequence).
            for j in range(1, len(values)):
                values[j] = min(values[j], values[j - 1] - 1e-8)
            # insert OLDEST-first: lithology IDs are numbered in isosurface insertion order
            # (after the event's base lithology), and events are numbered oldest->youngest,
            # so this keeps lithoID monotone with age (ascending ID = younger) across the
            # whole model -- the natural reading when inspecting exported volumes.
            for contact, iso in zip(reversed(meta.contacts), reversed(values)):
                ev.addIsosurface(contact.name, value=iso)
                out[f"{meta.name}_{contact.name}"] = iso
    return out


#: No-overturn weight for the **coupled** (soft-unit) path. Much lower than the per-field
#: :data:`_OVERTURN_WEIGHT` (30): there the sequential combine needs strongly monotone fields
#: to avoid basement leakage, but here the always-active cross-entropy already enforces the
#: stratigraphic ordering at every marker, so a strong magnitude prior is mostly redundant —
#: and, measured on wcsb, it actively *strangles the field fit* (it over-smooths, so a package's
#: 7–9 bands cannot separate in scalar space). Relaxing it from 30→~6 drops the NLL from ~1.06
#: to ~0.8 (matching GeoINR's near-zero regularisation) and lifts marker accuracy, while keeping
#: predicted column inversions low (~0.5 %); lower still (→1) fits marginally better but lets the
#: deep unconstrained region overturn (several-% inversions in ``GeoModel.predict``'s sequential
#: combine). 6 is the balance; tune via ``attach_unit_loss(overturn_weight=...)``.
#:
#: NOTE (seed-iso / hybrid path, e.g. skmb): use a **stiffer** weight (~12). There the interface
#: ``eq`` loss hard-pins the fields at the contact surfaces, and that curvature drives deep,
#: data-sparse regions to overturn as training continues — at 6 the predicted volume's
#: age-inversions *grow* with epochs (over-training), at 12 they *fall* (0.36 %→0.14 % at
#: 2000→3000 ep), i.e. robust to long training like the wcsb path. The interface ``eq`` already
#: separates the bands, so the stiffer prior costs little band accuracy (~0.03). Set it via
#: ``attach_unit_loss(overturn_weight=12)``.
_COUPLED_OVERTURN_WEIGHT = 6.0


def attach_unit_loss(M, *, tau=0.05, points_per_level=1024, iso_lr=None, weight=1.0,
                     overturn_weight=_COUPLED_OVERTURN_WEIGHT, seed_isos=False):
    """
    Build a :class:`~curlew.geology.softunit.UnitLoss` for a chain-path model and switch
    its fields onto the **coupled** recipe (SOFTUNIT_SPEC §4, §6).

    This is the entry point for the unit (soft-carve / NLL) coupling: instead of fitting
    each field independently with inequalities and estimating surfaces post-hoc, the
    returned loss couples **all** scalar fields in one cross-entropy against the true unit
    levels. Pass it to :meth:`~curlew.geology.geomodel.GeoModel.fit` as
    ``custom_loss=[loss]``, then call
    :meth:`~curlew.geology.softunit.UnitLoss.write_isosurfaces` before predicting
    (``estimate_isosurfaces`` is **not** used on this path).

    On the coupled path each field's ``iq_norm_weight`` is set to 0 (the cross-entropy
    supersedes the inequality ordering, mirroring GeoINR's ``include_unit_constraints=False``)
    and its ``overturn_weight`` is **relaxed** to :data:`_COUPLED_OVERTURN_WEIGHT` (the CE
    already enforces ordering at the data, so the per-field path's strong magnitude prior of
    30 is redundant here and over-smooths the fit — see that constant). ``HSet`` stays
    all-zero, as on the per-field path.

    Two iso regimes:

    - **learnable** (``seed_isos=False``, wcsb): the coupling owns shared, learnable
      iso-values and writes them post-fit. No on-contact data needed.
    - **seeded** (``seed_isos=True``, skmb): the iso-values are read from interface-seeded
      isosurfaces each step (no learnable isos). The fields keep their **interface**
      (``eq_norm``) loss so the contacts stay pinned to the on-contact data; nothing is
      written post-fit (the seeds already resolve at predict).

    Parameters
    ----------
    M : curlew.geology.geomodel.GeoModel
        A model built by :func:`build_geomodel`'s chain path (not yet fitted). Use a
        generous ``cap_per_unit`` (or ``None``): on the coupled path balance comes from
        ``points_per_level``, so the cap only limits the distinct geometry the fields see.
    tau : float, optional
        Soft-unit temperature; sharpness ``s = 1/τ`` (default 0.05 ⇒ ``s = 20``).
    points_per_level : int, optional
        Points sampled per level each epoch (balanced sampling, default 1024).
    iso_lr : float, optional
        Learning rate for the iso-value optimiser (default ~10× a coupled field's lr).
        Unused when ``seed_isos=True``.
    weight : float, optional
        Weight on the NLL loss term (default 1.0).
    overturn_weight : float, optional
        No-overturn weight applied to every coupled field, overriding the value set at
        build time. Defaults to :data:`_COUPLED_OVERTURN_WEIGHT`. Lower → better marker
        fit but more deep-region inversions in ``GeoModel.predict``; ``None`` leaves each
        field's built value untouched.
    seed_isos : bool, optional
        Read iso-values from interface-seeded isosurfaces instead of learning them (the
        hybrid skmb path; the model must have been built with interface observations).

    Returns
    -------
    curlew.geology.softunit.UnitLoss
        The coupling loss, ready to pass to ``M.fit(custom_loss=[loss])``.
    """
    from curlew.geology.softunit import UnitLoss

    assert getattr(M, "level_points", None) is not None, (
        "attach_unit_loss requires a model built by build_geomodel's chain path "
        "(M.level_points missing)."
    )
    # relax the (now mostly redundant) no-overturn prior so it does not over-smooth the fit.
    # The unit-only (learnable) path zeros iq_norm — the CE supersedes the inequality ordering
    # where unit points are dense. The seeded path instead **keeps iq_norm on**: the interface
    # data defines the surfaces directly, and the gradient-normalised inequalities order them
    # where the CE is silent — within a package (younger contact above older) and, crucially,
    # across unconformities (older surfaces stay below a younger iso), in the deep, unit-sparse
    # regions the CE cannot reach. The hinge is zero once satisfied, so it does not fight the
    # eq/CE fit; it only corrects the ordering violations (e.g. a younger unconformity cutting
    # below the Precambrian).
    for meta in M.field_meta:
        f = M[meta.name].getField(0)
        if hasattr(f, "iq_norm_weight"):
            f.iq_norm_weight = _IQ_WEIGHT if seed_isos else 0.0
        if (overturn_weight is not None) and hasattr(f, "overturn_weight"):
            f.overturn_weight = float(overturn_weight)
    return UnitLoss(M, tau=tau, points_per_level=points_per_level, iso_lr=iso_lr,
                    weight=weight, seed_isos=seed_isos)


def build_geomodel(strat_col_csv, observations, *, field="GeoINR",
                   scale="isometric", transform=None, cap_per_unit=None,
                   field_kwargs=None, hset=None, iq_samples=256,
                   reg_samples=5000, reg_grid_n=48, seed=0) -> GeoModel:
    """
    Build a ready-to-fit :class:`~curlew.geology.geomodel.GeoModel` from a
    stratigraphic-column CSV and point observations.

    The construction path is chosen automatically from the observations:

    - **On-surface path** (contact points present, e.g. ``multilayer_fold``): each
      depositional package is one ``strati`` event with ``eq`` traces, seed
      isosurfaces, ``gv`` gradients (where normals exist) and ``iq`` inequalities.
    - **Unit-only path** (only level-labelled unit points, e.g. ``wcsb``): the
      column is built as an alternating chain of erosional unconformities and
      depositional packages with synthesized region-only events at both ends. Each
      field is constrained purely by point-vs-point inequalities (``CSet.iq``:
      younger unit points above older ones) consumed by the GeoINR-family fields'
      gradient-normalised inequality loss, plus a no-overturn regularizer (see
      :func:`_build_unit_only_chain`; requires ``field="GeoINR"`` or ``"Siren"``).
      Surface iso-values are not learned — after ``M.fit``, call
      :func:`estimate_isosurfaces` to estimate and set them (required before
      ``M.predict``).

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
        Extra keyword arguments forwarded to each field's constructor (e.g.
        ``hidden_dim``, ``learning_rate``; on the unit-only path also
        ``iq_norm_weight`` / ``overturn_weight`` to override the default loss
        weights).
    hset : curlew.core.HSet, optional
        Hyperparameters for every event on the on-surface path; a minimal default
        is used when omitted. (The unit-only path keeps ``HSet`` all-zero — its loss
        comes from the GeoINR-side weights above.)
    iq_samples : int
        Number of random points drawn per inequality pool each evaluation (both paths).
        The GeoINR reference effectively uses *all* constraint points per epoch
        (``max_points=0``); with units capped at ``cap_per_unit`` ~1500, ``~1024``
        approaches that. Under-sampling leaves the deep regions of unconformity fields
        poorly controlled (floating islands of younger packages inside the basement).
    reg_samples : int
        Number of no-overturn grid points sampled per epoch on the unit-only path
        (GeoINR's ``n_grid_samples`` default is 5000). Unused on the on-surface path.
    reg_grid_n : int
        Resolution (divisions per axis) of the shared no-overturn grid the
        regularization points are drawn from. Unused on the on-surface path.
    seed : int
        Random seed for unit subsampling.

    Returns
    -------
    curlew.geology.geomodel.GeoModel
        Model with events, CSets, isosurfaces and the normalization transform
        attached. The parsed column and per-field metadata are stored on
        ``M.strat_units``, ``M.horizons``, ``M.scalar_fields``, ``M.field_meta``
        and (unit-only path) ``M.level_litho`` / ``M.level_points`` for
        inspection/visualisation. On the unit-only path, fit the model and then
        call :func:`estimate_isosurfaces` before predicting.
    """
    field_cls = _FIELD_TYPES[field] if isinstance(field, str) else field

    units = load_strat_column_csv(strat_col_csv)
    horizons = derive_horizons(units)
    sfields = derive_scalar_fields(units)

    ndim = observations.ndim
    if transform is None:
        transform = _normalization_transform(observations.bounds(), scale, ndim)
    Xm = transform.apply(observations.coords.astype(float))

    # path selection: the simple **on-surface** path handles an all-depositional column
    # with contact points (multilayer_fold); any column with unconformities or a
    # baselap-onto-conformal split goes through the **chain** path, which consumes
    # interface points when present (skmb) and falls back to unit-only (wcsb).
    has_structure = any(sf.kind == "erosional" or sf.onlap_package for sf in sfields)
    use_on_surface = bool(observations.is_interface.any()) and not has_structure

    level_litho, level_pts = {}, None
    if use_on_surface:
        # ---- on-surface path (multilayer_fold): depositional events only ----
        events, field_meta = [], []
        for sf in reversed(sfields):  # GeoModel expects oldest→youngest
            if sf.kind != "depositional" or not sf.horizon_unit_indices:
                raise NotImplementedError(
                    "The on-surface path supports depositional packages with contact "
                    f"points only; field '{sf.name}' is '{sf.kind}'. Datasets with only "
                    "unit points (no interface points) use the chain path instead."
                )
            ev, meta = _build_depositional_event(
                sf, units, observations, Xm, ndim, field_cls, field_kwargs, hset, iq_samples
            )
            events.append(ev)
            field_meta.append(meta)
    else:
        # ---- chain path (wcsb unit-only / skmb interface+unit hybrid) ----
        events, field_meta, level_litho, level_pts = _build_unit_only_chain(
            units, sfields, observations, Xm, ndim, field_cls, field_kwargs,
            iq_samples, cap_per_unit, seed, reg_samples, reg_grid_n,
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
    if level_pts is not None:
        # capped per-level unit pools (model coords) — needed by estimate_isosurfaces /
        # the soft-unit coupling (both the learnable wcsb path and the seeded skmb path)
        M.level_points = level_pts
    return M
