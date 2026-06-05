"""
Functions for performing common IO operations.
"""
import os
from pathlib import Path
import numpy as np
import torch

def saveModel(path, model):
    """
    Save a :class:`~curlew.geology.geomodel.GeoModel` or
    :class:`~curlew.geology.geoevent.GeoEvent` to disk.

    Uses :func:`torch.save`, which handles PyTorch tensors, ``nn.Module`` fields,
    and optimiser state in a way that is consistent with the rest of curlew.

    Parameters
    ----------
    path : str | os.PathLike
        Output file path (e.g. ``"my_model.pt"``).
    model : curlew.geology.geomodel.GeoModel | curlew.geology.geoevent.GeoEvent
        The geological model or event to save.
    """
    from curlew.geology.geoevent import GeoEvent
    from curlew.geology.geomodel import GeoModel

    if not isinstance(model, (GeoModel, GeoEvent)):
        raise TypeError(
            f"saveModel expects a GeoModel or GeoEvent, got {type(model).__name__}"
        )
    path = Path(path)
    if path.parent and str(path.parent) not in ("", "."):
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, path)

def loadModel(path, map_location=None):
    """
    Load a :class:`~curlew.geology.geomodel.GeoModel` or
    :class:`~curlew.geology.geoevent.GeoEvent` written by :func:`saveModel`.

    Parameters
    ----------
    path : str | os.PathLike
        Path to the saved model file.
    map_location : str | torch.device | callable, optional
        Device mapping passed to :func:`torch.load` (e.g. ``"cpu"`` or
        ``curlew.device``). Defaults to :data:`curlew.device`.

    Returns
    -------
    curlew.geology.geomodel.GeoModel | curlew.geology.geoevent.GeoEvent
        The restored model or event.
    """
    import curlew
    from curlew.geology.geoevent import GeoEvent
    from curlew.geology.geomodel import GeoModel

    path = Path(path)
    if map_location is None:
        map_location = curlew.device

    try:
        return torch.load(path, weights_only=False, map_location=map_location)
    except TypeError:
        return torch.load(path,  map_location=map_location)    

def saveOBJ(filename, xyz, rgb, faces):
    """
    Writes a mesh to an OBJ file.

    Parameters
    ---------------
    filename : str
        Output file path.
    xyz : np.ndarray | list
        List of vertex positions [(x, y, z), ...]
    rgb: np.ndarray | list
        List of vertex colors [(r, g, b), ...] or [(r, g, b, a), ...] in [0, 1] or [0, 255]
    faces: np.ndarray | list
        List of faces [(i1, i2, i3), ...] with 0-based indices
    """
    with open(filename, 'w') as f:
        f.write("# OBJ file with vertex colors (stored in comments)\n")

        for i, v in enumerate(xyz):
            x, y, z = v
            color_str = ""
            if (rgb is not None) and i < len(rgb):
                color = rgb[i]
                # Normalize to 0-1 if in 0-255
                if max(color) > 1:
                    color = [c / 255.0 for c in color[:3]]
                else:
                    color = color[:3]
                color_str = " # color {:.4f} {:.4f} {:.4f}".format(*color)
            f.write("v {:.6f} {:.6f} {:.6f}{}\n".format(x, y, z, color_str))

        for face in faces:
            # OBJ format is 1-indexed
            f.write("f {}\n".format(' '.join(str(i + 1) for i in face)))

def savePLY(path, xyz, rgb=None, normals=None, attr=None, names=None, faces=None):
    """
    Write a point cloud and associated RGB and scalar fields to .ply.

    Parameters
    ---------------
    Path : str
        File path for the created (or overwritten) .ply file
    xyz : np.ndarray
        Array of xyz points to add to the PLY file
    rgb : np.ndarray
        Array of 0-255 RGB values associated with these points, or None.
    normals : np.ndarray
        Array of normal vectors associated with each point, or None.
    attr : np.ndarray
        Array of float32 values associated with these points, or None
    attr_names : list 
        List containing names for each of the passed attributes, or None.
    faces : np.ndarray, optional
        Array of triangular faces (F x 3) with vertex indices, or None.
    """

    # make directories if need be
    os.makedirs(os.path.dirname( path ), exist_ok=True )

    try:
        from plyfile import PlyData, PlyElement
    except:
        assert False, "Please install plyfile (`pip install plyfile`) to export to PLY."

    sfmt='f4' # use float32 precision

    # create structured data arrays and derived PlyElements
    vertex = np.array(list(zip(xyz[:, 0], xyz[:, 1], xyz[:, 2])),
                      dtype=[('x', 'double'), ('y', 'double'), ('z', 'double')])
    ply = [PlyElement.describe(vertex, 'vertices')]

    # create RGB elements
    if rgb is not None:
        if (np.max(rgb) <= 1):
            irgb = np.clip((rgb * 255),0,255).astype(np.uint8)
        else:
            irgb = np.clip(rgb,0,255).astype(np.uint8)

        # convert to structured arrays and create elements
        irgb = np.array(list(zip(irgb[:, 0], irgb[:, 1], irgb[:, 2])),
                        dtype=[('r', 'u1'), ('g', 'u1'), ('b', 'u1')])
        ply.append(PlyElement.describe(irgb, 'color'))  # create ply elements

    # normal vectors
    if normals is not None:
        # convert to structured arrays
        norm = np.array(list(zip(normals[:, 0], normals[:, 1], normals[:, 2])),
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        ply.append(PlyElement.describe(norm, 'normals'))  # create ply elements

    # attributes
    if attr is not None:
        if names is None:
            names = ["SF%d"%(i+1) for i in range(attr.shape[-1])]
        
        # map scalar fields to required type and build data arrays
        data = attr.astype(np.float32)
        for b in range(data.shape[-1]):
            n = names[b].strip().replace(' ', '_') #remove spaces from n
            if 'scalar' in n: #name already includes 'scalar'?
                ply.append(PlyElement.describe(data[:, b].astype([('%s' % n, sfmt)]), '%s' % n))
            else: #otherwise prepend it (so CloudCompare recognises this as a scalar field).
                ply.append(PlyElement.describe(data[:, b].astype([('scalar_%s' % n, sfmt)]), 'scalar_%s' % n))
    
    # Append faces if present
    if faces is not None and len(faces) > 0:
        faces = np.asarray(faces, dtype=np.int32)
        face_data = np.array(
            [(list(face),) for face in faces],
            dtype=[('vertex_indices', 'i4', (3,))]
        )
        ply.append(PlyElement.describe(face_data, 'face'))
    
    PlyData(ply).write(path) # and, finally, write everything :-) 

def loadPLY(path):
    """
    Loads a PLY file from the specified path.
    """
    try:
        from plyfile import PlyData, PlyElement
    except:
        assert False, "Please install plyfile (pip install plyfile) to load PLY."
    data = PlyData.read(path) # load file!

    # extract data
    xyz = None
    rgb = None
    norm = None
    faces = None
    scalar = []
    scalar_names = []
    
    for e in data.elements:
        if 'face' in e.name.lower():
            faces = np.vstack( e['vertex_indices'])
        if 'vert' in e.name.lower():  # vertex data
            xyz = np.array([e['x'], e['y'], e['z']]).T
            if len(e.properties) > 3:  # vertices have more than just position
                names = e.data.dtype.names
                # colour?
                if 'red' in names and 'green' in names and 'blue' in names:
                    rgb = np.array([e['red'], e['green'], e['blue']], dtype=e['red'].dtype).T
                # normals?
                if 'nx' in names and 'ny' in names and 'nz' in names:
                    norm = np.array([e['nx'], e['ny'], e['nz']], dtype=e['nx'].dtype).T
                # load others as scalar
                mask = ['red', 'green', 'blue', 'nx', 'ny', 'nz', 'x', 'y', 'z']
                for n in names:
                    if not n in mask:
                        scalar_names.append(n)
                        scalar.append(e[n])
        elif 'color' in e.name.lower():  # rgb data
            rgb = np.array([e['r'], e['g'], e['b']], dtype=e['r'].dtype).T
        elif 'normals' in e.name.lower():  # normal data
            norm = np.array([e['x'], e['y'], e['z']], dtype=e['z'].dtype).T
        else:  # scalar data
            scalar_names.append(e.properties[0].name.strip().replace('scalar_',''))
            scalar.append(np.array(e[e.properties[0].name], dtype=e[e.properties[0].name].dtype))
    if len(scalar) > 0:
        scalar = np.vstack(scalar).T
    assert (not xyz is None) and (xyz.shape[0] > 0), "Error - PLY contains no geometry?"

    # TODO - also load faces if present
    
    # return everything needed
    out = dict( xyz = xyz, faces=faces, rgb=rgb, normals=norm, 
                attr=scalar, names=scalar_names )
    if len(scalar) == 0:
        del out['attr']
        del out['names']
    if rgb is None:
        del out['rgb']
    if norm is None:
        del out['normals']
    if faces is None:
        del out['faces']
    return out


# ---------------------------------------------------------------------------
# Observation loading (point observations for the strat-column builder)
# ---------------------------------------------------------------------------
# A thin loader producing per-level point arrays (coords + level), supporting
# the formats found in the GeoINR data directories:
#   - ``.vtp``  : points, a ``level`` point-array, optional ``normals`` array
#                 (read via pyvista, kept an optional/lazily-imported dependency)
#   - ``.csv``  : ``x,y[,z],level`` with optional ``is_interface`` and
#                 ``nx,ny[,nz]`` columns.
# Points are grouped by ``level`` so the builder can assemble per-interface
# seeds/traces and per-unit inequality pairs (see ``curlew.geology.stratbuilder``).

from dataclasses import dataclass, field as _dcfield


@dataclass
class Observations:
    """
    Point observations grouped by stratigraphic ``level``, used by
    :func:`curlew.geology.stratbuilder.build_geomodel`.

    Each row is one observed point. Points play one of three roles, distinguished
    by :attr:`is_interface` and whether a normal vector is present:

    - **interface / contact points** — ``is_interface`` is True; sampled on a
      stratigraphic contact (used for equality traces and seed isosurfaces).
    - **gradient (normal) points** — a finite vector in :attr:`normals`; an
      oriented bedding normal (used for ``gv`` gradient constraints).
    - **unit points** — ``is_interface`` is False and no normal; an observation of
      a unit's interior (used for inequality pairs / region labelling).

    Attributes
    ----------
    coords : np.ndarray
        ``(N, d)`` point positions in global (world) coordinates.
    level : np.ndarray
        ``(N,)`` integer stratigraphic level per point (``-1`` if unknown).
    normals : np.ndarray
        ``(N, d)`` bedding normals; rows without a normal are ``NaN``.
    is_interface : np.ndarray
        ``(N,)`` boolean flag, True for on-contact (interface) points.
    """

    coords: np.ndarray
    level: np.ndarray
    normals: np.ndarray
    is_interface: np.ndarray

    @property
    def ndim(self) -> int:
        """Spatial dimensionality of the observation coordinates."""
        return self.coords.shape[1]

    def __len__(self):
        return self.coords.shape[0]

    def _has_normal(self) -> np.ndarray:
        """Boolean mask of rows that carry a (finite) normal vector."""
        return np.isfinite(self.normals).all(axis=1)

    def levels(self, role: str = "all") -> list:
        """
        Sorted list of distinct (non-negative) levels present for the given role.

        Parameters
        ----------
        role : str
            One of ``"all"``, ``"interface"`` or ``"unit"``.
        """
        if role == "interface":
            mask = self.is_interface
        elif role == "unit":
            mask = (~self.is_interface) & (~self._has_normal())
        else:
            mask = np.ones(len(self), dtype=bool)
        lv = self.level[mask & (self.level >= 0)]
        return sorted(int(v) for v in np.unique(lv))

    def by_level(self, role: str = "interface") -> dict:
        """
        Group point coordinates by level for the requested role.

        Parameters
        ----------
        role : str
            ``"interface"`` (on-contact points), ``"unit"`` (interior unit points),
            or ``"all"``.

        Returns
        -------
        dict[int, np.ndarray]
            Maps each level to an ``(n, d)`` array of coordinates.
        """
        if role == "interface":
            base = self.is_interface
        elif role == "unit":
            base = (~self.is_interface) & (~self._has_normal())
        else:
            base = np.ones(len(self), dtype=bool)
        out = {}
        for L in self.levels(role):
            out[L] = self.coords[base & (self.level == L)]
        return out

    def interface_points(self):
        """Return ``(coords, level)`` for all on-contact (interface) points."""
        m = self.is_interface
        return self.coords[m], self.level[m]

    def unit_points(self):
        """Return ``(coords, level)`` for interior unit points (non-contact, no normal)."""
        m = (~self.is_interface) & (~self._has_normal())
        return self.coords[m], self.level[m]

    def normal_points(self):
        """Return ``(coords, normals, level)`` for points carrying a bedding normal."""
        m = self._has_normal()
        return self.coords[m], self.normals[m], self.level[m]

    def bounds(self):
        """Return ``(min, max)`` corner coordinates of the observed points."""
        return self.coords.min(axis=0), self.coords.max(axis=0)


def _read_vtp(path):
    """
    Read a ``.vtp`` polydata file with pyvista (lazily imported).

    Mirrors GeoINR's reader: any point-data array whose name contains ``"level"``
    (case-insensitive) is taken as the level array; any containing ``"normal"`` as
    the normals array.

    Returns
    -------
    tuple
        ``(points (N,3), level (N,) or None, normals (N,3) or None)``.
    """
    try:
        import pyvista as pv
    except ImportError as exc:  # keep pyvista optional per repo convention
        raise ImportError(
            "Reading .vtp observations requires pyvista. "
            "Install with `conda install -c conda-forge pyvista` (or `pip install pyvista`)."
        ) from exc

    poly = pv.read(str(path))
    level = None
    normals = None
    for name in poly.point_data.keys():
        low = name.lower()
        if "level" in low:
            level = np.asarray(poly.point_data[name]).reshape(-1)
        if "normal" in low:
            normals = np.asarray(poly.point_data[name])
    return np.asarray(poly.points), level, normals


def _read_csv_obs(path):
    """
    Read a ``.csv`` of point observations.

    Expects columns ``x, y[, z], level`` and optionally ``is_interface`` and
    normals (``nx, ny[, nz]``). Column matching is case-insensitive.

    Returns
    -------
    tuple
        ``(points (N,d), level (N,) or None, normals (N,d) or None,
        is_interface (N,) bool or None)``.
    """
    import csv as _csv

    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = _csv.DictReader(fh)
        cols = {c.strip().lower(): c for c in (reader.fieldnames or [])}
        rows = list(reader)

    def col(name):
        return cols.get(name)

    axes = [a for a in ("x", "y", "z") if a in cols]
    naxes = [a for a in ("nx", "ny", "nz") if a in cols]

    def fcol(row, name):
        return float(row[cols[name]])

    pts = np.array([[fcol(r, a) for a in axes] for r in rows], dtype=float)
    level = None
    if "level" in cols:
        level = np.array([int(float(r[col("level")])) for r in rows], dtype=int)
    normals = None
    if len(naxes) == len(axes) and len(naxes) > 0:
        normals = np.array([[fcol(r, a) for a in naxes] for r in rows], dtype=float)
    is_interface = None
    if "is_interface" in cols:
        is_interface = np.array(
            [str(r[col("is_interface")]).strip().lower() in ("1", "true", "yes") for r in rows],
            dtype=bool,
        )
    return pts, level, normals, is_interface


def _coerce_paths(arg):
    """Normalise a path / list-of-paths argument to a list (possibly empty)."""
    if arg is None:
        return []
    if isinstance(arg, (str, os.PathLike)):
        return [arg]
    return list(arg)


def loadObservations(interfaces=None, normals=None, units=None):
    """
    Load point observations for the strat-column builder, merging one or more
    files into a single :class:`Observations`.

    The *role* of each file is given explicitly (mirroring GeoINR's separate
    interface/unit/normal file arguments), since a level-only ``.vtp`` is
    ambiguous between contact points and unit markers. A ``.csv`` may further
    refine the role per-row via an ``is_interface`` column and may carry normals.

    Parameters
    ----------
    interfaces : str | os.PathLike | list, optional
        File(s) of on-contact points (``.vtp`` with a ``level`` array, or ``.csv``).
        Points are flagged ``is_interface=True`` unless a CSV ``is_interface`` column
        says otherwise.
    normals : str | os.PathLike | list, optional
        File(s) of oriented bedding normals (``.vtp`` with a ``normals`` array, or
        ``.csv`` with ``nx,ny[,nz]``). A ``level`` is used if present, else ``-1``.
    units : str | os.PathLike | list, optional
        File(s) of interior unit points (level labels only).

    Returns
    -------
    Observations
        Merged observations with per-point ``coords``, ``level``, ``normals`` and
        ``is_interface`` arrays.
    """
    coords_all, level_all, normals_all, isint_all = [], [], [], []

    def _add(path, role):
        path = Path(path)
        if path.suffix.lower() == ".vtp":
            pts, lvl, nrm = _read_vtp(path)
            isint = None
        elif path.suffix.lower() == ".csv":
            pts, lvl, nrm, isint = _read_csv_obs(path)
        else:
            raise ValueError(f"Unsupported observation format: {path.suffix} ({path}).")

        n, d = pts.shape
        if lvl is None:
            lvl = np.full(n, -1, dtype=int)
        else:
            lvl = np.asarray(lvl).round().astype(int).reshape(-1)
        full_nrm = np.full((n, d), np.nan, dtype=float)
        if nrm is not None:
            full_nrm[:] = np.asarray(nrm)[:, :d]
        if isint is None:
            isint = np.full(n, role == "interfaces", dtype=bool)

        coords_all.append(pts)
        level_all.append(lvl)
        normals_all.append(full_nrm)
        isint_all.append(isint)

    for p in _coerce_paths(interfaces):
        _add(p, "interfaces")
    for p in _coerce_paths(normals):
        _add(p, "normals")
    for p in _coerce_paths(units):
        _add(p, "units")

    if not coords_all:
        raise ValueError("loadObservations: no observation files provided.")

    return Observations(
        coords=np.vstack(coords_all),
        level=np.concatenate(level_all),
        normals=np.vstack(normals_all),
        is_interface=np.concatenate(isint_all),
    )