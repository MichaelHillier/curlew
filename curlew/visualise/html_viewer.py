"""
Self-contained **WebGL HTML viewer** for curlew model outputs — a browser-based
alternative to :class:`curlew.visualise.napari_viewer.NapariViewer` that needs no
Qt session and produces a portable, shareable artifact.

The viewer renders three model layers, all sharing one *unit-visibility* mask,
palette, vertical-exaggeration and X/Y/Z slicer (the same controls as the GeoINR
research prototype it is ported from):

```
grid volume   -> boundary faces between differing unit categories + a full-res slicer
surfaces      -> boundary/contact iso-surface meshes (curlew Grid.contour output)
unit points   -> an observation / marker point cloud
```

Two output modes (choose with ``mode=``):

- ``"self_contained"`` — one ``.html`` file with all geometry embedded as base64.
  Opens directly from ``file://``. Best for small models (multilayer_fold).
- ``"tiled"`` — a small ``.html`` plus a sibling ``<stem>_data/`` folder of binary
  tiles fetched over HTTP (no base64 bloat, progressive draw). Needs a local HTTP
  server (use :func:`serve_and_open`). Best for large models (wcsb).

Design notes
------------
* **numpy-only.** Unlike the prototype (which read ``.vts``/``.vtp`` via the
  ``vtk`` package), this module is fed **in-memory** from curlew objects, so it
  adds no dependency beyond numpy (``pyvista``/``vtk`` stay optional, used only by
  :func:`curlew.io.saveVTK`). The high-level entry point :func:`viewer_from_geode`
  adapts a :class:`curlew.core.Geode` directly; :func:`write_html_viewer` takes
  plain arrays.
* **Coordinate order.** curlew flattens grids C-order (last/z axis fastest); the
  viewer (like VTK) wants x fastest. The grid category volume is reordered with the
  same transpose :func:`curlew.io.saveVTK` uses (``reshape(shape).transpose(2,1,0)``).
* **Axis-aligned grids only.** The slicer/volume need a rectilinear grid; a rotated
  :class:`curlew.geometry.Grid` raises (mirroring saveVTK's ``.vti`` restriction).

The UI/shaders are a single shared template; only the data-loading "bootstrap" JS
differs between the two modes.
"""
from __future__ import annotations

import base64
import functools
import json
import shutil
import webbrowser
from pathlib import Path

import numpy as np

__all__ = ["write_html_viewer", "write_geomodel_viewer", "extract_geomodel_layers",
           "observation_layers", "serve_and_open"]

# Hard cap from the shaders (uPalette[64] / uUnitVisible[64] uniform arrays).
_MAX_CATEGORIES = 64

# Tiling defaults (tiled mode only).
_GRID_TILE_CELLS = 32          # grid-mesh faces are split into XY tiles of this many cells
_SURFACE_TILE_DIMS = (12, 12)  # surface triangles binned into this many XY tiles
_POINT_TILE_DIMS = (12, 12)    # points binned into this many XY tiles


# ===========================================================================
# small numeric helpers (ported from the GeoINR prototype, kept verbatim)
# ===========================================================================
def _b64(array: np.ndarray) -> str:
    """Base64-encode a contiguous numpy array's raw bytes (little-endian)."""
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def _make_mapper(category_values):
    """
    Build a lookup that maps arbitrary integer ids to compact ``uint8`` category
    indices ``0..len(category_values)-1`` (the indices the shaders use).
    """
    offset = int(min(category_values))
    lut = np.zeros(int(max(category_values)) - offset + 1, dtype=np.uint8)
    for idx, value in enumerate(category_values):
        lut[int(value) - offset] = idx

    def mapper(values: np.ndarray) -> np.ndarray:
        return lut[np.asarray(values).astype(np.int64, copy=False) - offset]

    return mapper


def _triangle_normals_i8(positions: np.ndarray) -> np.ndarray:
    """Per-triangle face normals, quantised to int8 (normalised in the shader)."""
    triangles = positions.reshape((-1, 3, 3))
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 0
    normals[valid] /= lengths[valid, None]
    normals[~valid] = np.array([0.0, 0.0, 1.0])
    normals = np.clip(np.rint(normals * 127.0), -127, 127).astype(np.int8)
    return np.repeat(normals, 3, axis=0)


def _tile_ids_from_xy(points_xy, bounds, dims):
    """Bin XY positions into a ``dims[0] x dims[1]`` tile grid → flat tile id."""
    xmin, xmax, ymin, ymax = bounds
    tx = np.floor((points_xy[:, 0] - xmin) / max(xmax - xmin, 1.0) * dims[0]).astype(np.int32)
    ty = np.floor((points_xy[:, 1] - ymin) / max(ymax - ymin, 1.0) * dims[1]).astype(np.int32)
    tx = np.clip(tx, 0, dims[0] - 1)
    ty = np.clip(ty, 0, dims[1] - 1)
    return tx + ty * dims[0]


def _grouped_tile_indices(tile_ids):
    """Return ``[(tile_id, indices), ...]`` grouping element indices by tile id."""
    if tile_ids.size == 0:
        return []
    order = np.argsort(tile_ids, kind="stable")
    sorted_ids = tile_ids[order]
    starts = np.r_[0, np.nonzero(sorted_ids[1:] != sorted_ids[:-1])[0] + 1]
    stops = np.r_[starts[1:], sorted_ids.size]
    return [(int(sorted_ids[s]), order[s:e]) for s, e in zip(starts, stops)]


# ===========================================================================
# grid → boundary faces  (vectorised; shared by both modes)
# ===========================================================================
def _append_grid_faces(parts, axis, sign, mask, cell_categories, xvec, yvec, zvec):
    """
    Append the quads (2 triangles each) for every cell flagged in ``mask`` whose
    face points along ``axis`` (`'x'|'y'|'z'`) in direction ``sign`` (±1).

    Vectorised port of the prototype's ``append_grid_faces`` (no tile offsets —
    operates on the full ``(nz-1, ny-1, nx-1)`` cell arrays). ``cell_categories``
    supplies the colour (the originating cell's category).
    """
    kk, jj, ii = np.nonzero(mask)
    n = int(len(kk))
    if n == 0:
        return
    cats = cell_categories[kk, jj, ii]
    positions = np.empty((n, 6, 3), dtype=np.float32)

    if axis == "x":
        plane_i = ii + (1 if sign > 0 else 0)
        x = xvec[plane_i]
        y0, y1, z0, z1 = yvec[jj], yvec[jj + 1], zvec[kk], zvec[kk + 1]
        if sign > 0:
            quad = ((x, y0, z0), (x, y0, z1), (x, y1, z1), (x, y0, z0), (x, y1, z1), (x, y1, z0))
            normal = (127, 0, 0)
        else:
            quad = ((x, y0, z0), (x, y1, z0), (x, y1, z1), (x, y0, z0), (x, y1, z1), (x, y0, z1))
            normal = (-127, 0, 0)
    elif axis == "y":
        plane_j = jj + (1 if sign > 0 else 0)
        x0, x1, y, z0, z1 = xvec[ii], xvec[ii + 1], yvec[plane_j], zvec[kk], zvec[kk + 1]
        if sign > 0:
            quad = ((x0, y, z0), (x1, y, z0), (x1, y, z1), (x0, y, z0), (x1, y, z1), (x0, y, z1))
            normal = (0, 127, 0)
        else:
            quad = ((x0, y, z0), (x0, y, z1), (x1, y, z1), (x0, y, z0), (x1, y, z1), (x1, y, z0))
            normal = (0, -127, 0)
    else:  # axis == "z"
        plane_k = kk + (1 if sign > 0 else 0)
        x0, x1, y0, y1, z = xvec[ii], xvec[ii + 1], yvec[jj], yvec[jj + 1], zvec[plane_k]
        if sign > 0:
            quad = ((x0, y0, z), (x0, y1, z), (x1, y1, z), (x0, y0, z), (x1, y1, z), (x1, y0, z))
            normal = (0, 0, 127)
        else:
            quad = ((x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y0, z), (x1, y1, z), (x0, y1, z))
            normal = (0, 0, -127)

    for v_idx, vertex in enumerate(quad):
        positions[:, v_idx, 0] = vertex[0]
        positions[:, v_idx, 1] = vertex[1]
        positions[:, v_idx, 2] = vertex[2]

    vc = n * 6
    parts.append((
        positions.reshape(vc, 3),
        np.tile(np.array(normal, dtype=np.int8), (vc, 1)),
        np.repeat(cats, 6).astype(np.uint8, copy=False),
    ))


def _extract_grid_faces(xvec, yvec, zvec, point_categories, active):
    """
    Extract the visible boundary faces of a structured category volume.

    A face is emitted (once, coloured by the lower-index cell — matching the
    prototype) wherever an **active** cell meets a neighbour of a *different*
    category, or the grid edge. Inactive cells (``active=False``, e.g. a "no unit"
    / hidden category) never originate faces, so the mesh wraps only the visible
    region.

    Parameters
    ----------
    xvec, yvec, zvec : (nx,), (ny,), (nz,) float arrays
        World coordinate axes (z already scaled).
    point_categories : (nz, ny, nx) uint8
        Compact category index at each grid point (x fastest).
    active : (nz-1, ny-1, nx-1) bool
        Per-cell visibility mask.

    Returns
    -------
    positions (M,3) float32, normals (M,3) int8, categories (M,) uint8
    """
    cell_cat = point_categories[:-1, :-1, :-1]
    if cell_cat.size == 0:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.int8), np.empty(0, np.uint8))
    sentinel = int(cell_cat.min()) - 1
    comp = np.where(active, cell_cat.astype(np.int64), sentinel)  # inactive ≠ everything real
    parts = []

    def faces(axis, sign, mask):
        _append_grid_faces(parts, axis, sign, mask, cell_cat, xvec, yvec, zvec)

    # +axis faces own internal boundaries (category change) and the far edge.
    # -axis faces are emitted only against an inactive/edge neighbour (the +axis
    # face of the neighbouring cell already covers active|active boundaries).
    # ---- x ----
    m = np.zeros_like(active)
    m[:, :, :-1] = active[:, :, :-1] & (comp[:, :, :-1] != comp[:, :, 1:])
    m[:, :, -1] = active[:, :, -1]
    faces("x", +1, m)
    m = np.zeros_like(active)
    m[:, :, 0] = active[:, :, 0]
    m[:, :, 1:] = active[:, :, 1:] & ~active[:, :, :-1]
    faces("x", -1, m)
    # ---- y ----
    m = np.zeros_like(active)
    m[:, :-1, :] = active[:, :-1, :] & (comp[:, :-1, :] != comp[:, 1:, :])
    m[:, -1, :] = active[:, -1, :]
    faces("y", +1, m)
    m = np.zeros_like(active)
    m[:, 0, :] = active[:, 0, :]
    m[:, 1:, :] = active[:, 1:, :] & ~active[:, :-1, :]
    faces("y", -1, m)
    # ---- z ----
    m = np.zeros_like(active)
    m[:-1, :, :] = active[:-1, :, :] & (comp[:-1, :, :] != comp[1:, :, :])
    m[-1, :, :] = active[-1, :, :]
    faces("z", +1, m)
    m = np.zeros_like(active)
    m[0, :, :] = active[0, :, :]
    m[1:, :, :] = active[1:, :, :] & ~active[:-1, :, :]
    faces("z", -1, m)

    if not parts:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.int8), np.empty(0, np.uint8))
    return (
        np.vstack([p[0] for p in parts]),
        np.vstack([p[1] for p in parts]),
        np.concatenate([p[2] for p in parts]),
    )


# ===========================================================================
# curlew adapters: Grid → axes + reordered category volume
# ===========================================================================
def _grid_axes_world(grid, z_scale):
    """World-coordinate axes of an axis-aligned 3D :class:`curlew.geometry.Grid`."""
    if grid.ndim != 3:
        raise ValueError(f"The HTML viewer requires a 3D grid; got ndim={grid.ndim}.")
    R = grid.matrix[:3, :3]
    if not np.allclose(R, np.diag(np.diag(R))):
        raise ValueError(
            "The HTML viewer requires an axis-aligned grid (no rotation); "
            "rotated grids are not supported."
        )
    cx, cy, cz = (float(grid.center[i]) for i in range(3))
    xvec = (np.asarray(grid.axes[0], float) + cx).astype(np.float32)
    yvec = (np.asarray(grid.axes[1], float) + cy).astype(np.float32)
    zvec = ((np.asarray(grid.axes[2], float) + cz) * float(z_scale)).astype(np.float32)
    return xvec, yvec, zvec


def _category_volume(grid, grid_categories, mapper):
    """
    Reorder a flat ``(N,)`` category array (curlew C-order, z fastest) into the
    viewer's ``(nz, ny, nx)`` volume (x fastest when ravelled) and map ids to
    compact indices. Mirrors :func:`curlew.io.saveVTK`'s reorder.
    """
    shape = tuple(grid.shape)  # (nx, ny, nz)
    a = np.asarray(grid_categories)
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else a
    vol_ids = np.ascontiguousarray(a.reshape(shape).transpose(2, 1, 0))  # (nz, ny, nx)
    return mapper(vol_ids).astype(np.uint8, copy=False)


# ===========================================================================
# surface / point assembly (in-memory, replaces the prototype's .vtp reads)
# ===========================================================================
def _assemble_surfaces(surfaces, mapper, z_scale):
    """
    Flatten a list of ``(verts, faces, category)`` triangle meshes into one
    interleaved triangle-soup (positions, int8 normals, uint8 categories).
    """
    if not surfaces:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.int8), np.empty(0, np.uint8))
    pos_parts, cat_parts = [], []
    for verts, faces, category in surfaces:
        verts = np.asarray(verts, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int64)
        if len(verts) == 0 or len(faces) == 0:
            continue
        tri = verts[faces.reshape(-1)].astype(np.float32, copy=True)
        tri[:, 2] *= z_scale
        pos_parts.append(tri)
        cat_parts.append(np.full(tri.shape[0], int(category), dtype=np.int64))
    if not pos_parts:
        return (np.empty((0, 3), np.float32), np.empty((0, 3), np.int8), np.empty(0, np.uint8))
    positions = np.vstack(pos_parts)
    normals = _triangle_normals_i8(positions)
    categories = mapper(np.concatenate(cat_parts)).astype(np.uint8, copy=False)
    return positions, normals, categories


def _assemble_points(points, mapper, z_scale):
    """Flatten ``(coords, categories)`` into positions + uint8 categories."""
    if points is None:
        return np.empty((0, 3), np.float32), np.empty(0, np.uint8)
    coords, categories = points
    coords = np.asarray(coords, dtype=np.float32).copy()
    if len(coords) == 0:
        return np.empty((0, 3), np.float32), np.empty(0, np.uint8)
    coords[:, 2] *= z_scale
    return coords, mapper(np.asarray(categories)).astype(np.uint8, copy=False)


def _normalize_point_layers(points):
    """
    Normalise the ``points`` argument into a list of ``(name, coords, categories)``
    layers — each becomes its own GUI on/off toggle. Accepts:

    * ``None`` → ``[]``;
    * ``(coords, cats)`` → one layer named ``"Unit points"``;
    * ``{name: (coords, cats)}`` → one layer per dict entry (insertion order);
    * ``[(name, coords, cats), ...]`` → as given.
    """
    if points is None:
        return []
    if isinstance(points, dict):
        return [(str(n), np.asarray(c), np.asarray(k)) for n, (c, k) in points.items()]
    if (isinstance(points, (list, tuple)) and len(points) > 0
            and isinstance(points[0], (list, tuple)) and len(points[0]) == 3):
        return [(str(n), np.asarray(c), np.asarray(k)) for n, c, k in points]
    coords, cats = points
    return [("Unit points", np.asarray(coords), np.asarray(cats))]


def _assemble_vectors(vectors, z_scale, length):
    """
    Turn ``(origins (V,3), directions (V,3))`` into GL_LINES endpoints
    ``(2V, 3)`` — each vector drawn as ``origin → origin + unit_dir * length``.
    """
    if vectors is None:
        return np.empty((0, 3), np.float32)
    origins = np.asarray(vectors[0], dtype=float)
    directions = np.asarray(vectors[1], dtype=float)
    if len(origins) == 0:
        return np.empty((0, 3), np.float32)
    n = directions / (np.linalg.norm(directions, axis=1, keepdims=True) + 1e-12)
    seg = np.empty((len(origins) * 2, 3), dtype=np.float32)
    seg[0::2] = origins
    seg[1::2] = origins + n * length
    seg[:, 2] *= z_scale
    return seg


def _collect_category_values(grid_vol_ids, surfaces, point_layers):
    """Sorted unique original ids across all layers (drives the compact mapping)."""
    vals = set(np.unique(np.asarray(grid_vol_ids)).tolist()) if grid_vol_ids is not None else set()
    for verts, faces, category in (surfaces or []):
        if len(np.asarray(faces)) and len(np.asarray(verts)):
            vals.add(int(category))
    for _name, coords, cats in (point_layers or []):
        if len(np.asarray(coords)):
            vals.update(int(v) for v in np.unique(np.asarray(cats)).tolist())
    return sorted(int(v) for v in vals)


def _scene_bounds(xvec, yvec, zvec, *position_arrays):
    """Scene centre + half-extent (the largest XY half-range) for camera framing."""
    mins = [float(xvec.min()), float(yvec.min()), float(zvec.min())]
    maxs = [float(xvec.max()), float(yvec.max()), float(zvec.max())]
    for arr in position_arrays:
        if arr is not None and len(arr):
            mins = np.minimum(mins, arr.min(axis=0)).tolist()
            maxs = np.maximum(maxs, arr.max(axis=0)).tolist()
    center = [(mn + mx) * 0.5 for mn, mx in zip(mins, maxs)]
    scale = max(maxs[0] - mins[0], maxs[1] - mins[1]) * 0.5
    return center, (scale if scale > 0 else 1.0), mins, maxs


# ===========================================================================
# public API
# ===========================================================================
def write_html_viewer(
    output_path,
    *,
    grid,
    grid_categories,
    surfaces=None,
    points=None,
    vectors=None,
    vector_scale=None,
    vector_name="Normals",
    mode="self_contained",
    legend=None,
    hidden_categories=None,
    z_scale=1.0,
    title="Curlew Model Viewer",
    initial_z_exaggeration=25,
    max_embedded_mb=250.0,
    allow_large=False,
    grid_tile_cells=_GRID_TILE_CELLS,
    surface_tile_dims=_SURFACE_TILE_DIMS,
    point_tile_dims=_POINT_TILE_DIMS,
):
    """
    Write a WebGL HTML viewer for an in-memory curlew model.

    Parameters
    ----------
    output_path : str | os.PathLike
        Output ``.html`` path. In tiled mode a sibling ``<stem>_data/`` directory
        is written next to it.
    grid : curlew.geometry.Grid
        Axis-aligned 3D grid the model was evaluated on (``geode.grid``).
    grid_categories : array-like, shape (N,)
        Integer unit/level id per grid point, in the grid's evaluation order
        (e.g. ``geode.lithoID`` or a derived ``level`` volume).
    surfaces : list[tuple[verts, faces, category]] | None
        Triangle meshes in world coordinates (e.g. from
        :meth:`curlew.geometry.Grid.contour`). ``verts`` is ``(M,3)``, ``faces``
        is ``(K,3)`` integer indices, ``category`` is the int id colouring the mesh.
    points : tuple | dict | list | None
        Observation point cloud(s). Each becomes a **named layer with its own GUI
        on/off toggle**. Accepts a single ``(coords, cats)`` (named ``"Unit points"``),
        a ``{name: (coords, cats)}`` dict, or a ``[(name, coords, cats), ...]`` list,
        where ``coords`` is ``(P,3)`` world positions and ``cats`` is ``(P,)`` int ids.
        Point categories are still masked by the per-unit legend checkboxes.
    vectors : tuple | None
        Optional oriented vectors (e.g. bedding normals) as ``(origins, directions)``,
        each ``(V,3)``. Drawn as fixed-colour line segments with their own ``vector_name``
        toggle in the GUI (not affected by the per-unit mask).
    vector_scale : float | None
        World-unit length of the drawn vectors. Defaults to ~3% of the scene's XY extent.
    vector_name : str
        Label for the vectors' GUI toggle (default ``"Normals"``).
    mode : {"self_contained", "tiled"}
        Output mode (see module docstring).
    legend : dict[int, str] | None
        Maps category id → display name for the unit legend/checkboxes. Ids without
        an entry fall back to ``"Unit <id>"``.
    hidden_categories : Iterable[int] | None
        Category ids to default to *hidden* (unchecked) and to exclude from the grid
        volume mesh — e.g. a ``-1`` "outside model" id. They still appear (toggleable)
        in the legend and slicer.
    z_scale : float
        Multiplier applied to all Z coordinates at write time. Curlew models are
        usually not vertically exaggerated, so the default ``1.0`` is right; the
        viewer's interactive *Vertical exag.* slider handles exaggeration instead.
    title : str
        Viewer window/title text.
    initial_z_exaggeration : int
        Initial value of the vertical-exaggeration slider (1–100).
    max_embedded_mb : float
        Self-contained mode aborts if the embedded payload would exceed this
        (pre-base64) size, unless ``allow_large=True`` — a nudge toward tiled mode.
    allow_large : bool
        Bypass the ``max_embedded_mb`` guard.
    grid_tile_cells, surface_tile_dims, point_tile_dims :
        Tiled-mode tiling granularity.

    Returns
    -------
    pathlib.Path
        The written ``.html`` path (pass to :func:`serve_and_open`).
    """
    if mode not in ("self_contained", "tiled"):
        raise ValueError(f"mode must be 'self_contained' or 'tiled', got {mode!r}")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hidden_categories = set(int(v) for v in (hidden_categories or ()))

    # --- coordinate axes + reordered id volume (original ids, for category set) ---
    xvec, yvec, zvec = _grid_axes_world(grid, z_scale)
    shape = tuple(grid.shape)  # (nx, ny, nz)
    grid_ids = np.asarray(grid_categories)
    grid_ids = grid_ids.detach().cpu().numpy() if hasattr(grid_ids, "detach") else grid_ids
    grid_vol_ids = np.ascontiguousarray(grid_ids.reshape(shape).transpose(2, 1, 0))  # (nz,ny,nx)

    raw_point_layers = _normalize_point_layers(points)
    category_values = _collect_category_values(grid_vol_ids, surfaces, raw_point_layers)
    if not category_values:
        raise ValueError("No categories found in grid_categories / surfaces / points.")
    if len(category_values) > _MAX_CATEGORIES:
        raise ValueError(
            f"The viewer supports up to {_MAX_CATEGORIES} categories; got "
            f"{len(category_values)}. Merge/relabel units to reduce the count."
        )
    mapper = _make_mapper(category_values)

    # --- compact category volume + per-cell active mask (hidden ids excluded) ---
    grid_point_categories = mapper(grid_vol_ids).astype(np.uint8, copy=False)  # (nz,ny,nx)
    visible_id = np.array([v not in hidden_categories for v in category_values], dtype=bool)
    cell_corner = grid_point_categories[:-1, :-1, :-1]
    active = visible_id[cell_corner] if cell_corner.size else np.zeros((0, 0, 0), bool)

    grid_positions, grid_normals, grid_categories_arr = _extract_grid_faces(
        xvec, yvec, zvec, grid_point_categories, active
    )
    surface_positions, surface_normals, surface_categories = _assemble_surfaces(
        surfaces, mapper, z_scale
    )
    # one entry per non-empty named point layer: (name, positions, categories)
    point_layers = []
    for name, coords, cats in raw_point_layers:
        pos, pc = _assemble_points((coords, cats), mapper, z_scale)
        if len(pos):
            point_layers.append((name, pos, pc))

    vec_origins = np.asarray(vectors[0], float) if vectors is not None and len(vectors[0]) else None
    center, scale, mins, maxs = _scene_bounds(
        xvec, yvec, zvec, surface_positions, vec_origins, *[p for _, p, _ in point_layers]
    )
    vlen = float(vector_scale) if vector_scale else 0.03 * max(maxs[0] - mins[0], maxs[1] - mins[1])
    vector_positions = _assemble_vectors(vectors, z_scale, vlen)

    legend = legend or {}
    scene = {
        "title": title,
        "categoryValues": category_values,
        "legend": [legend.get(v, f"Unit {v}") for v in category_values],
        "hidden": [(not vis) for vis in visible_id.tolist()],
        "gridDims": [int(shape[0]), int(shape[1]), int(shape[2])],  # [nx, ny, nz]
        "sceneCenter": [float(c) for c in center],
        "sceneScale": float(scale),
        "bounds": {"min": [float(v) for v in mins], "max": [float(v) for v in maxs]},
        "zScale": float(z_scale),
        "initialZExag": int(initial_z_exaggeration),
        "gridTriangleCount": int(grid_positions.shape[0] // 3),
        "surfaceTriangleCount": int(surface_positions.shape[0] // 3),
        "pointGroups": [n for n, _, _ in point_layers],
        "pointCount": int(sum(p.shape[0] for _, p, _ in point_layers)),
        "hasVectors": bool(vector_positions.shape[0]),
        "vectorName": vector_name,
        "vectorCount": int(vector_positions.shape[0] // 2),
    }

    if mode == "self_contained":
        # stale tiled data dir would confuse serve_and_open() — remove it.
        _safe_rmtree(output_path.with_name(output_path.stem + "_data"))
        _write_self_contained(
            output_path, scene, grid_positions, grid_normals, grid_categories_arr,
            surface_positions, surface_normals, surface_categories,
            point_layers, vector_positions, grid_point_categories,
            xvec, yvec, zvec, max_embedded_mb, allow_large,
        )
    else:
        _write_tiled(
            output_path, scene, grid_positions, grid_normals, grid_categories_arr,
            surface_positions, surface_normals, surface_categories,
            point_layers, vector_positions, grid_point_categories,
            xvec, yvec, zvec, grid_tile_cells, surface_tile_dims, point_tile_dims,
        )
    return output_path


def _as_numpy(a):
    return a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)


def extract_geomodel_layers(model, geode, grid, *, color_by="lithoID",
                            erode_mask=-2, unconformity_id=-2):
    """
    Build the grid colouring, legend and boundary surfaces for a fitted curlew
    model + its grid prediction — the per-notebook boilerplate, in one place.

    Returns a ``dict`` with ``grid_categories``, ``surfaces``, ``legend`` and
    ``hidden_categories``, ready to splat into :func:`write_html_viewer` (you still
    supply ``points`` and ``mode``). :func:`write_geomodel_viewer` does exactly that.

    The boundary surfaces are extracted with :meth:`curlew.geometry.Grid.contour`
    for every generative event, each coloured by the unit/level it bounds. (Surface
    colouring requires reconstructing the **event-prefixed** key
    ``f"{event.name}_{iso}"`` that :class:`curlew.geology.geoevent.GeoEvent` stores
    in ``geode.lithoLookup`` — using the bare iso name colours every surface alike.)

    Parameters
    ----------
    model : curlew.geology.geomodel.GeoModel
        The fitted model (provides ``events``).
    geode : curlew.core.Geode
        ``model.predict(grid)`` output (provides ``lithoID``/``structureID``/
        ``fields``/``lithoLookup``).
    grid : curlew.geometry.Grid
        The grid ``geode`` was evaluated on (used for ``contour``).
    color_by : {"lithoID", "level"}
        * ``"lithoID"`` — colour by curlew's lithology id (``geode.lithoID`` /
          ``geode.lithoLookup``). Works for any model.
        * ``"level"`` — colour by stratigraphic *level* (the GeoINR data convention;
          ``-1`` = no unit). Requires a strat-column model from
          :func:`curlew.geology.stratbuilder.build_geomodel` (``llookup`` /
          ``level_litho`` / ``strat_units``).
    erode_mask : int
        ``erodeMask`` passed to ``Grid.contour`` (dilation/erosion of the per-event
        region mask before contouring).
    unconformity_id : int
        Category assigned to surfaces with no lithology (erosional unconformities),
        labelled ``"(unconformity surface)"`` in the legend.
    """
    lithoID = _as_numpy(geode.lithoID)
    name2litho = {n: i for i, n in geode.lithoLookup.items()}  # event-prefixed keys

    if color_by == "lithoID":
        grid_categories = lithoID
        legend = dict(geode.lithoLookup)
        legend.setdefault(unconformity_id, "(unconformity surface)")
        fallback = int(lithoID.max()) if lithoID.size else 0
        def cat_of(e, iso):
            return name2litho.get(f"{e.name}_{iso}", fallback)
    elif color_by == "level":
        for attr in ("llookup", "level_litho", "strat_units"):
            if not hasattr(model, attr):
                raise ValueError(
                    "color_by='level' needs a strat-column GeoModel (build_geomodel) "
                    f"with .{attr}; use color_by='lithoID' for a generic model."
                )
        lut = np.full(max(model.llookup.values()) + 1, -1, dtype=np.int64)
        for L, key in model.level_litho.items():
            lut[model.llookup[key]] = L
        grid_categories = lut[lithoID]
        legend = {u.level: u.name for u in model.strat_units}
        legend[-1] = "(no unit)"
        legend[unconformity_id] = "(unconformity surface)"
        lithoID2level = {model.llookup[key]: L for L, key in model.level_litho.items()}
        def cat_of(e, iso):
            li = name2litho.get(f"{e.name}_{iso}")
            return lithoID2level.get(li, unconformity_id) if li is not None else unconformity_id
    else:
        raise ValueError("color_by must be 'lithoID' or 'level'")

    sid = _as_numpy(geode.structureID)
    surfaces = []
    for e in model.events:
        if not getattr(e, "isosurfaces", None):
            continue
        mask = sid == e.eid
        if not mask.any():
            continue
        for iso_name, iso_val in e.getIsovalues().items():
            try:
                verts, faces = grid.contour(geode.fields[e.name], iso=iso_val,
                                            mask=mask, erodeMask=erode_mask)
            except Exception:  # iso outside the grid's value range etc. — skip
                continue
            if len(verts):
                surfaces.append((verts, faces, cat_of(e, iso_name)))

    return {"grid_categories": grid_categories, "surfaces": surfaces,
            "legend": legend, "hidden_categories": {-1}}


def _level_to_category_fn(model, geode, color_by, name2litho):
    """Callable: observation stratigraphic level → viewer category, matching ``color_by``
    and the grid colouring (so observation points sit in same-coloured volume)."""
    if color_by == "level":
        return lambda L: int(L)
    # color_by == "lithoID": build level → lithoID from the model's own maps, falling back
    # to the depositional contacts (the on-surface path has an empty level_litho).
    m = {}
    if getattr(model, "level_litho", None) and getattr(model, "llookup", None):
        for L, key in model.level_litho.items():
            if key in model.llookup:
                m[int(L)] = int(model.llookup[key])
    for ev, meta in zip(model.events, getattr(model, "field_meta", []) or []):
        for c in (getattr(meta, "contacts", None) or []):
            lid = name2litho.get(f"{ev.name}_{c.name}")
            if lid is not None:
                m[int(c.level)] = int(lid)
    lithoID = np.asarray(geode.lithoID)
    fallback = int(lithoID.max()) if lithoID.size else 0
    return lambda L: m.get(int(L), fallback)


def observation_layers(obs, model, geode, *, color_by="lithoID", normals=True,
                       max_points=200000, seed=0):
    """
    Turn a :class:`curlew.io.Observations` object into viewer inputs:

    * named **point layers** — ``"Interface points"`` (on-contact points) and/or
      ``"Unit points"`` (interior unit points), each coloured by the unit/level it
      belongs to (consistent with ``color_by``) and subsampled to ``max_points``;
    * a **normals** vector tuple ``(origins, directions)`` if the observations carry
      bedding normals (else ``None``).

    Returns ``(points, vectors)`` — pass straight to :func:`write_html_viewer` /
    :func:`write_geomodel_viewer`.
    """
    name2litho = {n: i for i, n in geode.lithoLookup.items()}
    to_cat = _level_to_category_fn(model, geode, color_by, name2litho)
    rng = np.random.default_rng(seed)

    def layer(coords, levels):
        coords = np.asarray(coords, float)
        levels = np.asarray(levels)
        if len(coords) > max_points:
            idx = rng.choice(len(coords), size=max_points, replace=False)
            coords, levels = coords[idx], levels[idx]
        return coords, np.array([to_cat(L) for L in levels], dtype=np.int64)

    points = {}
    ic, il = obs.interface_points()
    if len(ic):
        points["Interface points"] = layer(ic, il)
    uc, ul = obs.unit_points()
    if len(uc):
        points["Unit points"] = layer(uc, ul)

    vectors = None
    if normals:
        nco, nvec, _ = obs.normal_points()
        if len(nco):
            vectors = (np.asarray(nco, float), np.asarray(nvec, float))
    return points, vectors


def write_geomodel_viewer(output_path, model, geode, grid, *, obs=None, points=None,
                          vectors=None, color_by="lithoID", mode="self_contained",
                          erode_mask=-2, normals=True, max_points=200000,
                          legend=None, hidden_categories=None, **kwargs):
    """
    One-call viewer for a fitted curlew model: derives the grid colouring, legend
    and boundary surfaces via :func:`extract_geomodel_layers`; optionally derives the
    observation point layers + bedding-normal vectors from ``obs`` via
    :func:`observation_layers`; then writes the viewer via :func:`write_html_viewer`.

    Pass ``obs`` (a :class:`curlew.io.Observations`) to get interface/unit points and
    normals automatically, or pass ``points`` / ``vectors`` explicitly (these take
    precedence). Presentation kwargs (``title``, ``initial_z_exaggeration``,
    ``vector_scale``, ``z_scale``, …) and ``legend`` / ``hidden_categories`` overrides
    are forwarded.

    Example
    -------
    >>> write_geomodel_viewer("outputs/model.html", M, geode, G,
    ...     obs=obs, color_by="level", mode="tiled", initial_z_exaggeration=100)
    """
    layers = extract_geomodel_layers(model, geode, grid, color_by=color_by, erode_mask=erode_mask)
    if obs is not None:
        obs_points, obs_vectors = observation_layers(
            obs, model, geode, color_by=color_by, normals=normals, max_points=max_points)
        if points is None:
            points = obs_points
        if vectors is None:
            vectors = obs_vectors
    if legend is not None:
        layers["legend"] = legend
    if hidden_categories is not None:
        layers["hidden_categories"] = hidden_categories
    return write_html_viewer(output_path, grid=grid, points=points, vectors=vectors,
                             mode=mode, **layers, **kwargs)


def serve_and_open(html_path, *, open_browser=True, port=0, bind="127.0.0.1"):
    """
    Open an HTML viewer in the default browser.

    * **Self-contained** viewers open directly via a ``file://`` URL and this
      returns ``None``.
    * **Tiled** viewers (detected by the sibling ``<stem>_data/`` directory) need
      HTTP: a background daemon :class:`http.server.ThreadingHTTPServer` is started
      on a free port, rooted at the HTML's parent directory, and the browser is
      pointed at it. The server object is **returned** — keep a reference so it
      stays alive, and call ``server.shutdown()`` to stop it.

    Parameters
    ----------
    html_path : str | os.PathLike
        Path returned by :func:`write_html_viewer`.
    open_browser : bool
        Open the system browser (set ``False`` for headless runs).
    port : int
        TCP port for tiled mode (``0`` = pick a free one).
    bind : str
        Bind address for tiled mode (loopback only by default — nothing is uploaded).
    """
    html_path = Path(html_path)
    data_dir = html_path.with_name(html_path.stem + "_data")

    if not data_dir.exists():  # self-contained
        if open_browser:
            webbrowser.open(html_path.resolve().as_uri())
        return None

    import http.server  # stdlib, lazily imported
    import threading

    root = str(html_path.parent.resolve())
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=root)
    httpd = http.server.ThreadingHTTPServer((bind, port), handler)
    actual_port = httpd.server_address[1]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"http://{bind}:{actual_port}/{html_path.name}"
    print(f"Serving {root}\n  -> {url}\n  (call .shutdown() on the returned server to stop it)")
    if open_browser:
        webbrowser.open(url)
    return httpd


# ===========================================================================
# writers
# ===========================================================================
def _payload_mb(*arrays) -> float:
    """Pre-base64 size of the embedded payload, in MB."""
    return sum(a.nbytes for a in arrays) / 1024.0 / 1024.0


def _write_self_contained(
    out, scene, gpos, gnorm, gcat, spos, snorm, scat, point_layers, vpos, gridpc,
    xvec, yvec, zvec, max_embedded_mb, allow_large,
):
    point_arrays = [a for _, p, c in point_layers for a in (p, c)]
    arrays = (gpos, gnorm, gcat, spos, snorm, scat, gridpc, xvec, yvec, zvec, vpos, *point_arrays)
    mb = _payload_mb(*arrays)
    if mb > max_embedded_mb and not allow_large:
        raise RuntimeError(
            f"Embedded payload is ~{mb:.1f} MB (> max_embedded_mb={max_embedded_mb:.1f}). "
            "Use mode='tiled', raise max_embedded_mb, or pass allow_large=True."
        )
    # base64 key:value parts — fixed grid/surface arrays + one (pos, cat) pair per point layer
    parts = [
        f"gridPositions:'{_b64(gpos.astype('<f4', copy=False))}'",
        f"gridNormals:'{_b64(gnorm.astype(np.int8, copy=False))}'",
        f"gridCategories:'{_b64(gcat.astype(np.uint8, copy=False))}'",
        f"surfacePositions:'{_b64(spos.astype('<f4', copy=False))}'",
        f"surfaceNormals:'{_b64(snorm.astype(np.int8, copy=False))}'",
        f"surfaceCategories:'{_b64(scat.astype(np.uint8, copy=False))}'",
        f"gridPointCategories:'{_b64(gridpc.astype(np.uint8, copy=False).ravel(order='C'))}'",
        f"xvec:'{_b64(xvec.astype('<f4', copy=False))}'",
        f"yvec:'{_b64(yvec.astype('<f4', copy=False))}'",
        f"zvec:'{_b64(zvec.astype('<f4', copy=False))}'",
        f"vectorPositions:'{_b64(vpos.astype('<f4', copy=False))}'",
    ]
    for i, (_name, pos, cats) in enumerate(point_layers):
        parts.append(f"pointPos{i}:'{_b64(pos.astype('<f4', copy=False))}'")
        parts.append(f"pointCat{i}:'{_b64(cats.astype(np.uint8, copy=False))}'")
    bootstrap = (
        "const META=" + json.dumps(scene, separators=(",", ":")) + ";\n"
        "const B64={" + ",".join(parts) + "};\n"
        "function dec(b){const s=atob(b),n=s.length,a=new Uint8Array(n),C=32768;"
        "for(let i=0;i<n;i+=C){const e=Math.min(i+C,n);for(let j=i;j<e;j++)a[j]=s.charCodeAt(j)}return a.buffer}\n"
        "setScene(META);\n"
        "if(initGL()){\n"
        " const gp=new Float32Array(dec(B64.gridPositions)),gn=new Int8Array(dec(B64.gridNormals)),gc=new Uint8Array(dec(B64.gridCategories));\n"
        " if(gp.length)addLayer('grid',createMeshVao(gp,gn,gc));\n"
        " const sp=new Float32Array(dec(B64.surfacePositions)),sn=new Int8Array(dec(B64.surfaceNormals)),sc=new Uint8Array(dec(B64.surfaceCategories));\n"
        " if(sp.length)addLayer('surfaces',createMeshVao(sp,sn,sc));\n"
        " for(let g=0;g<META.pointGroups.length;g++){const pp=new Float32Array(dec(B64['pointPos'+g])),pc=new Uint8Array(dec(B64['pointCat'+g]));if(pp.length){const d=createPointVao(pp,pc);d.group=g;addLayer('points',d)}}\n"
        " if(META.hasVectors){const vp=new Float32Array(dec(B64.vectorPositions));if(vp.length)addLayer('vectors',createLineVao(vp))}\n"
        " setGridData({xvec:new Float32Array(dec(B64.xvec)),yvec:new Float32Array(dec(B64.yvec)),zvec:new Float32Array(dec(B64.zvec)),categories:new Uint8Array(dec(B64.gridPointCategories)),dims:META.gridDims});\n"
        " startCommon();\n"
        "}\n"
    )
    html = _TEMPLATE.replace("__TITLE__", _escape(scene["title"])).replace("__BADGE__", "Self-contained")
    html = html.replace("__BOOTSTRAP__", bootstrap)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({out.stat().st_size / 1024 / 1024:.2f} MB, ~{mb:.1f} MB geometry)")


def _write_mesh_bin(path, positions, normals, categories):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(np.ascontiguousarray(positions.astype("<f4", copy=False)).tobytes())
        fh.write(np.ascontiguousarray(normals.astype(np.int8, copy=False)).tobytes())
        fh.write(np.ascontiguousarray(categories.astype(np.uint8, copy=False)).tobytes())
    return {"file": path.name, "vertexCount": int(positions.shape[0])}


def _write_point_bin(path, positions, categories):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(np.ascontiguousarray(positions.astype("<f4", copy=False)).tobytes())
        fh.write(np.ascontiguousarray(categories.astype(np.uint8, copy=False)).tobytes())
    return {"file": path.name, "pointCount": int(positions.shape[0])}


def _tile_mesh(out_dir, prefix, positions, normals, categories, xy_bounds, dims):
    """Split a triangle soup into XY tiles by triangle centroid; write .bin each."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if positions.shape[0] == 0:
        return []
    tris = positions.reshape(-1, 3, 3)
    centroids = tris[:, :, :2].mean(axis=1)
    tile_ids = _tile_ids_from_xy(centroids, xy_bounds, dims)
    tiles = []
    for tid, idx in _grouped_tile_indices(tile_ids):
        vi = (idx[:, None] * 3 + np.arange(3)).reshape(-1)  # vertices of selected triangles
        rel = f"{prefix}/tile_{tid:03d}.bin"
        meta = _write_mesh_bin(out_dir / f"tile_{tid:03d}.bin", positions[vi], normals[vi], categories[vi])
        meta["file"] = rel
        tiles.append(meta)
    return tiles


def _tile_points(out_dir, positions, categories, xy_bounds, dims, prefix):
    out_dir.mkdir(parents=True, exist_ok=True)
    if positions.shape[0] == 0:
        return []
    tile_ids = _tile_ids_from_xy(positions[:, :2], xy_bounds, dims)
    tiles = []
    for tid, idx in _grouped_tile_indices(tile_ids):
        meta = _write_point_bin(out_dir / f"tile_{tid:03d}.bin", positions[idx], categories[idx])
        meta["file"] = f"{prefix}/tile_{tid:03d}.bin"
        tiles.append(meta)
    return tiles


def _write_tiled(
    out, scene, gpos, gnorm, gcat, spos, snorm, scat, point_layers, vpos, gridpc,
    xvec, yvec, zvec, grid_tile_cells, surface_tile_dims, point_tile_dims,
):
    data_dir = out.with_name(out.stem + "_data")
    _safe_rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    xy_bounds = (float(xvec.min()), float(xvec.max()), float(yvec.min()), float(yvec.max()))

    # grid slice assets (full-resolution category volume + axes)
    grid_dir = data_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    (grid_dir / "xvec.f32").write_bytes(np.ascontiguousarray(xvec.astype("<f4")).tobytes())
    (grid_dir / "yvec.f32").write_bytes(np.ascontiguousarray(yvec.astype("<f4")).tobytes())
    (grid_dir / "zvec.f32").write_bytes(np.ascontiguousarray(zvec.astype("<f4")).tobytes())
    (grid_dir / "categories.u8").write_bytes(
        np.ascontiguousarray(gridpc.astype(np.uint8).ravel(order="C")).tobytes()
    )

    # tile grid-mesh by XY (centroid), with cell-sized tiles ≈ grid_tile_cells
    nx, ny = scene["gridDims"][0], scene["gridDims"][1]
    grid_dims = (max(1, nx // grid_tile_cells), max(1, ny // grid_tile_cells))
    grid_tiles = _tile_mesh(data_dir / "grid_mesh", "grid_mesh", gpos, gnorm, gcat, xy_bounds, grid_dims)
    surface_tiles = _tile_mesh(data_dir / "surfaces", "surfaces", spos, snorm, scat, xy_bounds, surface_tile_dims)
    point_groups = []
    for g, (name, pos, cats) in enumerate(point_layers):
        tiles = _tile_points(data_dir / "points" / f"group{g}", pos, cats,
                             xy_bounds, point_tile_dims, f"points/group{g}")
        point_groups.append({"name": name, "tiles": tiles})

    manifest = dict(scene)
    manifest["grid"] = {
        "dims": scene["gridDims"],
        "xvec": "grid/xvec.f32", "yvec": "grid/yvec.f32",
        "zvec": "grid/zvec.f32", "categories": "grid/categories.u8",
    }
    manifest["gridMesh"] = {"tiles": grid_tiles}
    manifest["surfaces"] = {"tiles": surface_tiles}
    manifest["points"] = {"groups": point_groups}
    if vpos.shape[0]:  # vectors are few — a single binary, no tiling
        (data_dir / "vectors").mkdir(parents=True, exist_ok=True)
        (data_dir / "vectors" / "lines.bin").write_bytes(
            np.ascontiguousarray(vpos.astype("<f4", copy=False)).tobytes())
        manifest["vectors"] = {"file": "vectors/lines.bin", "count": int(vpos.shape[0])}
    (data_dir / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")

    bootstrap = (
        f"const DATA_ROOT='{data_dir.name}/';\n"
        "async function fj(p){const r=await fetch(p);if(!r.ok)throw new Error(p+': '+r.status);return r.json()}\n"
        "async function fb(p){const r=await fetch(p);if(!r.ok)throw new Error(p+': '+r.status);return r.arrayBuffer()}\n"
        "function meshFromBuffer(b,c){const pos=new Float32Array(b,0,c*3),norm=new Int8Array(b,c*12,c*3),cat=new Uint8Array(b,c*12+c*3,c);return createMeshVao(pos,norm,cat)}\n"
        "function pointFromBuffer(b,c){const pos=new Float32Array(b,0,c*3),cat=new Uint8Array(b,c*12,c);return createPointVao(pos,cat)}\n"
        "async function boot(){let m;try{m=await fj(DATA_ROOT+'manifest.json')}catch(e){showError('Tiled viewers must be opened over a local HTTP server, not file://. Use curlew serve_and_open(). Details: '+e.message);return}\n"
        " setScene(m);\n"
        " if(!initGL()){showError('WebGL2 is unavailable; try current Chrome, Edge or Firefox.');return}\n"
        " const [xb,yb,zb,cb]=await Promise.all([fb(DATA_ROOT+m.grid.xvec),fb(DATA_ROOT+m.grid.yvec),fb(DATA_ROOT+m.grid.zvec),fb(DATA_ROOT+m.grid.categories)]);\n"
        " setGridData({xvec:new Float32Array(xb),yvec:new Float32Array(yb),zvec:new Float32Array(zb),categories:new Uint8Array(cb),dims:m.grid.dims});\n"
        " startCommon();\n"
        " for(const t of m.gridMesh.tiles){try{addLayer('grid',meshFromBuffer(await fb(DATA_ROOT+t.file),t.vertexCount));requestDraw()}catch(e){showError('Tile load failed: '+e.message)}}\n"
        " for(const t of m.surfaces.tiles){try{addLayer('surfaces',meshFromBuffer(await fb(DATA_ROOT+t.file),t.vertexCount));requestDraw()}catch(e){showError('Tile load failed: '+e.message)}}\n"
        " for(let g=0;g<m.points.groups.length;g++){for(const t of m.points.groups[g].tiles){try{const d=pointFromBuffer(await fb(DATA_ROOT+t.file),t.pointCount);d.group=g;addLayer('points',d);requestDraw()}catch(e){showError('Tile load failed: '+e.message)}}}\n"
        " if(m.vectors){try{addLayer('vectors',createLineVao(new Float32Array(await fb(DATA_ROOT+m.vectors.file))));requestDraw()}catch(e){showError('Vector load failed: '+e.message)}}\n"
        "}\n"
        "boot();\n"
    )
    html = _TEMPLATE.replace("__TITLE__", _escape(scene["title"])).replace("__BADGE__", "Tiled")
    html = html.replace("__BOOTSTRAP__", bootstrap)
    out.write_text(html, encoding="utf-8")
    data_mb = sum(p.stat().st_size for p in data_dir.rglob("*") if p.is_file()) / 1024 / 1024
    print(f"Wrote {out}  (+ {data_dir.name}/  {data_mb:.2f} MB)")


# ===========================================================================
# misc
# ===========================================================================
def _safe_rmtree(path: Path):
    """Remove a directory only if it looks like one of our generated data dirs."""
    if path.exists() and path.is_dir() and path.name.endswith("_data"):
        shutil.rmtree(path)


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ===========================================================================
# shared HTML / CSS / WebGL template
# ---------------------------------------------------------------------------
# UI, shaders, camera, palette and slicer are identical for both modes; only the
# injected __BOOTSTRAP__ block differs (decode base64 vs fetch tiles). The common
# JS operates on a generic `layers` registry + `gridData` + `SCENE` so each
# bootstrap just populates those and calls startCommon().
# ===========================================================================
_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>__TITLE__</title>
<style>
:root{color-scheme:dark;--panel:rgba(18,23,29,.93);--line:rgba(255,255,255,.14);--text:#edf2f7;--muted:#aeb9c4;--accent:#4fb7c5;--bg:#101419}
*{box-sizing:border-box}html,body{width:100%;height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--text);font:13px/1.35 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
#view{position:fixed;inset:0;width:100vw;height:100vh;display:block;background:radial-gradient(circle at 50% 42%,#27313a 0%,#151b21 48%,#0c1014 100%);cursor:crosshair}
#ui{position:fixed;left:14px;top:14px;width:min(376px,calc(100vw - 28px));max-height:calc(100vh - 28px);overflow:auto;background:var(--panel);border:1px solid var(--line);border-radius:8px;box-shadow:0 18px 45px rgba(0,0,0,.35);backdrop-filter:blur(8px)}
header{padding:12px 14px 10px;border-bottom:1px solid var(--line)}h1{margin:0 0 4px;font-size:15px;font-weight:650}.meta{color:var(--muted);font-size:12px}
.section{padding:12px 14px;border-bottom:1px solid var(--line)}.section:last-child{border-bottom:0}.row{display:grid;grid-template-columns:132px 1fr auto;gap:10px;align-items:center;margin:9px 0;min-height:28px}.row.two{grid-template-columns:132px 1fr}
label{color:var(--muted);font-size:12px}input[type="range"]{width:100%;accent-color:var(--accent)}input[type="checkbox"]{accent-color:var(--accent)}select,button{width:100%;color:var(--text);background:#202933;border:1px solid var(--line);border-radius:6px;min-height:30px;padding:4px 8px;font:inherit}button{cursor:pointer}button:hover,select:hover{border-color:rgba(79,183,197,.62)}
.value{color:var(--text);font-variant-numeric:tabular-nums;min-width:62px;text-align:right}.toggleLine{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px;margin:8px 0}.toggleLine label{display:flex;align-items:center;gap:8px;color:var(--text)}
#legend{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px 12px;max-height:200px;overflow:auto;padding-right:4px}.legendItem{display:grid;grid-template-columns:18px 18px 1fr;align-items:center;gap:7px;min-width:0;color:var(--muted);cursor:pointer}.legendItem input{margin:0}.swatch{width:16px;height:16px;border-radius:4px;border:1px solid rgba(255,255,255,.35)}
.unitHeader{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}.unitActions{display:flex;gap:6px}.unitActions button{width:auto;min-height:26px;padding:3px 9px;font-size:12px}
.pill{display:inline-block;color:#d7eef2;background:rgba(79,183,197,.16);border:1px solid rgba(79,183,197,.25);border-radius:999px;padding:1px 7px;margin-right:5px;font-size:11px}.small{color:var(--muted);font-size:12px}
#error{position:fixed;inset:auto 18px 18px 18px;padding:12px 14px;background:#3a1f1f;color:#ffe8e8;border:1px solid #8d4d4d;border-radius:8px;display:none}
@media(max-width:720px){#ui{left:8px;top:8px;width:calc(100vw - 16px)}#legend{grid-template-columns:repeat(3,minmax(0,1fr));max-height:96px}.toggleLine{grid-template-columns:1fr}}
</style>
</head>
<body>
<canvas id="view"></canvas>
<div id="ui">
<header><h1>__TITLE__</h1><div class="meta"><span class="pill">__BADGE__</span><span id="counts"></span></div></header>
<div class="section">
<div class="toggleLine">
<label><input id="showGrid" type="checkbox" checked> Grid volume</label>
<label><input id="showSurfaces" type="checkbox" checked> Boundary surfaces</label>
<label><input id="showSlice" type="checkbox" checked> Slice</label>
</div>
<div id="pointToggles" class="toggleLine"></div>
<div class="row two"><label for="palette">Palette</label><select id="palette"><option value="turbo" selected>Turbo</option><option value="viridis">Viridis</option><option value="strata">Stratigraphy</option><option value="colorblind">Colorblind safe</option><option value="contrast">High contrast</option><option value="gray">Gray</option></select></div>
<div class="row"><label for="zExag">Vertical exag.</label><input id="zExag" type="range" min="1" max="100" step="1" value="25"><span id="zExagValue" class="value">25x</span></div>
<div class="row"><label for="gridOpacity">Grid opacity</label><input id="gridOpacity" type="range" min="0" max="100" step="1" value="60"><span id="gridOpacityValue" class="value">0.60</span></div>
<div class="row"><label for="surfaceOpacity">Surface opacity</label><input id="surfaceOpacity" type="range" min="0" max="100" step="1" value="85"><span id="surfaceOpacityValue" class="value">0.85</span></div>
<div class="row"><label for="pointSize">Point size</label><input id="pointSize" type="range" min="1" max="18" step="1" value="6"><span id="pointSizeValue" class="value">6</span></div>
<div class="row"><label for="pointOpacity">Point opacity</label><input id="pointOpacity" type="range" min="0" max="100" step="1" value="90"><span id="pointOpacityValue" class="value">0.90</span></div>
</div>
<div class="section">
<div class="row two"><label for="sliceAxis">Slice axis</label><select id="sliceAxis"><option value="x">X</option><option value="y" selected>Y</option><option value="z">Z</option></select></div>
<div class="row"><label for="sliceIndex">Slice position</label><input id="sliceIndex" type="range" min="0" max="1" step="1" value="0"><span id="sliceIndexValue" class="value">0</span></div>
<div class="row"><label for="sliceOpacity">Slice opacity</label><input id="sliceOpacity" type="range" min="0" max="100" step="1" value="96"><span id="sliceOpacityValue" class="value">0.96</span></div>
<div class="toggleLine"><label><input id="mouseSlice" type="checkbox"> Move slice with mouse cursor</label></div>
<button id="resetCamera">Reset Camera</button>
</div>
<div class="section"><div id="status" class="small"></div></div>
<div class="section"><div class="unitHeader"><span class="small">Unit visibility</span><div class="unitActions"><button id="unitsAll">All</button><button id="unitsNone">None</button></div></div><div id="legend"></div></div>
</div>
<div id="error"></div>
<script>
'use strict';
let SCENE=null,gridData=null;
const layers={grid:[],surfaces:[],points:[],vectors:[]};
let gl=null,meshProgram=null,pointProgram=null,lineProgram=null,sliceVao=null,sliceBuffers=null,sliceVertexCount=0;
let frameCount=0,renderPending=false,currentColors=[],paletteFlat=new Float32Array(64*3),unitMaskFlat=new Float32Array(64),unitVisible=[];
let vectorsVisible=true;  // "Normals" vector layer visibility (toggle appears only when present)
const canvas=document.getElementById('view');
function gid(id){return document.getElementById(id)}
function showError(m){const e=gid('error');e.textContent=m;e.style.display='block'}
const controls={showGrid:gid('showGrid'),showSurfaces:gid('showSurfaces'),showSlice:gid('showSlice'),palette:gid('palette'),zExag:gid('zExag'),gridOpacity:gid('gridOpacity'),surfaceOpacity:gid('surfaceOpacity'),pointSize:gid('pointSize'),pointOpacity:gid('pointOpacity'),sliceAxis:gid('sliceAxis'),sliceIndex:gid('sliceIndex'),sliceOpacity:gid('sliceOpacity'),mouseSlice:gid('mouseSlice'),resetCamera:gid('resetCamera'),unitsAll:gid('unitsAll'),unitsNone:gid('unitsNone')};
let pointVisible=[];  // per point-group visibility, driven by the dynamic #pointToggles checkboxes
function buildLayerToggles(){const box=gid('pointToggles');box.textContent='';const groups=(SCENE&&SCENE.pointGroups)||[];if(pointVisible.length!==groups.length)pointVisible=groups.map(()=>true);const mk=(name,checked,onchange)=>{const lab=document.createElement('label');const cb=document.createElement('input');cb.type='checkbox';cb.checked=checked;cb.addEventListener('change',()=>{onchange(cb.checked);requestDraw()});const sp=document.createElement('span');sp.textContent=name;sp.title=name;lab.append(cb,sp);box.appendChild(lab)};groups.forEach((name,g)=>mk(name,pointVisible[g],v=>{pointVisible[g]=v}));if(SCENE&&SCENE.hasVectors)mk(SCENE.vectorName||'Normals',vectorsVisible,v=>{vectorsVisible=v})}
const labels={zExag:gid('zExagValue'),gridOpacity:gid('gridOpacityValue'),surfaceOpacity:gid('surfaceOpacityValue'),pointSize:gid('pointSizeValue'),pointOpacity:gid('pointOpacityValue'),sliceIndex:gid('sliceIndexValue'),sliceOpacity:gid('sliceOpacityValue')};
// ---- palette ----
function hexToRgb(hex){const m=hex.replace('#','');return[parseInt(m.slice(0,2),16)/255,parseInt(m.slice(2,4),16)/255,parseInt(m.slice(4,6),16)/255]}
function lerp(a,b,t){return a+(b-a)*t}
function stops(st,t){if(t<=0)return st[0];if(t>=1)return st[st.length-1];const p=t*(st.length-1),i=Math.floor(p),f=p-i,a=st[i],b=st[i+1];return[lerp(a[0],b[0],f),lerp(a[1],b[1],f),lerp(a[2],b[2],f)]}
function hsl(h,s,l){const c=(1-Math.abs(2*l-1))*s,hp=h/60,x=c*(1-Math.abs((hp%2)-1));let r=0,g=0,b=0;if(hp<1)[r,g,b]=[c,x,0];else if(hp<2)[r,g,b]=[x,c,0];else if(hp<3)[r,g,b]=[0,c,x];else if(hp<4)[r,g,b]=[0,x,c];else if(hp<5)[r,g,b]=[x,0,c];else[r,g,b]=[c,0,x];const m=l-c/2;return[r+m,g+m,b+m]}
function palette(name,n){const base={strata:['#1f6f8b','#67a9cf','#3f8f6a','#8bc34a','#d6c45d','#e07a2f','#b95d2a','#8e5ea2','#c48a9d','#9b4f19','#5a78b0','#9dbf6f','#d85c5c','#6e9d9a','#c08a3d','#7f6aa3'],colorblind:['#0072B2','#E69F00','#009E73','#D55E00','#CC79A7','#56B4E9','#F0E442','#000000'],gray:['#f2f2f2','#d9d9d9','#bdbdbd','#969696','#737373','#525252','#252525']},vir=['#440154','#482777','#3f4a8a','#31678e','#26838f','#1f9d8a','#6cce5a','#b6de2b','#fee825'].map(hexToRgb),turbo=['#30123b','#4662d8','#36a3ec','#1ae4b6','#71fe5f','#c7ef34','#faba39','#ef5a11','#a51601'].map(hexToRgb),colors=[];for(let i=0;i<n;i++){const t=n===1?0:i/(n-1);if(name==='viridis')colors.push(stops(vir,t));else if(name==='turbo')colors.push(stops(turbo,t));else if(name==='contrast')colors.push(hsl((i*137.508)%360,.74,.57));else{const b=(base[name]||base.strata).map(hexToRgb),c=b[i%b.length],round=Math.floor(i/b.length),mix=Math.min(.28,round*.085);colors.push([lerp(c[0],1,mix),lerp(c[1],1,mix),lerp(c[2],1,mix)])}}return colors}
function legendLabel(i){return(SCENE.legend&&SCENE.legend[i])?SCENE.legend[i]:('Unit '+SCENE.categoryValues[i])}
function syncUnitMask(){unitMaskFlat.fill(0);for(let i=0;i<Math.min(64,unitVisible.length);i++)unitMaskFlat[i]=unitVisible[i]?1:0}
function setAllUnits(v){unitVisible=unitVisible.map(()=>v);syncUnitMask();document.querySelectorAll('[data-unit-index]').forEach(el=>{el.checked=v});requestDraw()}
function updatePalette(){const n=SCENE.categoryValues.length;currentColors=palette(controls.palette.value,n);if(unitVisible.length!==n)unitVisible=SCENE.categoryValues.map((v,i)=>!(SCENE.hidden&&SCENE.hidden[i]));syncUnitMask();paletteFlat.fill(0);for(let i=0;i<Math.min(64,n);i++){paletteFlat[i*3]=currentColors[i][0];paletteFlat[i*3+1]=currentColors[i][1];paletteFlat[i*3+2]=currentColors[i][2]}const l=gid('legend');l.textContent='';SCENE.categoryValues.forEach((v,i)=>{const item=document.createElement('label');item.className='legendItem';const cb=document.createElement('input');cb.type='checkbox';cb.checked=unitVisible[i];cb.dataset.unitIndex=String(i);cb.addEventListener('change',()=>{unitVisible[i]=cb.checked;syncUnitMask();requestDraw()});const sw=document.createElement('span');sw.className='swatch';const c=currentColors[i];sw.style.backgroundColor=`rgb(${Math.round(c[0]*255)}, ${Math.round(c[1]*255)}, ${Math.round(c[2]*255)})`;const lab=document.createElement('span');lab.textContent=legendLabel(i);lab.title=legendLabel(i);item.append(cb,sw,lab);l.appendChild(item)});requestDraw()}
// ---- shaders ----
const meshVS=`#version 300 es
layout(location=0)in vec3 aPosition;layout(location=1)in vec3 aNormal;layout(location=2)in float aCategory;uniform mat4 uViewProj;uniform vec3 uSceneCenter;uniform float uSceneScale;uniform float uZExag;uniform vec3 uPalette[64];uniform float uUnitVisible[64];out vec4 vColor;out vec3 vNormal;flat out float vVisible;void main(){vec3 c=vec3(aPosition.x-uSceneCenter.x,aPosition.y-uSceneCenter.y,(aPosition.z-uSceneCenter.z)*uZExag);gl_Position=uViewProj*vec4(c/uSceneScale,1.0);int idx=clamp(int(aCategory+0.5),0,63);vColor=vec4(uPalette[idx],1.0);vVisible=uUnitVisible[idx];vNormal=normalize(vec3(aNormal.x,aNormal.y,aNormal.z/max(uZExag,1.0)));}`;
const meshFS=`#version 300 es
precision highp float;in vec4 vColor;in vec3 vNormal;flat in float vVisible;uniform float uOpacity;out vec4 outColor;void main(){if(vVisible<0.5)discard;vec3 l=normalize(vec3(-0.45,-0.35,0.82));float d=max(dot(normalize(vNormal),l),0.0),s=.58+d*.42;outColor=vec4(vColor.rgb*s,vColor.a*uOpacity);}`;
const pointVS=`#version 300 es
layout(location=0)in vec3 aPoint;layout(location=1)in float aCategory;uniform mat4 uViewProj;uniform vec3 uSceneCenter;uniform float uSceneScale;uniform float uZExag;uniform float uPointSize;uniform vec3 uPalette[64];uniform float uUnitVisible[64];out vec4 vColor;flat out float vVisible;void main(){vec3 c=vec3(aPoint.x-uSceneCenter.x,aPoint.y-uSceneCenter.y,(aPoint.z-uSceneCenter.z)*uZExag);gl_Position=uViewProj*vec4(c/uSceneScale,1.0);gl_PointSize=uPointSize;int idx=clamp(int(aCategory+0.5),0,63);vColor=vec4(uPalette[idx],1.0);vVisible=uUnitVisible[idx];}`;
const pointFS=`#version 300 es
precision highp float;in vec4 vColor;flat in float vVisible;uniform float uOpacity;out vec4 outColor;void main(){if(vVisible<0.5)discard;vec2 d=gl_PointCoord*2.0-1.0;float r2=dot(d,d);if(r2>1.0)discard;float s=.62+.38*sqrt(max(0.0,1.0-r2));outColor=vec4(vColor.rgb*s,vColor.a*uOpacity);}`;
const lineVS=`#version 300 es
layout(location=0)in vec3 aPosition;uniform mat4 uViewProj;uniform vec3 uSceneCenter;uniform float uSceneScale;uniform float uZExag;void main(){vec3 c=vec3(aPosition.x-uSceneCenter.x,aPosition.y-uSceneCenter.y,(aPosition.z-uSceneCenter.z)*uZExag);gl_Position=uViewProj*vec4(c/uSceneScale,1.0);}`;
const lineFS=`#version 300 es
precision highp float;uniform vec3 uColor;uniform float uOpacity;out vec4 outColor;void main(){outColor=vec4(uColor,uOpacity);}`;
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s));return s}
function program(vs,fs){const p=gl.createProgram();gl.attachShader(p,shader(gl.VERTEX_SHADER,vs));gl.attachShader(p,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(p));return p}
function buf(target,data){const b=gl.createBuffer();gl.bindBuffer(target,b);gl.bufferData(target,data,gl.STATIC_DRAW);return b}
function createMeshVao(positions,normals,categories){const vao=gl.createVertexArray();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buf(gl.ARRAY_BUFFER,positions));gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,12,0);gl.bindBuffer(gl.ARRAY_BUFFER,buf(gl.ARRAY_BUFFER,normals));gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.BYTE,true,3,0);gl.bindBuffer(gl.ARRAY_BUFFER,buf(gl.ARRAY_BUFFER,categories));gl.enableVertexAttribArray(2);gl.vertexAttribPointer(2,1,gl.UNSIGNED_BYTE,false,1,0);gl.bindVertexArray(null);return{vao,count:positions.length/3,type:'mesh'}}
function createPointVao(positions,categories){const vao=gl.createVertexArray();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buf(gl.ARRAY_BUFFER,positions));gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,12,0);gl.bindBuffer(gl.ARRAY_BUFFER,buf(gl.ARRAY_BUFFER,categories));gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,1,gl.UNSIGNED_BYTE,false,1,0);gl.bindVertexArray(null);return{vao,count:positions.length/3,type:'points'}}
function createLineVao(positions){const vao=gl.createVertexArray();gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,buf(gl.ARRAY_BUFFER,positions));gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,12,0);gl.bindVertexArray(null);return{vao,count:positions.length/3,type:'lines'}}
function setupSliceBuffers(){sliceBuffers={pos:gl.createBuffer(),norm:gl.createBuffer(),cat:gl.createBuffer()};sliceVao=gl.createVertexArray();gl.bindVertexArray(sliceVao);gl.bindBuffer(gl.ARRAY_BUFFER,sliceBuffers.pos);gl.enableVertexAttribArray(0);gl.vertexAttribPointer(0,3,gl.FLOAT,false,12,0);gl.bindBuffer(gl.ARRAY_BUFFER,sliceBuffers.norm);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,3,gl.BYTE,true,3,0);gl.bindBuffer(gl.ARRAY_BUFFER,sliceBuffers.cat);gl.enableVertexAttribArray(2);gl.vertexAttribPointer(2,1,gl.UNSIGNED_BYTE,false,1,0);gl.bindVertexArray(null)}
// ---- slicer (over gridData) ----
function pid(i,j,k){const NX=gridData.dims[0],NY=gridData.dims[1];return i+j*NX+k*NX*NY}
function pointCat(i,j,k){return gridData.categories[pid(i,j,k)]}
function putVertex(pos,norm,cats,o,x,y,z,nx,ny,nz,cat){const p=o*3;pos[p]=x;pos[p+1]=y;pos[p+2]=z;norm[p]=nx;norm[p+1]=ny;norm[p+2]=nz;cats[o]=cat}
function rebuildSlice(){if(!gridData||!gl)return;const NX=gridData.dims[0],NY=gridData.dims[1],NZ=gridData.dims[2],xv=gridData.xvec,yv=gridData.yvec,zv=gridData.zvec,axis=controls.sliceAxis.value;let idx=Number(controls.sliceIndex.value),qc=0;if(axis==='x'){idx=Math.max(0,Math.min(NX-1,idx));qc=(NY-1)*(NZ-1)}else if(axis==='y'){idx=Math.max(0,Math.min(NY-1,idx));qc=(NX-1)*(NZ-1)}else{idx=Math.max(0,Math.min(NZ-1,idx));qc=(NX-1)*(NY-1)}const vc=qc*6,pos=new Float32Array(vc*3),norm=new Int8Array(vc*3),cats=new Uint8Array(vc);let v=0;if(axis==='x'){const ci=Math.min(idx,NX-2),x=xv[idx];for(let k=0;k<NZ-1;k++)for(let j=0;j<NY-1;j++){const cat=pointCat(ci,j,k),y0=yv[j],y1=yv[j+1],z0=zv[k],z1=zv[k+1];putVertex(pos,norm,cats,v++,x,y0,z0,127,0,0,cat);putVertex(pos,norm,cats,v++,x,y0,z1,127,0,0,cat);putVertex(pos,norm,cats,v++,x,y1,z1,127,0,0,cat);putVertex(pos,norm,cats,v++,x,y0,z0,127,0,0,cat);putVertex(pos,norm,cats,v++,x,y1,z1,127,0,0,cat);putVertex(pos,norm,cats,v++,x,y1,z0,127,0,0,cat)}}else if(axis==='y'){const cj=Math.min(idx,NY-2),y=yv[idx];for(let k=0;k<NZ-1;k++)for(let i=0;i<NX-1;i++){const cat=pointCat(i,cj,k),x0=xv[i],x1=xv[i+1],z0=zv[k],z1=zv[k+1];putVertex(pos,norm,cats,v++,x0,y,z0,0,127,0,cat);putVertex(pos,norm,cats,v++,x1,y,z0,0,127,0,cat);putVertex(pos,norm,cats,v++,x1,y,z1,0,127,0,cat);putVertex(pos,norm,cats,v++,x0,y,z0,0,127,0,cat);putVertex(pos,norm,cats,v++,x1,y,z1,0,127,0,cat);putVertex(pos,norm,cats,v++,x0,y,z1,0,127,0,cat)}}else{const ck=Math.min(idx,NZ-2),z=zv[idx];for(let j=0;j<NY-1;j++)for(let i=0;i<NX-1;i++){const cat=pointCat(i,j,ck),x0=xv[i],x1=xv[i+1],y0=yv[j],y1=yv[j+1];putVertex(pos,norm,cats,v++,x0,y0,z,0,0,127,cat);putVertex(pos,norm,cats,v++,x0,y1,z,0,0,127,cat);putVertex(pos,norm,cats,v++,x1,y1,z,0,0,127,cat);putVertex(pos,norm,cats,v++,x0,y0,z,0,0,127,cat);putVertex(pos,norm,cats,v++,x1,y1,z,0,0,127,cat);putVertex(pos,norm,cats,v++,x1,y0,z,0,0,127,cat)}}sliceVertexCount=vc;gl.bindBuffer(gl.ARRAY_BUFFER,sliceBuffers.pos);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.DYNAMIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,sliceBuffers.norm);gl.bufferData(gl.ARRAY_BUFFER,norm,gl.DYNAMIC_DRAW);gl.bindBuffer(gl.ARRAY_BUFFER,sliceBuffers.cat);gl.bufferData(gl.ARRAY_BUFFER,cats,gl.DYNAMIC_DRAW);updateLabels();requestDraw()}
function updateSliceAxis(reset){if(!gridData)return;const axis=controls.sliceAxis.value,d=gridData.dims,max=axis==='x'?d[0]-1:axis==='y'?d[1]-1:d[2]-1;controls.sliceIndex.max=String(max);controls.sliceIndex.value=String(reset?Math.round(max/2):Math.max(0,Math.min(max,Number(controls.sliceIndex.value))));rebuildSlice()}
function setSliceFromMouse(ev){if(!gridData||!controls.showSlice.checked||!controls.mouseSlice.checked||dragging)return;const r=canvas.getBoundingClientRect(),t=Math.max(0,Math.min(1,(ev.clientX-r.left)/Math.max(1,r.width))),idx=Math.round(t*Number(controls.sliceIndex.max));if(idx!==Number(controls.sliceIndex.value)){controls.sliceIndex.value=String(idx);rebuildSlice()}}
// ---- camera ----
function persp(fovy,aspect,near,far){const f=1/Math.tan(fovy/2),nf=1/(near-far),o=new Float32Array(16);o[0]=f/aspect;o[5]=f;o[10]=(far+near)*nf;o[11]=-1;o[14]=2*far*near*nf;return o}
function normv(v){const l=Math.hypot(v[0],v[1],v[2])||1;return[v[0]/l,v[1]/l,v[2]/l]}
function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}
function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}
function lookAt(eye,center,up){const z=normv([eye[0]-center[0],eye[1]-center[1],eye[2]-center[2]]),x=normv(cross(up,z)),y=cross(z,x),o=new Float32Array(16);o[0]=x[0];o[1]=y[0];o[2]=z[0];o[3]=0;o[4]=x[1];o[5]=y[1];o[6]=z[1];o[7]=0;o[8]=x[2];o[9]=y[2];o[10]=z[2];o[11]=0;o[12]=-dot(x,eye);o[13]=-dot(y,eye);o[14]=-dot(z,eye);o[15]=1;return o}
function mmul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)o[c*4+r]=a[0*4+r]*b[c*4+0]+a[1*4+r]*b[c*4+1]+a[2*4+r]*b[c*4+2]+a[3*4+r]*b[c*4+3];return o}
// World-up (z) orbit camera: azimuth (yaw) + elevation (pitch), up = +z. This gives the
// natural "grab and drag" feel for both flat (wcsb) and vertical (multilayer_fold) models
// — drag right → model right, drag up → model up — with no roll/inversion. Elevation is
// clamped just shy of ±90° (the only world-up singularity, where up ∥ view dir).
const PITCH_LIMIT=1.5533;  // ~89° — effectively the full range without the gimbal flip
let yaw=-.95,pitch=.42,distance=3.15,dragging=false,lastX=0,lastY=0;
function resetCamera(){yaw=-.95;pitch=.42;distance=3.15}
function requestDraw(){if(!gl||renderPending)return;renderPending=true;setTimeout(()=>{try{render()}catch(e){renderPending=false;console.error(e);showError('Render failed: '+(e.message||e))}},0)}
canvas.addEventListener('pointerdown',ev=>{dragging=true;lastX=ev.clientX;lastY=ev.clientY;canvas.setPointerCapture(ev.pointerId)});
canvas.addEventListener('pointermove',ev=>{if(dragging){const dx=ev.clientX-lastX,dy=ev.clientY-lastY;lastX=ev.clientX;lastY=ev.clientY;yaw-=dx*.006;pitch=Math.max(-PITCH_LIMIT,Math.min(PITCH_LIMIT,pitch+dy*.006));requestDraw()}else setSliceFromMouse(ev)});
canvas.addEventListener('pointerup',()=>{dragging=false});canvas.addEventListener('pointerleave',()=>{dragging=false});
canvas.addEventListener('wheel',ev=>{ev.preventDefault();distance=Math.max(.7,Math.min(12,distance*Math.exp(ev.deltaY*.001)));requestDraw()},{passive:false});
controls.resetCamera.addEventListener('click',()=>{resetCamera();requestDraw()});controls.palette.addEventListener('change',updatePalette);controls.unitsAll.addEventListener('click',()=>setAllUnits(true));controls.unitsNone.addEventListener('click',()=>setAllUnits(false));controls.sliceAxis.addEventListener('change',()=>updateSliceAxis(true));controls.sliceIndex.addEventListener('input',rebuildSlice);
['input','change'].forEach(evt=>{for(const key of ['zExag','gridOpacity','surfaceOpacity','pointSize','pointOpacity','sliceOpacity'])controls[key].addEventListener(evt,()=>{updateLabels();requestDraw()})});
for(const key of ['showGrid','showSurfaces','showSlice'])controls[key].addEventListener('change',requestDraw);controls.mouseSlice.addEventListener('change',requestDraw);window.addEventListener('resize',requestDraw);
function updateLabels(){labels.zExag.textContent=`${controls.zExag.value}x`;labels.gridOpacity.textContent=(Number(controls.gridOpacity.value)/100).toFixed(2);labels.surfaceOpacity.textContent=(Number(controls.surfaceOpacity.value)/100).toFixed(2);labels.pointSize.textContent=controls.pointSize.value;labels.pointOpacity.textContent=(Number(controls.pointOpacity.value)/100).toFixed(2);labels.sliceOpacity.textContent=(Number(controls.sliceOpacity.value)/100).toFixed(2);if(gridData){const axis=controls.sliceAxis.value,idx=Number(controls.sliceIndex.value),coord=axis==='x'?gridData.xvec[idx]:axis==='y'?gridData.yvec[idx]:gridData.zvec[idx];labels.sliceIndex.textContent=`${idx} (${coord.toFixed(0)})`}}
function resizeCanvas(){const dpr=Math.min(window.devicePixelRatio||1,2),w=Math.max(1,Math.floor(canvas.clientWidth*dpr)),h=Math.max(1,Math.floor(canvas.clientHeight*dpr));if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h)}return dpr}
function setCommonUniforms(p,vp,z){gl.uniformMatrix4fv(gl.getUniformLocation(p,'uViewProj'),false,vp);gl.uniform3fv(gl.getUniformLocation(p,'uSceneCenter'),SCENE.sceneCenter);gl.uniform1f(gl.getUniformLocation(p,'uSceneScale'),SCENE.sceneScale);gl.uniform1f(gl.getUniformLocation(p,'uZExag'),z);gl.uniform3fv(gl.getUniformLocation(p,'uPalette[0]'),paletteFlat);gl.uniform1fv(gl.getUniformLocation(p,'uUnitVisible[0]'),unitMaskFlat)}
function beginMesh(opacity,vp,z){gl.useProgram(meshProgram);setCommonUniforms(meshProgram,vp,z);gl.uniform1f(gl.getUniformLocation(meshProgram,'uOpacity'),opacity);if(opacity<.99){gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false)}else{gl.disable(gl.BLEND);gl.depthMask(true)}}
function beginPoints(opacity,size,vp,z,dpr){gl.useProgram(pointProgram);setCommonUniforms(pointProgram,vp,z);gl.uniform1f(gl.getUniformLocation(pointProgram,'uOpacity'),opacity);gl.uniform1f(gl.getUniformLocation(pointProgram,'uPointSize'),size*dpr);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);gl.depthMask(false)}
function drawLayer(kind){let drawn=0;for(const t of layers[kind]){gl.bindVertexArray(t.vao);if(t.type==='points'){gl.drawArrays(gl.POINTS,0,t.count);drawn+=t.count}else{gl.drawArrays(gl.TRIANGLES,0,t.count);drawn+=t.count/3}}gl.bindVertexArray(null);return drawn}
function drawPoints(){let drawn=0;for(const t of layers.points){if(pointVisible[t.group]===false)continue;gl.bindVertexArray(t.vao);gl.drawArrays(gl.POINTS,0,t.count);drawn+=t.count}gl.bindVertexArray(null);return drawn}
function beginLines(vp,z){gl.useProgram(lineProgram);gl.uniformMatrix4fv(gl.getUniformLocation(lineProgram,'uViewProj'),false,vp);gl.uniform3fv(gl.getUniformLocation(lineProgram,'uSceneCenter'),SCENE.sceneCenter);gl.uniform1f(gl.getUniformLocation(lineProgram,'uSceneScale'),SCENE.sceneScale);gl.uniform1f(gl.getUniformLocation(lineProgram,'uZExag'),z);gl.uniform3fv(gl.getUniformLocation(lineProgram,'uColor'),[0.886,0.255,0.290]);gl.uniform1f(gl.getUniformLocation(lineProgram,'uOpacity'),0.95);gl.disable(gl.BLEND);gl.depthMask(true)}
function drawVectors(vp,z){if(!vectorsVisible||!layers.vectors.length)return 0;beginLines(vp,z);let n=0;for(const t of layers.vectors){gl.bindVertexArray(t.vao);gl.drawArrays(gl.LINES,0,t.count);n+=t.count/2}gl.bindVertexArray(null);return n}
function updateStatus(g,s,sl,p){gid('counts').textContent=`${g.toLocaleString()} grid tris, ${s.toLocaleString()} surf tris, ${p.toLocaleString()} pts`;gid('status').textContent=`grid ${layers.grid.length} / surfaces ${layers.surfaces.length} / points ${layers.points.length} buffers`}
function render(){renderPending=false;if(!gl||!SCENE)return;const dpr=resizeCanvas();updateLabels();gl.clearColor(.055,.075,.095,1);gl.clearDepth(1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.disable(gl.CULL_FACE);const aspect=canvas.width/Math.max(1,canvas.height),projection=persp(42*Math.PI/180,aspect,.02,100),cp=Math.cos(pitch),eye=[distance*cp*Math.cos(yaw),distance*cp*Math.sin(yaw),distance*Math.sin(pitch)],view=lookAt(eye,[0,0,0],[0,0,1]),vp=mmul(projection,view),z=Number(controls.zExag.value);let dg=0,ds=0,dsl=0,dp=0;if(controls.showGrid.checked){beginMesh(Number(controls.gridOpacity.value)/100,vp,z);dg=drawLayer('grid')}if(controls.showSurfaces.checked){beginMesh(Number(controls.surfaceOpacity.value)/100,vp,z);ds=drawLayer('surfaces')}if(controls.showSlice.checked&&sliceVertexCount>0){beginMesh(Number(controls.sliceOpacity.value)/100,vp,z);gl.bindVertexArray(sliceVao);gl.drawArrays(gl.TRIANGLES,0,sliceVertexCount);gl.bindVertexArray(null);dsl=sliceVertexCount/3}if(layers.points.length){beginPoints(Number(controls.pointOpacity.value)/100,Number(controls.pointSize.value),vp,z,dpr);dp=drawPoints()}let dv=0;if(layers.vectors.length){dv=drawVectors(vp,z)}gl.depthMask(true);++frameCount;updateStatus(dg,ds,dsl,dp);window.__viewerStatus={frameCount,drawnGridTriangles:dg,drawnSurfaceTriangles:ds,drawnSliceTriangles:dsl,drawnPoints:dp,drawnVectors:dv,glError:gl.getError()};document.body.setAttribute('data-viewer-status',JSON.stringify(window.__viewerStatus))}
// ---- registry API used by the injected bootstrap ----
function setScene(s){SCENE=s;if(typeof s.initialZExag==='number'){controls.zExag.value=String(s.initialZExag)}document.title=s.title||document.title}
function setGridData(g){gridData=g}
function addLayer(kind,d){if(d&&d.count>0)layers[kind].push(d)}
function initGL(){gl=canvas.getContext('webgl2',{antialias:true,alpha:false,premultipliedAlpha:false});if(!gl){showError('WebGL2 is not available in this browser. Try current Chrome, Edge, or Firefox.');return false}try{meshProgram=program(meshVS,meshFS);pointProgram=program(pointVS,pointFS);lineProgram=program(lineVS,lineFS)}catch(e){showError('Shader setup failed: '+e.message);return false}setupSliceBuffers();return true}
function startCommon(){buildLayerToggles();updatePalette();updateSliceAxis(true);updateLabels();requestDraw()}
// ---- data bootstrap (mode-specific) ----
__BOOTSTRAP__
</script>
</body>
</html>'''
