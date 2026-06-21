# `html_viewer` — in-browser WebGL viewer for curlew models

[`html_viewer.py`](html_viewer.py) writes a self-contained **WebGL HTML viewer** for a
curlew model — a portable, shareable alternative to
[`napari_viewer.py`](napari_viewer.py) that needs no Qt session. It renders the predicted
**lithology/level volume**, **boundary iso-surfaces**, **observation point clouds** and
oriented **vectors** (e.g. bedding normals), with an interactive palette, per-unit
visibility, vertical exaggeration and an X/Y/Z slicer.

It is **numpy-only**: unlike the original prototype (which read `.vts`/`.vtp` via `vtk`),
this module is fed **in-memory** from curlew objects, so it adds no dependency beyond numpy.
(`pyvista`/`vtk` remain optional and are only used by `curlew.io.saveVTK` / `loadObservations`.)

## Quick start

The one-call entry point takes a fitted `GeoModel`, its grid prediction (`Geode`), the
`Grid`, and the `Observations` it was fit to:

```python
from curlew.geometry import Grid
from curlew.visualise.html_viewer import write_geomodel_viewer, serve_and_open

geode = M.predict(G)                      # G is an axis-aligned curlew Grid

# small model → one self-contained .html (opens from file://)
html = write_geomodel_viewer(
    "outputs/model.html", M, geode, G,
    obs=obs, color_by="lithoID",          # or "level"
    mode="self_contained", initial_z_exaggeration=1,
)
serve_and_open(html)
```

```python
# large model → small .html + binary tile folder, served over local HTTP
html = write_geomodel_viewer(
    "outputs/model.html", M, geode, G,
    obs=obs, color_by="level", mode="tiled", initial_z_exaggeration=100,
)
server = serve_and_open(html)             # starts a background HTTP server; keep `server`
# ... view in the browser ...            # server.shutdown() to stop it (daemon thread otherwise)
```

See [`examples/multilayer_fold`](../../examples/multilayer_fold) (self-contained, with
interface points + normals) and [`examples/wcsb`](../../examples/wcsb) (tiled, unit markers)
for the worked patterns.

## What `obs=` derives automatically

Passing `obs` (a `curlew.io.Observations`) lets `write_geomodel_viewer` build the whole scene
without per-notebook boilerplate (via `extract_geomodel_layers` + `observation_layers`):

| Scene element | Source |
|---|---|
| grid volume colouring + legend | `geode.lithoID` / `geode.lithoLookup` (`color_by="lithoID"`) or a stratigraphic **level** volume + `level→name` legend (`color_by="level"`) |
| boundary iso-surfaces | one `Grid.contour` mesh per generative event, coloured by the unit/level it bounds (erosional unconformities share an `"(unconformity surface)"` colour) |
| `"Interface points"` / `"Unit points"` layers | `obs.interface_points()` / `obs.unit_points()`, coloured to match the volume, subsampled to `max_points` |
| `"Normals"` vector layer | `obs.normal_points()` (drawn as line segments) |

`color_by="level"` requires a strat-column model from
`curlew.geology.stratbuilder.build_geomodel` (uses `llookup` / `level_litho` / `strat_units`);
`color_by="lithoID"` works for any model.

## Output modes

| | `self_contained` | `tiled` |
|---|---|---|
| Output | one `.html` (geometry base64-embedded) | small `.html` + `<stem>_data/` binary tiles |
| Opens from | `file://` directly | **local HTTP only** (browsers block `fetch` from `file://`) |
| Best for | small models (within the embed guard) | large models / progressive draw |
| Guard | aborts above `max_embedded_mb` (250 MB) unless `allow_large=True` | — |

`serve_and_open` handles both: it opens self-contained files directly, and for tiled output
starts a loopback `http.server.ThreadingHTTPServer` (daemon thread) and returns it. Nothing is
uploaded — the server binds `127.0.0.1`.

## Public API

```python
write_geomodel_viewer(output_path, model, geode, grid, *, obs=None, points=None,
    vectors=None, color_by="lithoID", mode="self_contained", erode_mask=-2,
    normals=True, max_points=200000, legend=None, hidden_categories=None, **kwargs)
```
High-level one-call viewer (recommended). `**kwargs` and `legend`/`hidden_categories`
forward to / override `write_html_viewer`. `points`/`vectors` override the `obs`-derived ones.

```python
write_html_viewer(output_path, *, grid, grid_categories, surfaces=None, points=None,
    vectors=None, vector_scale=None, vector_name="Normals", mode="self_contained",
    legend=None, hidden_categories=None, z_scale=1.0, title="Curlew Model Viewer",
    initial_z_exaggeration=25, max_embedded_mb=250.0, allow_large=False,
    grid_tile_cells=32, surface_tile_dims=(12,12), point_tile_dims=(12,12)) -> Path
```
Low-level writer over plain arrays. Key inputs:
- `grid` — axis-aligned 3D `Grid`; `grid_categories` — `(N,)` int id per grid point (grid order).
- `surfaces` — list of `(verts (M,3), faces (K,3), category)` meshes (world coords).
- `points` — a single `(coords, cats)`, a `{name: (coords, cats)}` dict, or a
  `[(name, coords, cats), …]` list. Each becomes a **named layer with its own GUI toggle**.
- `vectors` — `(origins, directions)` arrays, drawn as `vector_name` line segments.
- `legend` — `{category_id: name}`; `hidden_categories` — ids hidden by default (e.g. `{-1}`).
- `z_scale` — applied to Z at write time (default 1.0; the interactive slider handles
  exaggeration instead).

```python
extract_geomodel_layers(model, geode, grid, *, color_by="lithoID",
    erode_mask=-2, unconformity_id=-2) -> dict   # {grid_categories, surfaces, legend, hidden_categories}
observation_layers(obs, model, geode, *, color_by="lithoID", normals=True,
    max_points=200000, seed=0) -> (points, vectors)
serve_and_open(html_path, *, open_browser=True, port=0, bind="127.0.0.1") -> server | None
```

## Viewer controls (in the browser)

- **Camera** — drag to orbit (world-up; drag direction = model direction), wheel to zoom,
  *Reset Camera* to recentre.
- **Layers** — *Grid volume*, *Boundary surfaces*, *Slice*, plus one checkbox per named
  point layer and a *Normals* toggle (when present).
- **Palette** — Turbo / Viridis / Stratigraphy / Colorblind safe / High contrast / Gray.
- **Per-unit visibility** — a legend checkbox per unit (with *All* / *None*); the mask applies
  to the volume, surfaces, slice **and** point layers (vectors are exempt).
- **Sliders** — vertical exaggeration, grid/surface/point opacity, point size, slice opacity.
- **Slicer** — full-resolution X/Y/Z slice through the category volume; optional
  *Move slice with mouse cursor* (off by default).

## Requirements & limits

- A current WebGL2 browser (Chrome / Edge / Firefox).
- The grid must be **axis-aligned** (no rotation) and 3D — like `saveVTK`'s `.vti` path.
- Up to **64** distinct categories (shader uniform-array limit); merge/relabel units if exceeded.
- Tiled viewers must be served over HTTP (use `serve_and_open`), not opened from `file://`.
- Vectors render as 1-px lines (most browsers ignore wider line widths).
