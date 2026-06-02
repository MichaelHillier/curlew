# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`curlew` is a PyTorch toolkit for building 2D/3D geological models with **neural fields** (and other learnable or analytical functions). Everything is differentiable so models can be fit to sparse geological data and used in inversion/optimisation workflows. It can also build synthetic models (neural fields swapped for closed-form functions) to generate test/benchmark datasets.

## Commands

```bash
pip install -e .[all]          # editable install with all optional deps (pytest, napari, matplotlib, plyfile, scipy)
pytest ./tests/                # full test suite (this is what CI runs)
pytest tests/test_geology.py                       # single file
pytest tests/test_geology.py::test_name -s         # single test, show stdout
bash benchmark.sh              # run pytest-benchmark suite + the hutton memory benchmark
pytest benchmarks/ --benchmark-only --benchmark-save <name>   # benchmarks only
```

CI (`.github/workflows/tests.yml`) runs the test suite on Python 3.11, then builds pdoc3 HTML docs for both `main` (stable) and `dev` (development) branches and deploys to GitHub Pages. Docstrings are written for pdoc — keep cross-references in the ``curlew.module.Class`` form used throughout, since pdoc turns them into links.

## Global configuration

Module-level globals in [curlew/__init__.py](curlew/__init__.py) control runtime behaviour and are read throughout the codebase — set them rather than passing args everywhere:
- `curlew.device` — `'cpu'` by default; set to a GPU device for parallelisation.
- `curlew.dtype` — `torch.float64`; drop to `float32` to save RAM.
- `curlew.default_dim` — `3`; default input dimensionality for new models.
- `curlew.batchSize` — arrays larger than this are chunked during evaluation to avoid OOM.
- `curlew.compile` — toggles `torch.compile`.

`_tensor()` / `_numpy()` in the same file are the canonical converters between numpy/torch using the global device+dtype; use them at API boundaries.

## Architecture

The model is a **graph of geological events evaluated in retro-deformation order**. Understanding this evaluation flow is the key to the codebase.

### Layers (bottom to top)

1. **Scalar fields — `curlew.fields.BaseSF`** ([curlew/fields/__init__.py](curlew/fields/__init__.py))
   The implicit building blocks. Two families:
   - *Neural / learnable* (for interpolation): `fourier.NFF` (Fourier-feature), `series.FSF` (Fourier-series, faster). `BaseNF` is their base.
   - *Analytical / closed-form* (mostly for synthetic models): `analytical.LinearField`, `QuadraticField`, `PeriodicField`, `ListricField`, `EllipsoidalField`. `BaseAF` is their base.

2. **Geological events — `curlew.geology.geoevent.GeoEvent`** ([curlew/geology/geoevent.py](curlew/geology/geoevent.py))
   Wraps one or more `BaseSF`s and gives them geological meaning. Each event defines how it interacts with older/younger events via an interaction object (below). Events also own **isosurfaces** (lithological contacts, fault surfaces) and **volumes** (regions, finite faults). Isosurfaces can be a fixed value *or* a "seed" — whatever value the field happens to take at a seed point — which decouples surfaces from absolute field values.

3. **Interactions — `curlew.geology.interactions`** ([curlew/geology/interactions.py](curlew/geology/interactions.py))
   Two relationship types:
   - `Overprint` — truncation relations: unconformities, onlaps, intrusions, domain boundaries. Determines how an event overprints older geology.
   - `OffsetBase` — deformations that **retro-deform** (transform query points *backwards*) before older events are evaluated. Subclasses: `VFieldOffset` → `FaultOffset`, `SheetOffset`; `FoldOffset`.

4. **Model — `curlew.geology.geomodel.GeoModel`** ([curlew/geology/geomodel.py](curlew/geology/geomodel.py))
   Builds and links the event graph (`_linkE`), defines evaluation order and topology, handles global↔local coordinate transforms, and exposes `fit`, `predict`, `drill`. Inherits `LearnableBase`. Save/load via `curlew.io.saveModel` / `loadModel`.

### Evaluation flow

When `GeoModel.predict(x)` runs, query points are passed *down* the event graph from youngest to oldest. Each deformation event **undeforms** (retro-deforms) the points before handing them to older events, so older geology is evaluated in its pre-deformation coordinates. Overprints then decide which event's values "win" at each point as results combine back up. This is why `GeoEvent.forward`/`predict` take an `undef` flag and a `field` index.

### Construction helpers

Don't instantiate `GeoEvent` directly in normal use — use the factory functions in [curlew/geology/__init__.py](curlew/geology/__init__.py): `strati`, `sheet`, `fault`, `fold`, `stock`, `domainBoundary`. Each wires up the appropriate field type and interaction.

### Supporting data classes ([curlew/core.py](curlew/core.py))

- `CSet` — **constraints** used to fit a field: value, gradient (strike/dip), property, inequality, equality (tangent traces), grid, and trend constraints. Typically one `CSet` per `GeoEvent`. Prefer gradient/inequality over value constraints — neural fields fit those far better.
- `HSet` — **hyperparameters** weighting the multi-objective loss terms per field. Keep most weights at 0 (~1–3 active terms) or tuning becomes intractable.
- `Geode` — **model outputs**: scalar values, lithology/structure IDs, predicted properties, applied deformations, grid, transforms, metadata. Has `topology`, `combine`, `concat`, surface extraction.
- `Pebble` — groups **losses and their optimisers**; only needed when defining custom losses or learnable objects. Custom losses can also be passed at `GeoModel.fit(custom_loss=...)` to optimise the whole stack at once (e.g. gravity inversion).
- `LearnableBase` — `nn.Module` base for everything trainable; provides `init_optim`, `bind`, coordinate-transform plumbing.

### Other modules

- [curlew/geometry.py](curlew/geometry.py) — `Grid`, `Transform` (coordinate systems), Poisson-disk sampling, `section`, wave functions, `compute_topology_adjacency`.
- [curlew/synthetic.py](curlew/synthetic.py) — named synthetic models (`steno`, `lehmann`, `hutton`, `playfair`, `walker`, `michell`, `anderson`, `seuss`) plus `sample()` and `extract_constraints()` to turn a synthetic model into a training dataset.
- [curlew/io.py](curlew/io.py) — model save/load and PLY/OBJ mesh export (`savePLY`, `saveOBJ`, `loadPLY`).
- [curlew/visualise/napari_viewer.py](curlew/visualise/napari_viewer.py) — optional napari-based 3D viewer (runs alongside Jupyter).

## Conventions

- Optional dependencies (`matplotlib`, `napari`, `plyfile`, `scipy`) must be imported lazily/guarded — the core must run with only `numpy`, `torch`, `tqdm`. See the `mpl` try/except pattern in `__init__.py`.
- API methods generally accept array-likes and return numpy by default (`to_numpy=True`); internal computation stays in torch. Respect the `to_numpy` / `transform` flags when adding methods.
- `v1.2` renamed the old `GeoField` to `GeoEvent` — don't reintroduce the old name.
