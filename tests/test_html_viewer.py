"""
Round-trip tests for the in-browser WebGL viewer writer
(:mod:`curlew.visualise.html_viewer`). numpy-only — no GPU / browser needed, so
this runs in CI. We check that both output modes emit well-formed artifacts with
all template markers filled and self-consistent geometry counts.
"""
import base64
import json

import numpy as np
import pytest

from curlew.geometry import Grid
from curlew.visualise import html_viewer as hv


def _toy_model():
    """A small axis-aligned grid with a two-layer model + a -1 'outside' cap."""
    G = Grid(dims=(12, 10, 8), step=1.0, center=(0.0, 0.0, 0.0))
    coords = G.coords()                       # (N,3), curlew C-order (z fastest)
    z = coords[:, 2]
    ids = np.where(z < 0.0, 0, 1).astype(np.int64)   # bottom unit 0, top unit 1
    ids[z > 2.5] = -1                                # a "no unit" cap to hide
    # one flat triangle surface near the 0|1 contact, coloured by unit 1
    verts = np.array([[-5, -4, 0], [5, -4, 0], [0, 4, 0]], float)
    faces = np.array([[0, 1, 2]], int)
    surfaces = [(verts, faces, 1)]
    # a few marker points coloured by unit
    pcoords = np.array([[0, 0, -2], [1, 1, -1], [-2, 0, 1], [2, -1, 2]], float)
    pcats = np.array([0, 0, 1, 1])
    return G, ids, surfaces, (pcoords, pcats)


def test_self_contained_roundtrip(tmp_path):
    G, ids, surfaces, points = _toy_model()
    out = hv.write_html_viewer(
        tmp_path / "toy.html", grid=G, grid_categories=ids, surfaces=surfaces,
        points=points, mode="self_contained", legend={0: "Lower", 1: "Upper", -1: "(none)"},
        hidden_categories={-1}, title="Toy <Model>",
    )
    assert out.exists()
    text = out.read_text(encoding="utf-8")

    # every template marker must be substituted
    for marker in ("__BOOTSTRAP__", "__TITLE__", "__BADGE__"):
        assert marker not in text
    # title HTML-escaped, self-contained bootstrap present, no sibling data dir
    assert "Toy &lt;Model&gt;" in text
    assert "const B64=" in text and "setScene(META)" in text
    assert not (tmp_path / "toy_data").exists()

    # the embedded scene metadata is recoverable and consistent
    meta = json.loads(text.split("const META=", 1)[1].split(";\n", 1)[0])
    assert meta["gridDims"] == [12, 10, 8]
    assert meta["hidden"][meta["categoryValues"].index(-1)] is True
    assert meta["gridTriangleCount"] > 0
    assert meta["surfaceTriangleCount"] == 1
    assert meta["pointCount"] == 4

    # a base64 payload decodes to the advertised vertex count (positions = 3 f32)
    b64 = text.split("gridPositions:'", 1)[1].split("'", 1)[0]
    n_floats = len(base64.b64decode(b64)) // 4
    assert n_floats == meta["gridTriangleCount"] * 3 * 3


def test_tiled_roundtrip(tmp_path):
    G, ids, surfaces, points = _toy_model()
    out = hv.write_html_viewer(
        tmp_path / "toy.html", grid=G, grid_categories=ids, surfaces=surfaces,
        points=points, mode="tiled", hidden_categories={-1},
    )
    data = tmp_path / "toy_data"
    assert out.exists() and data.is_dir()

    text = out.read_text(encoding="utf-8")
    assert "__BOOTSTRAP__" not in text
    assert "const DATA_ROOT='toy_data/'" in text and "boot()" in text

    # grid slice assets
    for f in ("grid/xvec.f32", "grid/yvec.f32", "grid/zvec.f32", "grid/categories.u8"):
        assert (data / f).exists()

    manifest = json.loads((data / "manifest.json").read_text())
    assert manifest["grid"]["dims"] == [12, 10, 8]
    assert manifest["gridMesh"]["tiles"], "expected at least one grid-mesh tile"
    # single (coords, cats) → one "Unit points" group
    assert manifest["pointGroups"] == ["Unit points"]
    assert manifest["points"]["groups"][0]["tiles"], "expected at least one point tile"

    # every referenced tile file exists and its byte size matches its vertex count
    for tile in manifest["gridMesh"]["tiles"]:
        p = data / tile["file"]
        assert p.exists()
        # mesh tile = positions(12B) + normals(3B) + category(1B) per vertex
        assert p.stat().st_size == tile["vertexCount"] * (12 + 3 + 1)
    total_grid = sum(t["vertexCount"] for t in manifest["gridMesh"]["tiles"]) // 3
    assert total_grid == manifest["gridTriangleCount"] > 0


def test_named_point_layers(tmp_path):
    """A dict of point layers → one toggleable group each, in both modes."""
    G, ids, surfaces, (pc, pcats) = _toy_model()
    layers = {"Interface points": (pc[:2], pcats[:2]), "Unit markers": (pc[2:], pcats[2:])}

    out = hv.write_html_viewer(tmp_path / "sc.html", grid=G, grid_categories=ids,
                               points=layers, mode="self_contained")
    meta = json.loads(out.read_text(encoding="utf-8").split("const META=", 1)[1].split(";\n", 1)[0])
    assert meta["pointGroups"] == ["Interface points", "Unit markers"]
    assert meta["pointCount"] == 4
    # one base64 (pos,cat) pair per group is embedded
    text = out.read_text(encoding="utf-8")
    assert "pointPos0:'" in text and "pointPos1:'" in text and "pointPos2:'" not in text

    out2 = hv.write_html_viewer(tmp_path / "tiled.html", grid=G, grid_categories=ids,
                                points=layers, mode="tiled")
    man = json.loads((tmp_path / "tiled_data" / "manifest.json").read_text())
    assert [g["name"] for g in man["points"]["groups"]] == ["Interface points", "Unit markers"]
    assert all(g["tiles"] for g in man["points"]["groups"])


def test_vectors_layer(tmp_path):
    """Bedding-normal vectors embed (self-contained) / write a bin (tiled)."""
    G, ids, surfaces, points = _toy_model()
    origins = np.array([[0, 0, 0], [1, 1, 1]], float)
    directions = np.array([[0, 0, 1], [1, 0, 0]], float)

    out = hv.write_html_viewer(tmp_path / "v.html", grid=G, grid_categories=ids,
                               vectors=(origins, directions), vector_name="Normals",
                               mode="self_contained")
    text = out.read_text(encoding="utf-8")
    meta = json.loads(text.split("const META=", 1)[1].split(";\n", 1)[0])
    assert meta["hasVectors"] is True and meta["vectorCount"] == 2
    assert meta["vectorName"] == "Normals" and "vectorPositions:'" in text

    out2 = hv.write_html_viewer(tmp_path / "vt.html", grid=G, grid_categories=ids,
                                vectors=(origins, directions), mode="tiled")
    man = json.loads((tmp_path / "vt_data" / "manifest.json").read_text())
    assert man["hasVectors"] is True
    assert man["vectors"]["count"] == 4  # 2 vectors × 2 endpoints
    assert (tmp_path / "vt_data" / "vectors" / "lines.bin").exists()


def test_no_vectors_by_default(tmp_path):
    G, ids, surfaces, points = _toy_model()
    out = hv.write_html_viewer(tmp_path / "n.html", grid=G, grid_categories=ids,
                               mode="self_contained")
    meta = json.loads(out.read_text(encoding="utf-8").split("const META=", 1)[1].split(";\n", 1)[0])
    assert meta["hasVectors"] is False and meta["vectorCount"] == 0


def test_rejects_rotated_grid(tmp_path):
    rot = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    G = Grid(dims=(8, 8, 8), step=1.0, center=(0, 0, 0), rotation=rot)
    ids = np.zeros(int(np.prod(G.shape)), int)
    with pytest.raises(ValueError, match="axis-aligned"):
        hv.write_html_viewer(tmp_path / "r.html", grid=G, grid_categories=ids)


def test_serve_and_open_self_contained_returns_none(tmp_path):
    G, ids, surfaces, points = _toy_model()
    out = hv.write_html_viewer(tmp_path / "toy.html", grid=G, grid_categories=ids,
                               mode="self_contained")
    assert hv.serve_and_open(out, open_browser=False) is None


def test_serve_and_open_tiled_serves(tmp_path):
    G, ids, surfaces, points = _toy_model()
    out = hv.write_html_viewer(tmp_path / "toy.html", grid=G, grid_categories=ids,
                               mode="tiled")
    server = hv.serve_and_open(out, open_browser=False)
    try:
        assert server is not None
        import urllib.request
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/toy_data/manifest.json", timeout=5) as r:
            assert json.loads(r.read())["grid"]["dims"] == [12, 10, 8]
    finally:
        if server is not None:
            server.shutdown()
