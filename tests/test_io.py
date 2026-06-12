import pytest
import numpy as np
import os

def test_PLY(tmp_path):
    try:
        import plyfile # only run test if plyfile is installed
    except:
        return
    
    from curlew.io import savePLY, loadPLY
    xyz = np.random.rand(100,3)*100
    normals = np.random.rand(100,3)
    normals = normals / np.linalg.norm(normals, axis=1)[:,None]
    rgb = np.clip( (np.random.rand(100,3) * 255), 0, 255 ).astype(np.uint8)
    data = np.random.rand(100,5)
    names = ['Field%d'%i for i in range(5)]
    for n in [None, normals]:
        for c in [None, rgb]:
            for d in [None, data]:
                for nm in [None, names]:
                    file_path = tmp_path / "test.ply"
                    savePLY( file_path, xyz, rgb=c, normals=n, attr=d, names=nm)
                    p = loadPLY( file_path )
                    assert np.max( np.abs(xyz - p['xyz'] ) ) < 1e-6 # check positions match
                    if n is not None:
                        assert np.max( np.abs(normals - p['normals'] ) ) < 1e-6 # check positions match
                    if c is not None:
                        assert np.max( np.abs(rgb - p['rgb'] ) ) < 1e-6 # check positions match
                    if d is not None:
                        assert np.max( np.abs(data - p['attr'] ) ) < 1e-6 # check positions match
                        if nm is not None:
                            assert np.all([names[i] == p['names'][i] for i in range(len(names))])

def test_OBJ(tmp_path):
    from curlew.io import saveOBJ
    xyz = np.random.rand(100,3)*100 # make some random points
    faces = [ np.random.choice(len(xyz), 3) for i in range(100) ] # make some random faces
    rgb = np.clip( (np.random.rand(100,3) * 255), 0, 255 ).astype(np.uint8)
    saveOBJ( tmp_path / "test.obj", xyz=xyz, rgb=rgb, faces=faces ) # check OBJ writes
    assert os.path.exists( tmp_path / "test.obj" ) # not the robust test; but better than nothing...

def test_saveVTK(tmp_path):
    pv = pytest.importorskip("pyvista")  # optional dependency
    from curlew.core import Geode
    from curlew.geometry import Grid
    from curlew.io import saveVTK

    # small axis-aligned 3D grid with a coordinate-dependent payload, so any
    # axis-ordering mistake in the export shows up as a value mismatch
    G = Grid(dims=(40.0, 30.0, 20.0), step=(10.0, 10.0, 5.0), center=(100.0, -50.0, 5.0))
    pts = G.coords()
    geode = Geode()
    geode.grid = G
    geode.scalar = pts[:, 2] * 2.0 + pts[:, 0]            # depends on x and z
    geode.lithoID = (pts[:, 2] > 5.0).astype(int) + 1     # 1 below mid-z, 2 above
    geode.lithoLookup = {1: "lower", 2: "upper"}
    geode.structureID = np.ones(len(pts), dtype=int)
    geode.structureLookup = {1: "event"}

    level = (pts[:, 0] > 100.0).astype(int) + 10  # extra named array

    for ext in ("vts", "vti"):
        path = tmp_path / f"test.{ext}"
        saveVTK(path, geode, extra={"level": level})
        m = pv.read(str(path))
        assert m.n_points == len(pts)
        # values must sit at the right coordinates (catches axis-order bugs)
        order = np.lexsort(m.points.T)        # sort both by (z, y, x)
        ref = np.lexsort(pts.T)
        assert np.allclose(np.asarray(m.points)[order], pts[ref])
        assert np.allclose(np.asarray(m.point_data["scalar"])[order], geode.scalar[ref])
        assert np.array_equal(np.asarray(m.point_data["lithoID"])[order], geode.lithoID[ref])
        assert np.array_equal(np.asarray(m.point_data["level"])[order], level[ref])
        legend = [str(s) for s in m.field_data["lithoID_legend"]]
        assert legend == ["1: lower", "2: upper"]

    # rotated grids must be rejected for .vti but work for .vts
    a = np.deg2rad(30)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    Gr = Grid(dims=(40.0, 30.0, 20.0), step=(10.0, 10.0, 5.0), center=(0, 0, 0), rotation=R)
    geode_r = Geode()
    geode_r.grid = Gr
    geode_r.scalar = Gr.coords()[:, 0]
    with pytest.raises(AssertionError):
        saveVTK(tmp_path / "rot.vti", geode_r)
    saveVTK(tmp_path / "rot.vts", geode_r)
    m = pv.read(str(tmp_path / "rot.vts"))
    order = np.lexsort(m.points.T); ref = np.lexsort(Gr.coords().T)
    assert np.allclose(np.asarray(m.point_data["scalar"])[order], geode_r.scalar[ref])


def test_model_io(tmp_path):
    import curlew
    from curlew.io import saveModel, loadModel
    from curlew.synthetic import steno
    
    # build synthetic model and check it saves / loads
    curlew.default_dim = 2
    M = steno()
    path = tmp_path / "steno.pt"
    saveModel(path, M)
    M2 = loadModel(path)
    assert M2.name == M.name
    assert len(M2.events) == len(M.events)
    xy = M.grid.coords()[:50]
    g1 = M.predict(xy)
    g2 = M2.predict(xy)
    assert np.max(np.abs(g1.scalar - g2.scalar)) < 1e-6

    # extract one event and check it also saves / loads
    path_evt = tmp_path / "s0.pt"
    saveModel(path_evt, M.events[0])
    E2 = loadModel(path_evt)
    assert E2.name == M.events[0].name
