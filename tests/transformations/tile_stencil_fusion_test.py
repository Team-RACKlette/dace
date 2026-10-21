# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for TileStencilFusion: a stencil producer run once per tile of its consumer, into a tile buffer."""
import numpy as np

import dace
from dace import dtypes
from dace.sdfg import nodes
from dace.transformation.dataflow.tile_stencil_fusion import TileStencilFusion
from tests.transformations.otf_stencil_fusion_test import N, V, _in_place_sdfg, _point_consumer_body, _stencil_sdfg, _top_maps

# On the CPU the tile buffers are ordinary scope transients and the maps inside a tile sequential loops.
CPU = dict(schedule=dtypes.ScheduleType.Sequential, storage=dtypes.StorageType.Default)


def _nested(state, entry):
    return [n for n in state.scope_subgraph(entry).nodes() if isinstance(n, nodes.NestedSDFG)]


def test_two_outputs_through_one_tile():
    sdfg, state, pexit, a, b, centry = _stencil_sdfg()
    sdfg.validate()
    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(8,), **CPU))
    sdfg.validate()
    # One map over the tiles is left; inside it the producer body once (for both outputs) and the consumer body.
    tops = _top_maps(state)
    assert len(tops) == 1
    assert len(_nested(state, tops[0])) == 2
    assert "a" not in {n.data for n in state.data_nodes()} and "b" not in {n.data for n in state.data_nodes()}
    # The producer runs over the tile plus its halo: a is read at j + 1, b at j, so 9 cells per tile of 8.
    tile_map = next(m for m in state.scope_subgraph(tops[0]).nodes() if isinstance(m, nodes.MapEntry)
                    and m is not tops[0] and m is not centry)
    assert sdfg.arrays[next(n.data for n in state.data_nodes() if n.data.startswith("a_tile"))].shape == (9,)
    assert tile_map.map.schedule == dtypes.ScheduleType.Sequential

    rng = np.random.default_rng(0)
    x = rng.random(N + 2)
    out = np.zeros(N)
    sdfg(x=x, out=out, n=N)
    expected = (x[1:N + 1] + x[2:N + 2]) - (x[0:N] * x[1:N + 1])
    assert np.allclose(out, expected)


def test_tile_does_not_divide_the_range():
    sdfg, state, pexit, a, b, centry = _stencil_sdfg()
    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(24,), **CPU))
    sdfg.validate()
    rng = np.random.default_rng(1)
    n = 50  # 50 = 2 tiles of 24 and one of 2
    x = rng.random(n + 2)
    out = np.zeros(n)
    sdfg(x=x, out=out, n=n)
    expected = (x[1:n + 1] + x[2:n + 2]) - (x[0:n] * x[1:n + 1])
    assert np.allclose(out, expected)


def test_gpu_transformation_keeps_the_tile_structure():
    sdfg, state, pexit, a, b, centry = _stencil_sdfg()
    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(32,)))
    sdfg.apply_gpu_transformations()
    outer = _top_maps(state)[0]
    assert outer.map.schedule == dtypes.ScheduleType.GPU_Device
    inner = [m for m in state.scope_subgraph(outer).nodes() if isinstance(m, nodes.MapEntry) and m is not outer]
    assert len(inner) == 2 and all(m.map.schedule == dtypes.ScheduleType.GPU_ThreadBlock for m in inner)
    tiles = {n.data: sdfg.arrays[n.data] for n in state.scope_subgraph(outer).data_nodes() if "_tile" in n.data}
    assert tiles and all(d.storage == dtypes.StorageType.GPU_Shared for d in tiles.values())
    sdfg.validate()


def test_producer_scope_with_a_tasklet_feeding_the_body():
    """The producer's scope holds a tasklet whose result the body reads: the tile copies the whole scope."""
    n = dace.symbol("n")
    sdfg = dace.SDFG("tile_scope_tasklet")
    sdfg.add_array("x", [n + 2], dace.float64)
    sdfg.add_transient("a", [n + 1], dace.float64)
    sdfg.add_transient("b", [n + 1], dace.float64)
    sdfg.add_scalar("s", dace.float64, transient=True)
    sdfg.add_array("out", [n], dace.float64)
    state = sdfg.add_state("main")
    x, a, b, out = state.add_read("x"), state.add_access("a"), state.add_access("b"), state.add_write("out")
    pentry, pexit = state.add_map("producer", {"i": "0:n + 1"})
    scale = state.add_tasklet("scale", {"v"}, {"o"}, "o = 2 * v")
    state.add_memlet_path(x, pentry, scale, dst_conn="v", memlet=dace.Memlet("x[i]"))
    s_node = state.add_access("s")
    state.add_edge(scale, "o", s_node, None, dace.Memlet("s[0]"))
    body = dace.SDFG("producer_body")
    body.add_array("x", [n + 2], dace.float64)
    body.add_scalar("s", dace.float64)
    body.add_array("a", [n + 1], dace.float64)
    body.add_array("b", [n + 1], dace.float64)
    body.add_symbol("i", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"x1", "sv"}, {"oa", "ob"}, "oa = sv + x1\nob = sv * x1")
    st.add_edge(st.add_read("x"), None, t, "x1", dace.Memlet("x[i + 1]"))
    st.add_edge(st.add_read("s"), None, t, "sv", dace.Memlet("s[0]"))
    st.add_edge(t, "oa", st.add_write("a"), None, dace.Memlet("a[i]"))
    st.add_edge(t, "ob", st.add_write("b"), None, dace.Memlet("b[i]"))
    pbody = state.add_nested_sdfg(body, {"x", "s"}, {"a", "b"}, {"i": "i", "n": "n"})
    state.add_memlet_path(x, pentry, pbody, dst_conn="x", memlet=dace.Memlet("x[0:n + 2]"))
    state.add_edge(s_node, None, pbody, "s", dace.Memlet("s[0]"))
    state.add_memlet_path(pbody, pexit, a, src_conn="a", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(pbody, pexit, b, src_conn="b", memlet=dace.Memlet("b[i]"))
    centry, cexit = state.add_map("consumer", {"j": "0:n"})
    cbody = state.add_nested_sdfg(_point_consumer_body(n), {"a", "b"}, {"out"}, {"j": "j", "n": "n"})
    state.add_memlet_path(a, centry, cbody, dst_conn="a", memlet=dace.Memlet("a[j + 1]"))
    state.add_memlet_path(b, centry, cbody, dst_conn="b", memlet=dace.Memlet("b[j]"))
    state.add_memlet_path(cbody, cexit, out, src_conn="out", memlet=dace.Memlet("out[j]"))
    sdfg.validate()
    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(8,), **CPU))
    sdfg.validate()
    assert len(_top_maps(state)) == 1
    rng = np.random.default_rng(2)
    xv = rng.random(N + 2)
    res = np.zeros(N)
    sdfg(x=xv, out=res, n=N)
    expected = (2 * xv[1:N + 1] + xv[2:N + 2]) - (2 * xv[0:N] * xv[1:N + 1])
    assert np.allclose(res, expected)


def test_member_read_of_a_slab_write_through_a_tile():
    from tests.transformations.otf_stencil_fusion_test import _slab_write_member_read_sdfg
    sdfg, state, pexit, a, centry = _slab_write_member_read_sdfg()
    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(8,), **CPU))
    sdfg.validate()
    assert len(_top_maps(state)) == 1
    rng = np.random.default_rng(4)
    x = rng.random((N + 1, 2))
    out = np.zeros(N)
    sdfg(x=x, out=out)
    assert np.allclose(out, x[1:N + 1, 1] * 2)


def test_member_reads_by_tasklets_through_a_tile():
    """The consumer is a tasklet per member (no nested SDFG), each reading one member of the slab the producer
    writes, and writes one of them back in place."""
    from tests.transformations.otf_stencil_fusion_test import V, _slab_body_all_members
    sdfg = dace.SDFG("tile_slab_member_tasklets")
    sdfg.add_array("x", [N + 1, V], dace.float64)
    sdfg.add_transient("a", [N + 1, V], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    state = sdfg.add_state("main")
    x, a, out = state.add_read("x"), state.add_access("a"), state.add_write("out")
    pentry, pexit = state.add_map("producer", {"i": f"0:{N + 1}"})
    pbody = state.add_nested_sdfg(_slab_body_all_members(), {"x"}, {"a"}, {"i": "i"})
    state.add_memlet_path(x, pentry, pbody, dst_conn="x", memlet=dace.Memlet(f"x[0:{N + 1}, 0:{V}]"))
    state.add_memlet_path(pbody, pexit, a, src_conn="a", memlet=dace.Memlet(f"a[i, 0:{V}]"))
    centry, cexit = state.add_map("consumer", {"j": f"0:{N}"})
    t0 = state.add_tasklet("t0", {"v"}, {"w"}, "w = v + 1")  # member 0 of cell j + 1, updated in place
    t1 = state.add_tasklet("t1", {"v", "u"}, {"o"}, "o = v - u")  # member 1 of cell j, member 0 of cell j + 1 (updated)
    state.add_memlet_path(a, centry, t0, dst_conn="v", memlet=dace.Memlet("a[j + 1, 0]"))
    a_upd = state.add_access("a")
    state.add_edge(t0, "w", a_upd, None, dace.Memlet("a[j + 1, 0]"))
    state.add_edge(a_upd, None, t1, "u", dace.Memlet("a[j + 1, 0]"))
    state.add_memlet_path(a, centry, t1, dst_conn="v", memlet=dace.Memlet("a[j, 1]"))
    state.add_memlet_path(t1, cexit, out, src_conn="o", memlet=dace.Memlet("out[j]"))
    sdfg.validate()
    rng = np.random.default_rng(5)
    xv = rng.random((N + 1, V))
    expected = np.zeros(N)
    sdfg(x=xv, out=expected)
    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(8,), **CPU))
    sdfg.validate()
    got = np.zeros(N)
    sdfg(x=xv, out=got)
    assert np.allclose(got, expected), (got[:4], expected[:4])


def test_in_place_consumer_takes_the_tile():
    """A consumer that reads the producer's array and writes it back through the same connector (a normalisation of
    the values, say) takes the array from the tile: the update stays local to the iteration, as the array is dead
    after the consumer."""
    sdfg, state, pexit, a, b, centry = _in_place_sdfg(read_later=False)
    sdfg.validate()
    n = 40
    rng = np.random.default_rng(6)
    xv = rng.random(n + 2)
    expected = np.zeros(n)
    sdfg(x=xv, out=expected, n=n)
    assert TileStencilFusion.can_be_applied_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                                               options=dict(tile_sizes=(8,), **CPU))
    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(8,), **CPU))
    sdfg.validate()
    got = np.zeros(n)
    sdfg(x=xv, out=got, n=n)
    assert np.allclose(got, expected)


def test_in_place_producer_is_left_to_the_recomputing_fusion():
    """A producer that reads the array it writes (``a[j] = f(a[j - 1], a[j + 1])``) feeds the consumer: its tile copy
    would rebind ``a`` to the tile buffer for the write and still read the array through the same connector."""
    sdfg, state, pexit, a, b, centry = _stencil_sdfg()
    pentry = state.entry_node(pexit)
    body = next(n for n in state.scope_subgraph(pentry).nodes() if isinstance(n, nodes.NestedSDFG))
    body.sdfg.add_array("a_old", sdfg.arrays["a"].shape, dace.float64)
    body.add_in_connector("a_old")
    pentry.add_in_connector("IN_a_old")
    pentry.add_out_connector("OUT_a_old")
    state.add_edge(state.add_read("a"), None, pentry, "IN_a_old", dace.Memlet.from_array("a", sdfg.arrays["a"]))
    state.add_edge(pentry, "OUT_a_old", body, "a_old", dace.Memlet(f"a[{pentry.map.params[0]}]"))
    sdfg.validate()
    assert not TileStencilFusion.can_be_applied_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                                                   options=dict(tile_sizes=(8,), **CPU))


def _variable_producer_body() -> dace.SDFG:
    """``a[i, v] = x[i] + v * x[i + 1]``: the producer runs over the points and the components."""
    body = dace.SDFG("variable_producer_body")
    body.add_array("x", [N + 2], dace.float64)
    body.add_array("a", [N + 1, V], dace.float64)
    body.add_symbol("i", dace.int64)
    body.add_symbol("v", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"x0", "x1"}, {"o"}, "o = x0 + v * x1")
    xr = st.add_read("x")
    st.add_edge(xr, None, t, "x0", dace.Memlet("x[i]"))
    st.add_edge(xr, None, t, "x1", dace.Memlet("x[i + 1]"))
    st.add_edge(t, "o", st.add_write("a"), None, dace.Memlet("a[i, v]"))
    return body


def _variable_consumer_body() -> dace.SDFG:
    """``out[j] = sum_v (v + 1) * a[j + 1, v]``: reads the whole slab of components at an offset."""
    body = dace.SDFG("variable_consumer_body")
    body.add_array("a", [N + 1, V], dace.float64)
    body.add_array("out", [N], dace.float64)
    body.add_symbol("j", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"c"}, {"r"}, "r = " + " + ".join(f"{v + 1} * c[{v}]" for v in range(V)))
    st.add_edge(st.add_read("a"), None, t, "c", dace.Memlet(f"a[j + 1, 0:{V}]"))
    st.add_edge(t, "r", st.add_write("out"), None, dace.Memlet("out[j]"))
    return body


def test_variable_dimension_of_the_producer_through_a_tile():
    """The producer's components are a map dimension (the variables of a point on threads); the consumer reads
    the whole slab of components at a shift: the tile holds every component, the producer runs over the tile and all
    components, the consumer takes its slab from the tile."""
    sdfg = dace.SDFG("tile_variable_dimension")
    sdfg.add_array("x", [N + 2], dace.float64)
    sdfg.add_transient("a", [N + 1, V], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    state = sdfg.add_state("main")
    x, a, out = state.add_read("x"), state.add_access("a"), state.add_write("out")
    pentry, pexit = state.add_map("producer", {"i": f"0:{N + 1}", "v": f"0:{V}"})
    pbody = state.add_nested_sdfg(_variable_producer_body(), {"x"}, {"a"}, {"i": "i", "v": "v"})
    state.add_memlet_path(x, pentry, pbody, dst_conn="x", memlet=dace.Memlet(f"x[0:{N + 2}]"))
    state.add_memlet_path(pbody, pexit, a, src_conn="a", memlet=dace.Memlet("a[i, v]"))
    centry, cexit = state.add_map("consumer", {"j": f"0:{N}"})
    cbody = state.add_nested_sdfg(_variable_consumer_body(), {"a"}, {"out"}, {"j": "j"})
    state.add_memlet_path(a, centry, cbody, dst_conn="a", memlet=dace.Memlet(f"a[j + 1, 0:{V}]"))
    state.add_memlet_path(cbody, cexit, out, src_conn="out", memlet=dace.Memlet("out[j]"))
    sdfg.validate()

    TileStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry,
                               options=dict(tile_sizes=(8,), **CPU))
    sdfg.validate()
    assert {m.map.label for m in _top_maps(state)} == {"tile_consumer"} or len(_top_maps(state)) == 1
    tile = sdfg.arrays[next(n.data for n in state.data_nodes() if n.data.startswith("a_tile"))]
    assert tuple(tile.shape) == (8, V)  # a tile of 8 points (the read at j + 1 shifts, no halo) and every component
    tile_map = next(m for m in state.scope_subgraph(_top_maps(state)[0]).nodes() if isinstance(m, nodes.MapEntry)
                    and m is not _top_maps(state)[0] and m is not centry)
    assert "v" in tile_map.map.params

    rng = np.random.default_rng(3)
    xv = rng.random(N + 2)
    res = np.zeros(N)
    sdfg(x=xv, out=res)
    expected = sum((v + 1) * (xv[1:N + 1] + v * xv[2:N + 2]) for v in range(V))
    assert np.allclose(res, expected)
