# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for OTFStencilFusion: a stencil producer recomputed inside its consumer."""

import numpy as np

import dace
from dace.sdfg import nodes
from dace.transformation.dataflow.otf_stencil_fusion import OTFStencilFusion

N = 64
V = 2


def _producer_body(n) -> dace.SDFG:
    """``a[i] = x[i] + x[i + 1]`` and ``b[i] = x[i] * x[i + 1]``: two outputs from one body."""
    body = dace.SDFG("producer_body")
    body.add_array("x", [n + 2], dace.float64)
    body.add_array("a", [n + 1], dace.float64)
    body.add_array("b", [n + 1], dace.float64)
    body.add_symbol("i", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"x0", "x1"}, {"oa", "ob"}, "oa = x0 + x1\nob = x0 * x1")
    rx = st.add_read("x")
    st.add_edge(rx, None, t, "x0", dace.Memlet("x[i]"))
    st.add_edge(rx, None, t, "x1", dace.Memlet("x[i + 1]"))
    st.add_edge(t, "oa", st.add_write("a"), None, dace.Memlet("a[i]"))
    st.add_edge(t, "ob", st.add_write("b"), None, dace.Memlet("b[i]"))
    return body


def _point_consumer_body(n) -> dace.SDFG:
    """``out[j] = a[j + 1] - b[j]``: the two outputs read at different offsets."""
    body = dace.SDFG("consumer_body")
    body.add_array("a", [n + 1], dace.float64)
    body.add_array("b", [n + 1], dace.float64)
    body.add_array("out", [n], dace.float64)
    body.add_symbol("j", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"va", "vb"}, {"o"}, "o = va - vb")
    st.add_edge(st.add_read("a"), None, t, "va", dace.Memlet("a[j + 1]"))
    st.add_edge(st.add_read("b"), None, t, "vb", dace.Memlet("b[j]"))
    st.add_edge(t, "o", st.add_write("out"), None, dace.Memlet("out[j]"))
    return body


def _stencil_sdfg():
    n = dace.symbol("n")
    sdfg = dace.SDFG("otf_stencil_point")
    sdfg.add_array("x", [n + 2], dace.float64)
    sdfg.add_transient("a", [n + 1], dace.float64)
    sdfg.add_transient("b", [n + 1], dace.float64)
    sdfg.add_array("out", [n], dace.float64)
    state = sdfg.add_state("main")
    x, a, b, out = state.add_read("x"), state.add_access("a"), state.add_access("b"), state.add_write("out")

    pentry, pexit = state.add_map("producer", {"i": "0:n + 1"})
    pbody = state.add_nested_sdfg(_producer_body(n), {"x"}, {"a", "b"}, {"i": "i", "n": "n"})
    state.add_memlet_path(x, pentry, pbody, dst_conn="x", memlet=dace.Memlet("x[0:n + 2]"))
    state.add_memlet_path(pbody, pexit, a, src_conn="a", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(pbody, pexit, b, src_conn="b", memlet=dace.Memlet("b[i]"))

    centry, cexit = state.add_map("consumer", {"j": "0:n"})
    cbody = state.add_nested_sdfg(_point_consumer_body(n), {"a", "b"}, {"out"}, {"j": "j", "n": "n"})
    state.add_memlet_path(a, centry, cbody, dst_conn="a", memlet=dace.Memlet("a[j + 1]"))
    state.add_memlet_path(b, centry, cbody, dst_conn="b", memlet=dace.Memlet("b[j]"))
    state.add_memlet_path(cbody, cexit, out, src_conn="out", memlet=dace.Memlet("out[j]"))
    return sdfg, state, pexit, a, b, centry


def _top_maps(state):
    scopes = state.scope_dict()
    return [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scopes[n] is None]


def test_two_outputs_read_at_different_offsets():
    sdfg, state, pexit, a, b, centry = _stencil_sdfg()
    sdfg.validate()
    OTFStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)
    # The producer stays while its other output has a reader.
    assert {m.map.label for m in _top_maps(state)} == {"producer", "consumer"}
    OTFStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=b, second_map_entry=centry)
    assert {m.map.label for m in _top_maps(state)} == {"consumer"}
    assert "a" not in {n.data for n in state.data_nodes() if not sdfg.arrays[n.data].transient}
    # Two recomputations: one per read offset.
    replicas = [n for n in state.scope_subgraph(centry).nodes() if isinstance(n, nodes.NestedSDFG)]
    assert len(replicas) == 3
    sdfg.validate()

    rng = np.random.default_rng(0)
    x = rng.random(N + 2)
    out = np.zeros(N)
    sdfg(x=x, out=out, n=N)
    expected = (x[1:N + 1] + x[2:N + 2]) - (x[0:N] * x[1:N + 1])
    assert np.allclose(out, expected)


def _slab_producer_body() -> dace.SDFG:
    """``a[i, v] = x[i, v] * (v + 1)``."""
    body = dace.SDFG("slab_producer_body")
    body.add_array("x", [N + 1, V], dace.float64)
    body.add_array("a", [N + 1, V], dace.float64)
    body.add_symbol("i", dace.int64)
    body.add_symbol("v", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"xv"}, {"av"}, "av = xv * (v + 1)")
    st.add_edge(st.add_read("x"), None, t, "xv", dace.Memlet("x[i, v]"))
    st.add_edge(t, "av", st.add_write("a"), None, dace.Memlet("a[i, v]"))
    return body


def _slab_consumer_body() -> dace.SDFG:
    """Doubles the slab ``a[j + 1, :]`` in place, then ``out[j] = a[j + 1, 0] + a[j + 1, 1]``."""
    body = dace.SDFG("slab_consumer_body")
    body.add_array("a", [N + 1, V], dace.float64)
    body.add_array("out", [N], dace.float64)
    body.add_symbol("j", dace.int64)
    first = body.add_state("double")
    t = first.add_tasklet("t", {"a0", "a1"}, {"o0", "o1"}, "o0 = 2 * a0\no1 = 2 * a1")
    ra, wa = first.add_read("a"), first.add_write("a")
    first.add_edge(ra, None, t, "a0", dace.Memlet("a[j + 1, 0]"))
    first.add_edge(ra, None, t, "a1", dace.Memlet("a[j + 1, 1]"))
    first.add_edge(t, "o0", wa, None, dace.Memlet("a[j + 1, 0]"))
    first.add_edge(t, "o1", wa, None, dace.Memlet("a[j + 1, 1]"))
    second = body.add_state_after(first, "sum")
    t2 = second.add_tasklet("t2", {"a0", "a1"}, {"o"}, "o = a0 + a1")
    ra2 = second.add_read("a")
    second.add_edge(ra2, None, t2, "a0", dace.Memlet("a[j + 1, 0]"))
    second.add_edge(ra2, None, t2, "a1", dace.Memlet("a[j + 1, 1]"))
    second.add_edge(t2, "o", second.add_write("out"), None, dace.Memlet("out[j]"))
    return body


def test_slab_read_with_in_place_update():
    sdfg = dace.SDFG("otf_stencil_slab")
    sdfg.add_array("x", [N + 1, V], dace.float64)
    sdfg.add_transient("a", [N + 1, V], dace.float64)
    sdfg.add_array("out", [N], dace.float64)
    state = sdfg.add_state("main")
    x, a, a2, out = state.add_read("x"), state.add_access("a"), state.add_access("a"), state.add_write("out")

    pentry, pexit = state.add_map("producer", {"i": f"0:{N + 1}", "v": f"0:{V}"})
    pbody = state.add_nested_sdfg(_slab_producer_body(), {"x"}, {"a"}, {"i": "i", "v": "v"})
    state.add_memlet_path(x, pentry, pbody, dst_conn="x", memlet=dace.Memlet(f"x[0:{N + 1}, 0:{V}]"))
    state.add_memlet_path(pbody, pexit, a, src_conn="a", memlet=dace.Memlet("a[i, v]"))

    centry, cexit = state.add_map("consumer", {"j": f"0:{N}"})
    cbody = state.add_nested_sdfg(_slab_consumer_body(), {"a"}, {"a", "out"}, {"j": "j"})
    state.add_memlet_path(a, centry, cbody, dst_conn="a", memlet=dace.Memlet(f"a[j + 1, 0:{V}]"))
    state.add_memlet_path(cbody, cexit, a2, src_conn="a", memlet=dace.Memlet(f"a[j + 1, 0:{V}]"))
    state.add_memlet_path(cbody, cexit, out, src_conn="out", memlet=dace.Memlet("out[j]"))
    sdfg.validate()

    OTFStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)
    assert {m.map.label for m in _top_maps(state)} == {"consumer"}
    assert all(n.data != "a" for n in state.data_nodes())  # the in-place update stays in the iteration
    inner = [n for n in state.scope_subgraph(centry).nodes() if isinstance(n, nodes.MapEntry) and n is not centry]
    assert len(inner) == 1 and inner[0].map.params == ["v"]  # the slab read became an inner iteration
    sdfg.validate()

    rng = np.random.default_rng(1)
    x = rng.random((N + 1, V))
    out = np.zeros(N)
    sdfg(x=x, out=out)
    expected = 2 * (x[1:, 0] * 1 + x[1:, 1] * 2)
    assert np.allclose(out, expected)


def _slab_write_producer_body() -> dace.SDFG:
    """``out[i, v] = inp[i] + v`` for every component ``v``: the body writes the whole slab ``out[i, 0:V]``."""
    body = dace.SDFG("slab_write_producer_body")
    body.add_array("inp", [N + 1], dace.float64)
    body.add_array("out", [N + 1, V], dace.float64)
    body.add_symbol("i", dace.int64)
    st = body.add_state("body")
    st.add_mapped_tasklet("components", {"v": f"0:{V}"}, {"xi": dace.Memlet("inp[i]")},
                          "ov = xi + v", {"ov": dace.Memlet("out[i, v]")},
                          external_edges=True)
    return body


def _slab_read_consumer_body() -> dace.SDFG:
    """``res[j] = sum_v (v + 1) * out[j + 1, v]``: reads the whole slab ``out[j + 1, 0:V]`` at an offset."""
    body = dace.SDFG("slab_read_consumer_body")
    body.add_array("out", [N + 1, V], dace.float64)
    body.add_array("res", [N], dace.float64)
    body.add_symbol("j", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"c"}, {"r"}, "r = " + " + ".join(f"{v + 1} * c[{v}]" for v in range(V)))
    st.add_edge(st.add_read("out"), None, t, "c", dace.Memlet(f"out[j + 1, 0:{V}]"))
    st.add_edge(t, "r", st.add_write("res"), None, dace.Memlet("res[j]"))
    return body


def test_slab_write_producer():
    sdfg = dace.SDFG("otf_stencil_slab_write")
    sdfg.add_array("inp", [N + 1], dace.float64)
    sdfg.add_transient("out", [N + 1, V], dace.float64)
    sdfg.add_array("res", [N], dace.float64)
    state = sdfg.add_state("main")
    inp, out, res = state.add_read("inp"), state.add_access("out"), state.add_write("res")

    # The producer is a map over the points only; its body writes all V components of a point per iteration.
    pentry, pexit = state.add_map("producer", {"i": f"0:{N + 1}"})
    pbody = state.add_nested_sdfg(_slab_write_producer_body(), {"inp"}, {"out"}, {"i": "i"})
    state.add_memlet_path(inp, pentry, pbody, dst_conn="inp", memlet=dace.Memlet(f"inp[0:{N + 1}]"))
    state.add_memlet_path(pbody, pexit, out, src_conn="out", memlet=dace.Memlet(f"out[i, 0:{V}]"))

    centry, cexit = state.add_map("consumer", {"j": f"0:{N}"})
    cbody = state.add_nested_sdfg(_slab_read_consumer_body(), {"out"}, {"res"}, {"j": "j"})
    state.add_memlet_path(out, centry, cbody, dst_conn="out", memlet=dace.Memlet(f"out[j + 1, 0:{V}]"))
    state.add_memlet_path(cbody, cexit, res, src_conn="res", memlet=dace.Memlet("res[j]"))
    sdfg.validate()

    OTFStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=out, second_map_entry=centry)
    assert {m.map.label for m in _top_maps(state)} == {"consumer"}
    assert all(n.data != "out" for n in state.data_nodes())  # the intermediate is gone
    # The slab stays whole in the per-iteration buffer; one recomputation of the producer body, no inner map.
    assert sdfg.arrays["out_otf"].shape == (V, )
    scope = state.scope_subgraph(centry).nodes()
    assert len([n for n in scope if isinstance(n, nodes.NestedSDFG)]) == 2
    assert not [n for n in scope if isinstance(n, nodes.MapEntry) and n is not centry]
    sdfg.validate()

    rng = np.random.default_rng(2)
    inp = rng.random(N + 1)
    res = np.zeros(N)
    sdfg(inp=inp, res=res)
    expected = sum((v + 1) * (inp[1:] + v) for v in range(V))
    assert np.allclose(res, expected)


def _in_place_consumer_body(n) -> dace.SDFG:
    """``a[j] *= 2`` and ``b[j] *= 3``: the consumer updates both producer outputs in place and writes nothing else."""
    body = dace.SDFG("in_place_consumer_body")
    body.add_array("a", [n + 1], dace.float64)
    body.add_array("b", [n + 1], dace.float64)
    body.add_symbol("j", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"va", "vb"}, {"oa", "ob"}, "oa = 2 * va\nob = 3 * vb")
    st.add_edge(st.add_read("a"), None, t, "va", dace.Memlet("a[j]"))
    st.add_edge(st.add_read("b"), None, t, "vb", dace.Memlet("b[j]"))
    st.add_edge(t, "oa", st.add_write("a"), None, dace.Memlet("a[j]"))
    st.add_edge(t, "ob", st.add_write("b"), None, dace.Memlet("b[j]"))
    return body


def _in_place_sdfg(read_later: bool):
    """The producer of ``_stencil_sdfg``, a consumer that only updates ``a`` and ``b`` in place, and a copy of ``x``
    into ``out``; with ``read_later``, a second state reads the updated ``a`` into ``out``."""
    n = dace.symbol("n")
    sdfg = dace.SDFG("otf_stencil_in_place")
    sdfg.add_array("x", [n + 2], dace.float64)
    sdfg.add_transient("a", [n + 1], dace.float64)
    sdfg.add_transient("b", [n + 1], dace.float64)
    sdfg.add_array("out", [n], dace.float64)
    state = sdfg.add_state("main")
    x, a, b = state.add_read("x"), state.add_access("a"), state.add_access("b")
    a2, b2 = state.add_access("a"), state.add_access("b")

    pentry, pexit = state.add_map("producer", {"i": "0:n + 1"})
    pbody = state.add_nested_sdfg(_producer_body(n), {"x"}, {"a", "b"}, {"i": "i", "n": "n"})
    state.add_memlet_path(x, pentry, pbody, dst_conn="x", memlet=dace.Memlet("x[0:n + 2]"))
    state.add_memlet_path(pbody, pexit, a, src_conn="a", memlet=dace.Memlet("a[i]"))
    state.add_memlet_path(pbody, pexit, b, src_conn="b", memlet=dace.Memlet("b[i]"))

    centry, cexit = state.add_map("consumer", {"j": "0:n"})
    cbody = state.add_nested_sdfg(_in_place_consumer_body(n), {"a", "b"}, {"a", "b"}, {"j": "j", "n": "n"})
    state.add_memlet_path(a, centry, cbody, dst_conn="a", memlet=dace.Memlet("a[j]"))
    state.add_memlet_path(b, centry, cbody, dst_conn="b", memlet=dace.Memlet("b[j]"))
    state.add_memlet_path(cbody, cexit, a2, src_conn="a", memlet=dace.Memlet("a[j]"))
    state.add_memlet_path(cbody, cexit, b2, src_conn="b", memlet=dace.Memlet("b[j]"))

    state.add_mapped_tasklet("copy", {"k": "0:n"}, {"v": dace.Memlet("x[k]")}, "o = v", {"o": dace.Memlet("out[k]")},
                             external_edges=True)
    if read_later:
        after = sdfg.add_state_after(state, "after")
        after.add_mapped_tasklet("use", {"k": "0:n"}, {"v": dace.Memlet("a[k]")}, "o = v",
                                 {"o": dace.Memlet("out[k]")}, external_edges=True)
    return sdfg, state, pexit, a, b, centry


def test_consumer_left_without_outputs_is_removed():
    sdfg, state, pexit, a, b, centry = _in_place_sdfg(read_later=False)
    sdfg.validate()
    OTFStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)
    # The consumer still writes b, so it stays, and so does the producer of b.
    assert {m.map.label for m in _top_maps(state)} == {"producer", "consumer", "copy_map"}
    OTFStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=b, second_map_entry=centry)
    # All the consumer wrote were in-place updates nothing reads: it goes, and with it the producer.
    assert {m.map.label for m in _top_maps(state)} == {"copy_map"}
    assert not {"a", "b"} & {n.data for n in state.data_nodes()}
    state.scope_tree()
    sdfg.validate()

    x = np.random.default_rng(3).random(N + 2)
    out = np.zeros(N)
    sdfg(x=x, out=out, n=N)
    assert np.allclose(out, x[:N])


def test_in_place_update_read_later_is_not_fused():
    sdfg, state, pexit, a, b, centry = _in_place_sdfg(read_later=True)
    sdfg.validate()
    # The update of a must reach the state after this one: it cannot stay local to the consumer's iteration.
    assert not OTFStencilFusion.can_be_applied_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)
    # Nothing reads b after the consumer, but the producer could not go (a is read later): fusing b alone would
    # add the producer's work to the consumer's.
    assert not OTFStencilFusion.can_be_applied_to(sdfg, first_map_exit=pexit, array=b, second_map_entry=centry)


if __name__ == "__main__":
    test_two_outputs_read_at_different_offsets()
    test_slab_read_with_in_place_update()
    test_slab_write_producer()
    test_consumer_left_without_outputs_is_removed()
    test_in_place_update_read_later_is_not_fused()


def _member_consumer_body() -> dace.SDFG:
    """``out[j] = a[j + 1, 1]``: one member of the slab the producer writes."""
    body = dace.SDFG("member_consumer_body")
    body.add_array("a", [N + 1, V], dace.float64)
    body.add_array("out", [N], dace.float64)
    body.add_symbol("j", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"a1"}, {"o"}, "o = a1")
    st.add_edge(st.add_read("a"), None, t, "a1", dace.Memlet("a[j + 1, 1]"))
    st.add_edge(t, "o", st.add_write("out"), None, dace.Memlet("out[j]"))
    return body


def _slab_write_member_read_sdfg():
    """The producer writes the slab ``a[i, 0:V]`` per iteration; the consumer reads the member ``a[j + 1, 1]``."""
    sdfg = dace.SDFG("otf_slab_member")
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
    cbody = state.add_nested_sdfg(_member_consumer_body(), {"a"}, {"out"}, {"j": "j"})
    state.add_memlet_path(a, centry, cbody, dst_conn="a", memlet=dace.Memlet("a[j + 1, 1]"))
    state.add_memlet_path(cbody, cexit, out, src_conn="out", memlet=dace.Memlet("out[j]"))
    return sdfg, state, pexit, a, centry


def _slab_body_all_members() -> dace.SDFG:
    """``a[i, v] = x[i, v] * (v + 1)`` for every member ``v`` in one body (a sequential map)."""
    body = dace.SDFG("slab_body_all_members")
    body.add_array("x", [N + 1, V], dace.float64)
    body.add_array("a", [N + 1, V], dace.float64)
    body.add_symbol("i", dace.int64)
    st = body.add_state("body")
    me, mx = st.add_map("members", {"v": f"0:{V}"}, schedule=dace.ScheduleType.Sequential)
    t = st.add_tasklet("t", {"xv"}, {"av"}, "av = xv * (v + 1)")
    st.add_memlet_path(st.add_read("x"), me, t, dst_conn="xv", memlet=dace.Memlet("x[i, v]"))
    st.add_memlet_path(t, mx, st.add_write("a"), src_conn="av", memlet=dace.Memlet("a[i, v]"))
    return body


def test_member_read_of_a_slab_write():
    sdfg, state, pexit, a, centry = _slab_write_member_read_sdfg()
    sdfg.validate()
    OTFStencilFusion.apply_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)
    sdfg.validate()
    assert {m.map.label for m in _top_maps(state)} == {"consumer"}
    rng = np.random.default_rng(3)
    x = rng.random((N + 1, V))
    out = np.zeros(N)
    sdfg(x=x, out=out)
    assert np.allclose(out, x[1:N + 1, 1] * 2)


def test_solve_with_a_symbolic_range_read():
    """A read spanning a symbolic range (a consumer tiled by an earlier fusion): the match is a range along the
    producer's parameter, whatever the order of its symbolic ends."""
    import sympy
    from dace import subsets, symbolic
    tq = symbolic.symbol("tq")
    write = subsets.Range([(symbolic.symbol("i"), symbolic.symbol("i"), 1)])
    read = subsets.Range([(tq, sympy.Min(200, tq + 31), 1)])
    matched, inner = OTFStencilFusion.solve(["i"], write, ["j"], read)
    assert matched == {} and set(inner) == {"i"}


def test_in_place_producer_is_not_fused():
    """A producer that reads the array it writes (``a[j] = f(a[j - 1], a[j + 1])``, an immersed-boundary correction)
    cannot be recomputed in its consumer: its copy would read ``a`` through the connector the recomputation narrows
    to the per-iteration buffer it writes."""
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
    assert not OTFStencilFusion.can_be_applied_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)


def _maybe_conditional_producer_body(n, conditional: bool) -> dace.SDFG:
    """``a[i] = 2 * x[i]``, under ``if x[i] > 0.5`` when ``conditional`` (THINC sharpens interface cells only)."""
    from dace.properties import CodeBlock
    from dace.sdfg.state import ConditionalBlock, ControlFlowRegion

    body = dace.SDFG("maybe_conditional_producer_body")
    body.add_array("x", [n + 2], dace.float64)
    body.add_array("a", [n + 1], dace.float64)
    body.add_symbol("i", dace.int64)
    region = body
    if conditional:
        cond = ConditionalBlock("sharpen")
        body.add_node(cond, is_start_block=True)
        region = ControlFlowRegion("interface", sdfg=body)
        cond.add_branch(CodeBlock("x[i] > 0.5"), region)
    st = region.add_state("write", is_start_block=True)
    t = st.add_tasklet("t", {"x0"}, {"o"}, "o = 2 * x0")
    st.add_edge(st.add_read("x"), None, t, "x0", dace.Memlet("x[i]"))
    st.add_edge(t, "o", st.add_write("a"), None, dace.Memlet("a[i]"))
    return body


def _sum_consumer_body(n) -> dace.SDFG:
    """``out[j] = a[j] + a[j + 1]``."""
    body = dace.SDFG("sum_consumer_body")
    body.add_array("a", [n + 1], dace.float64)
    body.add_array("out", [n], dace.float64)
    body.add_symbol("j", dace.int64)
    st = body.add_state("body")
    t = st.add_tasklet("t", {"a0", "a1"}, {"o"}, "o = a0 + a1")
    ra = st.add_read("a")
    st.add_edge(ra, None, t, "a0", dace.Memlet("a[j]"))
    st.add_edge(ra, None, t, "a1", dace.Memlet("a[j + 1]"))
    st.add_edge(t, "o", st.add_write("out"), None, dace.Memlet("out[j]"))
    return body


def _maybe_conditional_sdfg(conditional: bool):
    n = dace.symbol("n")
    sdfg = dace.SDFG("otf_stencil_conditional" if conditional else "otf_stencil_unconditional")
    sdfg.add_array("x", [n + 2], dace.float64)
    sdfg.add_transient("a", [n + 1], dace.float64)
    sdfg.add_array("out", [n], dace.float64)
    state = sdfg.add_state("main")
    x, a, out = state.add_read("x"), state.add_access("a"), state.add_write("out")
    pentry, pexit = state.add_map("producer", {"i": "0:n + 1"})
    pbody = state.add_nested_sdfg(_maybe_conditional_producer_body(n, conditional), {"x"}, {"a"}, {"i": "i", "n": "n"})
    state.add_memlet_path(x, pentry, pbody, dst_conn="x", memlet=dace.Memlet("x[0:n + 2]"))
    state.add_memlet_path(pbody, pexit, a, src_conn="a", memlet=dace.Memlet("a[i]"))
    centry, cexit = state.add_map("consumer", {"j": "0:n"})
    cbody = state.add_nested_sdfg(_sum_consumer_body(n), {"a"}, {"out"}, {"j": "j", "n": "n"})
    state.add_memlet_path(a, centry, cbody, dst_conn="a", memlet=dace.Memlet("a[j:j + 2]"))
    state.add_memlet_path(cbody, cexit, out, src_conn="out", memlet=dace.Memlet("out[j]"))
    sdfg.validate()
    return sdfg, pexit, a, centry


def test_a_producer_writing_under_a_condition_is_not_fused():
    sdfg, pexit, a, centry = _maybe_conditional_sdfg(conditional=False)
    assert OTFStencilFusion.can_be_applied_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)
    sdfg, pexit, a, centry = _maybe_conditional_sdfg(conditional=True)
    assert not OTFStencilFusion.can_be_applied_to(sdfg, first_map_exit=pexit, array=a, second_map_entry=centry)
