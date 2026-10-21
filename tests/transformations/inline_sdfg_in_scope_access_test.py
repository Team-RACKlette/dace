# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""InlineSDFG and a nested SDFG in a map whose state reads, inside an inner map, a node of an output it wrote."""
import dace
import numpy as np
from dace.sdfg import nodes
from dace.transformation.dataflow import MapCollapse
from dace.transformation.interstate import InlineSDFG

N = 16


def averaging_sdfg():
    """``for j: for k: q[1, 2, k, j] = q[0, 2, k, j] + q[0, 2, k, j - 1]; q[1, 3, k, j] = ...`` with the ``k`` loop
    a map in the nested SDFG of the ``j`` map, and both statements sharing one node of ``q`` in the ``k`` map."""
    sdfg = dace.SDFG("inline_in_scope_access")
    sdfg.add_array("q", [2, 4, N, N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("outer", dict(j=f"1:{N}"))

    inner = dace.SDFG("inner")
    inner.add_array("q", [2, 4, N, N], dace.float64)
    inner.add_symbol("j", dace.int64)
    istate = inner.add_state()
    ime, imx = istate.add_map("inner", dict(k=f"0:{N}"))
    first = istate.add_tasklet("first", {"a", "b"}, {"o"}, "o = a + b")
    second = istate.add_tasklet("second", {"a", "b"}, {"o"}, "o = a + b")
    shared = istate.add_access("q")
    q_in = istate.add_read("q")
    istate.add_memlet_path(q_in, ime, first, dst_conn="a", memlet=dace.Memlet("q[0, 2, k, j]"))
    istate.add_memlet_path(q_in, ime, first, dst_conn="b", memlet=dace.Memlet("q[0, 2, k, j - 1]"))
    istate.add_edge(first, "o", shared, None, dace.Memlet("q[1, 2, k, j]"))
    istate.add_edge(shared, None, second, "a", dace.Memlet("q[0, 3, k, j]"))
    istate.add_edge(shared, None, second, "b", dace.Memlet("q[0, 3, k, j - 1]"))
    istate.add_memlet_path(second, imx, istate.add_write("q"), src_conn="o", memlet=dace.Memlet("q[1, 3, k, j]"))

    node = state.add_nested_sdfg(inner, {"q"}, {"q"}, {"j": "j"})
    state.add_memlet_path(state.add_read("q"), me, node, dst_conn="q", memlet=dace.Memlet(f"q[0, 2:4, 0:{N}, j - 1:j + 1]"))
    state.add_memlet_path(node, mx, state.add_write("q"), src_conn="q", memlet=dace.Memlet(f"q[1, 2:4, 0:{N}, j]"))
    sdfg.validate()
    return sdfg, node


def test_a_node_of_an_output_read_inside_an_inner_map_is_inlined():
    sdfg, node = averaging_sdfg()
    q = np.random.rand(2, 4, N, N)
    ref = q.copy()
    for v in (2, 3):
        ref[1, v, :, 1:] = ref[0, v, :, 1:] + ref[0, v, :, :-1]

    assert InlineSDFG.can_be_applied_to(sdfg, nested_sdfg=node)
    InlineSDFG.apply_to(sdfg, nested_sdfg=node)
    assert not any(isinstance(n, nodes.NestedSDFG) for n in sdfg.start_state.nodes())
    # The point of inlining: the two maps now collapse into one.
    assert sdfg.apply_transformations(MapCollapse) == 1
    sdfg.validate()
    sdfg(q=q)
    assert np.allclose(q, ref)


def test_a_top_level_output_node_read_again_in_a_map_stays_nested():
    """The node the inlining would reconnect to the outer map exit has a further read: its write would not reach
    the exit, so the nested SDFG stays."""
    sdfg = dace.SDFG("inline_top_level_reread")
    sdfg.add_array("c", [N], dace.float64)
    sdfg.add_array("d", [N], dace.float64)
    state = sdfg.add_state()
    me, mx = state.add_map("outer", dict(i=f"0:{N}"))

    inner = dace.SDFG("inner")
    inner.add_array("c", [N], dace.float64)
    inner.add_array("d", [N], dace.float64)
    inner.add_symbol("i", dace.int64)
    istate = inner.add_state()
    init = istate.add_tasklet("init", {}, {"o"}, "o = 2")
    square = istate.add_tasklet("square", {"a"}, {"o"}, "o = a * a")
    c = istate.add_access("c")
    istate.add_edge(init, "o", c, None, dace.Memlet("c[i]"))
    istate.add_edge(c, None, square, "a", dace.Memlet("c[i]"))
    istate.add_edge(square, "o", istate.add_write("d"), None, dace.Memlet("d[i]"))

    node = state.add_nested_sdfg(inner, {}, {"c", "d"}, {"i": "i"})
    state.add_nedge(me, node, dace.Memlet())
    state.add_memlet_path(node, mx, state.add_write("c"), src_conn="c", memlet=dace.Memlet("c[i]"))
    state.add_memlet_path(node, mx, state.add_write("d"), src_conn="d", memlet=dace.Memlet("d[i]"))
    sdfg.validate()
    assert not InlineSDFG.can_be_applied_to(sdfg, nested_sdfg=node)
